"""Per-model inference parameters - what the model settings page edits.

State lives in state/model_params.json: one entry per model filename,
holding the hyperparameters audio-separator takes for that model's
architecture (segment_size, window_size, aggression, overlap, batch_size,
...). Which models to preload, the default, and the output format are NOT
here - they are plain defaults on app.config.Settings, since the main page
already picks a model and format per job.

This module is deliberately thin. The field names, defaults and (for
MDXC/MDX) validation already live in
vocal_remove.config.ModelConfig/SeparationConfig and their four
per-architecture subclasses, so nothing here redeclares them: which fields
an architecture has comes from their own_fields(), and what a saved value
becomes in practice from their read_applied(). This module only reads and
writes those fields as a plain dict. See load_params().

Importing vocal_remove.config does NOT pull in torch or audio_separator -
those arrive only through vocal_remove.separator, which vocal_remove's
__init__.py imports lazily (on first use of init_models/separate/...)
specifically so a config-only import like this one stays cheap in the web
process. See that module's docstring. This module is otherwise stdlib-only,
like db.py and jobs.py - see tests/test_jobs.py for why that matters.
"""
import json
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from vocal_remove.config import config_classes_for
from vocal_remove.errors import InvalidConfig

logger = logging.getLogger("app.model_settings")

PARAMS_FILENAME = "model_params.json"


class UnknownModelParam(ValueError):
    """A param name is not one this model's architecture supports."""


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file and rename over the real one.

    So a crash or a second writer mid-save leaves either the old file or the
    new one, never a truncated one - the same concern fetch_models.py has
    about model downloads, applied to a much smaller file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s (%s: %s)", path, type(exc).__name__, exc)
        return None


def _kind_for(annotation) -> str:
    """'bool' | 'text' | 'number', for the settings page's generic form.

    Every Optional field in vocal_remove.config is Optional[int] or
    Optional[float] (segment_size, overlap, batch_size) - None just means
    "the model's own default", so it renders the same as a number field with
    an empty value.
    """
    if annotation is bool:
        return "bool"
    if annotation is str:
        return "text"
    return "number"


def _arch_name(model_cls) -> str:
    """'mdxc' | 'mdx' | 'vr' | 'demucs', from the ModelConfig subclass name."""
    name = model_cls.__name__
    if name.endswith("ModelConfig"):
        name = name[:-len("ModelConfig")]
    return name.lower()


def arch_for(model_name: str) -> str:
    model_cls, _ = config_classes_for(model_name)
    return _arch_name(model_cls)


@dataclass
class ModelParams:
    """One model's editable params, plus enough shape for a generic form.

    `values` holds every load_time_fields + per_job_fields entry - saved
    where a save exists, the architecture's own default otherwise.
    """

    model: str
    arch: str
    load_time_fields: Tuple[str, ...]
    per_job_fields: Tuple[str, ...]
    kinds: Dict[str, str]
    values: Dict[str, Any]

    def load_time_kwargs(self) -> Dict[str, Any]:
        """Splat straight into a fresh ModelConfig(...) call."""
        return {k: self.values[k] for k in self.load_time_fields}

    def per_job_kwargs(self) -> Dict[str, Any]:
        """Splat straight into a per-job SeparationConfig(...) call."""
        return {k: self.values[k] for k in self.per_job_fields}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "arch": self.arch,
            "load_time_fields": list(self.load_time_fields),
            "per_job_fields": list(self.per_job_fields),
            "kinds": self.kinds,
            "values": self.values,
        }


def _field_specs(model_cls, sep_cls):
    load_fields = model_cls.own_fields()
    job_fields = sep_cls.own_fields()
    hints = {**typing.get_type_hints(model_cls), **typing.get_type_hints(sep_cls)}
    kinds = {name: _kind_for(hints[name]) for name in (*load_fields, *job_fields)}
    return load_fields, job_fields, kinds


def _construct(model_cls, sep_cls, load_fields, job_fields, values) -> None:
    """Build the real ModelConfig/SeparationConfig from `values`.

    This is what runs their own validation - e.g. MDXCSeparationConfig and
    MDXSeparationConfig both reject an overlap in the wrong units for their
    architecture - instead of anything reimplemented in this module. Raises
    InvalidConfig (their own exception) or TypeError (a value of the wrong
    Python type); callers decide whether that is a warning-and-fall-back
    (load_params) or a 400 (save_params, via the API).
    """
    model_cls(**{f: values[f] for f in load_fields})
    sep_cls(**{f: values[f] for f in job_fields})


def params_path_for(state_dir: Path) -> Path:
    return Path(state_dir) / PARAMS_FILENAME


def load_params(state_dir: Path, model_name: str) -> ModelParams:
    """This model's saved params, or its architecture's defaults for
    anything not saved (or not saved validly).

    Never raises: this runs on the worker's startup path, and a bad file
    should cost one model its overrides, not the server its start.
    """
    model_cls, sep_cls = config_classes_for(model_name)
    load_fields, job_fields, kinds = _field_specs(model_cls, sep_cls)
    known = set(load_fields) | set(job_fields)

    defaults = {}
    defaults.update({f: getattr(model_cls(), f) for f in load_fields})
    defaults.update({f: getattr(sep_cls(), f) for f in job_fields})

    raw = _read_json(params_path_for(state_dir)) or {}
    entry = raw.get(model_name, {})
    unknown = set(entry) - known
    if unknown:
        logger.warning("ignoring unknown param(s) for %s: %s",
                       model_name, sorted(unknown))

    values = dict(defaults)
    values.update({k: v for k, v in entry.items() if k in known})

    try:
        _construct(model_cls, sep_cls, load_fields, job_fields, values)
    except (TypeError, InvalidConfig) as exc:
        logger.warning("%s has invalid params for %s (%s); using defaults",
                       params_path_for(state_dir), model_name, exc)
        values = defaults

    return ModelParams(model=model_name, arch=_arch_name(model_cls),
                       load_time_fields=load_fields, per_job_fields=job_fields,
                       kinds=kinds, values=values)


def save_params(state_dir: Path, model_name: str,
                partial: Dict[str, Any]) -> ModelParams:
    """Merge `partial` onto the saved (or default) values, validate via the
    real vocal_remove config classes, then persist.

    Raises TypeError, InvalidConfig or UnknownModelParam without writing -
    the API turns any of those into a 400.
    """
    current = load_params(state_dir, model_name)
    known = set(current.load_time_fields) | set(current.per_job_fields)
    unknown = set(partial) - known
    if unknown:
        raise UnknownModelParam(
            f"unknown parameter(s) for {current.arch} model {model_name!r}: "
            f"{sorted(unknown)}")

    values = dict(current.values)
    values.update(partial)

    model_cls, sep_cls = config_classes_for(model_name)
    _construct(model_cls, sep_cls, current.load_time_fields,
              current.per_job_fields, values)

    path = params_path_for(state_dir)
    raw = _read_json(path) or {}
    raw[model_name] = values
    _atomic_write(path, json.dumps(raw, indent=2) + "\n")

    return ModelParams(model=model_name, arch=current.arch,
                       load_time_fields=current.load_time_fields,
                       per_job_fields=current.per_job_fields,
                       kinds=current.kinds, values=values)
