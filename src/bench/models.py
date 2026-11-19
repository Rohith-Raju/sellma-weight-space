"""Reads the model entries in config/models.toml.

Each [section] is one model version. The file is plain TOML; this module only adds two
things: relative paths are resolved against the project folder, and unknown names fail
with a message that lists the valid ones.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench import paths


@dataclass(frozen=True)
class ModelSpec:
    name: str                  # the section name, e.g. "qwen-9b-full"
    backend: str               # "hf", "llamacpp", "random" or "oracle"
    path: Path | None          # model weights (hf) or GGUF file (llamacpp); None for dry runs
    tokenizer: Path | None     # Hugging Face folder whose chat template formats the prompt
    raw: dict[str, Any] = field(default_factory=dict)   # the TOML table as written, for run.json

    def setting(self, key: str, default: Any = None) -> Any:
        """Any extra key in the entry, such as seed, parent or llama_cpp_bin."""
        return self.raw.get(key, default)


def load_model_specs(config_path: Path | None = None) -> dict[str, ModelSpec]:
    config_path = config_path or paths.MODELS_CONFIG
    with open(config_path, "rb") as f:
        tables = tomllib.load(f)
    return {name: _spec_from_table(name, table) for name, table in tables.items()}


def get_model(name: str) -> ModelSpec:
    specs = load_model_specs()
    if name not in specs:
        known = ", ".join(sorted(specs)) or "(none)"
        raise KeyError(f"unknown model {name!r} in config/models.toml; known models: {known}")
    return specs[name]


def _spec_from_table(name: str, table: dict[str, Any]) -> ModelSpec:
    if "backend" not in table:
        raise ValueError(f"model {name!r} in config/models.toml has no backend")
    return ModelSpec(
        name=name,
        backend=str(table["backend"]),
        path=_resolve(table.get("path")),
        tokenizer=_resolve(table.get("tokenizer")),
        raw=dict(table),
    )


def _resolve(value: str | None) -> Path | None:
    """Relative paths in the config are relative to the project folder, not the working directory."""
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else paths.ROOT / path
