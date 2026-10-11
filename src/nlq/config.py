"""Load YAML configuration, apply environment overrides, and hash the result.

The API key stays in the process environment. It is never a config field, so a
config hash can group identical runs without storing a secret.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Explicit map so we do not take a dependency on pydantic-settings.
# Precedence is YAML, then .env, then variables already set in the process.
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "NLQ_LLM_PROVIDER": ("llm", "provider"),
    "NLQ_LLM_MODEL": ("llm", "model"),
    "NLQ_LLM_MODEL_VERSION": ("llm", "model_version"),
    "NLQ_LLM_TEMPERATURE": ("llm", "temperature"),
    "NLQ_DATA_DIR": ("paths", "data_dir"),
    "NLQ_RUNS_DIR": ("paths", "runs_dir"),
    "NLQ_RESULTS_DIR": ("paths", "results_dir"),
    "NLQ_BUDGET_USD": ("experiment", "budget_usd"),
}


class ConfigError(Exception):
    """Configuration file, project root, or env override could not be applied."""


class LLMSettings(BaseModel):
    """Provider identity. The model string is the only place a run records which model answered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    api_key_env: str = Field(default="NLQ_LLM_API_KEY", min_length=1)
    max_api_retries: int = Field(default=5, ge=0)


class PathSettings(BaseModel):
    """Filesystem locations. Relative values are resolved against the project root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path
    runs_dir: Path
    results_dir: Path
    prompts_dir: Path


class TpchSettings(BaseModel):
    """TPC-H scale factors written as Parquet under ``data/tpch_sf{scale}/``.

    SF10 stays optional: lineitem is about 60 million rows. ``--all`` generates
    ``scales`` only, unless the CLI is asked to include ``optional_scales``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scales: list[float] = Field(default_factory=lambda: [0.01, 0.1, 1.0])
    optional_scales: list[float] = Field(default_factory=lambda: [10.0])
    children: int = Field(default=1, ge=1)

    @field_validator("scales")
    @classmethod
    def _required_scales(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("at least one scale factor is required")
        return _positive_scales(value)

    @field_validator("optional_scales")
    @classmethod
    def _optional_scales(cls, value: list[float]) -> list[float]:
        return _positive_scales(value)


def _positive_scales(value: list[float]) -> list[float]:
    if any(item <= 0 for item in value):
        raise ValueError("scale factors must be positive")
    return value


class SparkSettings(BaseModel):
    """Spark session defaults. The row cap is what later phases log when they truncate results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    app_name: str = Field(default="nlq-eval", min_length=1)
    sql_timeout_s: int = Field(default=120, gt=0)
    result_row_cap: int = Field(default=200, gt=0)


class ExperimentSettings(BaseModel):
    """Experiment knobs shared by later runners. ``budget_usd`` of None disables the cap."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 42
    repeats: int = Field(default=3, ge=1)
    budget_usd: float | None = Field(default=None, ge=0)
    warmup_discard: int = Field(default=1, ge=0)


class S3Settings(BaseModel):
    """LangGraph limits and ablation flags. Flags default to the full pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_sql_retries: int = Field(default=3, ge=0)
    max_answer_revisions: int = Field(default=2, ge=0)
    enable_sql_validation: bool = True
    enable_answer_verification: bool = True


class PricingSettings(BaseModel):
    """USD per million tokens. Placeholder zeros until real provider prices are set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)


class AppConfig(BaseModel):
    """Resolved experiment configuration after YAML, dotenv, and env overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompts_version: str = Field(min_length=1)
    llm: LLMSettings
    paths: PathSettings
    tpch: TpchSettings = Field(default_factory=TpchSettings)
    spark: SparkSettings
    experiment: ExperimentSettings
    s3: S3Settings
    pricing: PricingSettings

    def api_key(self) -> str | None:
        """Return the provider key from the environment, or None when it is unset.

        Read at call time so the secret is not stored on the model and cannot
        enter ``config_hash`` or a log record built from ``model_dump``.
        """
        value = os.environ.get(self.llm.api_key_env)
        if value is None or value.strip() == "":
            return None
        return value

    def canonical_json(self) -> str:
        """Stable JSON of the resolved config, used as the hash input."""
        payload = self.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def config_hash(self) -> str:
        """SHA-256 of :meth:`canonical_json`. Identical configs share a hash across runs."""
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return digest


def find_project_root(start: Path) -> Path:
    """Walk parents until ``pyproject.toml`` is found.

    Relative data paths are meaningless without that anchor, so a missing root
    is an error rather than a fallback to the current directory.
    """
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(
        f"Could not find pyproject.toml at or above {start}. "
        "Paths are resolved from the project root."
    )


def load_config(path: Path | str | None = None, *, load_dotenv_file: bool = True) -> AppConfig:
    """Load ``configs/default.yaml`` (or ``path``), then apply dotenv and env overrides.

    ``load_dotenv_file`` defaults to reading ``.env`` at the project root.
    Existing process variables win over that file (``override=False``).
    """
    if path is None:
        project_root = find_project_root(Path.cwd())
        config_path = project_root / "configs" / "default.yaml"
    else:
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        if not config_path.is_file():
            raise ConfigError(f"Config file not found: {config_path}")
        project_root = find_project_root(config_path)

    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    if load_dotenv_file:
        env_path = project_root / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False, encoding="utf-8")

    data = _read_yaml_mapping(config_path)
    _apply_env_overrides(data, os.environ)
    _resolve_paths(data, project_root)

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {config_path}: {exc}") from exc


def _read_yaml_mapping(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config file must be a YAML mapping: {config_path}")
    return loaded


def _apply_env_overrides(data: dict[str, Any], environ: Mapping[str, str]) -> None:
    for env_name, (section, field_name) in ENV_OVERRIDES.items():
        if env_name not in environ:
            continue
        section_value = data.get(section)
        if not isinstance(section_value, dict):
            raise ConfigError(f"Cannot apply {env_name}: config section '{section}' is missing")
        raw_value = environ[env_name]
        if env_name == "NLQ_BUDGET_USD" and raw_value.strip() == "":
            section_value[field_name] = None
        else:
            section_value[field_name] = raw_value


def _resolve_paths(data: dict[str, Any], project_root: Path) -> None:
    paths = data.get("paths")
    if not isinstance(paths, dict):
        return
    for key, value in list(paths.items()):
        if not isinstance(value, (str, Path)):
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        paths[key] = candidate.resolve()
