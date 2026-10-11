"""Config loading, env overrides, and secret-free hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlq.config import LLMSettings, ConfigError, load_config


def test_default_yaml_loads(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project_root)
    config = load_config(load_dotenv_file=False)

    assert config.llm.temperature == 0
    assert config.prompts_version == "1.0.0"
    assert config.llm.model == "REPLACE_ME"
    assert config.llm.model_version == "REPLACE_ME"
    assert config.spark.result_row_cap == 200
    assert config.tpch.scales == pytest.approx([0.01, 0.1, 1.0])
    assert config.tpch.optional_scales == pytest.approx([10.0])
    assert config.tpch.children == 1
    assert config.s3.max_sql_retries == 3
    assert config.s3.max_answer_revisions == 2
    assert config.paths.data_dir == (project_root / "data").resolve()
    assert config.paths.runs_dir == (project_root / "runs").resolve()
    assert config.paths.results_dir == (project_root / "results").resolve()
    assert config.paths.prompts_dir == (project_root / "prompts").resolve()


def test_llm_temperature_defaults_to_zero() -> None:
    settings = LLMSettings(provider="openai", model="m", model_version="v")
    assert settings.temperature == 0


def test_env_override_does_not_rewrite_yaml(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = project_root / "configs" / "default.yaml"
    before = config_path.read_bytes()
    monkeypatch.setenv("NLQ_LLM_MODEL", "override-model")
    monkeypatch.chdir(project_root)

    config = load_config(config_path, load_dotenv_file=False)

    assert config.llm.model == "override-model"
    assert config_path.read_bytes() == before


def test_relative_and_absolute_path_overrides(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(project_root)
    monkeypatch.setenv("NLQ_DATA_DIR", "custom_data")
    relative = load_config(load_dotenv_file=False)
    assert relative.paths.data_dir == (project_root / "custom_data").resolve()

    absolute = tmp_path / "abs-data"
    monkeypatch.setenv("NLQ_DATA_DIR", str(absolute))
    resolved = load_config(load_dotenv_file=False)
    assert resolved.paths.data_dir == absolute.resolve()


def test_empty_budget_override_clears_cap(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(project_root)
    monkeypatch.setenv("NLQ_BUDGET_USD", "")
    config = load_config(load_dotenv_file=False)
    assert config.experiment.budget_usd is None


def test_missing_config_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError, match="not found"):
        load_config(missing)


def test_missing_project_root_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "default.yaml"
    config_path.write_text("prompts_version: '0'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="pyproject.toml"):
        load_config(config_path)


def test_dotenv_fills_unset_and_process_env_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Pin the variable so dotenv mutations are undone with the test.
    monkeypatch.setenv("NLQ_LLM_MODEL", "sentinel")
    monkeypatch.delenv("NLQ_LLM_MODEL")
    root = _write_temp_project(tmp_path)
    (root / ".env").write_text("NLQ_LLM_MODEL=from-dotenv\n", encoding="utf-8")

    from_file = load_config(root / "configs" / "default.yaml")
    assert from_file.llm.model == "from-dotenv"

    monkeypatch.setenv("NLQ_LLM_MODEL", "from-process")
    from_process = load_config(root / "configs" / "default.yaml")
    assert from_process.llm.model == "from-process"


def test_config_hash_is_stable_and_excludes_api_key(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(project_root)
    monkeypatch.setenv("NLQ_LLM_API_KEY", "super-secret-value")
    config = load_config(load_dotenv_file=False)

    assert config.config_hash() == config.config_hash()
    assert len(config.config_hash()) == 64
    assert "super-secret-value" not in config.canonical_json()
    assert config.api_key() == "super-secret-value"

    changed = config.model_copy(
        update={"llm": config.llm.model_copy(update={"model": "other-model"})}
    )
    assert changed.config_hash() != config.config_hash()

    monkeypatch.setenv("NLQ_LLM_API_KEY", "a-different-secret")
    reloaded = load_config(load_dotenv_file=False)
    assert reloaded.config_hash() == config.config_hash()
    assert "a-different-secret" not in reloaded.canonical_json()


def _write_temp_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'temp'\n", encoding="utf-8")
    (root / "configs" / "default.yaml").write_text(
        """
prompts_version: "0.1.0"
llm:
  provider: openai
  model: yaml-model
  model_version: v1
  temperature: 0
paths:
  data_dir: data
  runs_dir: runs
  results_dir: results
  prompts_dir: prompts
tpch:
  scales: [0.01, 0.1, 1]
  optional_scales: [10]
  children: 1
spark:
  app_name: nlq-eval
  sql_timeout_s: 120
  result_row_cap: 200
experiment:
  seed: 42
  repeats: 3
  budget_usd: null
  warmup_discard: 1
s3:
  max_sql_retries: 3
  max_answer_revisions: 2
  enable_sql_validation: true
  enable_answer_verification: true
pricing:
  input_usd_per_million: 0.0
  output_usd_per_million: 0.0
""",
        encoding="utf-8",
    )
    return root
