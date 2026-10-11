"""TPC-H generation: scale labels, deterministic SF0.01 export, CLI scale selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlq.config import load_config
from nlq.data.generate_tpch import (
    TPCH_TABLES,
    dataset_directory,
    generate_tpch,
    scale_label,
    scales_to_generate,
)

SF001_ROWS = {
    "region": 5,
    "nation": 25,
    "supplier": 100,
    "customer": 1500,
    "part": 2000,
    "partsupp": 8000,
    "orders": 15000,
    "lineitem": 60175,
}


def test_scale_labels() -> None:
    assert scale_label(0.01) == "sf0.01"
    assert scale_label(0.1) == "sf0.1"
    assert scale_label(1) == "sf1"
    assert scale_label(10) == "sf10"
    assert dataset_directory(Path("data"), 0.01) == Path("data") / "tpch_sf0.01"
    with pytest.raises(ValueError):
        scale_label(0)


def test_scales_to_generate_keeps_sf10_optional(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project_root)
    config = load_config(load_dotenv_file=False)
    assert scales_to_generate(config, [0.01], all_scales=False, include_sf10=False) == [0.01]
    assert scales_to_generate(config, None, all_scales=True, include_sf10=False) == pytest.approx([0.01, 0.1, 1.0])
    with_sf10 = scales_to_generate(config, None, all_scales=True, include_sf10=True)
    assert with_sf10 == pytest.approx([0.01, 0.1, 1.0, 10.0])
    with pytest.raises(ValueError, match="not both"):
        scales_to_generate(config, [0.01], all_scales=True, include_sf10=False)


def test_sf001_export_is_deterministic(tmp_path: Path) -> None:
    first = generate_tpch(0.01, tmp_path / "a")
    second = generate_tpch(0.01, tmp_path / "b")

    assert first.scale_label == "sf0.01"
    assert first.seed_mode == "duckdb-tpch-reference-fixed"
    assert first.children == 1
    assert [table.name for table in first.tables] == list(TPCH_TABLES)
    counts = {table.name: table.row_count for table in first.tables}
    assert counts == SF001_ROWS
    assert first.content_hash == second.content_hash
    assert (tmp_path / "a" / "manifest.json").is_file()
    for table in TPCH_TABLES:
        assert (tmp_path / "a" / f"{table}.parquet").is_file()

    reread = type(first).read(tmp_path / "a")
    assert reread.content_hash == first.content_hash


def test_generate_refuses_existing_directory_without_overwrite(tmp_path: Path) -> None:
    directory = tmp_path / "tpch"
    generate_tpch(0.01, directory)
    with pytest.raises(FileExistsError):
        generate_tpch(0.01, directory, overwrite=False)
