"""Spark SQL execution for benchmark answers.

The session registers Parquet tables from a generated dataset and runs one
statement at a time. A job group is cancelled when the statement exceeds the
timeout, because Spark has no statement timeout that both stops the work and
returns control to the caller. Results are collected to the driver; the row
cap exists so a later narration prompt does not receive an unbounded frame.
The full row count is still computed, and ``truncated`` records the cap.

On Windows, PySpark's launcher scripts break when ``SPARK_HOME`` contains spaces
(common under OneDrive). Before the session starts we rewrite ``SPARK_HOME`` and
the Python executables to 8.3 short paths when those are available.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

from pyspark.sql import SparkSession

from nlq.config import SparkSettings
from nlq.data.generate_tpch import quote_ident
from nlq.data.views import VIEW_SELECTS, create_view_statements
from nlq.engine.results import QueryResult, call_with_timeout, json_safe, strip_trailing_semicolon


class SparkEngine:
    """One local Spark session. Execute calls are serial; the session is not shared across threads."""

    def __init__(
        self,
        settings: SparkSettings | None = None,
        *,
        warehouse_dir: Path | None = None,
        master: str = "local[*]",
    ) -> None:
        self._settings = settings if settings is not None else SparkSettings()
        self._warehouse = Path(warehouse_dir) if warehouse_dir is not None else Path(
            tempfile.mkdtemp(prefix="nlq-spark-")
        )
        self._warehouse.mkdir(parents=True, exist_ok=True)
        self._session = _build_session(self._settings.app_name, self._warehouse, master)
        self._registered: list[str] = []

    @property
    def session(self) -> SparkSession:
        return self._session

    def register_parquet_dataset(self, dataset_dir: Path) -> list[str]:
        """Register each ``*.parquet`` file as a temp view named after the file stem."""
        directory = Path(dataset_dir)
        files = _parquet_files(directory)
        names: list[str] = []
        for path in files:
            name = path.stem
            quote_ident(name)
            # Short paths avoid Windows launcher issues when the dataset sits under OneDrive.
            self._session.read.parquet(_windows_safe_path(path)).createOrReplaceTempView(name)
            names.append(name)
        self._registered = names
        return names

    def register_business_views(self) -> list[str]:
        """Add the renamed business views. The model is shown only these names."""
        for statement in create_view_statements(temporary=True):
            self._session.sql(statement)
        created = list(VIEW_SELECTS)
        self._registered.extend(created)
        return created

    def cache_state(self) -> list[str]:
        """Names of registered views that Spark still has cached."""
        cached: list[str] = []
        for name in self._registered:
            if self._session.catalog.isCached(name):
                cached.append(name)
        return cached

    def clear_cache(self) -> list[str]:
        """Drop cached tables and return whatever is still cached.

        Experiments compare cold runs. The returned list is the cache state
        to log after the clear; it should be empty.
        """
        self._session.catalog.clearCache()
        return self.cache_state()

    def execute(
        self,
        sql: str,
        *,
        timeout_s: float | None = None,
        apply_row_cap: bool = False,
        row_cap: int | None = None,
    ) -> QueryResult:
        """Run one statement. The cap limits rows returned, not the reported ``row_count``."""
        statement = strip_trailing_semicolon(sql)
        timeout = self._settings.sql_timeout_s if timeout_s is None else timeout_s
        cap = self._settings.result_row_cap if row_cap is None else row_cap
        if apply_row_cap and cap < 1:
            raise ValueError("row_cap must be positive when apply_row_cap is set")
        job_group = f"nlq-{uuid.uuid4().hex}"

        def action() -> QueryResult:
            # setJobGroup is thread-local, so it has to run on the worker thread.
            self._session.sparkContext.setJobGroup(job_group, "nlq", True)
            started = time.perf_counter()
            try:
                if apply_row_cap:
                    count_frame = self._session.sql(
                        f"SELECT COUNT(*) AS _nlq_n FROM ({statement}) AS _nlq_count"
                    )
                    total = int(count_frame.collect()[0][0])
                    data_frame = self._session.sql(
                        f"SELECT * FROM ({statement}) AS _nlq_cap LIMIT {int(cap)}"
                    )
                    raw_rows = data_frame.collect()
                    truncated = total > len(raw_rows)
                    used_cap: int | None = cap
                else:
                    data_frame = self._session.sql(statement)
                    raw_rows = data_frame.collect()
                    total = len(raw_rows)
                    truncated = False
                    used_cap = None
                columns = list(data_frame.columns)
                rows = [
                    {column: json_safe(row[column]) for column in columns}
                    for row in raw_rows
                ]
                return QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=total,
                    truncated=truncated,
                    row_cap=used_cap,
                    exec_time_s=time.perf_counter() - started,
                )
            finally:
                # Newer Spark builds dropped clearJobGroup; an empty group is enough.
                self._session.sparkContext.setJobGroup("", "", False)

        def cancel() -> None:
            self._session.sparkContext.cancelJobGroup(job_group)

        return call_with_timeout(action, timeout_s=timeout, cancel=cancel, sql=statement)

    def stop(self) -> None:
        self._session.stop()


def _build_session(app_name: str, warehouse: Path, master: str) -> SparkSession:
    """Start a local session. UI and shuffle defaults are trimmed for repeated benchmark runs."""
    _prepare_windows_spark_env()
    python = _windows_safe_path(sys.executable)
    os.environ["PYSPARK_PYTHON"] = python
    os.environ["PYSPARK_DRIVER_PYTHON"] = python
    active = SparkSession.getActiveSession()
    if active is not None:
        active.stop()
    warehouse = Path(_windows_safe_path(warehouse))
    derby = warehouse / "derby"
    local = warehouse / "local"
    derby.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.warehouse.dir", str(warehouse / "warehouse"))
        .config("spark.local.dir", str(local))
        .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={derby}")
        .getOrCreate()
    )


def _prepare_windows_spark_env() -> None:
    """Point ``SPARK_HOME`` at a space-free path so ``spark-class.cmd`` can start Java."""
    if os.name != "nt":
        return
    import pyspark

    spark_home = Path(pyspark.__file__).resolve().parent
    os.environ["SPARK_HOME"] = _windows_safe_path(spark_home)
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        os.environ["JAVA_HOME"] = _windows_safe_path(java_home)


def _windows_safe_path(path: Path | str) -> str:
    """Return an 8.3 short path when the long path contains spaces or parentheses."""
    text = str(Path(path).resolve())
    if os.name != "nt":
        return text
    if " " not in text and "(" not in text and ")" not in text:
        return text
    get_short = ctypes.windll.kernel32.GetShortPathNameW
    buffer = ctypes.create_unicode_buffer(520)
    length = get_short(text, buffer, len(buffer))
    if length == 0:
        return text
    return buffer.value


def _parquet_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Parquet dataset directory not found: {directory}")
    files = sorted(path for path in directory.glob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No Parquet files in {directory}")
    return files
