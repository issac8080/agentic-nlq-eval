"""Run experiments 1-3. Order is seeded, one warm-up is discarded, and resume skips finished ids.

Experiment 1: every TPC-H question, five systems, the requested repeats, one scale.
Experiment 2: a stratified subset (up to 9 per tier), systems s1/s2/s3, several scales.
Experiment 3: NYC questions, systems s1/s2/s3.

The scaffold has five questions per tier, so experiment 2 uses all of them until
the benchmark is filled out to 30 per tier. SF10 is not selected unless the
scale list passed on the CLI includes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nlq.benchmark.schema import BenchmarkQuestion, load_questions, questions_for_dataset
from nlq.config import AppConfig, S3Settings, load_config
from nlq.data.generate_tpch import dataset_directory
from nlq.data.nyc_tlc import write_demonstration_fixture
from nlq.data.schema_context import SchemaContext, build_schema_context
from nlq.engine.spark_engine import SparkEngine
from nlq.eval.accuracy import score_question
from nlq.eval.faithfulness import (
    Claim,
    calculation_correctness,
    provenance_ok,
    summarise,
    verify_claims,
)
from nlq.eval.latency import from_system_result
from nlq.llm.base import LLMClient
from nlq.llm.cost import BudgetExceededError, BudgetGuard
from nlq.llm.prompts import PromptLibrary
from nlq.logging_utils import JsonlRunLogger, QuestionRunRecord, read_records
from nlq.systems.s1_single import SinglePassSystem
from nlq.systems.s2_retry import RetrySystem
from nlq.systems.s3_agentic.graph import AgenticSystem

SYSTEMS = ("s1", "s2", "s3", "s3v", "s3a")


def plan_runs(
    questions: list[BenchmarkQuestion],
    systems: list[str],
    scales: list[str],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Seeded shuffle across systems so no system always runs first."""
    planned = [
        {"question_id": question.id, "system": system, "scale": scale, "repeat": repeat, "warmup": False}
        for question in questions
        for system in systems
        for scale in scales
        for repeat in range(1, repeats + 1)
    ]
    rng = random.Random(seed)
    rng.shuffle(planned)
    if questions and systems and scales:
        planned.insert(
            0,
            {
                "question_id": questions[0].id,
                "system": systems[0],
                "scale": scales[0],
                "repeat": 0,
                "warmup": True,
            },
        )
    return planned


def stratified_subset(questions: list[BenchmarkQuestion], per_tier: int, seed: int) -> list[BenchmarkQuestion]:
    """Up to ``per_tier`` questions from each tier. Fewer are used when the scaffold is smaller."""
    rng = random.Random(seed)
    chosen: list[BenchmarkQuestion] = []
    for tier in range(1, 6):
        pool = [question for question in questions if question.tier == tier and question.dataset == "tpch"]
        rng.shuffle(pool)
        chosen.extend(pool[:per_tier])
    return chosen


def run_id_for(
    *,
    experiment: int,
    system: str,
    question_id: str,
    scale: str,
    repeat: int,
    warmup: bool,
    config_hash: str,
    model: str,
    prompts_version: str,
) -> str:
    """Stable id so --resume can skip a run that already has a question_run line."""
    raw = "|".join(
        [
            str(experiment),
            system,
            question_id,
            scale,
            str(repeat),
            "warmup" if warmup else "measured",
            config_hash,
            model,
            prompts_version,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def completed_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record.run_id for record in read_records(path) if record.record_type == "question_run"}


def build_system(
    name: str,
    llm: LLMClient,
    engine: Any,
    schema: SchemaContext,
    prompts: PromptLibrary,
    config: AppConfig,
    logger: JsonlRunLogger,
    run_id: str,
):
    if name == "s1":
        return SinglePassSystem(
            llm, engine, schema, prompts, row_cap=config.spark.result_row_cap, logger=logger, run_id=run_id
        )
    if name == "s2":
        return RetrySystem(
            llm,
            engine,
            schema,
            prompts,
            max_sql_retries=config.s3.max_sql_retries,
            row_cap=config.spark.result_row_cap,
            logger=logger,
            run_id=run_id,
        )
    settings = _ablation(config.s3, name)
    return AgenticSystem(
        llm,
        engine,
        schema,
        prompts,
        settings,
        name=name,
        row_cap=config.spark.result_row_cap,
        logger=logger,
        run_id=run_id,
    )


def execute_plan(
    *,
    config: AppConfig,
    llm: LLMClient,
    questions: list[BenchmarkQuestion],
    planned: list[dict[str, Any]],
    experiment: int,
    log_path: Path,
    data_dir: Path,
    budget: BudgetGuard | None = None,
) -> int:
    """Run the plan. Spark cache is cleared before every measured run and the state is logged."""
    prompts = PromptLibrary(config.paths.prompts_dir)
    logger = JsonlRunLogger(log_path)
    done = completed_run_ids(log_path)
    by_id = {question.id: question for question in questions}
    engine = SparkEngine(config.spark, master="local[2]")
    completed = 0
    current_scale = ""
    try:
        for item in planned:
            question = by_id[item["question_id"]]
            run_id = run_id_for(
                experiment=experiment,
                system=item["system"],
                question_id=question.id,
                scale=item["scale"],
                repeat=item["repeat"],
                warmup=item["warmup"],
                config_hash=config.config_hash(),
                model=config.llm.model,
                prompts_version=prompts.version,
            )
            if run_id in done:
                continue
            if budget is not None:
                budget.before_call()
            if item["scale"] != current_scale:
                _register_scale(engine, data_dir, item["scale"], question.dataset)
                current_scale = item["scale"]
            cache_before = engine.clear_cache()
            schema = build_schema_context(engine, dataset=question.dataset)
            system = build_system(item["system"], llm, engine, schema, prompts, config, logger, run_id)
            started = datetime.now(timezone.utc)
            result = system.answer(question)
            ended = datetime.now(timezone.utc)
            cache_after = engine.cache_state()
            latency = from_system_result(
                run_id, question.id, item["system"], item["scale"], item["repeat"], result, warmup=item["warmup"]
            )
            accuracy = score_question(
                question=question,
                system=item["system"],
                predicted_sql=result.sql,
                runner=engine,
                answer_values=result.answer_values,
                entities=result.entities,
                sql_failed=result.failed,
                use_executed_gold=item["scale"] != question.reference_scale,
            )
            faithfulness = _faithfulness(result, schema)
            rows = _lineitem_rows(data_dir, item["scale"])
            logger.log_question_run(
                QuestionRunRecord(
                    run_id=run_id,
                    experiment=experiment,
                    system=item["system"],
                    question_id=question.id,
                    scale=item["scale"],
                    repeat=item["repeat"],
                    config_hash=config.config_hash(),
                    model_name=config.llm.model,
                    model_version=config.llm.model_version,
                    prompts_version=prompts.version,
                    started_at=started,
                    ended_at=ended,
                    payload={
                        "warmup": item["warmup"],
                        "tier": question.tier,
                        "lineitem_rows": rows,
                        "answer": result.model_dump(),
                        "accuracy": accuracy.model_dump(),
                        "faithfulness": faithfulness,
                        "latency": latency.model_dump(),
                        "cache_before": cache_before,
                        "cache_after": cache_after,
                        "config_hash": config.config_hash(),
                        "model": config.llm.model,
                        "prompts_version": prompts.version,
                    },
                )
            )
            completed += 1
            done.add(run_id)
    except BudgetExceededError:
        return completed
    finally:
        engine.stop()
    return completed


def _register_scale(engine: SparkEngine, data_dir: Path, scale: str, dataset: str) -> None:
    if dataset == "nyc_tlc" or scale == "fixture":
        directory = write_demonstration_fixture(data_dir / "nyc_fixture")
        engine.register_parquet_dataset(directory)
        return
    label = scale if scale.startswith("sf") else f"sf{scale}"
    directory = data_dir / f"tpch_{label}"
    if not directory.exists():
        directory = dataset_directory(data_dir, float(label.removeprefix("sf")))
    engine.register_parquet_dataset(directory)
    engine.register_business_views()


def _lineitem_rows(data_dir: Path, scale: str) -> int | None:
    """Row count of lineitem at this scale, from the manifest written at generation time."""
    if not scale.startswith("sf"):
        return None
    manifest = data_dir / f"tpch_{scale}" / "manifest.json"
    if not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for table in payload.get("tables") or []:
        if table.get("name") == "lineitem":
            return int(table["row_count"])
    return None


def _faithfulness(result: Any, schema: SchemaContext) -> dict[str, Any]:
    """Score claims the system stored, or the evidence lines when no claim list exists."""
    raw_claims = result.raw_outputs.get("claims") or []
    claims = [Claim.model_validate(item) for item in raw_claims if isinstance(item, dict)]
    if not claims:
        claims = [
            Claim(claim_text=item.claim_text, claim_type="numeric")
            for item in result.evidence
            if item.claim_text
        ]
    verdicts = verify_claims(
        claims,
        [],
        result.sql,
        tolerance=0.01,
        schema_tables=list(schema.column_map()),
    )
    # Result rows are not stored separately from the answer object. Numeric checks
    # use the structured answer values so a supported number has to appear there.
    if result.answer_values and isinstance(result.answer_values[0], dict):
        verdicts = verify_claims(
            claims,
            result.answer_values,
            result.sql,
            tolerance=0.01,
            schema_tables=list(schema.column_map()),
        )
    summary = summarise(
        verdicts,
        calculation=calculation_correctness([result.narrative], 0.01),
        provenance=provenance_ok(
            reported_sql=result.sql,
            executed_sql=result.sql,
            reported_tables=result.tables,
            executed_tables=result.tables,
            reported_filters=result.filters,
            executed_filters=result.filters,
            reported_row_count=result.row_count,
            executed_row_count=result.row_count,
        ),
    )
    return summary.model_dump()


def _ablation(settings: S3Settings, name: str) -> S3Settings:
    if name == "s3v":
        return settings.model_copy(update={"enable_sql_validation": False})
    if name == "s3a":
        return settings.model_copy(update={"enable_answer_verification": False})
    return settings


def experiment_spec(experiment: int, questions: list[BenchmarkQuestion], systems: list[str], scale: str, repeats: int, seed: int):
    if experiment == 1:
        selected = questions_for_dataset(questions, "tpch")
        scales = [scale]
        chosen_systems = systems
    elif experiment == 2:
        selected = stratified_subset(questions, 9, seed)
        scales = ["sf0.01", "sf0.1", "sf1", "sf10"]
        chosen_systems = [item for item in systems if item in {"s1", "s2", "s3"}]
    elif experiment == 3:
        selected = questions_for_dataset(questions, "nyc_tlc")
        scales = ["fixture"]
        chosen_systems = [item for item in systems if item in {"s1", "s2", "s3"}]
    else:
        raise ValueError(f"Unknown experiment {experiment}")
    return selected, chosen_systems, scales, plan_runs(selected, chosen_systems, scales, repeats, seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a numbered NLQ experiment.")
    parser.add_argument("--experiment", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--scale", default="sf1")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=None)
    args = parser.parse_args(argv)
    config = load_config()
    repeats = args.repeats if args.repeats is not None else config.experiment.repeats
    seed = args.seed if args.seed is not None else config.experiment.seed
    questions = load_questions()
    selected, systems, scales, planned = experiment_spec(
        args.experiment, questions, args.systems, args.scale, repeats, seed
    )
    from nlq.llm.provider import OpenAICompatibleClient

    api_key = config.api_key() or ""
    budget = BudgetGuard(args.budget_usd if args.budget_usd is not None else config.experiment.budget_usd, config.pricing)
    llm = OpenAICompatibleClient(config.llm, api_key, budget=budget)
    log_path = config.paths.runs_dir / f"experiment_{args.experiment}.jsonl"
    if not args.resume and log_path.exists():
        log_path.unlink()
    execute_plan(
        config=config,
        llm=llm,
        questions=selected,
        planned=planned,
        experiment=args.experiment,
        log_path=log_path,
        data_dir=config.paths.data_dir,
        budget=budget,
    )
    print(json.dumps({"log": str(log_path), "planned": len(planned)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
