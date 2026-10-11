# agentic-nlq-eval

Research harness comparing three ways of answering natural-language business questions over a relational dataset. It measures execution accuracy, claim-level faithfulness, and latency as the data grows. It is an experiment, not a product. Experiments stay on the command line. A separate read-only page can show the benchmark and finished logs.

The three systems share one model, one schema prompt, and one narration prompt:

- **S1** generates SQL once, executes it, and narrates the rows.
- **S2** is S1 plus an error-driven retry. The cap is three retries after the first attempt.
- **S3** is a LangGraph pipeline: question understanding, schema grounding, query planning, SQL generation, SQL validation, execution, result analysis, answer and provenance, and answer verification.
- **S3-V** is the same graph with SQL validation turned off, so the first SQL statement is executed.
- **S3-A** is the same graph with answer verification turned off.

## Demonstration

```bash
uv run python -m nlq.demo.server
```

Open `http://127.0.0.1:8765`. The page lists the benchmark questions and, when `runs/*.jsonl` exists, the measured accuracy, faithfulness, SQL, and latency. It does not call a model or start Spark.

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/). Java is required because Spark runs the answering queries.

```bash
uv sync
```

On a network that intercepts TLS, retry with `uv sync --system-certs`.

Copy `.env.example` to `.env` and set `NLQ_LLM_API_KEY`. The key is read at call time. It is not a config field, so it cannot enter the config hash or a log line.

Model name, model version, and temperature live in `configs/default.yaml`. The checked-in model is the placeholder `REPLACE_ME`. Temperature is 0. Pricing rates in that file are placeholders of zero until you set the provider's prices; the budget guard uses those rates.

Prompt text lives under `prompts/`. `prompts/VERSION` is the prompt library version written into every run. The config field `prompts_version` is kept in step with that file (`1.0.0`).

## Tests

```bash
uv run pytest
```

Tests never call a model API. The language model is a script. They do start DuckDB and, for the gold cross-check, a local Spark session. On Windows, if the checkout path contains spaces (OneDrive paths do), the Spark engine rewrites `SPARK_HOME` and the data paths to 8.3 short paths before the JVM starts.

## Generate TPC-H

```bash
uv run python -m nlq.data.generate_tpch --scale 0.01
uv run python -m nlq.data.generate_tpch --all
uv run python -m nlq.data.generate_tpch --all --include-sf10
```

Parquet lands in `data/tpch_sf{scale}/` with a `manifest.json`. DuckDB's `dbgen` has no seed argument. The reference generator is deterministic for a scale factor, and this harness calls it with `children=1`. `--all` writes SF0.01, SF0.1, and SF1. SF10 is about 60 million lineitem rows and stays opt-in.

The model never sees raw TPC-H names. Business views (`geography`, `customers`, `suppliers`, `parts`, `part_suppliers`, `sales_orders`, `order_lines`) are created in the engine session. Profit is defined once, on `order_lines`:

```text
profit = sale_amount * (1 - discount_rate) - unit_cost * quantity
```

## Validate gold SQL

Gold answers in the YAML files are the SF0.01 reference. The checker runs every TPC-H gold query on Spark and on DuckDB and fails if the normalised results differ. It also checks the YAML answer against DuckDB at the reference scale.

```bash
uv run python -m nlq.benchmark.validate_gold --scales 0.01 0.1 --data-dir data --output data/gold_hashes.json
```

Add `--nyc-dir data/nyc_fixture` to cross-check the NYC scaffold against the five-trip demonstration fixture. That fixture is what the NYC gold answers describe. A real TLC download is separate and does not run during tests:

```bash
uv run python -m nlq.data.nyc_tlc --directory data/nyc_raw
```

The download helper is `nlq.data.nyc_tlc.download_nyc_months`. After a real month is cleaned, recompute the NYC gold answers before treating them as results. Cleaning keeps January and February 2023, drops bad fares, distances, passenger counts, and unknown payment or zone ids, and keeps the first 50,000 rows by pickup time.

The question files under `src/nlq/benchmark/questions/` are a scaffold of five questions per tier and five NYC questions. They are original questions, not the published TPC-H query set. Experiment 2 asks for up to nine questions per tier and therefore uses all five until more questions are added.

## Run an experiment

```bash
uv run python -m nlq.eval.run_experiment --experiment 1 --systems s1 s2 s3 s3v s3a --scale sf1 --repeats 3 --seed 42 --resume
uv run python -m nlq.eval.run_experiment --experiment 2 --systems s1 s2 s3 --repeats 3 --seed 42 --budget-usd 25
uv run python -m nlq.eval.run_experiment --experiment 3 --systems s1 s2 s3 --repeats 3 --seed 42
```

- Experiment 1 runs every TPC-H question at the scale given by `--scale` (default `sf1`).
- Experiment 2 runs a stratified subset, up to nine questions per tier, on S1, S2, and S3 at SF0.01, SF0.1, SF1, and SF10.
- Experiment 3 runs the NYC questions on the demonstration fixture.

Run order is shuffled with `--seed`. One warm-up question is executed first and stored with `repeat` 0 and `warmup: true`. Analysis drops it. Spark's cache is cleared before every run and the cache state is written into the JSONL payload.

Each measured run has a stable id from the experiment, system, question, scale, repeat, warm-up flag, config hash, model, and prompt version. `--resume` keeps the existing JSONL and skips ids that already have a `question_run` line. Without `--resume`, the log file for that experiment is replaced.

`--budget-usd` (or `budget_usd` in the config, or `NLQ_BUDGET_USD`) aborts before the next model call once estimated spend has reached the cap. Spend is input and output tokens times the rates in `configs/default.yaml`. A call that starts under the cap can finish over it, because the token count is known only after the response.

Logs go to `runs/experiment_{n}.jsonl`. Each line is a `node_call` or a `question_run`. A question run stores the config hash, model name and version, prompt version, timestamps, the answer object, raw model text, accuracy, faithfulness, latency components, and the cache state.

At a scale other than the question's reference scale, execution accuracy and answer values are scored by re-running the gold SQL on that scale. The YAML values remain the SF0.01 audit.

## Figures

```bash
uv run python -m nlq.analysis.reproduce --runs runs/experiment_1.jsonl --out results
```

This writes tidy CSVs and PNG figures under `results/`. Warm-up rows are excluded. Repeats are majority-voted per question before McNemar's test, so three repeats of one question are not three independent samples. A tie counts as incorrect. Pairwise p-values are Holm-adjusted, with a bootstrap interval on the accuracy difference. Latency figures use medians. Faithfulness intervals, when computed in `nlq.analysis.stats`, resample whole questions.

## Configuration

`nlq.config.load_config` reads `configs/default.yaml`, then `.env` without overriding variables already set, then:

- `NLQ_LLM_PROVIDER`
- `NLQ_LLM_MODEL`
- `NLQ_LLM_MODEL_VERSION`
- `NLQ_LLM_TEMPERATURE`
- `NLQ_DATA_DIR`
- `NLQ_RUNS_DIR`
- `NLQ_RESULTS_DIR`
- `NLQ_BUDGET_USD` (empty clears the cap)

Relative paths resolve against the directory that contains `pyproject.toml`. `AppConfig.config_hash()` is a SHA-256 of that resolved config. The hash changes if the checkout moves, because the stored paths are absolute.
