"""Generate stakeholder PDFs for agentic-nlq-eval using fpdf2 (no extra install)."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUT_DIR = Path(__file__).resolve().parent

_ASCII_MAP = str.maketrans(
    {
        "\u2014": "-",  # em dash
        "\u2013": "-",  # en dash
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
        "\u2192": "->",
        "\u00d7": "x",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2248": "~",
    }
)


def to_ascii(text: str) -> str:
    return text.translate(_ASCII_MAP)


class DocPDF(FPDF):
    def __init__(self, title: str):
        super().__init__(format="Letter")
        self.doc_title = title
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, to_ascii(f"{self.doc_title}  |  Page {self.page_no()}"), align="C")

    def cover(self, title: str, lines: list[str]) -> None:
        self.add_page()
        self.ln(50)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(26, 35, 50)
        self.multi_cell(0, 10, to_ascii(title), align="C")
        self.ln(6)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(70, 85, 100)
        for line in lines:
            self.multi_cell(0, 7, to_ascii(line), align="C")
            self.ln(1)

    def h1(self, text: str) -> None:
        self.ln(4)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(26, 35, 50)
        self.multi_cell(0, 8, to_ascii(text))
        self.ln(1)

    def h2(self, text: str) -> None:
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(36, 52, 71)
        self.multi_cell(0, 7, to_ascii(text))
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, to_ascii(text))
        self.ln(1.5)

    def bold_body(self, label: str, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, to_ascii(f"{label}{text}"))
        self.ln(1)

    def bullets(self, items: list[str]) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        for item in items:
            self.set_x(self.l_margin)
            self.multi_cell(0, 5.5, to_ascii(f"  -  {item}"))
            self.ln(0.5)
        self.ln(1)

    def code(self, text: str) -> None:
        self.set_fill_color(244, 246, 248)
        self.set_font("Courier", "", 7.5)
        self.set_text_color(20, 20, 20)
        self.set_x(self.l_margin)
        self.multi_cell(self.epw, 4, to_ascii(text.rstrip() + "\n"), fill=True)
        self.ln(2)

    def qa(self, num: int, question: str, difficulty: str, category: str, answer: str, code: str | None = None) -> None:
        needed = 28 + (12 if code else 0)
        if self.get_y() > self.h - needed:
            self.add_page()
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(13, 33, 55)
        self.multi_cell(0, 5.5, to_ascii(f"Q{num}. {question}"))
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(90, 90, 90)
        self.multi_cell(0, 4.5, to_ascii(f"Difficulty: {difficulty}  |  Category: {category}"))
        self.ln(0.5)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.2, to_ascii(f"Answer. {answer}"))
        self.ln(1)
        if code:
            self.code(code)


def build_nontechnical(path: Path) -> None:
    pdf = DocPDF("Non-Technical Documentation")
    pdf.cover(
        "agentic-nlq-eval",
        [
            "Non-Technical Project Documentation",
            "Research harness for comparing AI systems that answer",
            "business questions in plain English",
            "",
            "Audience: stakeholders, clients, and new team members",
            "Aligned with project version 0.1.0",
        ],
    )

    pdf.add_page()
    pdf.h1("1. Executive Summary")
    pdf.body(
        'Imagine you ask a company database a question in everyday language—'
        '"Which region made the most profit last quarter?"—and an AI system turns that into '
        "a database query, runs it, and explains the answer in words. That capability is often "
        "called natural-language querying (NLQ)."
    )
    pdf.body(
        "agentic-nlq-eval is not a customer-facing product. It is a research test bench: a "
        "controlled lab where several AI answering strategies compete on the same questions, "
        "the same data, and the same scoring rules. The goal is to learn which approach is "
        "more accurate, more trustworthy, and faster as datasets get larger."
    )
    pdf.body(
        "Core value proposition: give decision-makers evidence—not opinions—about whether a "
        "simple one-shot AI, a retry-on-error AI, or a multi-step agent pipeline is worth the "
        "extra complexity and cost for business Q&A over structured data."
    )
    pdf.body("In plain terms, the project answers three business questions:")
    pdf.bullets(
        [
            "Does the system get the numbers right? (execution accuracy against known-correct answers)",
            "Does the written explanation stick to the evidence? (faithfulness of claims)",
            "How long and how expensive is each approach? (latency and optional spend caps)",
        ]
    )
    pdf.body(
        "Think of it like a standardized driving test for three different car designs: same "
        "road, same rules, same scoring sheet—so comparisons are fair."
    )

    pdf.h1("2. Core Features")
    pdf.h2("2.1 Three answering strategies (plus two variants)")
    pdf.body(
        "All strategies use the same language model, the same description of the business "
        "tables, and the same style of final explanation. That keeps the comparison focused "
        "on how the system works, not on which model or prompt library was lucky."
    )
    pdf.bullets(
        [
            "S1 — Single pass: Writes one database query, runs it, then explains the rows. No second chance. (Analogy: one draft, submit.)",
            "S2 — Retry: Like S1, but if the query fails checks or execution, the error is fed back. Up to three retries after the first try. (Analogy: teacher marks errors; student revises.)",
            "S3 — Agentic pipeline: Breaks the job into stages—understand the question, map it to tables, plan the query, write SQL, validate, run, analyze results, write the answer with sources, then verify claims. (Analogy: a checklist of specialist steps.)",
            "S3-V / S3-A: Ablation variants of S3. S3-V turns off SQL validation; S3-A turns off answer verification—so researchers can see what each check is worth.",
        ]
    )

    pdf.h2("2.2 Fair test data and business-friendly tables")
    pdf.body(
        "The main dataset is TPC-H—synthetic sales and supply-chain data that can be generated "
        "at different sizes (from tiny lab samples to tens of millions of line items). The AI "
        "never sees cryptic raw table names; it sees business views such as customers, suppliers, "
        "sales orders, and order lines. Profit is defined once in one place so every system "
        "computes it the same way."
    )
    pdf.body(
        "A smaller NYC taxi demonstration set supports a second experiment style: questions "
        "over trip data, first on a tiny fixture for demos and tests, with an optional path "
        "to download real public months later."
    )

    pdf.h2("2.3 Questions with known-correct (gold) answers")
    pdf.body(
        "Each benchmark question ships with a carefully checked correct answer at a reference "
        "data size. Those gold answers are cross-checked on two different query engines so "
        "scoring is trustworthy. Questions are original business-style prompts (not copied "
        "from the classic TPC-H query list) and are grouped into difficulty tiers."
    )

    pdf.h2("2.4 Scoring stakeholders can interpret")
    pdf.bullets(
        [
            "Accuracy — Did the system's numbers match the gold answer after fair normalization?",
            "Faithfulness — When the system writes prose, do its claims match the table it just computed?",
            "Latency — How long did planning, model calls, and database work take?",
            "Cost guard (optional) — Experiments can stop before the next AI call once an estimated dollar budget is reached.",
        ]
    )

    pdf.h2("2.5 Reproducible experiments and figures")
    pdf.body(
        "Runs are logged line-by-line to durable files. Order is shuffled with a seed so no "
        "system always goes first. A warm-up question is discarded from analysis. Researchers "
        "can pause and resume without redoing finished work. A reproduce step turns logs into "
        "tidy tables and charts (accuracy comparisons, latency pictures, statistical tests)."
    )

    pdf.h2("2.6 Local demonstration browser (optional)")
    pdf.body(
        "A small local demo server can show questions and completed run summaries in a browser. "
        "It is for inspection and demos; the research core remains a command-line experiment "
        "harness, not a hosted SaaS product."
    )

    pdf.h1("3. User Journey / Workflow")
    pdf.body(
        "Different people touch the project at different stages. Below is the end-to-end "
        "journey from empty laptop to charts leadership can discuss."
    )

    pdf.h2("Stage A — Set up the lab")
    pdf.bullets(
        [
            "Install the project environment (Python tooling and Java for the big-data engine).",
            "Copy the example secrets file and place the AI provider key in the local environment only—keys never go into result files or config fingerprints.",
            "Set the model name and pricing rates in the shared config file so cost estimates are meaningful.",
        ]
    )

    pdf.h2("Stage B — Prepare data and questions")
    pdf.bullets(
        [
            "Generate TPC-H Parquet datasets at the needed sizes (small scales for daily work; large scale optional).",
            "Optionally prepare the NYC demonstration fixture.",
            "Validate gold SQL so reference answers agree across engines.",
            "Expand the question bank over time (today's files are a scaffold of example questions per tier).",
        ]
    )

    pdf.h2("Stage C — Run a fair competition")
    pdf.body(
        "Experiment 1 — All TPC-H questions, all systems (including ablation variants), one "
        "chosen data size, several repeats."
    )
    pdf.body(
        "Experiment 2 — A balanced sample across difficulty tiers, systems S1/S2/S3, and "
        "several data sizes (including a very large optional size) to study how performance "
        "changes as data grows."
    )
    pdf.body("Experiment 3 — NYC taxi questions on the demonstration fixture.")
    pdf.body(
        "During a run: each system answers each scheduled question; results, timings, and "
        "scores are appended to a log. If a budget is set, the harness stops cleanly before "
        "starting another model call once spend has hit the cap."
    )

    pdf.h2("Stage D — Analyze and communicate")
    pdf.bullets(
        [
            "Reproduce figures and summary tables from the logs.",
            "Compare systems with proper statistics (paired tests on majority-voted repeats, adjusted p-values).",
            "Discuss trade-offs: S3 may be slower or costlier but more reliable; S1 may be cheapest but brittle; S2 sits in between.",
            "Optionally browse completed runs in the local demo UI for storytelling.",
        ]
    )

    pdf.h1("4. What Success Looks Like")
    pdf.bullets(
        [
            "Leadership can see clear accuracy / faithfulness / latency trade-offs between simple and agentic designs.",
            "Engineers can reproduce a past run from config fingerprints, prompt versions, and logs.",
            "Researchers can turn individual checks (SQL validation, answer verification) on or off to measure their contribution.",
            "Spend stays predictable when a budget guard is enabled.",
        ]
    )

    pdf.h1("5. Scope and Current Boundaries")
    pdf.bullets(
        [
            "This is an experiment harness, not a production chatbot or BI product.",
            "There is no multi-tenant cloud UI; the demo server is local and optional.",
            "The question bank is intentionally a scaffold until more original questions are authored.",
            "Live AI experiments require a real model name and API key; automated tests use a scripted fake model so they never spend money.",
        ]
    )

    pdf.h1("6. Glossary (plain language)")
    glossary = [
        ("SQL", "The language databases understand. The AI writes it; the engine runs it."),
        ("Schema / views", "The map of business tables and columns the AI is allowed to use."),
        ("Gold answer", "The known-correct reference used for scoring."),
        ("Faithfulness", "Whether the written story matches the computed evidence."),
        ("Latency", "Wall-clock time from start to answer."),
        ("Ablation", "Turning one feature off to measure how much it mattered."),
        ("JSONL log", "A durable diary of runs—one record per line—for later analysis."),
        ("Scale factor", "How large the synthetic dataset is (bigger = more rows, longer queries)."),
    ]
    for term, meaning in glossary:
        pdf.bold_body(f"{term} — ", meaning)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, "Document generated from the agentic-nlq-eval codebase for stakeholder onboarding.")

    pdf.output(str(path))


def build_interview(path: Path) -> None:
    pdf = DocPDF("Technical Interview Guide 2027")
    pdf.cover(
        "Technical Interview Guide",
        [
            "agentic-nlq-eval stack — 2027 standards",
            "50 high-yield questions with detailed answers",
            "and production-ready code snippets",
            "",
            "Python 3.11+ · Pydantic · Text-to-SQL · LangGraph",
            "Spark & DuckDB · sqlglot · Evaluation science · Testing",
        ],
    )

    pdf.add_page()
    pdf.h1("How to use this guide")
    pdf.body(
        "Questions are grouped by difficulty. Within each band, categories rotate so you "
        "practice conceptual, architectural, and coding prompts. Answers are written to the "
        "depth expected in senior interviews in 2026-2027: trade-offs, failure modes, and "
        "runnable patterns—not buzzwords."
    )
    pdf.body(
        "Stack context: a research harness that compares S1 (single-pass text-to-SQL), S2 "
        "(error-driven retry), and S3 (LangGraph multi-node pipeline with SQL validation and "
        "answer verification ablations)."
    )

    pdf.h1("Beginner (Q1-Q15)")

    pdf.qa(
        1,
        "What problem does a text-to-SQL (NLQ) system solve?",
        "Beginner",
        "NLQ concepts",
        "It lets non-SQL users ask questions in natural language; the system produces SQL, "
        "executes it against a relational schema, and often narrates the result. The hard parts "
        "are schema grounding, correct joins/filters, and not inventing columns.",
    )
    pdf.qa(
        2,
        "Why keep the API key out of config hashes and logs?",
        "Beginner",
        "Security / config",
        "Secrets must never be persisted in reproducible fingerprints or experiment diaries. "
        "In this project the key is read from the environment at call time via an env var name "
        "stored in config, so identical experiments share a config hash without embedding credentials.",
        "# Conceptual pattern\n"
        'api_key_env = "NLQ_LLM_API_KEY"  # stored in config\n'
        "api_key = os.environ.get(api_key_env)  # read at call time only\n"
        "# Never put api_key into AppConfig fields that are hashed or logged.",
    )
    pdf.qa(
        3,
        "What is the difference between S1 and S2 in this harness?",
        "Beginner",
        "Architecture",
        "S1 generates SQL once, validates/executes, then narrates—no retry. S2 feeds validator "
        "and execution errors back into the next SQL generation, up to a shared retry cap "
        "(three retries after the first attempt). Same model and prompts; different control loop.",
    )
    pdf.qa(
        4,
        "Explain Parquet in one interview-ready sentence, then one practical benefit.",
        "Beginner",
        "Data engineering",
        "Parquet is a columnar on-disk format for analytics tables. Benefit: engines like "
        "DuckDB/Spark can read only needed columns and prune row groups, which matters when "
        "TPC-H lineitem grows to tens of millions of rows.",
    )
    pdf.qa(
        5,
        "What does temperature=0 usually buy you in evaluation harnesses?",
        "Beginner",
        "LLM ops",
        "Lower temperature reduces sampling randomness, improving run-to-run stability for "
        "benchmarking. It does not guarantee determinism across providers or versions, so the "
        "harness still uses repeats and majority voting.",
    )
    pdf.qa(
        6,
        "Why use Pydantic models for configuration?",
        "Beginner",
        "Python",
        "Pydantic validates types and constraints at load time (e.g., positive scale factors, "
        "frozen configs), fails fast on typos via extra='forbid', and serializes cleanly for hashing.",
        "from pydantic import BaseModel, ConfigDict, Field\n\n"
        "class PricingSettings(BaseModel):\n"
        "    model_config = ConfigDict(extra='forbid', frozen=True)\n"
        "    input_usd_per_million: float = Field(ge=0)\n"
        "    output_usd_per_million: float = Field(ge=0)",
    )
    pdf.qa(
        7,
        'What is a "gold" answer in a benchmark?',
        "Beginner",
        "Evaluation",
        "A trusted reference result for a question at a reference scale. Systems are scored "
        "by comparing their execution output (normalized) to gold. This project also "
        "cross-checks gold SQL on Spark and DuckDB to catch engine drift.",
    )
    pdf.qa(
        8,
        "DuckDB vs Spark: when would you pick each in a research harness?",
        "Beginner",
        "Engines",
        "DuckDB: fast local analytics, great for gold checks and small/medium Parquet. Spark: "
        "answers at larger scales and matches production-like distributed SQL; needs a JVM. "
        "This harness uses Spark for answering runs and DuckDB heavily in validation/tests.",
    )
    pdf.qa(
        9,
        "What is JSONL and why use it for experiment logs?",
        "Beginner",
        "Observability",
        "JSON Lines stores one JSON object per line. You can append safely, stream partial "
        "files, and resume by scanning completed run IDs without loading a giant nested document.",
    )
    pdf.qa(
        10,
        "Write a tiny function that estimates USD cost from token counts.",
        "Beginner",
        "Coding",
        "Cost is linear in tokens per million for input and output rates—matching the harness BudgetGuard.",
        "def estimate_cost(input_tokens, output_tokens, in_per_m, out_per_m):\n"
        "    return (input_tokens / 1_000_000) * in_per_m + \\\n"
        "           (output_tokens / 1_000_000) * out_per_m\n\n"
        "assert abs(estimate_cost(1_000_000, 500_000, 1.0, 2.0) - 2.0) < 1e-9",
    )
    pdf.qa(
        11,
        "What is schema grounding?",
        "Beginner",
        "NLQ concepts",
        "Mapping question phrases to actual tables/columns (and join paths) that exist in the "
        "allowed schema. Failures here cause invented columns or wrong grain. S3 has an "
        "explicit schema_grounding node before SQL generation.",
    )
    pdf.qa(
        12,
        "Why expose business views instead of raw TPC-H names to the model?",
        "Beginner",
        "Product / NLQ",
        "Raw warehouse names are cryptic and leak benchmark identity. Business views "
        "(customers, order_lines, ...) mirror how analysts think and keep profit definitions "
        "centralized, reducing inconsistent formulas across systems.",
    )
    pdf.qa(
        13,
        'What does pytest\'s pythonpath = ["src"] accomplish in pyproject.toml?',
        "Beginner",
        "Testing",
        "It puts the src layout on sys.path so tests can import nlq without installing the "
        "package in editable mode for every CI job—common for src-layout projects.",
    )
    pdf.qa(
        14,
        "Explain SELECT-only validation in one paragraph.",
        "Beginner",
        "Security",
        "Before execution, parse SQL and reject DDL/DML/multiple statements so the model "
        "cannot drop tables or mutate data. This harness uses sqlglot with a Spark dialect "
        "and also checks tables/columns against the business schema and bans unconstrained cross joins.",
    )
    pdf.qa(
        15,
        "Implement a safe path join that rejects path traversal for log filenames.",
        "Beginner",
        "Coding",
        "Resolve the candidate path and require it to remain under the runs directory—same idea as the demo server's _is_safe_log.",
        "from pathlib import Path\n\n"
        "def safe_log_path(runs_dir: Path, name: str) -> Path:\n"
        "    path = (runs_dir / name).resolve()\n"
        "    path.relative_to(runs_dir.resolve())  # raises if outside\n"
        '    if not path.is_file() or path.suffix != ".jsonl":\n'
        "        raise FileNotFoundError(name)\n"
        "    return path",
    )

    pdf.add_page()
    pdf.h1("Intermediate (Q16-Q35)")

    pdf.qa(
        16,
        "Design a stable run_id for resumable experiments. What fields belong in the hash?",
        "Intermediate",
        "Experiment design",
        "Include experiment number, system, question, scale, repeat, warmup flag, config hash, "
        "model, and prompts version. Exclude wall-clock timestamps and secrets. Resume skips "
        "IDs already present as question_run lines in JSONL.",
        "import hashlib\n\n"
        "def run_id_for(**parts: str) -> str:\n"
        '    material = "|".join(f"{k}={parts[k]}" for k in sorted(parts))\n'
        "    return hashlib.sha256(material.encode()).hexdigest()[:32]",
    )
    pdf.qa(
        17,
        "Why majority-vote repeats before McNemar's test?",
        "Intermediate",
        "Statistics",
        "Three stochastic repeats of the same question are not three independent samples of "
        "the population of questions. Collapsing to one vote per question avoids "
        "pseudo-replication. Ties can be treated as incorrect (as this project does).",
    )
    pdf.qa(
        18,
        "Sketch a BudgetGuard that fails closed before the next LLM call.",
        "Intermediate",
        "Coding",
        "Charge after each call; before_call raises if spent already meets/exceeds the cap. "
        "Acknowledge that an in-flight call can still finish over budget because tokens are known only afterward.",
        "class BudgetExceededError(Exception):\n"
        "    pass\n\n"
        "class BudgetGuard:\n"
        "    def __init__(self, cap_usd):\n"
        "        self.cap_usd = cap_usd\n"
        "        self.spent_usd = 0.0\n\n"
        "    def before_call(self):\n"
        "        if self.cap_usd is not None and self.spent_usd >= self.cap_usd:\n"
        "            raise BudgetExceededError('budget reached')\n\n"
        "    def charge(self, cost: float):\n"
        "        self.spent_usd += cost",
    )
    pdf.qa(
        19,
        "How does LangGraph differ from a linear Python script of LLM calls?",
        "Intermediate",
        "Agents / LangGraph",
        "LangGraph models a stateful graph with nodes and conditional edges (retry loops, "
        "ablation shortcuts). State accumulates fields (SQL, feedback, timings). You get "
        "explicit control flow, recursion limits, and easier ablations by changing edges "
        "rather than rewriting node code—as S3-V/S3-A do.",
    )
    pdf.qa(
        20,
        "Show a minimal StateGraph pattern for validate-or-retry SQL generation.",
        "Intermediate",
        "Coding",
        "Conditional edges after validation either execute or return to generation until retries exhaust.",
        "from langgraph.graph import StateGraph, START, END\n\n"
        "def build(enable_validation: bool):\n"
        "    g = StateGraph(dict)\n"
        "    g.add_node('sql_generation', generate)\n"
        "    g.add_node('sql_validation', validate)\n"
        "    g.add_node('execute', execute)\n"
        "    g.add_edge(START, 'sql_generation')\n"
        "    g.add_conditional_edges(\n"
        "        'sql_generation',\n"
        "        lambda s: 'execute' if not enable_validation else 'sql_validation',\n"
        "        {'execute': 'execute', 'sql_validation': 'sql_validation'},\n"
        "    )\n"
        "    g.add_edge('execute', END)\n"
        "    return g.compile()",
    )
    pdf.qa(
        21,
        'What should a deterministic SQL validator check beyond "parses"?',
        "Intermediate",
        "Architecture",
        "Single statement; read-only; tables/columns subset of schema; resolve aliases/CTEs; "
        "reject ambiguous columns; reject CROSS JOIN / join without ON/USING. Keep it "
        "model-free so messages are safe feedback for retries.",
    )
    pdf.qa(
        22,
        "Implement a sqlglot-based SELECT-only gate (Spark dialect).",
        "Intermediate",
        "Coding",
        "Parse with read='spark', require one Select/Union-family statement.",
        "from sqlglot import exp, parse\n\n"
        "READ_ONLY = (exp.Select, exp.Union, exp.Intersect, exp.Except)\n\n"
        "def is_single_select(sql: str):\n"
        "    try:\n"
        "        stmts = [s for s in parse(sql, read='spark') if s is not None]\n"
        "    except Exception as e:\n"
        "        return False, f'parse error: {e}'\n"
        "    if len(stmts) != 1:\n"
        "        return False, 'Exactly one SQL statement is allowed'\n"
        "    if not isinstance(stmts[0], READ_ONLY):\n"
        "        return False, 'Only a SELECT query is allowed'\n"
        "    return True, 'ok'",
    )
    pdf.qa(
        23,
        "How do you score accuracy when evaluating at a scale different from gold YAML?",
        "Intermediate",
        "Evaluation",
        "Re-run the gold SQL at the evaluation scale and compare the system's result to that "
        "dynamic gold. Keep YAML values as the SF0.01 audit trail. Normalize types, rounding, "
        "and column order per question metadata.",
    )
    pdf.qa(
        24,
        "Define claim-level faithfulness labels used in this project.",
        "Intermediate",
        "Evaluation",
        "supported, contradicted, unsupported, non_verifiable. Numbers/comparisons are checked "
        "against logged results; explanatory claims without checkable numbers stay "
        "non_verifiable; inventing a new verification query is avoided—unsupported records "
        "that no extra query was run.",
    )
    pdf.qa(
        25,
        "Write a claim number extractor and membership check against result cells.",
        "Intermediate",
        "Coding",
        "A minimal faithfulness building block: extract numbers from claim text and see if they appear in normalized cells.",
        "import re\n\n"
        r"_NUM = re.compile(r'-?\d+(?:\.\d+)?')"
        "\n\n"
        "def numbers_in(text: str):\n"
        "    return _NUM.findall(text)\n\n"
        "def claim_supported_by_cells(claim, cells):\n"
        "    cell_set = set(cells)\n"
        "    found = numbers_in(claim)\n"
        "    return bool(found) and all(n in cell_set for n in found)",
    )
    pdf.qa(
        26,
        "Why clear Spark cache between measured runs and log cache state?",
        "Intermediate",
        "Performance / fairness",
        "Cached tables can make later systems look faster. Clearing between runs reduces "
        "cross-run contamination; logging cache state documents residual effects for analysis.",
    )
    pdf.qa(
        27,
        "How would you structure PromptLibrary so JSON braces in templates are safe?",
        "Intermediate",
        "LLM ops",
        "Avoid str.format on whole prompts. Use explicit placeholders like <<SCHEMA>> and "
        "replace them. Version prompts via prompts/VERSION written into every run record.",
        "def render(template: str, **tokens: str) -> str:\n"
        "    out = template\n"
        "    for key, value in tokens.items():\n"
        "        out = out.replace(f'<<{key.upper()}>>', value)\n"
        "    return out",
    )
    pdf.qa(
        28,
        "Describe a production-ready OpenAI-compatible client without the official SDK.",
        "Intermediate",
        "Coding",
        "Use urllib against /v1/chat/completions, map HTTP 408/429/5xx to retryable errors "
        "with tenacity, return text + token counts + latency. Keep temperature and model from config.",
        "import json, time, urllib.request\n\n"
        "def chat_complete(url, api_key, model, messages, temperature=0.0):\n"
        "    body = json.dumps({\n"
        "        'model': model, 'temperature': temperature,\n"
        "        'messages': messages,\n"
        "    }).encode()\n"
        "    req = urllib.request.Request(\n"
        "        url, data=body, method='POST',\n"
        "        headers={'Authorization': f'Bearer {api_key}',\n"
        "                 'Content-Type': 'application/json'})\n"
        "    t0 = time.perf_counter()\n"
        "    with urllib.request.urlopen(req, timeout=60) as resp:\n"
        "        payload = json.loads(resp.read().decode())\n"
        "    latency = time.perf_counter() - t0\n"
        "    text = payload['choices'][0]['message']['content']\n"
        "    usage = payload.get('usage') or {}\n"
        "    return text, usage.get('prompt_tokens', 0), \\\n"
        "           usage.get('completion_tokens', 0), latency",
    )
    pdf.qa(
        29,
        "What is an ablation study in agent systems, using S3-V and S3-A?",
        "Intermediate",
        "Architecture",
        "Disable one component at a time while keeping the rest fixed. S3-V disables SQL "
        "validation (first SQL executes); S3-A disables answer verification. Comparing to "
        "full S3 estimates each check's contribution to accuracy/faithfulness/latency.",
    )
    pdf.qa(
        30,
        "Implement seeded shuffle planning so systems are interleaved.",
        "Intermediate",
        "Coding",
        "Build the Cartesian product of question x system x scale x repeat, shuffle with Random(seed), insert warmup.",
        "import random\n\n"
        "def plan_runs(questions, systems, scales, repeats, seed):\n"
        "    planned = [\n"
        "        {'q': q, 'system': s, 'scale': sc, 'repeat': r, 'warmup': False}\n"
        "        for q in questions for s in systems for sc in scales\n"
        "        for r in range(1, repeats + 1)\n"
        "    ]\n"
        "    rng = random.Random(seed)\n"
        "    rng.shuffle(planned)\n"
        "    planned.insert(0, {**planned[0], 'repeat': 0, 'warmup': True})\n"
        "    return planned",
    )
    pdf.qa(
        31,
        "How do you keep unit tests from calling a real LLM API?",
        "Intermediate",
        "Testing",
        "Inject a scripted LLMClient that returns predetermined JSON for each node/prompt. "
        "Tests still exercise DuckDB/Spark locally. Assert on SQL validation, scoring, and "
        "graph routing without network or spend.",
    )
    pdf.qa(
        32,
        "Normalize numeric cells for comparison: show half-even rounding.",
        "Intermediate",
        "Coding",
        "Use decimal with ROUND_HALF_EVEN when question metadata demands banker's rounding.",
        "from decimal import Decimal, ROUND_HALF_EVEN\n\n"
        "def round_half_even(value: float, places: int) -> Decimal:\n"
        "    q = Decimal('1').scaleb(-places)\n"
        "    return Decimal(str(value)).quantize(q, rounding=ROUND_HALF_EVEN)\n\n"
        "assert round_half_even(2.5, 0) == Decimal('2')\n"
        "assert round_half_even(3.5, 0) == Decimal('4')",
    )
    pdf.qa(
        33,
        "Windows + Spark + spaces in OneDrive paths: what goes wrong and how do you mitigate?",
        "Intermediate",
        "Platform",
        "JVM/Spark often break on paths with spaces. Mitigation used here: rewrite SPARK_HOME "
        "and data paths to 8.3 short paths before starting the JVM so CI and laptops with "
        "OneDrive layouts still run.",
    )
    pdf.qa(
        34,
        "Config load precedence: YAML, dotenv, process env—who wins?",
        "Intermediate",
        "Config",
        "Load YAML defaults, apply .env without clobbering variables already set in the "
        "process, then apply explicit NLQ_* overrides. Document precedence; test it. "
        "Empty NLQ_BUDGET_USD clears the cap.",
    )
    pdf.qa(
        35,
        "Write a stratified sampler: up to N questions per tier.",
        "Intermediate",
        "Coding",
        "Group by tier, shuffle each pool with a seed, take min(N, available)—matches experiment 2's scaffold behavior.",
        "import random\n"
        "from collections import defaultdict\n\n"
        "def stratified(questions, per_tier, seed):\n"
        "    buckets = defaultdict(list)\n"
        "    for q in questions:\n"
        "        buckets[q['tier']].append(q)\n"
        "    rng = random.Random(seed)\n"
        "    out = []\n"
        "    for tier in sorted(buckets):\n"
        "        pool = buckets[tier][:]\n"
        "        rng.shuffle(pool)\n"
        "        out.extend(pool[:per_tier])\n"
        "    return out",
    )

    pdf.add_page()
    pdf.h1("Advanced (Q36-Q50)")

    pdf.qa(
        36,
        "Architect a fair comparison between agentic and single-pass text-to-SQL. Name five controls.",
        "Advanced",
        "Architecture",
        "Controls: (1) same model+temperature, (2) same schema prompt surface, (3) same "
        "narration prompt, (4) shared SQL validator rules where applicable, (5) seeded "
        "interleaved schedule + warmup discard + cache clear + identical row caps/timeouts. "
        "Log config hash and prompt version on every run.",
    )
    pdf.qa(
        37,
        "Where can agentic pipelines still hallucinate despite SQL validation?",
        "Advanced",
        "Failure modes",
        "Valid SQL can be semantically wrong (wrong filter year, wrong grain, subtle join). "
        "Narration can invent claims not in the result set. Answer verification and "
        "claim-level faithfulness catch some of this; execution accuracy against gold catches "
        "others. Validation is not correctness.",
    )
    pdf.qa(
        38,
        "Implement Holm-Bonferroni adjustment for a list of p-values.",
        "Advanced",
        "Coding / stats",
        "Sort p-values ascending; adjust with (m-i+1) factor; enforce monotonicity.",
        "def holm(pvals):\n"
        "    m = len(pvals)\n"
        "    order = sorted(range(m), key=lambda i: pvals[i])\n"
        "    adj = [0.0] * m\n"
        "    prev = 0.0\n"
        "    for rank, i in enumerate(order):\n"
        "        val = min(1.0, (m - rank) * pvals[i])\n"
        "        val = max(val, prev)\n"
        "        adj[i] = val\n"
        "        prev = val\n"
        "    return adj",
    )
    pdf.qa(
        39,
        "Design result truncation safely (row_cap=200) without silent wrong answers.",
        "Advanced",
        "Systems design",
        "Cap protects memory/latency, but aggregates must be computed in SQL—not client-side "
        "on truncated rows. Log truncation flags. Prefer pushing GROUP BY/ORDER BY/LIMIT into "
        "the engine. For entity lists, document that narratives based on capped rows may be incomplete.",
    )
    pdf.qa(
        40,
        "Show a typed SystemResult assembly pattern after SQL + narration.",
        "Advanced",
        "Coding",
        "Centralize assembly so S1/S2/S3 share logging fields: SQL, validation, rows, narrative, retries, timings, failure.",
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class SystemResult:\n"
        "    sql: str\n"
        "    narrative: str\n"
        "    answer_values: list\n"
        "    retries: int = 0\n"
        "    failed: bool = False\n"
        "    exec_time_s: float = 0.0\n"
        "    llm_calls: list = field(default_factory=list)\n\n"
        "def assemble(sql, narrative, values, **meta):\n"
        "    return SystemResult(sql=sql, narrative=narrative,\n"
        "                        answer_values=values, **meta)",
    )
    pdf.qa(
        41,
        "How would you detect flaky gold disagreements between Spark and DuckDB?",
        "Advanced",
        "Evaluation eng",
        "Normalize schemas/types, compare sorted multisets or ordered frames per question "
        "flags, hash canonical CSV of results, fail CI on mismatch. Investigate dialect "
        "functions, null sorting, timestamp precision, and floating tolerance metadata.",
    )
    pdf.qa(
        42,
        "Propose a circuit-breaker around LLM retries distinct from SQL retries.",
        "Advanced",
        "Reliability",
        "SQL retries are semantic (bad query). API retries are transport (429/5xx) via "
        "tenacity with jittered backoff and max_api_retries. Do not conflate them in one "
        "counter—or a transient 429 will burn your SQL attempt budget.",
        "from tenacity import retry, stop_after_attempt, \\\n"
        "    wait_random_exponential, retry_if_exception_type\n\n"
        "class TransientLLMError(Exception):\n"
        "    pass\n\n"
        "@retry(reraise=True, stop=stop_after_attempt(5),\n"
        "       wait=wait_random_exponential(multiplier=1, max=20),\n"
        "       retry=retry_if_exception_type(TransientLLMError))\n"
        "def complete_with_transport_retries(client, messages):\n"
        "    return client.complete(messages)",
    )
    pdf.qa(
        43,
        "Cluster bootstrap for faithfulness intervals—why resample whole questions?",
        "Advanced",
        "Statistics",
        "Claims nested inside questions are dependent. Resampling claims i.i.d. understates "
        "variance. Cluster bootstrap resamples questions (clusters) to keep within-question "
        "dependence, matching how the harness computes faithfulness intervals.",
    )
    pdf.qa(
        44,
        "Implement a simple cluster bootstrap CI for a mean rate per question.",
        "Advanced",
        "Coding",
        "Each question contributes one rate; resample questions with replacement.",
        "import random, statistics\n\n"
        "def cluster_bootstrap_ci(rates, n_boot=2000, alpha=0.05, seed=0):\n"
        "    rng = random.Random(seed)\n"
        "    n = len(rates)\n"
        "    stats = []\n"
        "    for _ in range(n_boot):\n"
        "        sample = [rates[rng.randrange(n)] for _ in range(n)]\n"
        "        stats.append(statistics.fmean(sample))\n"
        "    stats.sort()\n"
        "    lo = stats[int(alpha/2 * n_boot)]\n"
        "    hi = stats[int((1 - alpha/2) * n_boot)]\n"
        "    return statistics.fmean(rates), lo, hi",
    )
    pdf.qa(
        45,
        "How do recursion_limit and max_sql_retries interact in LangGraph S3?",
        "Advanced",
        "Agents",
        "max_sql_retries bounds the business loop; recursion_limit is a framework safety net "
        "against unbounded graphs. Set recursion_limit high enough for worst-case node visits "
        "(understanding to retries to revisions) but still finite. Prefer explicit counters "
        "in state over relying only on recursion_limit.",
    )
    pdf.qa(
        46,
        "Design provenance for answers so faithfulness checks are auditable.",
        "Advanced",
        "Architecture",
        "Store SQL, truncated result snapshot, evidence items (row refs/aggregates), raw model "
        "texts, and claim verdicts in the question_run record. Narration must cite evidence "
        "IDs. Verification compares claims to logged evidence—not to a fresh silent query—unless "
        "explicitly designed otherwise.",
    )
    pdf.qa(
        47,
        "Write an idempotent JSONL resume checker.",
        "Advanced",
        "Coding",
        "Scan existing file for question_run IDs; skip planned runs already completed.",
        "import json\n"
        "from pathlib import Path\n\n"
        "def completed_ids(path: Path) -> set[str]:\n"
        "    done = set()\n"
        "    if not path.is_file():\n"
        "        return done\n"
        "    with path.open(encoding='utf-8') as f:\n"
        "        for line in f:\n"
        "            rec = json.loads(line)\n"
        "            if rec.get('record_type') == 'question_run':\n"
        "                done.add(rec['run_id'])\n"
        "    return done\n\n"
        "def should_run(run_id, done, resume):\n"
        "    return not (resume and run_id in done)",
    )
    pdf.qa(
        48,
        "Discuss privacy and compliance when logging raw model text in NLQ evals.",
        "Advanced",
        "Governance",
        "Raw outputs aid reproducibility but may contain sensitive question content or "
        "regurgitated rows. Mitigations: local-only runs dirs, gitignore data/runs/results, "
        "no secrets in logs, retention policy, redact PII fields for taxi data, separate "
        "demo fixtures from production extracts.",
    )
    pdf.qa(
        49,
        "Optimize experiment 2 cost without destroying statistical validity.",
        "Advanced",
        "Experiment design",
        "Stratify tiers; cap questions per tier; use cheaper scales for smoke; reserve SF10 "
        "for a subset; temperature 0; resume; budget guard; majority-vote 3 repeats rather "
        "than 10; pre-validate SQL to avoid paying for doomed executions—but report that "
        "policy in the paper.",
    )
    pdf.qa(
        50,
        "System design prompt: outline modules for an NLQ eval harness like this one.",
        "Advanced",
        "Architecture",
        "Modules: config (validated, hashed), data (TPC-H/NYC + business views + schema "
        "context), engines (Spark/DuckDB adapters + QueryResult), benchmark (YAML questions + "
        "gold validate), llm (client, prompts, cost guard), systems (S1/S2/S3 graph), eval "
        "(accuracy, faithfulness, latency), logging (JSONL), analysis (stats/plots/reproduce), "
        "tests (scripted LLM). Keep secrets out of hashes; make ablations flags, not forks.",
        "nlq/\n"
        "  config.py\n"
        "  logging_utils.py\n"
        "  data/       # generate_tpch, views, schema_context, nyc_tlc\n"
        "  engine/     # spark_engine, duckdb_engine, results\n"
        "  benchmark/  # schema, questions/*.yaml, validate_gold\n"
        "  llm/        # base, provider, prompts, cost\n"
        "  systems/    # s1_single, s2_retry, s3_agentic/, sql_validation\n"
        "  eval/       # accuracy, faithfulness, latency, run_experiment\n"
        "  analysis/   # stats, plots, reproduce\n"
        "  demo/       # optional local server",
    )

    pdf.h2("Appendix — Quick study map")
    pdf.bullets(
        [
            "Beginner: NLQ basics, config/secrets, S1 vs S2, Parquet, gold answers, JSONL.",
            "Intermediate: LangGraph edges, sqlglot validation, budget guards, stratified plans, faithfulness labels.",
            "Advanced: fair experiment controls, Holm/bootstrap, provenance, resume semantics, governance.",
        ]
    )
    pdf.body(
        "Practice tip: re-implement S1 and a tiny validate-retry loop in under 100 lines, "
        "then explain what S3's extra nodes buy you on semantic errors that still pass SQL validation."
    )

    pdf.output(str(path))


def main() -> None:
    nontech = OUT_DIR / "Non_Technical_Project_Documentation.pdf"
    interview = OUT_DIR / "Technical_Interview_Guide_2027.pdf"
    build_nontechnical(nontech)
    build_interview(interview)
    print(f"Wrote {nontech}")
    print(f"Wrote {interview}")
    print(f"Sizes: {nontech.stat().st_size} bytes, {interview.stat().st_size} bytes")


if __name__ == "__main__":
    main()
