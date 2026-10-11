"""Student-friendly colorful technical report for agentic-nlq-eval."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from fpdf import FPDF

OUT = Path(__file__).resolve().parent
ASSETS = OUT / "assets"
ASSETS.mkdir(exist_ok=True)

# Color palette
C = {
    "navy": "#1B3A4B",
    "teal": "#1F7A8C",
    "sky": "#4ECDC4",
    "coral": "#FF6B6B",
    "orange": "#FF9F1C",
    "gold": "#FFD166",
    "green": "#06D6A0",
    "purple": "#7B68EE",
    "pink": "#F72585",
    "blue": "#4361EE",
    "light": "#F0F7F4",
    "cream": "#FFF8F0",
    "gray": "#5C677D",
    "white": "#FFFFFF",
    "s1": "#4ECDC4",
    "s2": "#FF9F1C",
    "s3": "#7B68EE",
}

_ASCII = str.maketrans(
    {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
        "\u2192": "->",
        "\u00d7": "x",
    }
)


def A(text: str) -> str:
    return text.translate(_ASCII)


# ---------------------------------------------------------------------------
# Diagram helpers
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, fc, ec=None, fontsize=8, bold=False):
    ec = ec or C["navy"]
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, facecolor=fc, edgecolor=ec, mutation_aspect=0.3,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, color=C["navy"], fontweight=weight,
        wrap=True, family="DejaVu Sans",
    )
    return (x + w / 2, y, x + w / 2, y + h)  # bottom mid, top mid helpers unused


def _arrow(ax, x1, y1, x2, y2, color=None):
    color = color or C["gray"]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=12),
    )


def _save(fig, name: str) -> Path:
    path = ASSETS / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def diagram_big_picture() -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=C["cream"])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Big Picture: How the Research Lab Works", fontsize=14, fontweight="bold", color=C["navy"], pad=10)

    _box(ax, 0.3, 4.6, 2.2, 1.1, "You ask a\nbusiness question\nin English", C["gold"], fontsize=8, bold=True)
    _box(ax, 3.0, 4.6, 2.4, 1.1, "AI System\n(S1 or S2 or S3)\nwrites SQL", C["sky"], fontsize=8, bold=True)
    _box(ax, 5.9, 4.6, 2.2, 1.1, "Database Engine\n(Spark)\nruns the query", C["orange"], fontsize=8, bold=True)
    _box(ax, 8.5, 4.6, 2.2, 1.1, "Answer + Score\naccuracy &\nfaithfulness", C["green"], fontsize=8, bold=True)
    _arrow(ax, 2.5, 5.15, 3.0, 5.15, C["teal"])
    _arrow(ax, 5.4, 5.15, 5.9, 5.15, C["teal"])
    _arrow(ax, 8.1, 5.15, 8.5, 5.15, C["teal"])

    # Bottom layers
    _box(ax, 0.3, 2.6, 3.3, 1.4, "DATA\nTPC-H sales tables\n(+ optional NYC taxi)\nstored as Parquet files", "#B8E0D2", fontsize=8)
    _box(ax, 3.9, 2.6, 3.3, 1.4, "RULES\nSame model, same schema\nsame narration prompt\nso the race is fair", "#CDB4DB", fontsize=8)
    _box(ax, 7.5, 2.6, 3.2, 1.4, "LOGS\nJSONL run diary\nthen charts &\nstatistics", "#FFCAD4", fontsize=8)

    _arrow(ax, 1.95, 4.6, 1.95, 4.0, C["gray"])
    _arrow(ax, 5.55, 4.6, 5.55, 4.0, C["gray"])
    _arrow(ax, 9.1, 4.6, 9.1, 4.0, C["gray"])

    ax.text(5.5, 1.6, "Goal: compare simple AI vs retry AI vs multi-step agent AI", ha="center",
            fontsize=10, color=C["navy"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C["white"], edgecolor=C["teal"], lw=2))
    ax.text(5.5, 0.7, "This is a research harness (lab), not a public website product.",
            ha="center", fontsize=9, color=C["gray"], style="italic")
    return _save(fig, "01_big_picture.png")


def diagram_system_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=C["light"])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_title("System Architecture (Layers)", fontsize=14, fontweight="bold", color=C["navy"], pad=8)

    layers = [
        (6.4, "Presentation / Demo (optional local browser)", C["gold"],
         "demo/server.py  |  static UI to browse questions & run logs"),
        (5.2, "Experiment Runner", C["coral"],
         "eval/run_experiment.py  |  plans runs, scores answers, writes JSONL"),
        (4.0, "Answering Systems", C["purple"],
         "S1 single-pass  |  S2 retry  |  S3 LangGraph agent (+ S3-V, S3-A)"),
        (2.8, "Intelligence & Safety", C["blue"],
         "LLM client + prompts  |  sqlglot SQL validator  |  BudgetGuard"),
        (1.6, "Data Engines", C["teal"],
         "Spark (answer queries)  |  DuckDB (gold checks / tests)"),
        (0.4, "Data & Config Foundation", C["orange"],
         "Parquet TPC-H / NYC  |  business views  |  YAML config  |  prompts/"),
    ]
    for y, title, color, detail in layers:
        _box(ax, 0.4, y, 10.2, 1.0, f"{title}\n{detail}", color, fontsize=8, bold=False)
    for y in [1.4, 2.6, 3.8, 5.0, 6.2]:
        _arrow(ax, 5.5, y + 0.2, 5.5, y + 0.55, C["navy"])
    return _save(fig, "02_system_architecture.png")


def diagram_package_map() -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.8), facecolor=C["cream"])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    ax.set_title("Code Map: Where Things Live (src/nlq/)", fontsize=14, fontweight="bold", color=C["navy"])

    packages = [
        (0.3, 5.0, "config.py", "Loads YAML +\nenv settings\n& hashes them", C["gold"]),
        (2.5, 5.0, "data/", "TPC-H generate\nviews, schema\nNYC helper", C["sky"]),
        (4.7, 5.0, "engine/", "Spark &\nDuckDB\nadapters", C["orange"]),
        (6.9, 5.0, "benchmark/", "Questions YAML\ngold validate\nschema models", C["green"]),
        (9.0, 5.0, "llm/", "API client\nprompts\ncost guard", C["pink"]),
        (1.0, 2.4, "systems/", "S1, S2, S3\nsql_validation\nanswer helpers", C["purple"]),
        (3.5, 2.4, "eval/", "run experiment\naccuracy\nfaithfulness\nlatency", C["coral"]),
        (6.0, 2.4, "analysis/", "stats tests\nplots\nreproduce CLI", C["blue"]),
        (8.3, 2.4, "logging +\ndemo", "JSONL diary\nlocal demo\nserver", C["teal"]),
    ]
    for x, y, title, body, color in packages:
        _box(ax, x, y, 1.9, 1.8, f"{title}\n\n{body}", color, fontsize=7, bold=True)
    ax.text(5.5, 1.3, "Tip for students: start at config.py -> data/ -> systems/ -> eval/ -> analysis/",
            ha="center", fontsize=9, color=C["navy"],
            bbox=dict(boxstyle="round", facecolor=C["white"], edgecolor=C["teal"]))
    ax.text(5.5, 0.5, "Tests live in tests/ and use a fake (scripted) LLM so they cost $0.",
            ha="center", fontsize=8, color=C["gray"], style="italic")
    return _save(fig, "03_package_map.png")


def diagram_s1_s2_s3() -> Path:
    fig, ax = plt.subplots(figsize=(11, 7.2), facecolor=C["light"])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title("Three Systems Side by Side", fontsize=14, fontweight="bold", color=C["navy"])

    # S1 column
    ax.add_patch(FancyBboxPatch((0.25, 0.4), 3.3, 6.4, boxstyle="round,pad=0.05", facecolor="#E8FFFC", edgecolor=C["s1"], lw=2.5))
    ax.text(1.9, 6.5, "S1  Single Pass", ha="center", fontsize=11, fontweight="bold", color=C["teal"])
    steps1 = ["Question", "Generate SQL\n(1 try)", "Validate\n(safety)", "Execute\non Spark", "Narrate\nanswer", "Done"]
    ys = [5.6, 4.6, 3.6, 2.6, 1.6, 0.7]
    for y, t in zip(ys, steps1):
        _box(ax, 0.55, y, 2.7, 0.75, t, C["s1"], fontsize=7)
    for i in range(len(ys) - 1):
        _arrow(ax, 1.9, ys[i], 1.9, ys[i + 1] + 0.75, C["teal"])

    # S2 column
    ax.add_patch(FancyBboxPatch((3.85, 0.4), 3.3, 6.4, boxstyle="round,pad=0.05", facecolor="#FFF4E6", edgecolor=C["s2"], lw=2.5))
    ax.text(5.5, 6.5, "S2  Retry Loop", ha="center", fontsize=11, fontweight="bold", color="#C46B00")
    steps2 = ["Question", "Generate SQL", "Validate /\nExecute", "Error?\nfeed back", "Retry up to\n3 more times", "Narrate"]
    for y, t in zip(ys, steps2):
        _box(ax, 4.15, y, 2.7, 0.75, t, C["s2"], fontsize=7)
    for i in range(len(ys) - 1):
        _arrow(ax, 5.5, ys[i], 5.5, ys[i + 1] + 0.75, C["orange"])
    # retry arc hint
    ax.annotate("", xy=(6.7, 4.9), xytext=(6.7, 2.0),
                arrowprops=dict(arrowstyle="-|>", color=C["coral"], lw=1.5,
                                connectionstyle="arc3,rad=-0.4"))
    ax.text(7.05, 3.5, "retry", fontsize=7, color=C["coral"], rotation=90)

    # S3 column
    ax.add_patch(FancyBboxPatch((7.45, 0.4), 3.3, 6.4, boxstyle="round,pad=0.05", facecolor="#F3F0FF", edgecolor=C["s3"], lw=2.5))
    ax.text(9.1, 6.5, "S3  Agentic Graph", ha="center", fontsize=11, fontweight="bold", color=C["purple"])
    steps3 = ["Understand +\nGround schema", "Plan query", "Generate SQL", "Validate &\nExecute (+retry)", "Analyze +\nWrite answer", "Verify claims"]
    for y, t in zip(ys, steps3):
        _box(ax, 7.75, y, 2.7, 0.75, t, C["s3"], fontsize=7)
    for i in range(len(ys) - 1):
        _arrow(ax, 9.1, ys[i], 9.1, ys[i + 1] + 0.75, C["purple"])
    return _save(fig, "04_s1_s2_s3.png")


def diagram_s3_graph() -> Path:
    fig, ax = plt.subplots(figsize=(11, 8.2), facecolor="#F8F5FF")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.set_title("S3 LangGraph Pipeline (Detailed Flowchart)", fontsize=14, fontweight="bold", color=C["navy"])

    nodes = [
        (4.5, 7.4, "START", C["green"]),
        (4.5, 6.4, "1. Question\nUnderstanding", "#BDE0FE"),
        (4.5, 5.4, "2. Schema\nGrounding", "#A2D2FF"),
        (4.5, 4.4, "3. Query\nPlanning", "#90E0EF"),
        (4.5, 3.4, "4. SQL\nGeneration", C["purple"]),
        (1.2, 2.4, "5. SQL\nValidation", C["gold"]),
        (4.5, 2.4, "6. Execute\non Spark", C["orange"]),
        (7.8, 2.4, "7. Result\nAnalysis", C["sky"]),
        (7.8, 1.3, "8. Answer +\nProvenance", C["coral"]),
        (4.5, 1.3, "9. Answer\nVerification", C["pink"]),
        (1.2, 1.3, "END", C["green"]),
    ]
    for x, y, t, c in nodes:
        _box(ax, x, y, 2.0, 0.8, t, c, fontsize=7, bold=True)

    # main vertical
    for y1, y2 in [(7.4, 7.2), (6.4, 6.2), (5.4, 5.2), (4.4, 4.2), (3.4, 3.2)]:
        _arrow(ax, 5.5, y1, 5.5, y2, C["navy"])
    _arrow(ax, 4.5, 3.8, 3.2, 3.0, C["navy"])  # gen -> validation
    _arrow(ax, 3.2, 2.4, 4.5, 2.8, C["green"])  # validation pass -> execute
    _arrow(ax, 5.5, 2.4, 7.8, 2.8, C["navy"])  # execute -> analysis
    _arrow(ax, 8.8, 2.4, 8.8, 2.1, C["navy"])
    _arrow(ax, 7.8, 1.7, 6.5, 1.7, C["navy"])  # answer -> verify
    _arrow(ax, 4.5, 1.7, 3.2, 1.7, C["green"])  # verify ok -> end

    # retry loops
    ax.annotate("fail: retry SQL\n(max 3)", xy=(5.5, 3.7), xytext=(0.3, 3.7),
                fontsize=7, color=C["coral"],
                arrowprops=dict(arrowstyle="-|>", color=C["coral"], lw=1.4,
                                connectionstyle="arc3,rad=0.25"))
    ax.annotate("exec error:\nretry SQL", xy=(5.0, 2.5), xytext=(0.3, 0.4),
                fontsize=7, color=C["coral"],
                arrowprops=dict(arrowstyle="-|>", color=C["coral"], lw=1.3,
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate("claims weak:\nrevise answer\n(max 2)", xy=(8.8, 1.5), xytext=(9.5, 3.5),
                fontsize=7, color=C["pink"],
                arrowprops=dict(arrowstyle="-|>", color=C["pink"], lw=1.3,
                                connectionstyle="arc3,rad=0.3"))

    ax.text(5.5, 0.25,
            "Ablations: S3-V skips box 5 (goes straight to Execute).  S3-A skips box 9 (ends after Answer).",
            ha="center", fontsize=8, color=C["navy"],
            bbox=dict(boxstyle="round", facecolor=C["white"], edgecolor=C["purple"]))
    return _save(fig, "05_s3_graph.png")


def diagram_data_flow() -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=C["cream"])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("Data Flow: From Raw Tables to Fair Scoring", fontsize=14, fontweight="bold", color=C["navy"])

    flow = [
        (0.3, 4.6, "dbgen\nTPC-H", C["orange"]),
        (2.3, 4.6, "Parquet\nfiles", C["gold"]),
        (4.3, 4.6, "Business\nViews", C["sky"]),
        (6.3, 4.6, "Schema\nContext\n(for AI)", C["purple"]),
        (8.5, 4.6, "AI writes\nSQL", C["coral"]),
    ]
    for x, y, t, c in flow:
        _box(ax, x, y, 1.8, 1.3, t, c, fontsize=8, bold=True)
    for i in range(len(flow) - 1):
        _arrow(ax, flow[i][0] + 1.8, 5.25, flow[i + 1][0], 5.25, C["teal"])

    _box(ax, 0.3, 2.2, 3.2, 1.5, "Gold SQL\n(known correct query)\nchecked on Spark + DuckDB", C["green"], fontsize=8)
    _box(ax, 4.0, 2.2, 3.2, 1.5, "System result rows\nnormalized\n(types, rounding, order)", C["blue"], fontsize=8)
    _box(ax, 7.7, 2.2, 3.0, 1.5, "Scorecard\nAccuracy\nFaithfulness\nLatency", C["pink"], fontsize=8)
    _arrow(ax, 3.5, 2.95, 4.0, 2.95, C["navy"])
    _arrow(ax, 7.2, 2.95, 7.7, 2.95, C["navy"])
    _arrow(ax, 9.4, 4.6, 9.2, 3.7, C["gray"])

    ax.text(5.5, 1.1, "Profit formula (same for everyone):", ha="center", fontsize=9, color=C["navy"], fontweight="bold")
    ax.text(5.5, 0.5, "profit = sale_amount * (1 - discount_rate) - unit_cost * quantity",
            ha="center", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="#E8FFFC", edgecolor=C["teal"]))
    return _save(fig, "06_data_flow.png")


def diagram_experiment() -> Path:
    fig, ax = plt.subplots(figsize=(11, 7.0), facecolor=C["light"])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.0)
    ax.axis("off")
    ax.set_title("Experiment Workflow (How a Study Run Works)", fontsize=14, fontweight="bold", color=C["navy"])

    steps = [
        (0.4, 5.5, "1. Load\nconfig", C["gold"]),
        (2.3, 5.5, "2. Load\nquestions", C["sky"]),
        (4.2, 5.5, "3. Plan runs\n(seed shuffle)", C["orange"]),
        (6.1, 5.5, "4. Warm-up\n(discard later)", C["coral"]),
        (8.2, 5.5, "5. Measured\nruns", C["purple"]),
    ]
    for x, y, t, c in steps:
        _box(ax, x, y, 1.7, 1.1, t, c, fontsize=8, bold=True)
    for i in range(len(steps) - 1):
        _arrow(ax, steps[i][0] + 1.7, 6.05, steps[i + 1][0], 6.05, C["navy"])

    _box(ax, 0.4, 3.4, 4.8, 1.5,
         "Each measured run:\nclear Spark cache -> system answers ->\nscore accuracy/faithfulness/latency -> append JSONL",
         "#CDB4DB", fontsize=8)
    _box(ax, 5.6, 3.4, 5.0, 1.5,
         "Guards:\n--resume skips finished run IDs\n--budget-usd stops before next LLM call\nstable run_id from config+model+prompt version",
         "#FFCAD4", fontsize=8)

    # three experiments
    _box(ax, 0.4, 1.2, 3.2, 1.6, "Experiment 1\nAll TPC-H questions\nAll systems\nOne scale (e.g. SF1)", C["s1"], fontsize=8, bold=True)
    _box(ax, 3.9, 1.2, 3.2, 1.6, "Experiment 2\nStratified by tier\nS1/S2/S3\nMany scales (growth study)", C["s2"], fontsize=8, bold=True)
    _box(ax, 7.4, 1.2, 3.2, 1.6, "Experiment 3\nNYC taxi questions\nDemo fixture\nS1/S2/S3", C["s3"], fontsize=8, bold=True)
    ax.text(5.5, 0.4, "Finally: analysis/reproduce.py makes CSV tables + PNG charts",
            ha="center", fontsize=9, color=C["navy"], fontweight="bold")
    return _save(fig, "07_experiment.png")


def diagram_evaluation() -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="#FFF8F0")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("How We Grade the AI (Three Score Types)", fontsize=14, fontweight="bold", color=C["navy"])

    _box(ax, 0.4, 3.2, 3.2, 2.5,
         "ACCURACY\n\nDid the numbers match\nthe gold answer?\n\nCompare normalized\nrows / values",
         C["green"], fontsize=8, bold=True)
    _box(ax, 3.9, 3.2, 3.2, 2.5,
         "FAITHFULNESS\n\nDid the written story\nstick to the table?\n\nClaims: supported /\ncontradicted /\nunsupported",
         C["gold"], fontsize=8, bold=True)
    _box(ax, 7.4, 3.2, 3.2, 2.5,
         "LATENCY\n\nHow long did it take?\n\nModel time +\nSQL time +\ntotal wall time",
         C["sky"], fontsize=8, bold=True)

    ax.text(5.5, 2.3, "Statistics used later (don't worry - tools do the math):",
            ha="center", fontsize=9, color=C["navy"], fontweight="bold")
    _box(ax, 0.4, 0.4, 3.2, 1.5, "Majority vote\nover 3 repeats\n(tie = wrong)", "#BDE0FE", fontsize=8)
    _box(ax, 3.9, 0.4, 3.2, 1.5, "McNemar test +\nHolm-adjusted\np-values", "#CDB4DB", fontsize=8)
    _box(ax, 7.4, 0.4, 3.2, 1.5, "Bootstrap CIs\n(question clusters)\nMedian latency plots", "#FFCAD4", fontsize=8)
    return _save(fig, "08_evaluation.png")


def diagram_llm_call() -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=C["light"])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.set_title("One LLM Call (Inside the Client)", fontsize=14, fontweight="bold", color=C["navy"])

    boxes = [
        (0.3, 3.8, "BudgetGuard\nbefore_call()", C["coral"]),
        (2.6, 3.8, "Build chat\nmessages", C["gold"]),
        (4.9, 3.8, "HTTP POST\n/v1/chat/\ncompletions", C["blue"]),
        (7.2, 3.8, "Parse text +\ntoken counts", C["sky"]),
        (9.2, 3.8, "charge()\ncost estimate", C["green"]),
    ]
    for x, y, t, c in boxes:
        _box(ax, x, y, 2.0, 1.4, t, c, fontsize=7, bold=True)
    for i in range(len(boxes) - 1):
        _arrow(ax, boxes[i][0] + 2.0, 4.5, boxes[i + 1][0], 4.5, C["navy"])

    _box(ax, 2.6, 1.3, 5.8, 1.8,
         "If HTTP 408 / 429 / 5xx  ->  TransientLLMError\ntenacity retries with exponential backoff\n(SQL retries are SEPARATE - do not mix the counters!)",
         "#FFD6A5", fontsize=8)
    _arrow(ax, 5.9, 3.8, 5.5, 3.1, C["orange"])
    ax.text(5.5, 0.5, "API key is read from environment only. It never enters config hash or JSONL logs.",
            ha="center", fontsize=9, color=C["coral"], fontweight="bold")
    return _save(fig, "09_llm_call.png")


def diagram_student_journey() -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.0), facecolor=C["cream"])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    ax.set_title("Student Learning Path (Recommended Order)", fontsize=14, fontweight="bold", color=C["navy"])

    path = [
        (0.3, 4.2, "A\nIdea\nof NLQ", C["gold"]),
        (2.2, 4.2, "B\nData &\nViews", C["orange"]),
        (4.1, 4.2, "C\nS1 then\nS2", C["sky"]),
        (6.0, 4.2, "D\nS3 graph\nnodes", C["purple"]),
        (7.9, 4.2, "E\nScoring\nmetrics", C["green"]),
        (9.5, 4.2, "F\nRun &\nCharts", C["coral"]),
    ]
    for x, y, t, c in path:
        _box(ax, x, y, 1.6, 1.3, t, c, fontsize=8, bold=True)
    for i in range(len(path) - 1):
        _arrow(ax, path[i][0] + 1.6, 4.85, path[i + 1][0], 4.85, C["navy"])

    tips = [
        "Read README.md once for the map.",
        "Open one tier YAML question and its gold SQL.",
        "Trace S1.answer() in s1_single.py line by line.",
        "Compare S2 retry loop with S1.",
        "Walk S3 nodes in graph.py like a checklist.",
        "Run pytest (fake LLM) before any paid API call.",
    ]
    for i, tip in enumerate(tips):
        y = 3.2 - i * 0.45
        ax.text(0.5, y, f"{i+1}. {tip}", fontsize=9, color=C["navy"],
                bbox=dict(boxstyle="round,pad=0.2", facecolor=C["white"], edgecolor="#DDDDDD"))
    return _save(fig, "10_student_journey.png")


def build_all_diagrams() -> dict[str, Path]:
    return {
        "big": diagram_big_picture(),
        "arch": diagram_system_architecture(),
        "packages": diagram_package_map(),
        "systems": diagram_s1_s2_s3(),
        "s3": diagram_s3_graph(),
        "data": diagram_data_flow(),
        "experiment": diagram_experiment(),
        "eval": diagram_evaluation(),
        "llm": diagram_llm_call(),
        "journey": diagram_student_journey(),
    }


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class Report(FPDF):
    def __init__(self):
        super().__init__(format="Letter")
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(16, 16, 16)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, A(f"agentic-nlq-eval  |  Student Technical Report  |  Page {self.page_no()}"), align="C")

    def banner(self, text: str, r=31, g=122, b=140):
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 10, A(f"  {text}"), ln=1, fill=True)
        self.ln(3)
        self.set_text_color(30, 30, 30)

    def h2(self, text: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(27, 58, 75)
        self.multi_cell(0, 6, A(text))
        self.ln(1)

    def p(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(35, 35, 35)
        self.multi_cell(0, 5.4, A(text))
        self.ln(1.5)

    def tip(self, text: str):
        self.set_fill_color(255, 249, 230)
        self.set_draw_color(255, 159, 28)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(80, 50, 0)
        self.set_x(self.l_margin)
        self.multi_cell(self.epw, 5.2, A(f"Student tip: {text}"), border=1, fill=True)
        self.ln(3)
        self.set_text_color(35, 35, 35)

    def bullets(self, items: list[str]):
        self.set_font("Helvetica", "", 10)
        for item in items:
            self.set_x(self.l_margin)
            self.multi_cell(0, 5.2, A(f"  *  {item}"))
            self.ln(0.4)
        self.ln(1.5)

    def img(self, path: Path, w: float = 180):
        if self.get_y() > 200:
            self.add_page()
        self.image(str(path), x=self.l_margin, w=w)
        self.ln(4)


def build_pdf(diagrams: dict[str, Path], out_path: Path) -> None:
    pdf = Report()

    # Cover
    pdf.add_page()
    pdf.set_fill_color(27, 58, 75)
    pdf.rect(0, 0, 216, 279, "F")
    pdf.set_fill_color(31, 122, 140)
    pdf.rect(0, 0, 216, 55, "F")
    pdf.set_fill_color(78, 205, 196)
    pdf.rect(0, 55, 216, 8, "F")
    pdf.set_y(80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 26)
    pdf.multi_cell(0, 12, A("agentic-nlq-eval"), align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 209, 102)
    pdf.multi_cell(0, 8, A("Student Technical Report"), align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(230, 240, 245)
    pdf.multi_cell(0, 6, A("A colorful, easy-English guide to the architecture,\nsystems, data flow, experiments, and scoring."), align="C")
    pdf.ln(14)
    pdf.set_fill_color(255, 255, 255)
    pdf.set_text_color(27, 58, 75)
    pdf.set_font("Helvetica", "", 10)
    box = (
        "What you will learn\n"
        "1. What natural-language querying means\n"
        "2. How S1, S2, and S3 differ\n"
        "3. How data, engines, and the LLM fit together\n"
        "4. How experiments stay fair and reproducible\n"
        "5. How answers are graded"
    )
    pdf.set_x(35)
    pdf.multi_cell(140, 6, A(box), border=0, fill=True, align="L")
    pdf.ln(10)
    pdf.set_text_color(200, 220, 230)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, A("Based on project version 0.1.0  |  Research harness (not a product UI)"), align="C")

    # TOC
    pdf.add_page()
    pdf.banner("Contents")
    toc = [
        "1. Welcome: what this project is (in plain English)",
        "2. Big-picture diagram",
        "3. System architecture layers",
        "4. Code map (folders)",
        "5. Data flow and business views",
        "6. The three answering systems (S1, S2, S3)",
        "7. S3 LangGraph flowchart (detailed)",
        "8. LLM client, prompts, and budget safety",
        "9. SQL validation (the safety gate)",
        "10. Experiment workflow",
        "11. Evaluation: accuracy, faithfulness, latency",
        "12. Configuration and secrets",
        "13. Student learning path",
        "14. Glossary",
    ]
    pdf.bullets(toc)

    # 1 Welcome
    pdf.add_page()
    pdf.banner("1. Welcome: what this project is", 27, 58, 75)
    pdf.p(
        "Normal people ask questions in English. Databases understand SQL "
        "(a special query language). A natural-language query (NLQ) system is a bridge: "
        "it reads your English question, writes SQL, runs it, and explains the answer."
    )
    pdf.p(
        "agentic-nlq-eval is a research lab for that bridge. It is not a shiny website for "
        "customers. It is a fair race track where three designs compete on the same questions "
        "and the same data, so we can measure which design is better."
    )
    pdf.h2("The three racers (short version)")
    pdf.bullets(
        [
            "S1: one try - write SQL, run it, explain. Fast and simple.",
            "S2: like S1, but if SQL fails, show the error and try again (up to 3 retries).",
            "S3: a multi-step agent (LangGraph) that plans, validates, executes, explains, and checks claims.",
        ]
    )
    pdf.tip(
        "If this feels big, treat it like cooking: S1 is one recipe step, S2 is tasting and fixing, "
        "S3 is a full kitchen checklist with a quality inspector at the end."
    )

    # 2 Big picture
    pdf.add_page()
    pdf.banner("2. Big-picture diagram", 78, 205, 196)
    pdf.p(
        "Read this diagram left to right. A question goes into an AI system. The system writes SQL. "
        "Spark runs SQL on Parquet data. We score the answer. Underneath, shared data, fair rules, "
        "and logs keep the race honest."
    )
    pdf.img(diagrams["big"], 178)

    # 3 Architecture
    pdf.add_page()
    pdf.banner("3. System architecture layers", 67, 97, 238)
    pdf.p(
        "Think of the software as a stack of layers. The bottom holds data and config. "
        "The middle talks to databases and the AI model. Near the top, S1/S2/S3 answer questions. "
        "The runner runs experiments. An optional demo server only helps you look at results."
    )
    pdf.img(diagrams["arch"], 178)
    pdf.p(
        "Important rule for fairness: all systems share one model, one schema description, and one "
        "narration prompt style. We change the control flow, not the 'brain brand'."
    )

    # 4 Packages
    pdf.add_page()
    pdf.banner("4. Code map (folders)", 123, 104, 238)
    pdf.p(
        "When you open the repo, almost everything useful is under src/nlq/. "
        "This map shows which folder owns which job."
    )
    pdf.img(diagrams["packages"], 178)

    # 5 Data
    pdf.add_page()
    pdf.banner("5. Data flow and business views", 255, 159, 28)
    pdf.p(
        "TPC-H is a standard fake sales/supply-chain dataset. We can generate it at different "
        "sizes (scale factors). Small sizes are for quick tests. Bigger sizes stress performance."
    )
    pdf.p(
        "The AI never sees ugly raw TPC-H names. The engine creates friendly business views such as "
        "customers, suppliers, sales_orders, and order_lines. That matches how a business analyst thinks."
    )
    pdf.img(diagrams["data"], 178)
    pdf.bullets(
        [
            "Parquet = columnar file format, good for analytics.",
            "Spark runs the answering SQL in experiments.",
            "DuckDB helps validate gold answers and supports tests.",
            "Gold answer = known-correct reference used for scoring.",
        ]
    )

    # 6 Systems
    pdf.add_page()
    pdf.banner("6. The three answering systems", 255, 107, 107)
    pdf.p(
        "Same question, same data, three strategies. Study the middle column carefully: "
        "S2's power is the feedback loop. S3's power is many specialized steps."
    )
    pdf.img(diagrams["systems"], 178)
    pdf.h2("Shared pieces")
    pdf.bullets(
        [
            "generate_sql() builds one SELECT using the schema text and optional error list.",
            "validation_for() / sqlglot rejects unsafe or unknown tables/columns.",
            "run_sql() executes with a row cap (default 200) and timeout.",
            "narrate() turns result rows into a readable answer.",
        ]
    )

    # 7 S3 detail
    pdf.add_page()
    pdf.banner("7. S3 LangGraph flowchart (detailed)", 123, 104, 238)
    pdf.p(
        "LangGraph is a library for building AI workflows as a graph: boxes (nodes) and arrows "
        "(edges). Conditional arrows mean 'if this, go there'. S3 stores working memory in state "
        "(question, SQL, feedback, timings, token usage)."
    )
    pdf.img(diagrams["s3"], 178)
    pdf.h2("Ablation variants (science experiments)")
    pdf.bullets(
        [
            "S3-V: turn OFF SQL validation - first SQL goes to execute. Shows value of the validator.",
            "S3-A: turn OFF answer verification - skip claim checking. Shows value of the final check.",
            "Flags change edges; node code stays the same. That is clean experiment design.",
        ]
    )
    pdf.tip(
        "recursion_limit is a safety net so the graph cannot loop forever. Business limits "
        "(max_sql_retries, max_answer_revisions) are the real policy knobs."
    )

    # 8 LLM
    pdf.add_page()
    pdf.banner("8. LLM client, prompts, and budget safety", 247, 37, 133)
    pdf.p(
        "LLM means Large Language Model (the AI text model). This project talks to an "
        "OpenAI-compatible chat API with the standard library (urllib), not a heavy SDK."
    )
    pdf.img(diagrams["llm"], 178)
    pdf.h2("Prompts")
    pdf.p(
        "Prompt files live in prompts/ (question understanding, schema grounding, SQL generation, "
        "narration, claim extraction, ...). Placeholders look like <<SCHEMA>> so JSON braces "
        "inside prompts stay safe. prompts/VERSION is stored on every run for reproducibility."
    )
    pdf.h2("BudgetGuard")
    pdf.p(
        "If you set a dollar budget, the guard estimates cost from token counts and pricing rates. "
        "It blocks the NEXT call when the cap is already reached. One call that started under the "
        "cap can still finish a little over, because tokens are known only after the response."
    )

    # 9 SQL validation
    pdf.add_page()
    pdf.banner("9. SQL validation (the safety gate)", 6, 214, 160)
    pdf.p(
        "Before trusting model SQL, a deterministic checker (sqlglot, Spark dialect) inspects it. "
        "No AI is involved in this check - that keeps messages reliable for retries."
    )
    pdf.h2("What gets rejected?")
    pdf.bullets(
        [
            "More than one statement",
            "Anything that is not a read-only SELECT-family query (no DROP/UPDATE)",
            "Unknown tables or columns outside the business schema",
            "Ambiguous columns",
            "Cross joins / joins without a condition",
        ]
    )
    pdf.tip(
        "Passing validation does NOT mean the SQL is semantically correct. It can still filter "
        "the wrong year or join the wrong way. That is why we also score against gold answers."
    )

    # 10 Experiments
    pdf.add_page()
    pdf.banner("10. Experiment workflow", 67, 97, 238)
    pdf.p(
        "An experiment is a planned list of (question, system, scale, repeat) jobs. "
        "Order is shuffled with a seed so no system always runs first. One warm-up is recorded "
        "then dropped in analysis so cold-start noise does not spoil charts."
    )
    pdf.img(diagrams["experiment"], 178)

    # 11 Evaluation
    pdf.add_page()
    pdf.banner("11. Evaluation: how answers are graded", 255, 209, 102)
    pdf.set_text_color(27, 58, 75)
    pdf.p(
        "Grading has three lenses. Accuracy asks 'are the numbers right?'. Faithfulness asks "
        "'does the story stick to the table?'. Latency asks 'how long did we wait?'."
    )
    pdf.img(diagrams["eval"], 178)
    pdf.h2("Faithfulness labels (easy meanings)")
    pdf.bullets(
        [
            "supported - the claim matches the result evidence",
            "contradicted - the claim fights the evidence",
            "unsupported - claim has a checkable fact that is not in the result",
            "non_verifiable - soft explanation without a checkable number/comparison",
        ]
    )

    # 12 Config
    pdf.add_page()
    pdf.banner("12. Configuration and secrets", 28, 58, 75)
    pdf.h2("Where settings live")
    pdf.bullets(
        [
            "configs/default.yaml - model name, temperature, paths, Spark caps, S3 flags, pricing rates",
            ".env - NLQ_LLM_API_KEY only (copy from .env.example)",
            "NLQ_* environment variables can override selected fields",
        ]
    )
    pdf.h2("Config hash")
    pdf.p(
        "AppConfig.config_hash() is a SHA-256 fingerprint of the resolved config. It helps group "
        "identical runs. Absolute paths mean the hash can change if you move the project folder. "
        "The API key is intentionally NOT part of the hashed config."
    )
    pdf.tip(
        "Never commit a real API key. Tests use a scripted fake model, so CI can stay free."
    )

    # 13 Journey
    pdf.add_page()
    pdf.banner("13. Student learning path", 78, 205, 196)
    pdf.p(
        "Do not try to learn every file on day one. Follow this path. After step C you already "
        "understand most of the project idea."
    )
    pdf.img(diagrams["journey"], 178)
    pdf.h2("Handy commands (after setup)")
    pdf.set_font("Courier", "", 8.5)
    pdf.set_fill_color(240, 247, 244)
    pdf.multi_cell(
        0, 4.5,
        A(
            "uv sync\n"
            "uv run pytest\n"
            "uv run python -m nlq.data.generate_tpch --scale 0.01\n"
            "uv run python -m nlq.eval.run_experiment --experiment 1 --systems s1 s2 s3 --scale sf0.01 --repeats 1\n"
            "uv run python -m nlq.analysis.reproduce --runs runs/experiment_1.jsonl --out results"
        ),
        fill=True,
    )
    pdf.ln(3)

    # 14 Glossary
    pdf.add_page()
    pdf.banner("14. Glossary (keep this open while reading code)", 123, 104, 238)
    glossary = [
        ("Agent / agentic", "A workflow with multiple steps, memory/state, and decision loops - not one single prompt."),
        ("Ablation", "Turn one feature off to measure how much it mattered."),
        ("Benchmark", "A fixed set of questions with known answers for fair comparison."),
        ("Config hash", "Fingerprint of settings used for a run."),
        ("DuckDB", "Fast local analytics database engine."),
        ("Faithfulness", "Whether the written answer sticks to computed evidence."),
        ("Gold answer", "Trusted correct result used for scoring."),
        ("JSONL", "Text file with one JSON object per line (easy to append and resume)."),
        ("LangGraph", "Library to build AI pipelines as graphs of nodes and edges."),
        ("Latency", "How long a run takes (wall-clock and parts of it)."),
        ("LLM", "Large Language Model - the AI that writes SQL/text."),
        ("NLQ", "Natural-Language Querying - ask databases in English."),
        ("Parquet", "Columnar data file format used for TPC-H tables here."),
        ("Prompt", "Instructions/template sent to the LLM."),
        ("Provenance", "Trace of where an answer came from (SQL + evidence)."),
        ("Scale factor", "Size knob for TPC-H data (bigger = more rows)."),
        ("Schema", "Map of tables and columns the AI may use."),
        ("Spark", "Distributed SQL/analytics engine used to run answering queries."),
        ("sqlglot", "SQL parser/toolkit used for validation."),
        ("Temperature", "Sampling randomness for the LLM; 0 is more stable for evals."),
        ("Text-to-SQL", "Task of turning English into SQL."),
        ("Warm-up run", "First throwaway run to reduce cold-start noise."),
    ]
    for term, meaning in glossary:
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(27, 58, 75)
        pdf.write(5.2, A(f"{term}: "))
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5.2, A(meaning))
        pdf.ln(0.8)

    pdf.ln(4)
    pdf.set_fill_color(27, 58, 75)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 8, A("  You made it. Re-open the diagrams whenever a code file feels confusing."), fill=True)

    pdf.output(str(out_path))


def main() -> None:
    print("Drawing diagrams...")
    diagrams = build_all_diagrams()
    out = OUT / "Student_Technical_Report.pdf"
    print("Building PDF...")
    build_pdf(diagrams, out)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    print(f"Diagrams in {ASSETS}")


if __name__ == "__main__":
    main()
