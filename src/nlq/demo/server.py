"""Local demonstration server. Standard library only, so the eval stack stays unchanged."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from nlq.benchmark.schema import load_questions
from nlq.config import find_project_root
from nlq.logging_utils import read_records

STATIC_DIR = Path(__file__).resolve().parent / "static"
SYSTEMS = (
    {"id": "s1", "name": "S1", "summary": "One SQL generation, then execution and narration. No retry."},
    {"id": "s2", "name": "S2", "summary": "S1 plus error feedback. Up to three retries after the first attempt."},
    {"id": "s3", "name": "S3", "summary": "Plan, ground, generate, validate, execute, narrate, and check claims."},
    {"id": "s3v", "name": "S3-V", "summary": "Same graph as S3 with SQL validation turned off."},
    {"id": "s3a", "name": "S3-A", "summary": "Same graph as S3 with answer verification turned off."},
)


def question_payload() -> list[dict]:
    return [question.model_dump(mode="json") for question in load_questions()]


def list_run_files(runs_dir: Path) -> list[dict]:
    if not runs_dir.is_dir():
        return []
    files = []
    for path in sorted(runs_dir.glob("*.jsonl")):
        if not _is_safe_log(runs_dir, path):
            continue
        files.append({"name": path.name, "bytes": path.stat().st_size})
    return files


def load_run(runs_dir: Path, name: str) -> dict:
    path = (runs_dir / name).resolve()
    if not _is_safe_log(runs_dir, path) or not path.is_file():
        raise FileNotFoundError(name)
    rows = []
    for record in read_records(path):
        if record.record_type != "question_run":
            continue
        payload = record.payload
        answer = payload.get("answer") or {}
        rows.append(
            {
                "run_id": record.run_id,
                "experiment": record.experiment,
                "system": record.system,
                "question_id": record.question_id,
                "scale": record.scale,
                "repeat": record.repeat,
                "model_name": record.model_name,
                "prompts_version": record.prompts_version,
                "warmup": bool(payload.get("warmup")),
                "tier": payload.get("tier"),
                "lineitem_rows": payload.get("lineitem_rows"),
                "accuracy": payload.get("accuracy") or {},
                "faithfulness": payload.get("faithfulness") or {},
                "latency": payload.get("latency") or {},
                "sql": answer.get("sql", ""),
                "narrative": answer.get("narrative", ""),
                "answer_values": answer.get("answer_values") or [],
                "failed": bool(answer.get("failed")),
                "retries": answer.get("retries", 0),
            }
        )
    return {"name": path.name, "rows": rows}


def _is_safe_log(runs_dir: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(runs_dir.resolve())
    except ValueError:
        return False
    return path.suffix == ".jsonl" and path.name == Path(path.name).name


class DemoHandler(BaseHTTPRequestHandler):
    project_root: Path = Path(".")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"/", "/index.html"}:
            self._file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/styles.css":
            self._file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
            return
        if path == "/app.js":
            self._file(STATIC_DIR / "app.js", "text/javascript; charset=utf-8")
            return
        if path == "/api/bootstrap":
            self._json(
                {
                    "systems": list(SYSTEMS),
                    "questions": question_payload(),
                    "runs": list_run_files(self.project_root / "runs"),
                }
            )
            return
        if path.startswith("/api/runs/"):
            name = path.removeprefix("/api/runs/")
            try:
                self._json(load_run(self.project_root / "runs", name))
            except FileNotFoundError:
                self._json({"error": "run not found"}, status=404)
            return
        self._json({"error": "not found"}, status=404)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._json({"error": "missing file"}, status=404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(project_root: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    DemoHandler.project_root = project_root.resolve()
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"demo http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open the read-only NLQ demonstration.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    root = find_project_root(Path.cwd())
    serve(root, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
