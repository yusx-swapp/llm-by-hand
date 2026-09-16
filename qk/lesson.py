"""The one-lesson prototype: source content, three-stage progress, and verification API."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, distribution, version
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
LESSON = ROOT / "lessons" / "llama_attention"
STAGES = ("follow", "cloze", "recall")
Stage = Literal["follow", "cloze", "recall"]


def read_json(name):
    return json.loads((LESSON / name).read_text(encoding="utf-8"))


def source_hash(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def cloze_source(reference, gaps):
    lines = reference.splitlines()
    for gap in reversed(gaps):
        lines[gap["start"] - 1:gap["end"]] = gap["replacement"].splitlines()
    return "\n".join(lines) + "\n"


def recall_source(reference):
    """Expose only class/method contracts, never the original implementation bodies."""
    lines = reference.splitlines()
    cls = next(node for node in ast.parse(reference).body if isinstance(node, ast.ClassDef))
    result = [lines[cls.lineno - 1]]
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"__init__", "forward"}:
            result.extend(lines[node.lineno - 1:node.body[0].lineno - 1])
            result.extend(['        raise NotImplementedError("请独立实现这个方法")', ""])
    return "\n".join(result)


def token_locations(reference):
    def path(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = path(node.value)
            return f"{parent}.{node.attr}" if parent else None

    tokens = []
    for node in ast.walk(ast.parse(reference)):
        if isinstance(node, (ast.Name, ast.Attribute)) and node.lineno == node.end_lineno:
            name = node.id if isinstance(node, ast.Name) else node.attr
            tokens.append({"line": node.end_lineno, "start": node.end_col_offset - len(name) + 1,
                           "end": node.end_col_offset + 1, "name": name, "path": path(node),
                           "access": "write" if isinstance(node.ctx, ast.Store) else "read"})
        elif isinstance(node, ast.arg):
            tokens.append({"line": node.lineno, "start": node.col_offset + 1,
                           "end": node.col_offset + len(node.arg) + 1, "name": node.arg,
                           "path": node.arg, "access": "parameter"})
    return tokens


class LessonStore:
    def __init__(self, path):
        self.path = path

    @contextmanager
    def db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                connection.execute("CREATE TABLE IF NOT EXISTS lesson_stages ("
                                   "stage TEXT PRIMARY KEY, source TEXT, passed INTEGER NOT NULL DEFAULT 0, "
                                   "report TEXT, updated_at REAL NOT NULL)")
                yield connection
        finally:
            connection.close()

    def state(self):
        with self.db() as db:
            rows = {row["stage"]: dict(row) for row in db.execute("SELECT * FROM lesson_stages")}
        return {stage: {"source": rows.get(stage, {}).get("source"),
                        "passed": bool(rows.get(stage, {}).get("passed")),
                        "report": json.loads(rows[stage]["report"]) if rows.get(stage, {}).get("report") else None}
                for stage in STAGES}

    def save(self, stage, source):
        with self.db() as db:
            db.execute("INSERT INTO lesson_stages(stage, source, updated_at) VALUES (?,?,?) "
                       "ON CONFLICT(stage) DO UPDATE SET source=excluded.source, updated_at=excluded.updated_at",
                       (stage, source, time.time()))

    def record(self, stage, report):
        with self.db() as db:
            db.execute("INSERT INTO lesson_stages(stage, passed, report, updated_at) VALUES (?,?,?,?) "
                       "ON CONFLICT(stage) DO UPDATE SET passed=MAX(lesson_stages.passed, excluded.passed), "
                       "report=excluded.report, updated_at=excluded.updated_at",
                       (stage, int(report["passed"]), json.dumps(report, ensure_ascii=False), time.time()))


class StageRequest(BaseModel):
    stage: Stage


class DraftRequest(StageRequest):
    source: str = Field(max_length=150_000)


def run_verification(source):
    """Only this subprocess executes candidate Python; it is NOT a security sandbox."""
    started = time.monotonic()
    # Incomplete typing / unfilled gaps should not wait for a cold PyTorch import.
    try:
        tree = ast.parse(source)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LlamaAttention"]
        if len(classes) != 1:
            raise ValueError("请定义一个 LlamaAttention 类；空草稿不会自动使用 reference。")
        methods = {node.name for node in classes[0].body if isinstance(node, ast.FunctionDef)}
        if not {"__init__", "forward"} <= methods:
            raise ValueError("请先完成 __init__ 和 forward 的接口与方法体。")
        if any(isinstance(node, ast.Name) and node.id.startswith("__BLANK_") for node in ast.walk(tree)):
            raise ValueError("还有 __BLANK_ 空缺未填写。先补全这些核心位置，再验证。")
    except (SyntaxError, ValueError) as error:
        return {"passed": False, "tests": [], "error": f"{type(error).__name__}: {error}",
                "line": getattr(error, "lineno", None), "source_sha256": source_hash(source),
                "wall_seconds": round(time.monotonic() - started, 2), "verified_at": time.time()}
    with tempfile.TemporaryDirectory(prefix="qk-lesson-") as temporary:
        source_file = Path(temporary) / "learner.py"
        output_file = Path(temporary) / "result.json"
        source_file.write_text(source, encoding="utf-8")
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
        try:
            process = subprocess.run(
                [sys.executable, str(ROOT / "qk" / "lesson_worker.py"), "verify", "--source", str(source_file),
                 "--output", str(output_file)], cwd=ROOT, env=environment, capture_output=True,
                encoding="utf-8", errors="replace", timeout=120)
            if not output_file.exists():
                raise RuntimeError((process.stderr or "验证进程未返回结果。")[-1600:])
            report = json.loads(output_file.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            report = {"passed": False, "tests": [], "error": "运行超过 120 秒。检查是否存在死循环或过大的张量。"}
        except (OSError, ValueError, RuntimeError) as error:
            report = {"passed": False, "tests": [], "error": str(error)}
    report["source_sha256"] = source_hash(source)
    report["wall_seconds"] = round(time.monotonic() - started, 2)
    report["verified_at"] = time.time()
    return report


def create_lesson_router(database):
    router = APIRouter(prefix="/api/lesson", tags=["source lesson"])
    store = LessonStore(database)
    lock = threading.Lock()
    provenance = read_json("provenance.json")
    teaching = read_json("lesson.json")
    reference = (LESSON / "reference.py").read_text(encoding="utf-8")
    starters = {"follow": "", "cloze": cloze_source(reference, teaching["gaps"]), "recall": recall_source(reference)}

    def runtime_issue():
        try:
            current = version("transformers")
            module = distribution("transformers").locate_file("transformers/models/llama/modeling_llama.py")
            if current != provenance["version"] or hashlib.sha256(Path(module).read_bytes()).hexdigest() != provenance["module_sha256"]:
                return "本课固定使用原版 Transformers 5.1.0；当前安装与已核对的源码不一致。请先检查环境，不要沿用旧运行观察。"
        except (OSError, ValueError, PackageNotFoundError) as error:
            return f"无法检查本课运行环境：{error}"
        return None

    def authorize(stage, state):
        index = STAGES.index(stage)
        if index and not state[STAGES[index - 1]]["passed"]:
            raise HTTPException(403, "先完成并验证上一阶段，再进入本阶段。")

    @router.get("")
    def lesson(stage: Stage = "follow"):
        state = store.state()
        authorize(stage, state)
        issue = runtime_issue()
        payload = {"stage": stage, "provenance": provenance, "subtitle": teaching["subtitle"],
                   "context": teaching["context"], "outcomes": teaching["outcomes"],
                   "runtime_context": teaching["runtime_context"], "runtime_issue": issue,
                   "progress": {key: {"passed": value["passed"], "unlocked": i == 0 or state[STAGES[i - 1]]["passed"]}
                                for i, (key, value) in enumerate(state.items())},
                   "source": state[stage]["source"] if state[stage]["source"] is not None else starters[stage],
                   "report": state[stage]["report"]}
        if stage == "follow":
            observations = read_json("observations.json")
            valid = (observations.get("source_sha256") == provenance["sha256"]
                     and observations.get("worker_sha256") == hashlib.sha256((ROOT / "qk/lesson_worker.py").read_bytes()).hexdigest()
                     and hashlib.sha256((LESSON / "reference.py").read_bytes()).hexdigest() == provenance["sha256"])
            payload.update(reference=reference, notes=teaching["line_notes"], symbols=teaching["symbols"],
                           tokens=token_locations(reference), dependencies=read_json("dependencies.json"),
                           observations=observations if valid and not issue else None,
                           observation_issue=None if valid and not issue else issue or "参考运行记录已过期，需重新生成。")
        elif stage == "cloze":
            payload["gaps"] = [{"id": gap["id"], "title": gap["title"], "goal": gap["goal"]} for gap in teaching["gaps"]]
        return payload

    @router.post("/draft")
    def draft(req: DraftRequest):
        authorize(req.stage, store.state())
        store.save(req.stage, req.source)
        return {"ok": True, "source_sha256": source_hash(req.source)}

    @router.post("/verify")
    def verify(req: DraftRequest):
        authorize(req.stage, store.state())
        issue = runtime_issue()
        if issue:
            raise HTTPException(409, issue)
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "正在验证另一个实现，请等它完成后再试。")
        try:
            report = run_verification(req.source)
            store.record(req.stage, report)  # never overwrite a newer draft with the submitted snapshot
            report["progress"] = {key: value["passed"] for key, value in store.state().items()}
            return report
        finally:
            lock.release()

    @router.post("/reset")
    def reset(req: StageRequest):
        authorize(req.stage, store.state())
        store.save(req.stage, starters[req.stage])
        return {"source": starters[req.stage], "message": "本阶段草稿已恢复；已通过的学习记录保留。"}

    return router
