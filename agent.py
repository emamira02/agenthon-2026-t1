"""Agenthon 2026 Track 1 coding agent.

Reads a task directory, asks the House model for a self-contained Python
solution, runs it, checks the deliverables and repairs the script on failure.
Always exits 0; every diagnostic goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

START = time.monotonic()
MAX_REQUESTS = 20          # House allows 25 admitted requests per unit; keep a margin
MAX_OUTPUT_TOKENS = 4000   # per-request output cap on the House route
DEFAULT_BUDGET_S = 1800.0
SAFETY_MARGIN_S = 150.0
MAX_ATTEMPTS = 6
OUTPUT_ROOTS = ("/app/output", "/output")
BACKUP_NAME = ".agent_prev_ok"


def log(msg: str) -> None:
    print(f"[agent {time.monotonic() - START:7.1f}s] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- task input


def read_budget(task_dir: Path) -> float:
    card = task_dir / "card.toml"
    try:
        import tomllib

        data = tomllib.loads(card.read_text())
        value = float(data.get("agent", {}).get("timeout_sec", DEFAULT_BUDGET_S))
        return max(120.0, value)
    except Exception:
        return DEFAULT_BUDGET_S


def path_map(task_dir: Path) -> dict[str, str]:
    """Map container paths named in the task (e.g. /app/prices.csv) to mounted files."""
    mapping: dict[str, str] = {}
    env = task_dir / "environment"
    dockerfile = env / "Dockerfile"
    if dockerfile.is_file():
        for line in dockerfile.read_text(errors="replace").splitlines():
            parts = line.strip().split()
            if len(parts) < 3 or parts[0].upper() not in {"COPY", "ADD"}:
                continue
            args = [p for p in parts[1:] if not p.startswith("--")]
            if len(args) < 2:
                continue
            *srcs, dest = args
            for src in srcs:
                src_path = (env / src).resolve()
                if not str(src_path).startswith(str(env.resolve())) or not src_path.exists():
                    continue
                if src_path.is_dir():
                    mapping[dest.rstrip("/") or "/"] = str(src_path)
                elif dest.endswith("/") or len(srcs) > 1:
                    mapping[dest.rstrip("/") + "/" + src_path.name] = str(src_path)
                else:
                    mapping[dest] = str(src_path)
    data_dir = env / "data"
    if data_dir.is_dir():
        mapping.setdefault("/app/data", str(data_dir))
        for f in data_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(data_dir).as_posix()
                mapping.setdefault(f"/app/{rel}", str(f))
                mapping.setdefault(f"/app/data/{rel}", str(f))
    return mapping


def preview_file(path: Path, limit: int = 1500) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in {".parquet", ".pqt"}:
            import pandas as pd

            df = pd.read_parquet(path)
            return (
                f"shape={df.shape}\ndtypes:\n{df.dtypes.to_string()}\n"
                f"head:\n{df.head(5).to_string()}"
            )[:limit * 2]
        if suffix in {".xlsx", ".xls"}:
            import pandas as pd

            sheets = pd.read_excel(path, sheet_name=None, nrows=5)
            return "\n".join(f"sheet {k}:\n{v.to_string()}" for k, v in sheets.items())[:limit * 2]
        with path.open("r", errors="replace") as fh:
            text = fh.read(limit)
        if suffix in {".csv", ".tsv", ".txt"}:
            try:
                n_lines = sum(1 for _ in path.open("rb"))
                text += f"\n... ({n_lines} lines total)"
            except Exception:
                pass
        return text
    except Exception as exc:  # binary or unreadable
        return f"<preview unavailable: {type(exc).__name__}>"


def describe_inputs(task_dir: Path, mapping: dict[str, str]) -> str:
    env = task_dir / "environment"
    files = sorted(p for p in env.rglob("*") if p.is_file() and p.name != "Dockerfile") if env.is_dir() else []
    reverse: dict[str, list[str]] = {}
    for logical, actual in mapping.items():
        reverse.setdefault(actual, []).append(logical)
    chunks = []
    total = 0
    for f in files[:40]:
        size = f.stat().st_size
        names = sorted(n for n in reverse.get(str(f), []))
        header = f"### {f}  ({size} bytes)" + (f"  — referenced in the task as {', '.join(names)}" if names else "")
        body = preview_file(f) if total < 24000 else "<preview omitted>"
        total += len(body)
        chunks.append(f"{header}\n```\n{body}\n```")
    if len(files) > 40:
        chunks.append(f"... and {len(files) - 40} more files under {env}")
    return "\n\n".join(chunks) if chunks else "(no input data files)"


IGNORED_OUTPUTS = {"reward.json", "reward.txt", "pytest_report.json"}
OUT_EXT = r"(?:json|csv|parquet|pqt|html|png|txt|pkl|npy|npz|md|xlsx|svg)"


def expected_outputs(instruction: str, input_names: set[str] | None = None) -> list[str]:
    """Best-effort list of deliverables; used only as a soft check and a hint."""
    input_names = input_names or set()
    found: list[str] = []

    def add(path: str) -> None:
        name = Path(path).name
        if name in IGNORED_OUTPUTS or name in input_names or any(ch in path for ch in "{}*<>"):
            return
        if path not in found:
            found.append(path)

    for m in re.finditer(r"(?<![\w/])(?:/app/output|/output)/([A-Za-z0-9_\-./]+?\." + OUT_EXT + r")\b", instruction):
        add("/app/output/" + m.group(1))
    # Bare file names listed in an output section, e.g. "1. **`summary.json`**".
    lower = instruction.lower()
    idx = max(lower.find("save all results"), lower.find("## output"), lower.find("output files"), lower.find("deliverables"))
    if idx >= 0:
        section = instruction[idx: idx + 6000]
        for m in re.finditer(r"`([A-Za-z0-9_\-]+\." + OUT_EXT + r")`", section):
            add("/app/output/" + m.group(1))
    return found[:20]


# ---------------------------------------------------------------- model calls


class Model:
    def __init__(self) -> None:
        self.endpoint = os.environ.get("MODEL_ENDPOINT", "").rstrip("/")
        self.name = os.environ.get("MODEL_NAME", "")
        self.token = os.environ.get("MODEL_TOKEN", "")
        self.requests = 0

    @property
    def available(self) -> bool:
        return bool(self.endpoint and self.name)

    def chat(self, messages: list[dict], max_tokens: int = MAX_OUTPUT_TOKENS, timeout: float = 600.0) -> tuple[str, str]:
        if self.requests >= MAX_REQUESTS:
            raise RuntimeError("request budget exhausted")
        body = {
            "model": self.name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        url = self.endpoint + "/v1/chat/completions"
        last_err: Exception | None = None
        for attempt in range(3):
            self.requests += 1
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode())
                choice = payload["choices"][0]
                text = choice.get("message", {}).get("content") or ""
                return strip_thinking(text), choice.get("finish_reason") or ""
            except urllib.error.HTTPError as exc:
                last_err = exc
                log(f"model HTTP {exc.code}")
                if exc.code in (401, 403):
                    self.requests -= 1  # refused before admission
                    break
                if exc.code not in (429, 500, 502, 503, 504) or self.requests >= MAX_REQUESTS:
                    break
            except Exception as exc:
                last_err = exc
                log(f"model call failed: {type(exc).__name__}: {exc}")
                if self.requests >= MAX_REQUESTS:
                    break
            time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"model unavailable: {last_err}")


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_code(text: str, keep_tail: bool = False) -> str:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len).strip("\n")
    open_block = re.search(r"```(?:python|py)?[ \t]*\n(.*)$", text, flags=re.S)
    if open_block:
        body = open_block.group(1)
        return body if keep_tail else body.strip("\n")
    return text if keep_tail else text.strip()


def strip_fences(text: str) -> str:
    text = re.sub(r"^\s*```(?:python|py)?[ \t]*\n", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text


SYSTEM_PROMPT = """You are a senior quantitative developer. You solve quantitative-finance coding tasks by writing ONE complete, self-contained Python 3.13 script.

Rules:
- Follow the task specification exactly: every formula, convention, rounding rule, column name, key name, ordering, file name and output path matters, because automated tests check them.
- Read inputs only from the actual file paths listed under "Input files". The task text may name them differently (for example /app/prices.csv); use the actual paths given.
- Write every deliverable to the exact output path the task names. The output directory is os.environ.get("OUTPUT_DIR", "/app/output") (/app/output and /output are the same directory). Create directories with os.makedirs(..., exist_ok=True).
- Available libraries: numpy, pandas, scipy, statsmodels, scikit-learn, arch, polars, pyarrow, openpyxl, matplotlib, seaborn, plotly, backtrader, talib. No network access, no package installation, no GPU code.
- Be deterministic (fix random seeds when randomness is involved). Use float64. Convert numpy types to plain Python types before json.dump.
- If the task asks you to write and run a script such as /app/solve.py, ignore that mechanic: your script is executed directly; just produce the deliverables.
- Keep the script efficient (it must finish within a few minutes) and print a short summary of what was written.
- Each reply is capped at about 4,000 tokens; if the script is long, keep going and you will be asked to continue. Prefer compact code without long comments or docstrings.
- Reply with the full script in a single ```python code block and nothing else."""


def initial_messages(instruction: str, inputs: str, outputs: list[str]) -> list[dict]:
    out_hint = "\n".join(f"- {o}" for o in outputs) if outputs else "- (see the task text)"
    user = (
        f"# Task\n\n{instruction}\n\n"
        f"# Input files (actual paths inside your runtime)\n\n{inputs}\n\n"
        f"# Deliverables detected in the task\n{out_hint}\n\n"
        "Write the complete Python script now."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def ask_for_script(model: Model, messages: list[dict]) -> str | None:
    text, finish = model.chat(messages)
    code_parts = [text]
    # The House route caps each reply; continue a truncated script up to twice.
    continuations = 0
    while finish == "length" and continuations < 4 and model.requests < MAX_REQUESTS - 1:
        continuations += 1
        cont = messages + [
            {"role": "assistant", "content": "".join(code_parts)},
            {"role": "user", "content": "Your reply was cut off. Continue the script exactly where it stopped, with no repetition and no commentary, inside a ```python block."},
        ]
        more, finish = model.chat(cont)
        code_parts.append(strip_fences(more))
    code = extract_code(code_parts[0], keep_tail=True) + "".join(code_parts[1:])
    return code if code.strip() else None


# ---------------------------------------------------------------- execution


def rewrite_paths(code: str, mapping: dict[str, str]) -> str:
    for logical in sorted(mapping, key=len, reverse=True):
        actual = mapping[logical]
        if logical in ("/", "/app"):
            continue
        code = re.sub(re.escape(logical) + r"(?=[\"'/\s)]|$)", actual.replace("\\", "\\\\"), code)
    code = re.sub(r"(?<![\w/])/output(?=[/\"'])", "/app/output", code)
    return code


def run_script(code: str, timeout: float, out_dir: Path) -> tuple[int, str, str]:
    workdir = Path("/tmp/agent_work")
    workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ("MODEL_TOKEN",):
        env.pop(key, None)
    env.update({"OUTPUT_DIR": str(out_dir), "MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/mpl", "HOME": "/tmp", "PYTHONUNBUFFERED": "1"})
    try:
        proc = subprocess.run(
            [sys.executable, "-"],
            input=code,
            text=True,
            capture_output=True,
            timeout=max(30.0, timeout),
            cwd=str(workdir),
            env=env,
        )
        return proc.returncode, proc.stdout[-4000:], proc.stderr[-6000:]
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        return -9, (out or "")[-2000:], f"TIMEOUT after {timeout:.0f}s — the script is too slow."
    except Exception as exc:
        return -1, "", f"runner error: {exc}"


def check_outputs(expected: list[str]) -> list[str]:
    problems = []
    for p in expected:
        path = Path(p)
        if not path.exists():
            problems.append(f"missing deliverable {p}")
        elif path.is_file() and path.stat().st_size == 0:
            problems.append(f"empty deliverable {p}")
        elif path.suffix == ".json":
            try:
                json.loads(path.read_text())
            except Exception as exc:
                problems.append(f"{p} is not valid JSON ({exc})")
    return problems


def snapshot(out_dir: Path) -> None:
    backup = out_dir / BACKUP_NAME
    shutil.rmtree(backup, ignore_errors=True)
    backup.mkdir(parents=True, exist_ok=True)
    for item in out_dir.iterdir():
        if item.name == BACKUP_NAME:
            continue
        target = backup / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def restore(out_dir: Path) -> None:
    backup = out_dir / BACKUP_NAME
    if not backup.is_dir():
        return
    for item in list(out_dir.iterdir()):
        if item.name == BACKUP_NAME:
            continue
        shutil.rmtree(item, ignore_errors=True) if item.is_dir() else item.unlink(missing_ok=True)
    for item in backup.iterdir():
        shutil.move(str(item), str(out_dir / item.name))


def summarize_outputs(expected: list[str]) -> str:
    lines = []
    for p in expected[:12]:
        path = Path(p)
        if path.is_file():
            lines.append(f"## {p}\n```\n{preview_file(path, 800)}\n```")
    return "\n".join(lines)


# ---------------------------------------------------------------- main loop


def solve(task_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = read_budget(task_dir)
    deadline = START + budget - SAFETY_MARGIN_S
    instruction_file = task_dir / "instruction.md"
    if not instruction_file.is_file():
        log("instruction.md not found; nothing to do")
        return
    instruction = instruction_file.read_text(errors="replace")
    mapping = path_map(task_dir)
    input_names = {Path(v).name for v in mapping.values()}
    expected = expected_outputs(instruction, input_names)
    log(f"budget={budget:.0f}s, mapped paths={len(mapping)}, expected outputs={expected}")

    model = Model()
    if not model.available:
        log("MODEL_ENDPOINT/MODEL_NAME not set; cannot solve without the House model")
        return

    messages = initial_messages(instruction, describe_inputs(task_dir, mapping), expected)
    best_ok = False
    attempts = 0
    reviewed = False
    last_problems: list[str] = []
    while attempts < MAX_ATTEMPTS and model.requests < MAX_REQUESTS:
        remaining = deadline - time.monotonic()
        if remaining < 90:
            log("time budget nearly spent; stopping")
            break
        attempts += 1
        try:
            code = ask_for_script(model, messages)
        except Exception as exc:
            log(f"stopping: {exc}")
            break
        if not code:
            messages.append({"role": "user", "content": "No code was found. Reply with the full script in one ```python block."})
            continue
        code = rewrite_paths(code, mapping)
        run_timeout = min(600.0, deadline - time.monotonic() - 30)
        if run_timeout < 30:
            break
        rc, stdout, stderr = run_script(code, run_timeout, out_dir)
        problems = check_outputs(expected)
        log(f"attempt {attempts}: rc={rc}, problems={problems[:3]}, requests={model.requests}")
        only_missing = bool(problems) and all(p.startswith("missing") for p in problems)
        ok = rc == 0 and (not problems or (only_missing and attempts >= 2 and problems == last_problems))
        last_problems = problems
        if ok:
            best_ok = True
            snapshot(out_dir)
            if reviewed or code.count("\n") > 250 or model.requests >= MAX_REQUESTS - 4 or deadline - time.monotonic() < 400:
                break
            # One self-review pass against the specification.
            reviewed = True
            messages = messages[:2] + [
                {"role": "assistant", "content": f"```python\n{code}\n```"},
                {
                    "role": "user",
                    "content": (
                        "The script ran successfully. Output preview:\n"
                        f"{summarize_outputs(expected)}\n\nstdout tail:\n```\n{stdout[-1500:]}\n```\n\n"
                        "Now audit it line by line against EVERY requirement in the task (formulas, conventions, "
                        "edge cases, column/key names, units, rounding, ordering, file formats). "
                        "If everything is correct, reply with exactly NO_CHANGES. Otherwise reply with the full corrected "
                        "script in one ```python block."
                    ),
                },
            ]
            try:
                text, _ = model.chat(messages)
            except Exception as exc:
                log(f"review skipped: {exc}")
                break
            if "NO_CHANGES" in text and "```" not in text:
                log("review: no changes")
                break
            revised = rewrite_paths(extract_code(text), mapping)
            if not revised.strip():
                break
            rc2, _, stderr2 = run_script(revised, min(600.0, deadline - time.monotonic() - 30), out_dir)
            problems2 = check_outputs(expected)
            log(f"review run: rc={rc2}, problems={problems2[:3]}")
            if rc2 != 0 or len(problems2) > len(problems):
                log("review version failed; restoring the previous outputs")
                restore(out_dir)
            break
        feedback = []
        if rc != 0:
            feedback.append(f"The script exited with code {rc}.\nstderr (tail):\n```\n{stderr[-3500:]}\n```")
        if problems:
            feedback.append("Deliverable problems:\n" + "\n".join(f"- {p}" for p in problems))
        if stdout:
            feedback.append(f"stdout (tail):\n```\n{stdout[-1000:]}\n```")
        messages = messages[:2] + [
            {"role": "assistant", "content": f"```python\n{code}\n```"},
            {"role": "user", "content": "\n\n".join(feedback) + "\n\nFix the problem and reply with the full corrected script in one ```python block."},
        ]
    shutil.rmtree(out_dir / BACKUP_NAME, ignore_errors=True)
    log(f"done: success={best_ok}, attempts={attempts}, requests={model.requests}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agenthon 2026 Track 1 agent")
    parser.add_argument("verb", choices=["solve"])
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        solve(Path(args.task_dir), Path(args.out))
    except Exception as exc:  # never fail the unit on an agent bug
        log(f"unexpected error: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
