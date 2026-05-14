#!/usr/bin/env python3
"""
Velqua Orchestrator — Autonomous agent loop.

Runs Claude Code sessions iteratively until a task meets all completion
criteria. Agents self-prompt based on actual failure output, not
hardcoded instructions. Writes status to the Mesh noteboard so the
dashboard shows what's happening.

Usage:
    python orchestrator.py --task tasks/velqua_mesh.json
    python orchestrator.py --task tasks/velqua_mesh.json --dry-run
    python orchestrator.py --list
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
TASKS_DIR = ROOT / "tasks"
LOGS_DIR = ROOT / "orchestrator_logs"
CLAUDE_BIN = Path("/home/phnix/.local/bin/claude")

TASKS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ITERATIONS_PER_CRITERION = 5   # escalate after this many failed attempts
MAX_TOTAL_ITERATIONS = 30          # hard stop regardless
CLAUDE_TIMEOUT = 600               # 10 min per claude session
ITERATION_DELAY = 5                # seconds between iterations

# ── Task schema ───────────────────────────────────────────────────────────────

TASK_SCHEMA = {
    "task": "Human-readable task name",
    "description": "What needs to be built or fixed",
    "working_dir": "Path to work in (absolute or relative to orchestrator.py)",
    "agent": "velqua",             # agent identity for noteboard
    "criteria": [                  # completion criteria — each must pass
        {
            "name": "tests_pass",
            "type": "command",      # run a command, check exit code
            "command": "python -m pytest tests/ -q",
            "expect": "exit_0",
        },
        {
            "name": "no_todos",
            "type": "grep_absent",  # grep must find nothing
            "pattern": "TODO|FIXME|HACK",
            "paths": ["backend/mesh/"],
        },
        {
            "name": "readme_has_mesh",
            "type": "grep_present", # grep must find something
            "pattern": "## Mesh",
            "paths": ["README.md"],
        },
    ],
    "initial_prompt": "Full instructions for the first iteration",
    "max_iterations": MAX_TOTAL_ITERATIONS,
}


# ── Criterion checking ─────────────────────────────────────────────────────────

def check_criterion(criterion: dict, working_dir: Path) -> tuple[bool, str]:
    """
    Check a single criterion. Returns (passed, detail).
    detail is the failure description used to generate the next prompt.
    """
    ctype = criterion["type"]
    name = criterion["name"]

    if ctype == "command":
        cmd = criterion["command"]
        expect = criterion.get("expect", "exit_0")
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=working_dir,
                capture_output=True, text=True, timeout=120
            )
            if expect == "exit_0":
                if result.returncode == 0:
                    return True, ""
                output = (result.stdout + result.stderr)[-3000:]
                return False, f"Command failed (exit {result.returncode}):\n{output}"
            elif expect == "exit_nonzero":
                if result.returncode != 0:
                    return True, ""
                return False, f"Command should have failed but succeeded"
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after 120s: {cmd}"
        except Exception as e:
            return False, f"Command error: {e}"

    elif ctype == "grep_absent":
        pattern = criterion["pattern"]
        paths = criterion.get("paths", ["."])
        found_lines = []
        for path_str in paths:
            path = working_dir / path_str
            if not path.exists():
                continue
            targets = list(path.rglob("*.py")) if path.is_dir() else [path]
            for f in targets:
                try:
                    result = subprocess.run(
                        ["grep", "-n", "-E", pattern, str(f)],
                        capture_output=True, text=True
                    )
                    if result.stdout.strip():
                        for line in result.stdout.strip().splitlines()[:5]:
                            found_lines.append(f"{f.relative_to(working_dir)}:{line}")
                except Exception:
                    pass
        if not found_lines:
            return True, ""
        sample = "\n".join(found_lines[:10])
        return False, f"Pattern '{pattern}' still present:\n{sample}"

    elif ctype == "grep_present":
        pattern = criterion["pattern"]
        paths = criterion.get("paths", ["."])
        for path_str in paths:
            path = working_dir / path_str
            if not path.exists():
                return False, f"File not found: {path_str}"
            targets = list(path.rglob("*.py")) if path.is_dir() else [path]
            for f in targets:
                try:
                    result = subprocess.run(
                        ["grep", "-l", "-E", pattern, str(f)],
                        capture_output=True, text=True
                    )
                    if result.stdout.strip():
                        return True, ""
                except Exception:
                    pass
        return False, f"Pattern '{pattern}' not found in {paths}"

    elif ctype == "file_exists":
        path = working_dir / criterion["path"]
        if path.exists():
            return True, ""
        return False, f"Required file missing: {criterion['path']}"

    elif ctype == "python_import":
        module = criterion["module"]
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import {module}; print('ok')"],
                cwd=working_dir, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return True, ""
            return False, f"Cannot import {module}:\n{result.stderr}"
        except Exception as e:
            return False, f"Import check error: {e}"

    return False, f"Unknown criterion type: {ctype}"


def check_all_criteria(task: dict, working_dir: Path) -> list[dict]:
    """Check all criteria. Returns list of results."""
    results = []
    for criterion in task["criteria"]:
        passed, detail = check_criterion(criterion, working_dir)
        results.append({
            "name": criterion["name"],
            "passed": passed,
            "detail": detail,
        })
    return results


# ── Prompt generation ──────────────────────────────────────────────────────────

def generate_prompt(task: dict, iteration: int, failed_criteria: list[dict],
                    iteration_history: dict) -> str:
    """
    Generate the next Claude prompt based on what specifically failed.
    Not hardcoded — derived from actual failure output.
    """
    if iteration == 1:
        return task["initial_prompt"]

    lines = [
        f"# Velqua Orchestrator — Iteration {iteration}",
        f"",
        f"Task: {task['task']}",
        f"",
        f"The previous iteration did not satisfy all completion criteria.",
        f"Fix ONLY the issues listed below. Do not refactor unrelated code.",
        f"",
        f"## Failing criteria:",
        f"",
    ]

    for fc in failed_criteria:
        name = fc["name"]
        detail = fc["detail"]
        attempts = iteration_history.get(name, 0)

        lines.append(f"### {name} (attempt {attempts}/{MAX_ITERATIONS_PER_CRITERION})")
        lines.append(f"")
        lines.append(detail)
        lines.append(f"")

        if attempts >= MAX_ITERATIONS_PER_CRITERION - 1:
            lines.append(f"⚠️  This criterion has failed {attempts} times.")
            lines.append(f"Try a completely different approach.")
            lines.append(f"")

    lines += [
        f"## Constraints",
        f"- All existing passing tests must still pass",
        f"- Do not change the public API",
        f"- Fix root cause, not symptoms",
        f"",
        f"When all criteria pass, say: DONE",
    ]

    return "\n".join(lines)


# ── Noteboard integration ──────────────────────────────────────────────────────

def post_to_noteboard(agent: str, content: str, tags: list[str] = None):
    """Write status to Velqua Mesh noteboard if available."""
    try:
        import sys
        sys.path.insert(0, str(ROOT / "backend"))
        from mesh.noteboard import Noteboard
        nb = Noteboard()
        nb.post(from_agent=agent, to_agent="any", content=content,
                tags=tags or ["orchestrator"])
    except Exception:
        pass  # noteboard is optional


# ── Claude runner ──────────────────────────────────────────────────────────────

def run_claude(prompt: str, working_dir: Path, task_name: str,
               iteration: int, dry_run: bool = False) -> str:
    """
    Run a Claude Code session with the given prompt.
    Returns the output text.
    """
    log_file = LOGS_DIR / f"{task_name}_iter{iteration:03d}_{datetime.now():%Y%m%d_%H%M%S}.log"

    if dry_run:
        print(f"\n[DRY RUN] Would run Claude with prompt:")
        print(f"{'─'*60}")
        print(prompt[:500] + ("..." if len(prompt) > 500 else ""))
        print(f"{'─'*60}")
        return "DRY RUN — no output"

    print(f"  Running Claude (timeout {CLAUDE_TIMEOUT}s)...")
    print(f"  Log: {log_file.name}")

    try:
        result = subprocess.run(
            [str(CLAUDE_BIN), "--print", "--dangerously-skip-permissions", prompt],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT,
            env={**os.environ, "CLAUDE_WORKING_DIR": str(working_dir)},
        )
        output = result.stdout + result.stderr
        log_file.write_text(
            f"ITERATION {iteration}\n"
            f"PROMPT:\n{prompt}\n\n"
            f"OUTPUT:\n{output}\n"
        )
        return output
    except subprocess.TimeoutExpired:
        msg = f"Claude timed out after {CLAUDE_TIMEOUT}s"
        log_file.write_text(f"TIMEOUT: {msg}\n")
        return msg
    except Exception as e:
        msg = f"Claude run error: {e}"
        log_file.write_text(f"ERROR: {msg}\n")
        return msg


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_task(task_path: Path, dry_run: bool = False):
    """Run the orchestration loop for a task."""
    task = json.loads(task_path.read_text())
    task_name = task_path.stem

    # Resolve working directory
    working_dir_str = task.get("working_dir", str(ROOT))
    working_dir = Path(working_dir_str)
    if not working_dir.is_absolute():
        working_dir = ROOT / working_dir
    working_dir = working_dir.resolve()

    if not working_dir.exists():
        print(f"[!] Working directory not found: {working_dir}")
        sys.exit(1)

    max_iterations = task.get("max_iterations", MAX_TOTAL_ITERATIONS)
    agent = task.get("agent", "orchestrator")

    print(f"\n{'='*60}")
    print(f"ORCHESTRATOR: {task['task']}")
    print(f"Working dir:  {working_dir}")
    print(f"Max iter:     {max_iterations}")
    print(f"Criteria:     {len(task['criteria'])}")
    print(f"{'='*60}\n")

    post_to_noteboard(agent, f"Orchestrator started: {task['task']}", ["start"])

    iteration_history = {}  # criterion_name → consecutive failure count

    for iteration in range(1, max_iterations + 1):
        print(f"\n── Iteration {iteration}/{max_iterations} ─────────────────────────")

        # Check criteria
        print(f"  Checking criteria...")
        results = check_all_criteria(task, working_dir)

        passed = [r for r in results if r["passed"]]
        failed = [r for r in results if not r["passed"]]

        for r in results:
            status = "✓" if r["passed"] else "✗"
            print(f"  {status} {r['name']}")

        if not failed:
            print(f"\n{'='*60}")
            print(f"✓ ALL CRITERIA PASSED — Task complete in {iteration-1} iterations")
            print(f"{'='*60}")
            post_to_noteboard(agent,
                f"Task complete: {task['task']} ({iteration-1} iterations)",
                ["complete", "success"])

            # Update task file with completion
            task["status"] = "complete"
            task["completed_at"] = datetime.now().isoformat()
            task["iterations_used"] = iteration - 1
            task_path.write_text(json.dumps(task, indent=2))
            return True

        # Update iteration history
        failed_names = {f["name"] for f in failed}
        for r in results:
            if not r["passed"]:
                iteration_history[r["name"]] = iteration_history.get(r["name"], 0) + 1
            else:
                iteration_history[r["name"]] = 0  # reset on pass

        # Check if any criterion is stuck
        stuck = [name for name, count in iteration_history.items()
                 if count >= MAX_ITERATIONS_PER_CRITERION]
        if stuck:
            msg = (f"Stuck on criteria after {MAX_ITERATIONS_PER_CRITERION} attempts: "
                   f"{', '.join(stuck)}. Human review needed.")
            print(f"\n[!] {msg}")
            post_to_noteboard(agent, msg, ["stuck", "needs_human"])
            task["status"] = "stuck"
            task["stuck_on"] = stuck
            task["stuck_at"] = datetime.now().isoformat()
            task_path.write_text(json.dumps(task, indent=2))
            return False

        # Generate next prompt
        prompt = generate_prompt(task, iteration, failed, iteration_history)

        # Post status to noteboard
        post_to_noteboard(agent,
            f"Iter {iteration}: {len(failed)}/{len(results)} criteria failing. "
            f"Working on: {', '.join(f['name'] for f in failed)}",
            ["progress"])

        # Run Claude
        output = run_claude(prompt, working_dir, task_name, iteration, dry_run)

        if "DONE" in output.upper() and not failed:
            print("  Claude reported DONE")

        # Small delay before next check
        if not dry_run:
            time.sleep(ITERATION_DELAY)

    # Hit max iterations
    msg = f"Max iterations ({max_iterations}) reached without completing task."
    print(f"\n[!] {msg}")
    post_to_noteboard(agent, msg, ["timeout", "needs_human"])
    task["status"] = "timeout"
    task_path.write_text(json.dumps(task, indent=2))
    return False


# ── Task templates ─────────────────────────────────────────────────────────────

def create_task_template(name: str):
    """Write a task template JSON to tasks/."""
    template = {
        "task": name,
        "description": "Describe what needs to be built or fixed",
        "working_dir": str(ROOT),
        "agent": "orchestrator",
        "status": "pending",
        "max_iterations": MAX_TOTAL_ITERATIONS,
        "criteria": [
            {
                "name": "tests_pass",
                "type": "command",
                "command": "python -m pytest tests/ -q --tb=short",
                "expect": "exit_0",
            },
            {
                "name": "example_file_exists",
                "type": "file_exists",
                "path": "examples/demo.py",
            },
            {
                "name": "no_todos",
                "type": "grep_absent",
                "pattern": "TODO|FIXME",
                "paths": ["backend/mesh/"],
            },
        ],
        "initial_prompt": (
            f"# Task: {name}\n\n"
            f"Describe what to build here. Be specific.\n\n"
            f"When all criteria pass, say: DONE"
        ),
    }
    out = TASKS_DIR / f"{name.lower().replace(' ', '_')}.json"
    out.write_text(json.dumps(template, indent=2))
    print(f"Created task template: {out}")
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Velqua Orchestrator")
    parser.add_argument("--task", type=Path, help="Path to task JSON file")
    parser.add_argument("--new", type=str, metavar="NAME",
                        help="Create a new task template")
    parser.add_argument("--list", action="store_true",
                        help="List all tasks and their status")
    parser.add_argument("--check", type=Path, metavar="TASK",
                        help="Check criteria for a task without running Claude")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check criteria and show prompts, don't run Claude")
    return parser.parse_args()


def list_tasks():
    tasks = list(TASKS_DIR.glob("*.json"))
    if not tasks:
        print("No tasks found in tasks/")
        return
    print(f"\n{'Task':<40} {'Status':<12} {'Iterations'}")
    print(f"{'─'*40} {'─'*12} {'─'*10}")
    for t in sorted(tasks):
        data = json.loads(t.read_text())
        status = data.get("status", "pending")
        iters = data.get("iterations_used", "—")
        print(f"{t.stem:<40} {status:<12} {iters}")


def main():
    args = parse_args()

    if args.new:
        create_task_template(args.new)
        return

    if args.list:
        list_tasks()
        return

    if args.check:
        task = json.loads(args.check.read_text())
        working_dir = Path(task.get("working_dir", ROOT)).resolve()
        results = check_all_criteria(task, working_dir)
        print(f"\nCriteria check: {args.check.name}")
        for r in results:
            status = "✓" if r["passed"] else "✗"
            print(f"  {status} {r['name']}")
            if not r["passed"] and r["detail"]:
                for line in r["detail"].splitlines()[:5]:
                    print(f"      {line}")
        return

    if args.task:
        if not args.task.exists():
            print(f"[!] Task file not found: {args.task}")
            sys.exit(1)
        success = run_task(args.task, dry_run=args.dry_run)
        sys.exit(0 if success else 1)

    print("No action specified. Use --help")


if __name__ == "__main__":
    main()
