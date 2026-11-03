"""Run up to 30 fixed BigCodeBench-Hard tasks through main.py.

Requires datasets. Outputs go to bigcodebench_runs/ for separate evaluation.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "bigcode/bigcodebench-hard"
DATASET_VERSION = "v0.1.4"
TASK_COUNT = 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("single", "orchestrator"), default="orchestrator"
    )
    parser.add_argument("--task-ids", nargs="+", help="Run only these task IDs from the fixed subset")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Install the dataset reader with: python -m pip install datasets") from error

    tasks = list(load_dataset(DATASET, split=DATASET_VERSION).select(range(TASK_COUNT)))
    if args.task_ids:
        requested = set(args.task_ids)
        unknown = requested - {task["task_id"] for task in tasks}
        if unknown:
            parser.error("Task IDs outside the fixed subset: " + ", ".join(sorted(unknown)))
        tasks = [task for task in tasks if task["task_id"] in requested]
    run_dir = ROOT / "bigcodebench_runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    samples_path = run_dir / "samples.jsonl"
    (run_dir / "task_ids.txt").write_text(
        "".join(task["task_id"] + "\n" for task in tasks), encoding="utf-8"
    )
    environment = {**os.environ, "AGENT_MODE": args.mode}
    request = "Read TASK.md fully implement it with specificity in solution.py."

    with (
        samples_path.open("w", encoding="utf-8") as samples,
        (run_dir / "references.jsonl").open("w", encoding="utf-8") as references,
    ):
        for task in tasks:
            task_id = task["task_id"]
            references.write(json.dumps({
                "task_id": task_id,
                "canonical_solution": task["canonical_solution"],
            }) + "\n")
            project = run_dir / "tasks" / task_id.replace("/", "_")
            project.mkdir(parents=True)
            (project / "TASK.md").write_text(
                task["instruct_prompt"], encoding="utf-8"
            )

            print(f"\n=== {task_id} ===", flush=True)
            result = subprocess.run(
                [sys.executable, str(ROOT / "main.py")],
                cwd=ROOT,
                input=f"{project}\n{request}\nexit\n",
                text=True,
                env=environment,
            )

            solution_path = project / "solution.py"
            solution = (
                solution_path.read_text(encoding="utf-8")
                if solution_path.is_file()
                else ""
            )
            samples.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
            samples.flush()
            if result.returncode:
                print(f"Harness exited with status {result.returncode}; continuing.")

    print(f"\nSamples: {samples_path}")
    print(f"Run folder (tasks, references and task IDs): {run_dir}")
    print("Evaluate this file with BigCodeBench's evaluator when the run finishes.")


if __name__ == "__main__":
    main()
