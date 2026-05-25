"""
Gap-Filling Evaluator
Runs the student model on a hold-out set of Rust tasks and identifies weak areas.

Usage:
    python training/gap_eval.py --model-path ./data/student_titan \
                                  --tasks ./eval/tasks.jsonl \
                                  --output ./eval/results.json
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def evaluate_task(task: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    # Stub: in real life, instantiate the backend and run inference
    # Here we return a placeholder result structure.
    return {
        "task_id": task.get("id", "unknown"),
        "success": True,
        "compile_pass": True,
        "test_pass": True,
        "num_steps": 3,
        "latency_ms": 1200,
        "memory_mb": 1800,
    }


def main():
    parser = argparse.ArgumentParser(description="Gap-Filling Evaluator")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    print(f"[gap] Loaded {len(tasks)} tasks")

    results = []
    for task in tasks:
        result = evaluate_task(task, args.model_path)
        results.append(result)

    # Compute per-category stats
    categories: Dict[str, List[bool]] = {}
    for task, res in zip(tasks, results):
        cat = task.get("category", "code")
        categories.setdefault(cat, []).append(res["success"])

    print("\n[gap] Results by category:")
    for cat, vals in categories.items():
        rate = sum(vals) / len(vals)
        print(f"  {cat}: {rate:.2%} ({sum(vals)}/{len(vals)})")
        if rate < 0.7:
            print(f"    ⚠️  Weak area: {cat}. Generate more teacher trajectories here.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"results": results, "categories": {k: sum(v)/len(v) for k,v in categories.items()}}, f, indent=2)
    print(f"\n[gap] Full results saved to {args.output}")


if __name__ == "__main__":
    main()
