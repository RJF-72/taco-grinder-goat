"""
LARIAT — TITAN Training Pipeline
Distillation: 680B/70B teacher → 1.5B–6B student
Stages:
  1) Collect teacher trajectories
  2) Build SFT dataset with weighted loss
  3) Multi-task fine-tuning (problem solving, tools, NLU, NLP, vision)
  4) MMIE self-improvement loop
Usage:
  python -m training.distill \
    --teacher-dir ./data/teacher_trajectories \
    --output-dir ./data/student_train \
    --mix-ratio 45,20,15,15,5
"""
import json
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

DATA_MIX = {
    "problem_solving": 0.45,
    "tools": 0.20,
    "nlu": 0.15,
    "nlp": 0.15,
    "vision": 0.05,
}

LOSS_WEIGHTS = {
    "tool_json": 2.0,
    "code": 2.0,
    "plan": 1.2,
    "nlu": 1.0,
    "nlp": 0.8,
    "vision": 1.0,
    "explain": 0.5,
}

def load_trajectories(teacher_dir: Path) -> List[Dict[str, Any]]:
    """Load all step files from teacher trajectory directories."""
    steps = []
    for traj_dir in teacher_dir.iterdir():
        if not traj_dir.is_dir():
            continue
        for step_file in traj_dir.glob("step_*.json"):
            with open(step_file, "r", encoding="utf-8") as f:
                steps.append(json.load(f))
    return steps

def bucket_steps(steps: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = {k: [] for k in DATA_MIX}
    for s in steps:
        mode = (s.get("mode") or "CODE").lower()
        # heuristic bucketing by mode
        if mode in ("requirement", "spec"):
            buckets["nlu"].append(s)
        elif mode == "explain":
            buckets["nlp"].append(s)
        elif mode == "vision":
            buckets["vision"].append(s)
        elif "tool" in s.get("output", "") or s.get("tools_used"):
            buckets["tools"].append(s)
        else:
            buckets["problem_solving"].append(s)
    return buckets

def build_sft_pairs(buckets: Dict[str, List[Dict[str, Any]]], target_total: int = 50_000) -> List[Dict[str, Any]]:
    """Sample from buckets according to DATA_MIX and format SFT pairs."""
    out = []
    for bucket_name, ratio in DATA_MIX.items():
        pool = buckets.get(bucket_name, [])
        if not pool:
            continue
        needed = int(target_total * ratio)
        chosen = random.choices(pool, k=min(needed, len(pool)))
        for step in chosen:
            pair = {
                "input": step.get("observation_refs", []),
                "system": "You are TITAN, a Rust engineer inside Lariat.",
                "instruction": step.get("task", ""),
                "output": step.get("output", ""),
                "tools": step.get("tools_used", []),
                "meta": {
                    "mode": step.get("mode", "CODE"),
                    "weight": LOSS_WEIGHTS.get(bucket_name, 1.0),
                }
            }
            out.append(pair)
    random.shuffle(out)
    return out

def save_dataset(pairs: List[Dict[str, Any]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save as jsonl for easy streaming training
    with open(output_dir / "distill_train.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    # Save a small validation split
    val_size = max(1, len(pairs) // 20)
    with open(output_dir / "distill_val.jsonl", "w", encoding="utf-8") as f:
        for p in pairs[:val_size]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Dataset saved: {len(pairs)} train, {val_size} val")

def main():
    parser = argparse.ArgumentParser(description="TITAN Distillation Dataset Builder")
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./data/student_train"))
    parser.add_argument("--target-total", type=int, default=50_000)
    args = parser.parse_args()

    print("Loading teacher trajectories...")
    steps = load_trajectories(args.teacher_dir)
    print(f"Loaded {len(steps)} steps")

    print("Bucketing...")
    buckets = bucket_steps(steps)
    for k, v in buckets.items():
        print(f"  {k}: {len(v)}")

    print("Building SFT pairs...")
    pairs = build_sft_pairs(buckets, target_total=args.target_total)

    print("Saving...")
    save_dataset(pairs, args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
