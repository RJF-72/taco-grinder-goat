"""
MMIE Offline Retraining Scheduler
Periodically re-evaluates trajectory rewards and triggers SFT on the top-K set.

Usage:
    python training/mmie_retrain.py --trajectories .lariat/training/trajectories \
                                      --model-path ./data/student_titan \
                                      --output-dir ./data/student_titan_v2 \
                                      --min-reward 1.5
"""
import json
import argparse
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def load_trajectory_dirs(root: Path) -> List[Path]:
    dirs = []
    for entry in root.iterdir():
        if entry.is_dir() and (entry / "summary.json").exists():
            dirs.append(entry)
    return dirs


def load_summary(traj_dir: Path) -> Dict[str, Any]:
    with open(traj_dir / "summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_steps(traj_dir: Path) -> List[Dict[str, Any]]:
    steps = []
    for p in sorted(traj_dir.glob("step_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            steps.append(json.load(f))
    return steps


def build_sft_pair(step: Dict[str, Any], system: str = "You are TITAN.") -> Dict[str, str]:
    return {
        "system": system,
        "instruction": step.get("task", ""),
        "output": step.get("output", ""),
        "tools": step.get("tools_used", []),
    }


def main():
    parser = argparse.ArgumentParser(description="MMIE Offline Retraining Scheduler")
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-reward", type=float, default=1.5)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    traj_dirs = load_trajectory_dirs(args.trajectories)
    print(f"[mmie] Found {len(traj_dirs)} trajectory dirs")

    summaries = [(d, load_summary(d)) for d in traj_dirs]
    summaries.sort(key=lambda x: x[1].get("final_reward", -999), reverse=True)

    selected = [s for s in summaries if s[1].get("final_reward", -999) >= args.min_reward]
    selected = selected[:args.top_k]
    print(f"[mmie] Selected {len(selected)} trajectories for retraining")

    if args.dry_run:
        print("[mmie] Dry run — pairs would be written but not training triggered.")
        return

    pairs = []
    for d, _ in selected:
        steps = load_steps(d)
        for step in steps:
            pairs.append(build_sft_pair(step))

    # Write out new training file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"mmie_batch_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[mmie] Wrote {len(pairs)} pairs to {out_dir / 'train.jsonl'}")

    # Trigger training (optional subprocess call or queue job)
    # In a real deployment, this would submit a job to a scheduler.
    print(f"[mmie] Next: launch training/train.py with data_files.train={out_dir / 'train.jsonl'}")


if __name__ == "__main__":
    main()
