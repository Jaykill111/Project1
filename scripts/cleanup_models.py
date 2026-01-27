"""
Cleanup old model files in api/models, keeping only the latest per threshold.
Supports:
- Corners classifiers: ou*.pkl
- Goals classifiers: goals_ou*.pkl
- Goals regression: model_v*_goals*.pkl (keep only latest)

Usage:
  python scripts/cleanup_models.py            # apply cleanup
  python scripts/cleanup_models.py --dry-run  # show plan without deleting
"""

from __future__ import annotations
import os
import sys
import re
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "api" / "models"

def human_size(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}TB"

def list_files(pattern: str) -> List[Path]:
    return list(MODEL_DIR.glob(pattern))

def parse_threshold_from_pickle(path: Path) -> float | None:
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
        th = data.get("threshold")
        if th is not None:
            return float(th)
    except Exception:
        return None
    return None

def parse_threshold_from_name(path: Path) -> float | None:
    m = re.search(r"ou([0-9]+\.[0-9]+|[0-9]+)", path.name)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def group_by_threshold(files: List[Path]) -> Dict[float, List[Path]]:
    groups: Dict[float, List[Path]] = {}
    for f in files:
        th = parse_threshold_from_pickle(f)
        if th is None:
            th = parse_threshold_from_name(f)
        if th is None:
            # skip files without threshold
            continue
        groups.setdefault(th, []).append(f)
    # sort each group by mtime desc
    for th in list(groups.keys()):
        groups[th].sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return groups

def cleanup_group(groups: Dict[float, List[Path]], dry_run: bool = False) -> Tuple[int, int, int]:
    kept, deleted, bytes_reclaimed = 0, 0, 0
    for th, files in groups.items():
        if not files:
            continue
        # keep latest
        keep = files[0]
        kept += 1
        to_delete = files[1:]
        for f in to_delete:
            size = f.stat().st_size if f.exists() else 0
            if dry_run:
                print(f"DRY-RUN: delete {f.name} ({human_size(size)})")
            else:
                try:
                    f.unlink()
                    deleted += 1
                    bytes_reclaimed += size
                    print(f"Deleted {f.name} ({human_size(size)})")
                except Exception as e:
                    print(f"Failed to delete {f}: {e}")
        print(f"Keep threshold {th}: {keep.name}")
    return kept, deleted, bytes_reclaimed

def cleanup_regression(dry_run: bool = False) -> Tuple[int, int, int]:
    files = list_files("model_v*_goals*.pkl")
    if not files:
        return (0, 0, 0)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    keep = files[0]
    kept, deleted, bytes_reclaimed = 1, 0, 0
    for f in files[1:]:
        size = f.stat().st_size if f.exists() else 0
        if dry_run:
            print(f"DRY-RUN: delete {f.name} ({human_size(size)})")
        else:
            try:
                f.unlink()
                deleted += 1
                bytes_reclaimed += size
                print(f"Deleted {f.name} ({human_size(size)})")
            except Exception as e:
                print(f"Failed to delete {f}: {e}")
    print(f"Keep regression: {keep.name}")
    return kept, deleted, bytes_reclaimed

def main(argv: List[str]) -> int:
    dry = "--dry-run" in argv or "-n" in argv
    print("="*70)
    print("SCOPE Model Cleanup")
    print("="*70)
    print(f"Model directory: {MODEL_DIR}")
    if not MODEL_DIR.exists():
        print("No model directory found.")
        return 1

    # Corners
    corner_files = list_files("ou*.pkl")
    corner_groups = group_by_threshold(corner_files)
    print(f"\nCorners: found {len(corner_files)} files, {len(corner_groups)} thresholds")
    c_kept, c_deleted, c_bytes = cleanup_group(corner_groups, dry_run=dry)

    # Goals classifiers
    goals_files = list_files("goals_ou*.pkl")
    goals_groups = group_by_threshold(goals_files)
    print(f"\nGoals classifiers: found {len(goals_files)} files, {len(goals_groups)} thresholds")
    g_kept, g_deleted, g_bytes = cleanup_group(goals_groups, dry_run=dry)

    # Goals regression
    print("\nGoals regression:")
    r_kept, r_deleted, r_bytes = cleanup_regression(dry_run=dry)

    total_deleted = c_deleted + g_deleted + r_deleted
    total_bytes = c_bytes + g_bytes + r_bytes

    print("\n" + "-"*70)
    print(f"Kept: corners={c_kept}, goals_cls={g_kept}, goals_reg={r_kept}")
    print(f"Deleted: {total_deleted} files, reclaimed {human_size(total_bytes)}")
    if dry:
        print("Run without --dry-run to apply changes.")
    else:
        print("Cleanup applied.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
