"""
Robust check script to locate models/labels.json, infer project root,
and print sample counts per class under data/keypoints.

Usage:
    python check.py
"""
from pathlib import Path
import json
import sys


def find_labels_json(start_path: Path = None, max_levels: int = 8) -> Path:
    """
    From start_path (or this file's parent) search upward for models/labels.json.
    Return the Path if found, else None.
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    p = start_path
    for _ in range(max_levels):
        candidate = p / "models" / "labels.json"
        if candidate.exists():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    return None


def infer_and_create_labels(project_root: Path, default_labels=None) -> list:
    """
    If models/labels.json is missing, infer labels from data/keypoints folder names,
    or fall back to default_labels. Save the labels.json for future runs.
    """
    if default_labels is None:
        default_labels = ['good', 'left', 'number1', 'number2', 'number3', 'heart', 'right', 'stop']

    data_dir = project_root / "data" / "keypoints"
    labels = None
    if data_dir.exists() and data_dir.is_dir():
        dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        if dirs:
            labels = dirs
    if not labels:
        labels = default_labels

    labels_path = project_root / "models" / "labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Saved inferred labels.json to: {labels_path}")
    return labels


def count_samples_per_label(project_root: Path, labels: list):
    data_dir = project_root / "data" / "keypoints"
    print("\nCounting .npy samples under:", data_dir)
    total = 0
    counts = {}
    missing_dirs = []
    for lbl in labels:
        p = data_dir / lbl
        if p.exists() and p.is_dir():
            cnt = len(list(p.glob("*.npy")))
            counts[lbl] = cnt
            total += cnt
        else:
            counts[lbl] = 0
            missing_dirs.append(lbl)

    print("\nSample counts per class (using models/labels.json order):")
    for lbl in labels:
        print(f"  {lbl}: {counts[lbl]}")
    print(f"\nTotal samples counted: {total}")

    # Detect unexpected directories present in data/keypoints not in labels.json
    unexpected = []
    if data_dir.exists() and data_dir.is_dir():
        for d in sorted([d.name for d in data_dir.iterdir() if d.is_dir()]):
            if d not in labels:
                unexpected.append(d)
    if missing_dirs:
        print("\nWarning: the following label directories listed in labels.json are missing under data/keypoints:")
        for m in missing_dirs:
            print("  -", m)
    if unexpected:
        print("\nNote: the following directories exist in data/keypoints but are NOT in models/labels.json:")
        for u in unexpected:
            print("  -", u)
    print()


def main():
    # locate labels.json
    labels_path = find_labels_json()
    if labels_path:
        try:
            labels = json.load(open(labels_path, "r", encoding="utf-8"))
            print("Found labels.json at:", labels_path)
            print("Labels order loaded:", labels)
            project_root = labels_path.parent.parent  # models -> project root
        except Exception as e:
            print("Error reading labels.json:", e)
            sys.exit(1)
    else:
        # not found, infer from data/keypoints or use default and create labels.json
        print("models/labels.json not found — attempting to infer labels from data/keypoints or use default.")
        project_root = Path(__file__).resolve().parent
        labels = infer_and_create_labels(project_root)

    # Count samples
    count_samples_per_label(project_root, labels)


if __name__ == "__main__":
    main()