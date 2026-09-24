#!/usr/bin/env python3
"""Create a child-only facial emotion dataset from the unified manifest.

Only the Children_Emotion source is retained because it is the source verified
as pediatric in the dataset audit. Micro_Expressions is excluded because its
subject ages were not verified.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.dataset_root / "emotion_manifest.csv"
    frame = pd.read_csv(manifest_path)
    required = {"source", "new_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    child = frame[frame["source"].eq("Children_Emotion")].copy()
    if child.empty:
        raise ValueError("No verified child images found")

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)

    for rel_path in child["new_path"]:
        source = args.dataset_root / rel_path
        if not source.exists():
            raise FileNotFoundError(source)
        target = args.output_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    child.to_csv(args.output_root / "emotion_manifest.csv", index=False)
    (args.output_root / "README.txt").write_text(
        "CHUWI child-only facial emotion dataset.\n"
        "Retained source: Children_Emotion.\n"
        "Excluded source: Micro_Expressions (child age not verified).\n",
        encoding="utf-8",
    )

    print(f"Retained verified child images: {len(child)}")
    print(f"Excluded non-verified-child images: {len(frame) - len(child)}")


if __name__ == "__main__":
    main()
