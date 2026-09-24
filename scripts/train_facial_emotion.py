#!/usr/bin/env python3
"""Compatibility entry point for Chuwi facial-model training.

Use scripts/benchmark_facial_models.py for the official reproducible experiment.
Pass --models with one model name when a single architecture is required.
"""

from scripts.benchmark_facial_models import main


if __name__ == "__main__":
    main()
