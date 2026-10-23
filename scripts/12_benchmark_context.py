#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.benchmark import MODEL_NAMES, run_benchmark, summarize_benchmark
from amp_rebuild.curation import ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["sample", "expanded"], default="sample")
    parser.add_argument("--protocols", nargs="+")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, choices=range(1, 5),
                        help="Independent split workers; each uses at most four numerical threads.")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize_benchmark(ROOT / "results/reconstruction_2026-10-20" / args.scope / "benchmark")
    else:
        workers = args.workers or (3 if args.scope == "expanded" else 1)
        run_benchmark(args.scope, args.protocols, args.seeds, args.models, workers)


if __name__ == "__main__":
    main()
