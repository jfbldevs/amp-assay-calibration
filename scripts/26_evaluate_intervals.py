#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.intervals import evaluate_intervals

parser = argparse.ArgumentParser()
parser.add_argument("--scope", choices=["sample","expanded"], default="sample")
args = parser.parse_args()
print(evaluate_intervals(args.scope), flush=True)
