#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.verification import audit_benchmark, audit_benchmark_summaries, structural_alias_audit

parser = argparse.ArgumentParser()
parser.add_argument("--scope", choices=["sample", "expanded"], default="sample")
parser.add_argument("--structures", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--summaries", action="store_true")
args = parser.parse_args()
if args.structures:
    structural_alias_audit(args.scope)
if args.benchmark:
    audit_benchmark(args.scope)
if args.summaries:
    audit_benchmark_summaries(args.scope)
