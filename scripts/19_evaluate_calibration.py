#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.evaluation import evaluate_saved
from amp_rebuild.intervals import evaluate_intervals
from amp_rebuild.verification import audit_benchmark_summaries

parser = argparse.ArgumentParser()
parser.add_argument("--scope", choices=["sample", "expanded"], default="sample")
scope = parser.parse_args().scope
evaluate_saved(scope)
subprocess.run([sys.executable, str(Path(__file__).with_name("30_audit_secondary_evaluation.py")),
                "--scope", scope], check=True)
evaluate_intervals(scope)
audit_benchmark_summaries(scope)
subprocess.run([sys.executable, str(Path(__file__).with_name("31_review_prediction_intervals.py")),
                "--scope", scope], check=True)
