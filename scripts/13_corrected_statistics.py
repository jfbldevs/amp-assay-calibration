#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.statistics import run_statistics

parser = argparse.ArgumentParser()
parser.add_argument("--scope", choices=["sample", "expanded"], default="sample")
run_statistics(parser.parse_args().scope)
