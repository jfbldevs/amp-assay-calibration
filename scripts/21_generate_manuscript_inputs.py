#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.reporting import generate_manuscript_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--scope", choices=["sample", "expanded"], default="sample")
generate_manuscript_inputs(parser.parse_args().scope)
