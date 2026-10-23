#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.bibliography import verify_bibliography

parser = argparse.ArgumentParser()
parser.add_argument("--expanded", action="store_true")
verify_bibliography(parser.parse_args().expanded)
