#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.posterior_checks import check_scope
parser = argparse.ArgumentParser()
parser.add_argument("--scope", choices=["sample", "expanded"], default="sample")
check_scope(parser.parse_args().scope)
