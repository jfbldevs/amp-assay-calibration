#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.curation import curate

parser = argparse.ArgumentParser()
parser.add_argument("--expanded", action="store_true")
parser.add_argument("--offline-taxonomy", action="store_true")
parser.add_argument("--refresh-publications", action="store_true",
                    help="Use the latest verified crosswalk; this invalidates existing analysis hashes.")
args = parser.parse_args()
curate(args.expanded, not args.offline_taxonomy, args.refresh_publications)
