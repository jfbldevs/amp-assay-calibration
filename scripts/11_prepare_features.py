#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.features import chemical_feature_store, cluster_scope, embed_catalog

parser = argparse.ArgumentParser()
parser.add_argument("--scope", choices=["sample", "expanded"], default="sample")
parser.add_argument("--embeddings", action="store_true")
parser.add_argument("--clusters", action="store_true")
parser.add_argument("--chemical", action="store_true")
args = parser.parse_args()
if args.embeddings:
    embed_catalog()
if args.clusters:
    cluster_scope(args.scope)
if args.chemical:
    chemical_feature_store(args.scope == "expanded")
