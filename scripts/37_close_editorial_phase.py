#!/usr/bin/env python3
"""Document and close the additive editorial phase after actual visual review."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amp_rebuild.editorial_release import document, close

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--document-only",action="store_true")
    args=parser.parse_args()
    document() if args.document_only else close()
