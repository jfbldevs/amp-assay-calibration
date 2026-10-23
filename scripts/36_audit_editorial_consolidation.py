#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amp_rebuild.editorial_audit import check_calibration, check_tail, check_deep

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--part", choices=["calibration","tail","deep","all"],default="all")
    args=parser.parse_args()
    for name, function in [("calibration",check_calibration),("tail",check_tail),("deep",check_deep)]:
        if args.part in [name,"all"]:
            function()
