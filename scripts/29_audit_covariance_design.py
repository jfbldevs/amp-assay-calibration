#!/usr/bin/env python3
"""Exploratory latent-covariance design audit; no model is fitted or changed."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amp_rebuild.censored import select_analysis_rows, preparation_fingerprint, active_stan_path

BLOCKS = {
    "chemical_entity": ["chemical_id"],
    "publication": ["publication_id"],
    "exact_target": ["target_exact"],
    "chemical_by_target": ["chemical_id", "target_exact"],
    "chemical_by_medium": ["chemical_id", "medium"],
}


def covariance_gram(frame, blocks):
    """Frobenius products of same-level indicator matrices, including residual I."""
    keys = list(blocks.values())
    gram = np.full((len(keys)+1, len(keys)+1), len(frame), dtype=np.int64)
    for i, left in enumerate(keys):
        for j in range(i, len(keys)):
            columns = list(dict.fromkeys(left+keys[j]))
            counts = frame.groupby(columns, dropna=False, sort=False).size().to_numpy(dtype=np.int64)
            gram[i, j] = gram[j, i] = np.square(counts).sum()
    return gram


def describe(gram):
    scale = np.sqrt(np.diag(gram).astype(float))
    normalized = gram/np.outer(scale, scale)
    eigenvalues = np.linalg.eigvalsh(normalized)
    tolerance = len(gram)*np.finfo(float).eps*float(eigenvalues[-1])
    rank = int((eigenvalues > tolerance).sum())
    return {
        "gram_integer": gram.tolist(), "normalized_gram": normalized.tolist(),
        "normalized_gram_eigenvalues": eigenvalues.tolist(), "rank_tolerance": tolerance,
        "rank": rank, "component_count": len(gram),
        "normalized_basis_condition_number": (
            float(np.sqrt(eigenvalues[-1]/eigenvalues[0])) if rank == len(gram) else None),
    }


def verify_calculation():
    # An independent dense calculation checks the counting identity on a small design.
    toy = pd.DataFrame({"chem": ["a","a","b","b","c","c"],
                        "pub": ["p","p","p","p","q","q"]})
    blocks = {"chemical": ["chem"], "publication": ["pub"]}
    matrices = [toy[c].to_numpy()[:, None] == toy[c].to_numpy()[None, :]
                for c in ["chem", "pub"]]+[np.eye(len(toy), dtype=bool)]
    direct = np.array([[np.logical_and(a,b).sum() for b in matrices] for a in matrices])
    np.testing.assert_array_equal(covariance_gram(toy, blocks), direct)
    nested_rank = describe(direct)["rank"]
    assert nested_rank == 3
    unique = pd.DataFrame({"chem": list("abcdef"), "pub": list("uvwxyz")})
    singleton_rank = describe(covariance_gram(unique, blocks))["rank"]
    assert singleton_rank == 1
    return {"dense_counting_identity_passed": True,
            "nested_no_cross_publication_replication_rank": nested_rank,
            "all_singleton_components_rank": singleton_rank,
            "note": "The nested example distinguishes three variance matrices without any chemical shared across publications; the all-singleton example correctly has only one distinct matrix."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["sample", "expanded"], default="expanded")
    args = parser.parse_args()
    source = ROOT/"data/curated"/args.scope/"activities.csv"
    columns = ["chemical_id", "publication_id", "target_exact", "medium",
               "eligible_censored_analysis", "parse_kind"]
    original = pd.read_csv(source, usecols=columns, keep_default_na=False, low_memory=False)
    cohorts = {}
    for subset in ["all", "exact", "multi_publication", "no_reported_ranges"]:
        data = select_analysis_rows(original, subset)
        if data.empty:
            raise ValueError(f"Empty covariance audit cohort: {subset}")
        cohorts[subset] = {"records": len(data), **describe(covariance_gram(data, BLOCKS))}
    answer = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "scope": args.scope,
        "analysis_status": "Exploratory diagnostic specified after reviewing the first expanded posterior",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preparation_source_sha256": preparation_fingerprint(),
        "stan_source_sha256": hashlib.sha256(active_stan_path(args.scope).read_bytes()).hexdigest(),
        "components": list(BLOCKS)+["residual"], "verification": verify_calculation(),
        "cohorts": cohorts,
        "derivation": "For K_j[i,k]=1 when records i and k share level j, <K_j,K_l>_F equals the sum of squared joint-level counts. <K_j,I>_F=<I,I>_F=N. These are the component matrices of the marginal latent Gaussian covariance sum_j tau_j^2 K_j + sigma^2 I.",
        "normalization": "Each component matrix is scaled to unit Frobenius norm. The condition number is the square root of the normalized Gram eigenvalue ratio; it is not a posterior or Fisher-information condition number.",
        "limitations": [
            "This checks linear independence of full latent Gaussian covariance matrices before fixed-effect projection; it is not a REML information-matrix audit.",
            "Censoring, finite data and parameter values can reduce inferential precision even when these matrices are linearly independent.",
            "Neither algebraic independence nor good posterior precision validates exchangeability, distributional assumptions or causal biological attribution.",
            "The two prior-scale fits use the same all-record covariance design.",
        ],
    }
    output = ROOT/"results/reconstruction_2026-10-20"/args.scope/"statistics/covariance_design_audit.json"
    output.write_text(json.dumps(answer, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"scope": args.scope, "verification": answer["verification"],
                      "cohorts": {k: {name:v[name] for name in ["records","rank","component_count","normalized_basis_condition_number"]}
                                  for k,v in cohorts.items()}}), flush=True)


if __name__ == "__main__":
    main()
