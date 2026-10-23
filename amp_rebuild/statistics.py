"""Dependence-aware descriptive reliability, matched ranks and design support."""
from itertools import combinations
from datetime import datetime, timezone
import hashlib
import json
import math

import numpy as np
import pandas as pd

from .curation import ROOT, stable_hash
from .features import UnionFind


def icc_from_sufficient(stats, multiplicity=None):
    w = np.ones(len(stats)) if multiplicity is None else np.asarray(multiplicity)
    n, total, squares = stats["n"].to_numpy(), stats["total"].to_numpy(), stats["squares"].to_numpy()
    N, G = np.sum(w*n), np.sum(w)
    if G <= 1 or N <= G:
        return math.nan
    mean = np.sum(w*total) / N
    within = np.sum(w*(squares-total**2/n))
    between = np.sum(w*n*(total/n-mean)**2)
    msw, msb = within/(N-G), between/(G-1)
    n0 = (N-np.sum(w*n*n)/N)/(G-1)
    denominator = msb + (n0-1)*msw
    return (msb-msw)/denominator if denominator > 0 else math.nan


def reliability_summary(data, groups, outcome, cluster, repeats=2000):
    working = data.copy()
    working["_sq"] = working[outcome]**2
    grouped = working.groupby(groups, dropna=False)
    stats = grouped.agg(n=(outcome, "size"), total=(outcome, "sum"), squares=("_sq", "sum"),
                        minimum=(outcome, "min"), maximum=(outcome, "max"),
                        cluster=(cluster, "first")).reset_index(drop=True)
    stats = stats[stats.n >= 2].reset_index(drop=True)
    if len(stats) < 2:
        return {"groups": len(stats), "records": int(stats.n.sum()), "icc": None}
    labels, unique = pd.factorize(stats.cluster)
    rng = np.random.default_rng(20261020)
    draws = []
    for _ in range(repeats):
        multiplicity = rng.multinomial(len(unique), np.ones(len(unique))/len(unique))[labels]
        value = icc_from_sufficient(stats, multiplicity)
        if np.isfinite(value):
            draws.append(value)
    return {
        "groups": len(stats), "records": int(stats.n.sum()), "sequence_clusters": len(unique),
        "icc": float(icc_from_sufficient(stats)), "icc_ci95": np.quantile(draws, [.025, .975]).tolist(),
        "zero_dispersion_groups": int(stats.minimum.eq(stats.maximum).sum()),
        "interpretation": "descriptive record agreement under this grouping; not a universal predictive ceiling",
        "bootstrap_unit": cluster, "bootstrap_repeats": len(draws),
    }


def analyze_gramPA(output):
    d = pd.read_csv(ROOT / "data/raw/grampa.csv")
    for field in ["sequence", "bacterium", "strain", "modifications", "url_source", "database"]:
        d[field] = d[field].fillna("").astype(str)
    d["chemical_proxy"] = [stable_hash([s, m], "GRAMPA:") for s, m in zip(d.sequence, d.modifications)]
    d = d[np.isfinite(d.value)].copy()
    known_strain = d[d.strain.ne("")]
    same_source = known_strain[known_strain.url_source.ne("")]
    result = {
        "rows": len(d), "outcome": "log10 MIC as supplied by GRAMPA",
        "all_target_records": reliability_summary(d, ["chemical_proxy", "bacterium"], "value", "sequence"),
        "reported_strain_only": reliability_summary(known_strain, ["chemical_proxy", "bacterium", "strain"],
                                                    "value", "sequence"),
        "same_source_and_reported_strain": reliability_summary(
            same_source, ["chemical_proxy", "bacterium", "strain", "url_source"], "value", "sequence"),
        "limitations": [
            "Source URLs identify database cards, not necessarily independent papers or experiments.",
            "Medium, dose-grid censoring and exact assay provenance are unavailable.",
            "Equal recorded outcomes are retained; equality alone does not establish a duplicate.",
            "Strain and modification metadata are incomplete, so between-record variation is not pure measurement error.",
        ],
    }
    d["active10"] = d.value <= 1
    counts = d.groupby(["chemical_proxy", "bacterium"]).active10.agg(["sum", "count"])
    counts = counts[counts["count"] >= 2]
    counts["discordant_ordered_pairs"] = 2*counts["sum"]*(counts["count"]-counts["sum"])
    counts["ordered_pairs"] = counts["count"]*(counts["count"]-1)
    result["binary_10uM"] = {
        "group_any_discordance": float(((counts["sum"] > 0) & (counts["sum"] < counts["count"])).mean()),
        "uniform_within_group_pair_discordance": float((counts.discordant_ordered_pairs/counts.ordered_pairs).mean()),
        "all_pairs_weighted_discordance": float(counts.discordant_ordered_pairs.sum()/counts.ordered_pairs.sum()),
        "note": "These estimands differ; none is labeled a clone-free classification limit.",
    }
    (output / "grampa_reliability.json").write_text(json.dumps(result, indent=2))
    return result


def design_support(data):
    publications = data.groupby("chemical_id").publication_id.nunique()
    chemicals = data.groupby("publication_id").chemical_id.nunique()
    p = sorted(data.chemical_id.unique())
    a = sorted(data.publication_id.unique())
    pi, ai = {x: i for i, x in enumerate(p)}, {x: i+len(p) for i, x in enumerate(a)}
    union = UnionFind(len(p)+len(a))
    for row in data[["chemical_id", "publication_id"]].drop_duplicates().itertuples(index=False):
        union.union(pi[row.chemical_id], ai[row.publication_id])
    sizes = pd.Series([union.find(i) for i in range(len(p)+len(a))]).value_counts()
    repeat_targets = data.groupby(["chemical_id", "target_exact"]).publication_id.nunique()
    return {
        "assay_cells": len(data), "chemical_entities": len(p), "publications": len(a),
        "single_publication_entities": int(publications.eq(1).sum()),
        "single_publication_fraction": float(publications.eq(1).mean()),
        "multi_publication_entities": int(publications.gt(1).sum()),
        "single_chemical_publications": int(chemicals.eq(1).sum()),
        "chemical_exact_target_pairs_in_multiple_publications": int(repeat_targets.gt(1).sum()),
        "bipartite_components": len(sizes), "largest_component_nodes": int(sizes.max()),
        "total_graph_nodes": len(p)+len(a),
        "interpretation": "Publication is a study proxy. This observational design does not identify a causal laboratory effect.",
    }


def curation_sensitivity_membership(data, activities, output):
    """Exploratory subset sensitivities; never overwrite primary assay cells."""
    exact = activities[activities.eligible_exact_analysis].copy()
    native = exact.groupby("assay_cell_id").conversion_rule.agg(
        lambda values: values.eq("native_molar").all())
    candidate = exact.groupby("assay_cell_id").duplicate_candidate.any()
    quantiles = data.log2_mic.quantile([.01, .99]).to_numpy()
    chemical = sorted(data.chemical_id.unique())
    publications = sorted(data.publication_id.unique())
    index = {v: i for i, v in enumerate(chemical+publications)}
    graph = UnionFind(len(index))
    for row in data[["chemical_id", "publication_id"]].drop_duplicates().itertuples(index=False):
        graph.union(index[row.chemical_id], index[row.publication_id])
    roots = data.chemical_id.map(lambda v: graph.find(index[v]))
    # Use node count, not outcomes, to select the largest connected component.
    sizes = pd.Series([graph.find(i) for i in range(len(index))]).value_counts()
    largest = sizes.index[0]
    flags = pd.DataFrame({"assay_cell_id": data.assay_cell_id})
    flags["native_molar_only"] = data.assay_cell_id.map(native).fillna(False)
    flags["no_candidate_duplicates"] = ~data.assay_cell_id.map(candidate).fillna(False)
    flags["central_98pct_outcomes"] = data.log2_mic.between(*quantiles)
    flags["at_least_two_context_fields"] = data.context_reporting_count.ge(2)
    flags["largest_connected_component"] = roots.eq(largest)
    flags["excluding_endpoint_ambiguity_example"] = data.publication_id.ne("PMID:25733609")
    flags.to_csv(output/"curation_sensitivity_membership.csv", index=False)
    report = {
        "status": "Exploratory sensitivities specified after inspecting the sample; not independent confirmation.",
        "primary_assay_cells": len(data),
        "subset_cell_counts": {key: int(flags[key].sum()) for key in flags.columns[1:]},
        "central_98pct_bounds_log2": quantiles.tolist(),
        "definitions": {
            "native_molar_only": "Every contributing exact record reports a native molar unit.",
            "no_candidate_duplicates": "Remove whole cells containing a flagged identical-record candidate; equality is not treated as proof of duplication.",
            "central_98pct_outcomes": "Retain observed log2 MIC between its empirical 1st and 99th percentiles; this deliberately changes the outcome distribution.",
            "at_least_two_context_fields": "At least two among the three recorded inoculum, supplement and pH fields are nonempty.",
            "largest_connected_component": "Largest chemical-publication bipartite component by number of nodes, with ties resolved by the deterministic node order.",
            "excluding_endpoint_ambiguity_example": "Exclude PMID:25733609, whose bacteriocin assay uses 50% inhibition while the table calls results MIC. Targeted source review; not a complete endpoint audit.",
        },
        "interpretation": "Subsets change both support and population. Differences do not identify bias or a causal effect.",
    }
    (output/"curation_sensitivity_design.json").write_text(json.dumps(report, indent=2))
    return report


RANK_COMPARISON_ATOL_LOG2 = 1e-12


def rank_inversion_flags(a, b, threshold=0.0):
    """Exclude machine-roundoff ties at a strict scientific threshold."""
    a, b = np.asarray(a), np.asarray(b)
    return (a*b < 0) & (np.minimum(abs(a), abs(b)) > threshold+RANK_COMPARISON_ATOL_LOG2)


def matched_rank_changes(data, output, repeats=5000):
    # Every contrast comes from the same publication and exact target label;
    # all OTHER recorded conditions, including notes, must match.
    matching = ["publication_id", "target_exact", "cfu_raw", "salt_type_raw",
                "ionic_strength_raw", "ph_raw", "note_raw"]
    working = data.copy()
    working[matching] = working[matching].fillna("")
    records = []
    for key, block in working.groupby(matching, dropna=False, sort=True):
        pivot = block.pivot_table(index="chemical_id", columns="medium", values="log2_mic", aggfunc="mean")
        for med_a, med_b in combinations(sorted(pivot.columns), 2):
            common = pivot[[med_a, med_b]].dropna()
            if len(common) < 2:
                continue
            block_id = stable_hash([key, med_a, med_b], "RANK:")
            complete = all(bool(k) for k in key[2:6])
            for chem_a, chem_b in combinations(common.index, 2):
                da = float(common.loc[chem_a, med_a]-common.loc[chem_b, med_a])
                db = float(common.loc[chem_a, med_b]-common.loc[chem_b, med_b])
                records.append({
                    "block": block_id, "publication_id": key[0], "target_exact": key[1],
                    "medium_a": med_a, "medium_b": med_b, "chemical_a": chem_a, "chemical_b": chem_b,
                    "difference_a_log2": da, "difference_b_log2": db,
                    "any_inversion": bool(rank_inversion_flags(da, db)),
                    "strict_inversion": bool(rank_inversion_flags(da, db, 1.)),
                    "all_cfu_salt_ionic_ph_reported": complete,
                    "cfu_reported": bool(key[2]),
                })
    pairs = pd.DataFrame(records)
    pairs.to_csv(output / "matched_rank_pairs.csv", index=False)
    if pairs.empty:
        return {"pairs": 0, "inference": "No supported matched rank comparisons"}
    result = {}
    for name, frame in [
        ("all_matching_reported_fields", pairs),
        ("cfu_reported", pairs[pairs.cfu_reported]),
        ("all_four_fields_reported", pairs[pairs.all_cfu_salt_ionic_ph_reported]),
    ]:
        if frame.empty:
            result[name] = {"pairs": 0}
            continue
        pub_idx, pubs = pd.factorize(frame.publication_id)
        chemicals = sorted(set(frame.chemical_a) | set(frame.chemical_b))
        mapping = {v: i for i, v in enumerate(chemicals)}
        a, b = frame.chemical_a.map(mapping).to_numpy(), frame.chemical_b.map(mapping).to_numpy()
        y = frame.strict_inversion.to_numpy()
        rng = np.random.default_rng(20261020)
        draws = []
        for _ in range(repeats):
            chem_w = rng.poisson(1, len(chemicals))
            weights = rng.poisson(1, len(pubs))[pub_idx]*chem_w[a]*chem_w[b]
            if weights.sum():
                draws.append(float(np.average(y, weights=weights)))
        result[name] = {
            "pairs": len(frame), "blocks": frame.block.nunique(), "publications": len(pubs),
            "chemical_entities": len(chemicals), "inversions": int(y.sum()),
            "rate": float(y.mean()), "crossed_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist(),
            "limited_independent_support": len(pubs) < 10,
        }
    result["inference"] = (
        "Descriptive matched contrasts with publication-by-peptide dependence-aware resampling. "
        "Matching missing fields does not establish equivalent protocols. No Fisher test against "
        "a nonexchangeable pooled replicate null is performed."
    )
    result["strict_threshold_log2"] = 1.
    result["numerical_comparison_atol_log2"] = RANK_COMPARISON_ATOL_LOG2
    (output / "matched_rank_summary.json").write_text(json.dumps(result, indent=2))
    rank_threshold_sensitivity(output, repeats=repeats)
    return result


def rank_threshold_sensitivity(output, repeats=5000):
    """Exploratory threshold robustness after inspecting a mass-conversion boundary."""
    source = output/"matched_rank_pairs.csv"
    pairs = pd.read_csv(source)
    thresholds = np.array([1., 1.01, 1.05, 1.1, 1.25, 1.5, 2.])
    a, b = pairs.difference_a_log2.to_numpy(), pairs.difference_b_log2.to_numpy()
    flags = rank_inversion_flags(a[:, None], b[:, None], thresholds[None, :])
    assert np.array_equal(flags[:, 0], pairs.strict_inversion.to_numpy())
    pub_idx, pubs = pd.factorize(pairs.publication_id)
    chemicals = sorted(set(pairs.chemical_a) | set(pairs.chemical_b))
    lookup = {v: i for i,v in enumerate(chemicals)}
    ca, cb = pairs.chemical_a.map(lookup).to_numpy(), pairs.chemical_b.map(lookup).to_numpy()
    rng = np.random.default_rng(20261020)
    draws = []
    for _ in range(repeats):
        chemical_weights = rng.poisson(1, len(chemicals))
        weights = rng.poisson(1, len(pubs))[pub_idx]*chemical_weights[ca]*chemical_weights[cb]
        if weights.sum():
            draws.append(np.average(flags, weights=weights, axis=0))
    intervals = np.quantile(draws, [.025,.975], axis=0)
    rows = [{"threshold_log2": float(t), "pairs": len(pairs),
             "inversions": int(flags[:,i].sum()), "rate": float(flags[:,i].mean()),
             "publications_with_inversions": int(pairs.loc[flags[:,i],"publication_id"].nunique()),
             "ci_low": float(intervals[0,i]), "ci_high": float(intervals[1,i])}
            for i,t in enumerate(thresholds)]
    primary = json.loads((output/"matched_rank_summary.json").read_text())["all_matching_reported_fields"]
    assert rows[0]["inversions"] == primary["inversions"]
    assert np.allclose(intervals[:,0], primary["crossed_bootstrap_ci95"], rtol=0, atol=1e-12)
    pd.DataFrame(rows).to_csv(output/"rank_threshold_sensitivity.csv", index=False)
    result = {"input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "status": "exploratory; thresholds chosen after inspecting a near-boundary converted MIC contrast",
              "primary_threshold_unchanged": 1., "primary_result_reproduced": True,
              "numerical_comparison_atol_log2": RANK_COMPARISON_ATOL_LOG2,
              "scale": "pairwise differences in log2 micromolar MIC, not original dilution-step counts",
              "bootstrap": "publication-by-chemical multiplier bootstrap; common draws across thresholds",
              "seed": 20261020, "valid_bootstrap_draws": len(draws), "results": rows,
              "created_utc": datetime.now(timezone.utc).isoformat()}
    (output/"rank_threshold_sensitivity.json").write_text(json.dumps(result, indent=2))
    return result


def run_statistics(scope):
    output = ROOT / "results/reconstruction_2026-10-20" / scope / "statistics"
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data/curated" / scope / "all_exact_assay_cells.csv",
                       low_memory=False, keep_default_na=False, na_values=[""])
    support = design_support(data)
    (output / "design_support.json").write_text(json.dumps(support, indent=2))
    activities = pd.read_csv(ROOT/"data/curated"/scope/"activities.csv",
                             low_memory=False, keep_default_na=False, na_values=[""])
    curation_sensitivity_membership(data, activities, output)
    rank = matched_rank_changes(data, output)
    if scope == "sample":
        analyze_gramPA(output)
    print(json.dumps({"support": support, "rank": rank}, indent=2), flush=True)
