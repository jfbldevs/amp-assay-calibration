"""Observation-window sensitivities of the five CLOSED censored posteriors.

No sampler is run and no original posterior file is changed. The share of a
predictive discrepancy caused by selection is not identifiable when the assay
window of exact observations is unrecorded. Report conditional sensitivities,
source-unit compatibility and residual-shape perturbations, not a causal split.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from threadpoolctl import threadpool_limits

from .editorial_common import ROOT, BASE, OUT, sha, utc, read_json, write_json, lock_json

DEST = OUT/"posterior_tail_gaps"
SEED, DRAWS = 20261020, 200
EFFECTS = [("z_chem", "chem"), ("z_pub", "pub"), ("z_target", "target_id"),
           ("z_chem_target", "chem_target"), ("z_chem_medium", "chem_medium")]
CONFIG = {
    "draws": DRAWS, "seed": SEED,
    "common_upper_log2_uM_quantiles_of_recorded_right_limits": [.5, .75, .9],
    "source_unit_blocks": ["publication_id", "target_exact", "unit_raw"],
    "local_limit_scenarios": ["unique_right_limit_compatible", "maximum_right_limit_compatible"],
    "support_envelope_margins_log2": [0., 1., 2.],
    "student_t_residual_degrees_freedom": [3, 5, 10],
    "residual_shape_comparison": "Variance-matched Student-t residual perturbation at the Gaussian posterior parameters; integrated singleton effects remain Gaussian. Not a refitted model or Bayes factor.",
    "share_identified": False,
    "estimand": "Change in exact-subset q95 under declared observation-window assumptions, paired by posterior draw and common uniforms.",
    "prohibited_inference": "No unique percentage of the observed gap can be assigned to selection versus Gaussian misspecification from these data.",
    "selection_formula": "f_exact(y|x,theta) = f(y|x,theta) pi_exact(y|x) / integral f(t|x,theta) pi_exact(t|x) dt",
    "truncated_gaussian": "For L < Y <= U: F_selected(y) = [Phi((y-mu)/s)-Phi((L-mu)/s)]/[Phi((U-mu)/s)-Phi((L-mu)/s)].",
    "support_envelope_warning": "The smallest compatible block limit uses observed outcomes. These envelopes are sensitivity bounds, not observed assay limits, fitted selection probabilities or held-out validation.",
}


def extract_selected(result_path):
    result = read_json(result_path)
    assert result["diagnostics"]["passed"]
    label = result_path.parent.parent.name
    folder = DEST/label
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder/"extraction.json"
    cache_path = folder/"selected_draws.npz"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        assert receipt["result_sha256"] == sha(result_path)
        assert receipt["cache_sha256"] == sha(cache_path)
        return dict(np.load(cache_path)), receipt
    total = result["draws_per_chain"]*result["chains"]
    rng = np.random.default_rng(SEED)
    selected = np.sort(rng.choice(total, size=DRAWS, replace=False))
    selected_set = set(selected)
    matrices, chain_receipts, header = [], [], None
    for chain, source_string in enumerate(result["csv_files"]):
        source = Path(source_string)
        before = source.stat()
        digest = hashlib.sha256()
        row, closed, warmup = 0, False, None
        seen_header = False
        with source.open("rb") as handle:
            for line in handle:
                digest.update(line)
                if line.startswith(b"#"):
                    closed |= b"Elapsed Time:" in line
                    if b"save_warmup =" in line:
                        warmup = line.split(b"=")[1].strip().split()[0] in (b"1", b"true")
                    continue
                if not line.strip():
                    continue
                if not seen_header:
                    current = line.decode().strip().split(",")
                    if header is None:
                        header = current
                    assert current == header
                    seen_header = True
                    continue
                index = chain*result["draws_per_chain"]+row
                if index in selected_set:
                    values = np.fromstring(line.decode(), sep=",")
                    assert len(values) == len(header) and np.isfinite(values).all()
                    matrices.append(values)
                row += 1
        after = source.stat()
        assert before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        assert closed and warmup is False and row == result["draws_per_chain"]
        chain_receipts.append({"source": str(source.relative_to(ROOT)), "sha256": digest.hexdigest(),
                               "rows": row, "closed_footer": closed, "bytes": after.st_size})
    matrix = np.array(matrices)
    assert matrix.shape == (DRAWS, len(header))
    columns = {name:i for i,name in enumerate(header)}
    out = {"selected_indices": selected}
    for name in ["alpha", "sigma", "beta", "tau", *[n for n,k in EFFECTS]]:
        if name in columns:
            out[name] = matrix[:, columns[name]]
        else:
            positions = [i for i,h in enumerate(header) if h.startswith(name+".")]
            positions.sort(key=lambda i: int(header[i].split(".")[-1]))
            out[name] = matrix[:, positions]
    np.savez_compressed(cache_path, **out)
    receipt = {"created_utc": utc(), "result_sha256": sha(result_path), "source_chains": chain_receipts,
               "draws": DRAWS, "selected_indices": selected.tolist(), "cache_sha256": sha(cache_path),
               "ordering": "CmdStanPy concatenates complete chains in file order (Fortran flattening of draws x chains x parameters)."}
    write_json(receipt_path, receipt)
    return out, receipt


def source_window_scenarios(full_rows):
    """Constraints in ORIGINAL assay units, transformed separately for each row."""
    assert full_rows.record_id.is_unique
    factor = full_rows.conversion_factor.to_numpy(float)
    assert np.isfinite(factor).all() and (factor > 0).all()
    log_factor = np.log2(factor)
    native_value = full_rows.log2_lower.to_numpy(float)-log_factor
    right = full_rows.parse_kind.eq("right").to_numpy()
    exact = full_rows.parse_kind.eq("exact").to_numpy()
    limits = {"unique_right_limit_compatible": np.full(len(full_rows), np.inf),
              "maximum_right_limit_compatible": np.full(len(full_rows), np.inf)}
    envelope = np.full(len(full_rows), np.inf)
    ledger = []
    for block, idx in full_rows.groupby(CONFIG["source_unit_blocks"], dropna=False).groups.items():
        idx = np.asarray(idx, int)
        e, r = idx[exact[idx]], idx[right[idx]]
        if not len(e):
            continue
        exact_max = float(native_value[e].max())
        right_max = float(native_value[r].max()) if len(r) else None
        right_min = float(native_value[r].min()) if len(r) else None
        unique = bool(len(r) and np.max(native_value[r])-np.min(native_value[r]) < 1e-6)
        compatible = bool(len(r) and exact_max <= right_max+1e-9)
        if compatible:
            limits["maximum_right_limit_compatible"][e] = right_max+log_factor[e]
            if unique:
                limits["unique_right_limit_compatible"][e] = right_max+log_factor[e]
        envelope_value = max(exact_max, right_max) if right_max is not None else exact_max
        envelope[e] = envelope_value+log_factor[e]
        ledger.append({"publication_id": block[0], "target_exact": block[1], "unit_raw": block[2],
            "exact_records": len(e), "right_records": len(r), "exact_max_native_log2": exact_max,
            "right_min_native_log2": right_min, "right_max_native_log2": right_max,
            "unique_right_limit": unique, "maximum_right_limit_compatible": compatible,
            "exact_records_above_maximum_right_limit": int(np.sum(native_value[e]>right_max+1e-9)) if right_max is not None else 0,
            "support_envelope_native_log2": envelope_value})
    for margin in CONFIG["support_envelope_margins_log2"]:
        limits[f"compatible_support_envelope_plus_{int(margin)}"] = envelope+margin
    for q in CONFIG["common_upper_log2_uM_quantiles_of_recorded_right_limits"]:
        value = float(np.quantile(full_rows.loc[right, "log2_lower"], q))
        limits[f"counterfactual_common_right_q{int(q*100)}"] = np.full(len(full_rows), value)
    return limits, pd.DataFrame(ledger)


def truncated_gaussian_from_uniform(mu, sd, uniform, upper):
    mass = ndtr((upper-mu)/sd)
    # Bounds used here are upper-only. Very small nonzero tails are retained;
    # clip solely at machine probability limits, with any clipping audited.
    probability = uniform*mass
    clipped = np.clip(probability, np.finfo(float).tiny, np.nextafter(1., 0.))
    return mu+sd*ndtri(clipped), mass, int(np.sum(probability != clipped))


def summaries(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values)),
            "q05": float(np.quantile(values, .05)), "median": float(np.median(values)),
            "q95": float(np.quantile(values, .95))}


def analyze_fit(result_path, full_rows, limits):
    label = result_path.parent.parent.name
    folder = DEST/label
    if (folder/"completion.json").exists():
        return pd.read_csv(folder/"summary.csv")
    draws, extraction = extract_selected(result_path)
    result = read_json(result_path)
    payload = read_json(result_path.parent/"stan_data.json")
    source_rows = pd.read_csv(result_path.parent/"analysis_rows.csv.gz", keep_default_na=False, na_values=[""], low_memory=False)
    assert len(source_rows) == payload["N"]
    kinds = np.asarray(payload["kind"])
    exact = kinds == 1
    observed = np.asarray(payload["lower_bound"])[exact]
    observed_stats = summaries(observed)
    assert np.allclose(source_rows.loc[exact, "log2_lower"], observed, rtol=0, atol=1e-12)
    full_indices = pd.Index(full_rows.record_id).get_indexer(source_rows.loc[exact, "record_id"])
    assert (full_indices >= 0).all()
    upper_scenarios = {key: values[full_indices] for key,values in limits.items()}
    scenario_metadata = []
    for scenario, upper in upper_scenarios.items():
        scenario_metadata.append({"scenario": scenario, "finite_limit_exact_records": int(np.isfinite(upper).sum()),
            "exact_records_contradicted": int(np.sum(observed > upper+1e-9)),
            "finite_limit_median_log2_uM": float(np.median(upper[np.isfinite(upper)])) if np.isfinite(upper).any() else None,
            "status": "Counterfactual common molar threshold; may contradict observed exact values" if scenario.startswith("counterfactual") else
                      "Data-derived compatible support envelope; not an observed assay window" if "envelope" in scenario else
                      "Assumes a shared original-unit assay limit only in blocks compatible with recorded right bounds and exact values"})
    X = np.asarray(payload["X"])[exact]
    singleton = np.asarray(payload["singleton"])[exact]
    effect_indices = {name: np.asarray(payload[key])[exact]-1 for name,key in EFFECTS}
    effects = {name: np.c_[np.zeros(DRAWS), draws[name]] for name,key in EFFECTS}
    rng = np.random.default_rng(SEED)
    selected = np.sort(rng.choice(result["draws_per_chain"]*result["chains"], DRAWS, replace=False))
    assert np.array_equal(selected, draws["selected_indices"])
    shape_rng = np.random.default_rng(SEED+1)
    rows = []
    with threadpool_limits(limits=2):
        for i in range(DRAWS):
            mu = draws["alpha"][i]+X@draws["beta"][i]
            for j,(name,key) in enumerate(EFFECTS):
                mu += draws["tau"][i,j]*effects[name][i,effect_indices[name]]
            singleton_variance = singleton@(draws["tau"][i]**2)
            sd = np.sqrt(draws["sigma"][i]**2+singleton_variance)
            # This is the exact original RNG sequence on the exact subset.
            gaussian_noise = rng.normal(0, sd)
            replicated = mu+gaussian_noise
            baseline = summaries(replicated)
            rows.append({"draw": i, "scenario": "gaussian_unselected", "mean_excluded_mass_on_exact_covariates": 0., "numerical_probability_clips": 0, **baseline})
            uniform = ndtr(gaussian_noise/sd)
            for scenario, upper in upper_scenarios.items():
                simulated, mass, clipped = truncated_gaussian_from_uniform(mu, sd, uniform, upper)
                assert np.all(simulated <= upper+1e-8)
                rows.append({"draw": i, "scenario": scenario, "mean_excluded_mass_on_exact_covariates": float(np.mean(1-mass)),
                             "numerical_probability_clips": clipped, **summaries(simulated)})
            for nu in CONFIG["student_t_residual_degrees_freedom"]:
                residual = shape_rng.standard_t(nu, len(mu))*draws["sigma"][i]*np.sqrt((nu-2)/nu)
                random_singleton = shape_rng.normal(0, np.sqrt(singleton_variance))
                rows.append({"draw": i, "scenario": f"student_t_{nu}_variance_matched_unselected",
                    "mean_excluded_mass_on_exact_covariates": 0., "numerical_probability_clips": 0,
                    **summaries(mu+random_singleton+residual)})
    frame = pd.DataFrame(rows)
    baseline = frame[frame.scenario.eq("gaussian_unselected")].set_index("draw")
    original = read_json(result_path.parent/"posterior_predictive_checks.json")
    reconciliation = {}
    for metric in observed_stats:
        actual = np.quantile(baseline[metric], [.025,.975])
        expected = original["exact_subset_checks"][metric]["replicated_interval95"]
        difference = float(np.max(np.abs(actual-expected)))
        assert difference < 1e-8, (label, metric, actual, expected)
        reconciliation[metric] = {"maximum_absolute_difference": difference, "passed": True}
    table = []
    meta = {r["scenario"]:r for r in scenario_metadata}
    baseline_gap = baseline.q95.to_numpy()-observed_stats["q95"]
    for scenario, block in frame.groupby("scenario"):
        block = block.sort_values("draw")
        difference = baseline.q95.to_numpy()-block.q95.to_numpy()
        positive_gap = bool(np.all(baseline_gap > 0.05))
        fraction = difference/baseline_gap if positive_gap else None
        row = {"fit": label, "n_records": result["n_records"], "exact_records": int(exact.sum()),
            "scenario": scenario, **meta.get(scenario, {}),
            "observed_q05": observed_stats["q05"], "observed_q95": observed_stats["q95"],
            "q95_gap_median_log2": float(np.median(block.q95-observed_stats["q95"])),
            "q95_reduction_from_unselected_median_log2": float(np.median(difference)),
            "q95_reduction_ci_low": float(np.quantile(difference, .025)), "q95_reduction_ci_high": float(np.quantile(difference, .975)),
            "positive_gap_fraction_defined": positive_gap,
            "fraction_positive_gap_removed_median": float(np.median(fraction)) if positive_gap else np.nan,
            "fraction_positive_gap_removed_ci_low": float(np.quantile(fraction, .025)) if positive_gap else np.nan,
            "fraction_positive_gap_removed_ci_high": float(np.quantile(fraction, .975)) if positive_gap else np.nan,
            "mean_excluded_mass_on_exact_covariates": float(block.mean_excluded_mass_on_exact_covariates.mean()),
            "numerical_probability_clips": int(block.numerical_probability_clips.sum())}
        for metric in observed_stats:
            row[metric+"_median"] = float(block[metric].median())
            row[metric+"_ci_low"], row[metric+"_ci_high"] = np.quantile(block[metric], [.025,.975])
        table.append(row)
    pd.DataFrame(table).to_csv(folder/"summary.csv", index=False)
    frame.to_csv(folder/"draw_level_sensitivities.csv.gz", index=False)
    write_json(folder/"completion.json", {"created_utc": utc(), "passed": True,
        "original_predictive_checks_reconciliation": reconciliation, "observed": observed_stats,
        "source_sha256": {str(p.relative_to(ROOT)):sha(p) for p in [result_path, result_path.parent/"stan_data.json", result_path.parent/"analysis_rows.csv.gz", result_path.parent/"posterior_predictive_checks.json"]},
        "summary_sha256": sha(folder/"summary.csv"), "draws_sha256": sha(folder/"draw_level_sensitivities.csv.gz"),
        "causal_fraction_identified": False,
        "interpretation": "Ratios are unbounded, assumption-dependent sensitivity summaries; values above one indicate overshooting the observed q95. Undefined for the exact-only fit because its unselected q95 gap is negative. Student-t perturbations do not constitute fitted posterior alternatives."})
    print(f"Tail sensitivities complete: {label}", flush=True)
    return pd.DataFrame(table)


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    lock_json(DEST/"configuration.json", CONFIG)
    paths = sorted((BASE/"censored").glob("*/*/result.json"))
    assert len(paths) == 5
    primary = next(p for p in paths if p.parent.parent.name == "all_prior1.5")
    source = primary.parent/"analysis_rows.csv.gz"
    full_rows = pd.read_csv(source, keep_default_na=False, na_values=[""], low_memory=False)
    limits, ledger = source_window_scenarios(full_rows)
    ledger.to_csv(DEST/"source_unit_window_compatibility.csv.gz", index=False)
    window_table = pd.DataFrame({"record_id": full_rows.record_id, **limits})
    window_table.to_csv(DEST/"assumed_upper_limits_by_record.csv.gz", index=False)
    tables = [analyze_fit(p, full_rows, limits) for p in paths]
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(DEST/"tail_gap_summary.csv", index=False)
    write_json(DEST/"completion.json", {"created_utc": utc(), "passed": True, "fits": len(paths),
        "posterior_draws_per_fit": DRAWS, "scenario_rows": len(combined),
        "source_rows_sha256": sha(source), "algorithm_sha256": sha(Path(__file__)),
        "summary_sha256": sha(DEST/"tail_gap_summary.csv"), "configuration_sha256": sha(DEST/"configuration.json"),
        "window_blocks": len(ledger), "blocks_with_right_records": int((ledger.right_records>0).sum()),
        "blocks_with_exact_above_maximum_right_limit": int((ledger.exact_records_above_maximum_right_limit>0).sum()),
        "exact_records_in_contradicted_blocks": int(ledger.loc[ledger.exact_records_above_maximum_right_limit>0, "exact_records"].sum()),
        "causal_fraction_identified": False,
        "closure": "Observation-window selection can be quantified only conditionally on unobserved-window assumptions; the original model-adequacy flags are retained. Mechanistic and universal variance-partition claims remain unsupported."})
