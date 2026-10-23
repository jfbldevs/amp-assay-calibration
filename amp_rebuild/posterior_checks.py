"""Conditional posterior predictive checks, kept separate from transfer tests."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import ndtr
from cmdstanpy import from_csv

from .curation import ROOT
from .reporting import posterior_results


def bfmi_by_chain(energy):
    """E-BFMI from post-warmup energies in draws-by-chains order."""
    energy = np.asarray(energy, dtype=float)
    if energy.ndim != 2 or energy.shape[0] < 2 or not np.isfinite(energy).all():
        raise ValueError("Finite post-warmup energies are required for every chain.")
    variance = energy.var(axis=0, ddof=1)
    if np.any(variance <= 0):
        raise ValueError("A constant energy chain cannot establish adequate exploration.")
    return np.square(np.diff(energy, axis=0)).mean(axis=0)/variance


def check_sampling_precision(result, result_path, fit=None):
    """Retain tail precision and chain-specific energy checks beyond the initial gate."""
    folder = result_path.parent
    scientific_path = folder/"scientific_parameters.csv"
    all_parameters_path = folder/"posterior_summary.csv"
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [result_path, scientific_path, all_parameters_path]}
    destination = folder/"sampling_precision.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["source_sha256"] == source_hashes:
            return previous
    scientific = pd.read_csv(scientific_path, index_col=0)
    all_parameters = pd.read_csv(all_parameters_path, index_col=0)
    if fit is None:
        fit = from_csv(result["csv_files"])
    energy = fit.method_variables()["energy__"]
    if energy.shape != (result["draws_per_chain"], result["chains"]):
        raise ValueError("Unexpected post-warmup energy array shape.")
    bfmi = bfmi_by_chain(energy)
    finite_rhat = bool(np.isfinite(all_parameters.R_hat).all())
    finite_scientific = bool(np.isfinite(scientific[["ESS_bulk", "ESS_tail", "MCSE", "StdDev"]]).all().all())
    flags = []
    if not finite_rhat or not finite_scientific:
        flags.append("nonfinite_sampling_diagnostic")
    if (scientific.ESS_tail < 400).any():
        flags.append("scientific_tail_ess_below_400")
    if (bfmi < .3).any():
        flags.append("chain_bfmi_below_0_3")
    fraction_rows = scientific[scientific.index.str.startswith("conditional_variance_fraction[")]
    answer = {
        "source_sha256": source_hashes,
        "initial_sampling_gate_passed": result["diagnostics"]["passed"],
        "finite_rhat_all_parameters": finite_rhat,
        "finite_scientific_diagnostics": finite_scientific,
        "min_tail_ess_scientific_parameters": float(scientific.ESS_tail.min()),
        "min_tail_ess_parameter": str(scientific.ESS_tail.idxmin()),
        "e_bfmi_by_chain": bfmi.tolist(),
        "max_mcse_to_sd_scientific_parameters": float((scientific.MCSE/scientific.StdDev).max()),
        "max_mcse_fraction_mean": float(fraction_rows.MCSE.max()),
        "flags": flags,
        "passed": bool(result["diagnostics"]["passed"] and not flags),
        "interpretation": "Additional computational diagnostics, not evidence of model adequacy or identification. Tail ESS covers the 5% and 95% quantiles and is not a direct MCSE for the reported 2.5% and 97.5% limits.",
        "review_thresholds": {"min_scientific_tail_ess": 400, "min_chain_e_bfmi": .3},
        "threshold_status": "Additional review criteria; the original sampler gate is unchanged.",
        "bfmi_formula": "mean(diff(energy)**2) / var(energy, ddof=1), separately within each post-warmup chain",
        "sources": ["https://mc-stan.org/learn-stan/diagnostics-warnings.html",
                    "https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/diagnostics.py"],
    }
    destination.write_text(json.dumps(answer, indent=2, allow_nan=False))
    return answer


def check_fit(result, result_path, draws=200):
    if not result["diagnostics"]["passed"]:
        raise ValueError("Do not interpret a posterior that failed its sampling gate.")
    folder = result_path.parent
    source = folder/"stan_data.json"
    if not source.exists():
        raise FileNotFoundError(source)
    signature = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = folder/"posterior_predictive_checks.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["stan_data_sha256"] == signature and previous["posterior_draws_checked"] == draws:
            check_sampling_precision(result, result_path)
            return previous
    payload = json.loads(source.read_text())
    fit = from_csv(result["csv_files"])
    check_sampling_precision(result, result_path, fit)
    rng = np.random.default_rng(20261020)
    total = result["draws_per_chain"]*result["chains"]
    selected = np.sort(rng.choice(total, size=min(draws, total), replace=False))
    alpha = fit.stan_variable("alpha")[selected]
    beta = fit.stan_variable("beta")[selected]
    sigma = fit.stan_variable("sigma")[selected]
    tau = fit.stan_variable("tau")[selected]
    effects = []
    for name, key in [("z_chem", "chem"), ("z_pub", "pub"), ("z_target", "target_id"),
                      ("z_chem_target", "chem_target"), ("z_chem_medium", "chem_medium")]:
        effects.append((np.column_stack([np.zeros(len(selected)), fit.stan_variable(name)[selected]]),
                        np.asarray(payload[key])-1))
    X = np.asarray(payload["X"])
    lower, upper = np.asarray(payload["lower_bound"]), np.asarray(payload["upper_bound"])
    kinds = np.asarray(payload["kind"])
    singleton = np.asarray(payload["singleton"])
    exact = kinds == 1
    observed = lower[exact]
    summarize = lambda y: {
        "mean": float(np.mean(y)), "sd": float(np.std(y)), "q05": float(np.quantile(y, .05)),
        "median": float(np.median(y)), "q95": float(np.quantile(y, .95)),
    }
    replicated = []
    probabilities = {name: [] for name in ["left", "right", "interval"]}
    fitted_mean = np.zeros(payload["N"])
    for k in range(len(selected)):
        mu = alpha[k]+X@beta[k]
        for j, (z, index) in enumerate(effects):
            mu += tau[k, j]*z[k, index]
        observation_sd = np.sqrt(sigma[k]**2 + singleton@(tau[k]**2))
        fitted_mean += mu/len(selected)
        replicated.append(summarize(mu[exact]+rng.normal(0, observation_sd[exact])))
        for name, code in [("left", 2), ("right", 3), ("interval", 4)]:
            mask = kinds == code
            if not mask.any():
                continue
            lo = (lower[mask]-mu[mask])/observation_sd[mask]
            hi = (upper[mask]-mu[mask])/observation_sd[mask]
            if name == "left":
                prob = ndtr(hi)
            elif name == "right":
                prob = ndtr(-lo)
            else:
                prob = np.where(lo > 0, ndtr(-lo)-ndtr(-hi), ndtr(hi)-ndtr(lo))
            probabilities[name].append(float(np.mean(prob)))
    observed_stats = summarize(observed)
    checks = {}
    for name, value in observed_stats.items():
        simulated = np.array([row[name] for row in replicated])
        tail = float(np.mean(simulated >= value))
        checks[name] = {"observed": value, "replicated_interval95": np.quantile(simulated, [.025,.975]).tolist(),
                        "posterior_predictive_upper_tail": tail, "extreme_tail_flag": tail < .01 or tail > .99}
    source_rows = pd.read_csv(folder/"analysis_rows.csv.gz", keep_default_na=False, na_values=[""], low_memory=False)
    source_rows["posterior_conditional_mean"] = fitted_mean
    source_rows["conditional_residual"] = source_rows.log2_lower-fitted_mean
    residuals = source_rows[source_rows.parse_kind.eq("exact")]
    group_tables = []
    for grouping in ["medium", "publication_id", "target_exact"]:
        table = residuals.groupby(grouping).conditional_residual.agg(["size", "mean", "std"])
        table = table[table["size"] >= 10].reset_index().rename(columns={grouping: "level"})
        table["grouping"] = grouping
        group_tables.append(table)
    pd.concat(group_tables).to_csv(folder/"conditional_residual_groups.csv", index=False)
    answer = {
        "stan_data_sha256": signature, "posterior_draws_checked": len(selected),
        "exact_observations_checked": int(exact.sum()), "exact_subset_checks": checks,
        "any_extreme_tail_flag": any(c["extreme_tail_flag"] for c in checks.values()),
        "mean_probability_of_recorded_censoring_event": {
            k: {"median": float(np.median(v)), "interval95": np.quantile(v, [.025,.975]).tolist()}
            for k, v in probabilities.items() if v},
        "interpretation": "Conditional in-sample model checks, not external predictive accuracy or calibrated hypothesis tests.",
        "limitations": [
            "Exact-record subset selection can depend on the unrecorded assay range.",
            "Replications are conditional on fitted random effects for observed entities.",
            "Posterior predictive tail probabilities are diagnostics, not frequentist p-values.",
            "Passing these summaries does not establish full model adequacy; group residuals are retained.",
        ],
    }
    destination.write_text(json.dumps(answer, indent=2))
    return answer


def check_scope(scope):
    reports = []
    for (subset, prior), (result, path) in posterior_results(scope).items():
        if result["diagnostics"]["passed"]:
            checked = check_fit(result, path)
            row = {"subset": subset, "prior": prior, "path": str(path.parent.relative_to(ROOT)),
                   "extreme_tail_flag": checked["any_extreme_tail_flag"],
                   "sampling_precision_passed": json.loads((path.parent/"sampling_precision.json").read_text())["passed"]}
            reports.append(row)
            print(json.dumps(row), flush=True)
    (ROOT/"results/reconstruction_2026-10-20"/scope/"posterior_check_summary.json").write_text(
        json.dumps(reports, indent=2))
    return reports
