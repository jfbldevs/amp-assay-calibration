"""Fit and diagnose the censored multilevel model; never relabel publication as lab."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, set_cmdstan_path

from .curation import ROOT
ACTIVE_STAN_PATH = ROOT / "models/context_censored_collapsed.stan"


def active_stan_path(scope):
    # Identical model and prepared data. The expanded categorical matrix is
    # multiplied in CSR form to avoid repeated arithmetic on its many zeros.
    return (ROOT/"models/context_censored_sparse.stan"
            if scope == "expanded" else ACTIVE_STAN_PATH)


def select_analysis_rows(original, subset):
    data = original[original.eligible_censored_analysis].copy()
    if subset == "multi_publication":
        counts = data.groupby("chemical_id").publication_id.nunique()
        data = data[data.chemical_id.isin(counts[counts > 1].index)]
    elif subset == "exact":
        data = data[data.parse_kind == "exact"]
    elif subset == "no_reported_ranges":
        data = data[data.parse_kind != "interval"]
    elif subset != "all":
        raise ValueError(subset)
    return data


def preparation_fingerprint():
    return hashlib.sha256((inspect.getsource(select_analysis_rows)+
                           inspect.getsource(prepare_stan_data)).encode()).hexdigest()


def prepare_stan_data(frame, prior_scale):
    data = frame.copy()
    data["chem_target"] = data.chemical_id + "|" + data.target_exact
    data["chem_medium"] = data.chemical_id + "|" + data.medium
    blocks = {"chem": "chemical_id", "pub": "publication_id", "target_id": "target_exact",
              "chem_target": "chem_target", "chem_medium": "chem_medium"}
    J_names = {"chem": "J_chem", "pub": "J_pub", "target_id": "J_target",
               "chem_target": "J_chem_target", "chem_medium": "J_chem_medium"}
    payload = {"N": len(data), "prior_scale": prior_scale}
    metadata = {"levels": {}, "sampled_levels": {}, "marginalized_singleton_levels": {},
                "fixed_effects": [], "numeric_scaling": {}}
    singleton = []
    for index, name in blocks.items():
        codes, labels = pd.factorize(data[name], sort=True)
        frequencies = np.bincount(codes, minlength=len(labels))
        retained = np.flatnonzero(frequencies > 1)
        lookup = np.ones(len(labels), dtype=int)
        lookup[retained] = np.arange(len(retained))+2
        payload[index] = lookup[codes].tolist()
        payload[J_names[index]] = len(retained)
        singleton.append((frequencies[codes] == 1).astype(int))
        metadata["levels"][index] = list(labels)
        metadata["sampled_levels"][index] = list(labels[retained])
        metadata["marginalized_singleton_levels"][index] = int((frequencies == 1).sum())
    payload["singleton"] = np.column_stack(singleton).tolist()
    metadata["singleton_method"] = (
        "Exact Gaussian integration: a unique level contributes tau_j^2 to its observation variance. "
        "The likelihood and scale priors match the uncollapsed model.")
    # Pool infrequent fixed categories using counts only, not outcomes.
    columns = {}
    for name in ["medium", "supplement_category"]:
        values = data[name].fillna("Unreported")
        counts = values.value_counts()
        kept = counts[counts >= 20].index
        collapsed = values.where(values.isin(kept), "Other_sparse")
        reference = "MHB" if name == "medium" and "MHB" in kept else collapsed.value_counts().idxmax()
        for level in sorted(set(collapsed) - {reference}):
            columns[f"{name}={level} (reference={reference})"] = (collapsed == level).to_numpy().astype(float)
    for name in ["log10_cfu", "ph", "nacl_reported_concentration"]:
        value = pd.to_numeric(data[name], errors="coerce")
        missing = value.isna()
        if missing.any() and (~missing).any():
            columns[f"{name}_unreported"] = missing.to_numpy().astype(float)
        if (~missing).sum() >= 10 and value.std() > 1e-9:
            mean, sd = float(value.mean()), float(value.std())
            columns[f"{name}_standardized"] = ((value-mean)/sd).fillna(0).to_numpy()
            metadata["numeric_scaling"][name] = {"mean": mean, "sd": sd}
    X = np.column_stack(list(columns.values()))
    payload.update({
        "K": X.shape[1], "X": X.tolist(),
        "kind": data.parse_kind.map({"exact": 1, "left": 2, "right": 3, "interval": 4}).tolist(),
        "lower_bound": data.log2_lower.replace([-np.inf, np.inf], 0).fillna(0).tolist(),
        "upper_bound": data.log2_upper.replace([-np.inf, np.inf], 0).fillna(0).tolist(),
    })
    for name in ["exact", "left", "right", "interval"]:
        index = np.flatnonzero(data.parse_kind.eq(name).to_numpy())+1
        payload["N_"+name] = len(index)
        payload[name+"_index"] = index.tolist()
    metadata["fixed_effects"] = list(columns)
    return payload, metadata


def fit_censored(scope="sample", subset="all", prior_scale=1.5, warmup=750, draws=750,
                 adapt_delta=.95, max_treedepth=12):
    stan_paths = sorted((ROOT / ".tools/cmdstan").glob("cmdstan-*"))
    if not stan_paths:
        raise FileNotFoundError("CmdStan is not installed in .tools/cmdstan")
    set_cmdstan_path(str(stan_paths[-1]))
    source = ROOT / "data/curated" / scope / "activities.csv"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    stan_source = active_stan_path(scope)
    stan_hash = hashlib.sha256(stan_source.read_bytes()).hexdigest()
    preparation_hash = preparation_fingerprint()
    run_hash = hashlib.sha256(
        f"{source_hash}|{stan_hash}|{preparation_hash}|{subset}|{prior_scale}|{warmup}|{draws}|{adapt_delta}|{max_treedepth}".encode()).hexdigest()[:12]
    output = (ROOT / "results/reconstruction_2026-10-20" / scope / "censored" /
              f"{subset}_prior{prior_scale:g}" / run_hash)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["diagnostics"]["passed"]:
            print(f"Verified posterior already cached: {result_path}", flush=True)
            return previous
        print("Cached posterior fails convergence gates; increase sampling settings.", flush=True)
        return previous
    original = pd.read_csv(source, low_memory=False, keep_default_na=False, na_values=[""])
    data = select_analysis_rows(original, subset)
    payload, metadata = prepare_stan_data(data, prior_scale)
    prepared_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    # A compatible pre-fingerprint fit can be reused only after exact payload
    # comparison; neither its original metadata nor its posterior is relabeled.
    for previous_path in output.parent.glob("*/result.json"):
        old = json.loads(previous_path.read_text())
        matching = {
            "input_sha256": source_hash, "stan_source_sha256": stan_hash,
            "subset": subset, "prior_scale_log2": prior_scale,
            "warmup_per_chain": warmup, "draws_per_chain": draws,
            "adapt_delta": adapt_delta, "max_treedepth": max_treedepth,
        }
        if any(old.get(k) != v for k,v in matching.items()):
            continue
        original_payload = json.loads((previous_path.parent/"stan_data.json").read_text())
        if original_payload == payload:
            print(f"Equivalent prepared input already has a posterior: {previous_path}", flush=True)
            return old
    data.to_csv(output / "analysis_rows.csv.gz", index=False)
    (output / "stan_data.json").write_text(json.dumps(payload, allow_nan=False))
    (output / "design.json").write_text(json.dumps(metadata, indent=2))
    model = CmdStanModel(stan_file=str(stan_source))
    fit = model.sample(
        data=payload, chains=4, parallel_chains=4, threads_per_chain=1,
        iter_warmup=warmup, iter_sampling=draws, seed=20261020,
        adapt_delta=adapt_delta, max_treedepth=max_treedepth, output_dir=str(output),
        show_progress=False, show_console=True, refresh=100,
    )
    summary = fit.summary()
    summary.to_csv(output / "posterior_summary.csv")
    diagnostic = fit.diagnose()
    (output / "diagnostics.txt").write_text(diagnostic)
    selected = [x for x in summary.index
                if x in {"alpha", "sigma"} or x.startswith(("tau[", "beta[", "conditional_variance_fraction["))]
    summary.loc[selected].to_csv(output / "scientific_parameters.csv")
    method = fit.method_variables()
    divergences = int(np.asarray(method["divergent__"]).sum())
    treedepth = int((np.asarray(method["treedepth__"]) >= max_treedepth).sum())
    quality = {
        "divergences": divergences, "max_treedepth_hits": treedepth,
        "max_rhat_all_parameters": float(summary.R_hat.max()),
        "min_bulk_ess_scientific_parameters": float(summary.loc[selected, "ESS_bulk"].min()),
        "max_rhat_scientific_parameters": float(summary.loc[selected, "R_hat"].max()),
    }
    quality["passed"] = bool(divergences == 0 and treedepth == 0 and
                             quality["max_rhat_all_parameters"] < 1.01 and
                             quality["min_bulk_ess_scientific_parameters"] >= 400)
    # Parameters may remain weakly identified even if computational diagnostics pass.
    fractions = fit.stan_variable("conditional_variance_fraction")
    names = ["chemical_entity", "publication", "exact_target", "chemical_by_target",
             "chemical_by_medium", "residual"]
    components = {}
    for i, name in enumerate(names):
        components[name] = {"median": float(np.median(fractions[:, i])),
                            "ci95": np.quantile(fractions[:, i], [.025, .975]).tolist()}
    result = {
        "scope": scope, "subset": subset, "n_records": len(data),
        "kind_counts": data.parse_kind.value_counts().to_dict(),
        "input_sha256": source_hash, "prior_scale_log2": prior_scale,
        "stan_source_sha256": stan_hash,
        "stan_source_path": str(stan_source.relative_to(ROOT)),
        "preparation_source_sha256": preparation_hash, "prepared_data_sha256": prepared_hash,
        "warmup_per_chain": warmup, "draws_per_chain": draws, "chains": 4,
        "adapt_delta": adapt_delta, "max_treedepth": max_treedepth,
        "reported_range_assumption": "Reported numeric ranges are treated as bounds on one latent assay value; no_reported_ranges is the sensitivity to this assumption.",
        "diagnostics": quality, "conditional_variance_fractions": components,
        "interpretation": "Model-dependent observational variance components, not causal laboratory contributions or universal ML ceilings",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "csv_files": fit.runset.csv_files,
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result
