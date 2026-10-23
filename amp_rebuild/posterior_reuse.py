"""Explicitly certify reuse when an amended source cohort leaves a fit unchanged."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from .curation import ROOT
from .censored import active_stan_path, preparation_fingerprint, prepare_stan_data, select_analysis_rows


def reuse_identical_prepared_fit(previous_result_path):
    previous_result_path = Path(previous_result_path)
    old = json.loads(previous_result_path.read_text())
    scope = old["scope"]
    source = ROOT/"data/curated"/scope/"activities.csv"
    input_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    stan_hash = hashlib.sha256(active_stan_path(scope).read_bytes()).hexdigest()
    preparation_hash = preparation_fingerprint()
    if stan_hash != old["stan_source_sha256"] or preparation_hash != old["preparation_source_sha256"]:
        raise ValueError("A changed model or preparation algorithm cannot use this equivalence certificate.")
    original = pd.read_csv(source, keep_default_na=False, na_values=[""], low_memory=False)
    selected = select_analysis_rows(original, old["subset"])
    payload, design = prepare_stan_data(selected, old["prior_scale_log2"])
    previous_folder = previous_result_path.parent
    if payload != json.loads((previous_folder/"stan_data.json").read_text()):
        raise ValueError("Amended cohort changes prepared model data; refitting is required.")
    if design != json.loads((previous_folder/"design.json").read_text()):
        raise ValueError("Amended cohort changes parameter labels or design semantics.")
    old_rows = pd.read_csv(previous_folder/"analysis_rows.csv.gz", usecols=["record_id"])
    if old_rows.record_id.tolist() != selected.record_id.tolist():
        raise ValueError("Analysis record membership or order changed.")
    signature = (
        f"{input_hash}|{stan_hash}|{preparation_hash}|{old['subset']}|{old['prior_scale_log2']}|"
        f"{old['warmup_per_chain']}|{old['draws_per_chain']}|{old['adapt_delta']}|{old['max_treedepth']}"
    )
    folder = previous_folder.parent/hashlib.sha256(signature.encode()).hexdigest()[:12]
    if folder == previous_folder:
        return previous_result_path
    folder.mkdir(parents=True,exist_ok=True)
    for name in ["stan_data.json","design.json","analysis_rows.csv.gz","posterior_summary.csv",
                 "scientific_parameters.csv","diagnostics.txt"]:
        shutil.copy2(previous_folder/name,folder/name)
    certificate = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "previous_result": str(previous_result_path.relative_to(ROOT)),
        "previous_result_sha256": hashlib.sha256(previous_result_path.read_bytes()).hexdigest(),
        "execution_input_sha256": old["input_sha256"],
        "current_input_sha256": input_hash,
        "exact_payload_equal": True, "exact_design_equal": True, "ordered_record_ids_equal": True,
        "prepared_payload_sha256": hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        "interpretation": "Revalidation of the original draws against an amended source cohort. The actual fit inputs and parameter labels are identical. Original diagnostics, execution date, chain paths and result file are preserved; no new sampling is claimed.",
    }
    (folder/"posterior_reuse.json").write_text(json.dumps(certificate,indent=2))
    result = {**old, "input_sha256":input_hash, "reuse_provenance":certificate}
    result_path=folder/"result.json"
    result_path.write_text(json.dumps(result,indent=2))
    return result_path
