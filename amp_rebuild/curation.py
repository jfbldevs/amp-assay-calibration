"""Source-preserving assay curation; no assumed molecular weights or protocols."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.warning")
AA = set("ACDEFGHIKLMNPQRSTVWY")
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ROOT = Path(__file__).resolve().parents[1]


def text(value):
    if isinstance(value, dict):
        value = value.get("name", "")
    return "" if value is None else str(value).strip()


def stable_hash(value, prefix=""):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return prefix + hashlib.sha256(payload.encode()).hexdigest()[:24]


@dataclass
class Concentration:
    lower: float = math.nan
    upper: float = math.nan
    point: float = math.nan
    kind: str = "unparsed"
    status: str = "manual_review"
    reported_error: float = math.nan


def parse_concentration(raw):
    """Strictly parse values, inequalities, ranges and reported mean +/- error.

    Ranges are not turned into exact measurements. A reported SD/SE is retained
    separately and is never interpreted as an interval containing the true MIC.
    """
    s = unicodedata.normalize("NFKC", text(raw)).replace("−", "-").replace("–", "-")
    s = re.sub(r"\s+", "", s).replace("≥", ">=").replace("≤", "<=")
    if re.fullmatch(r"\d+,\d+(?:[eE][+-]?\d+)?", s):
        s = s.replace(",", ".")
    if not s:
        return Concentration(status="missing")
    match = re.fullmatch(rf"({NUMBER})(?:±|\+/-)({NUMBER})", s)
    if match:
        val, err = map(float, match.groups())
        if val > 0:
            return Concentration(val, val, val, "summary_mean", "parsed", err)
    match = re.fullmatch(rf"(>=|<=|>|<)?({NUMBER})", s)
    if match:
        op, value = match.groups()
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            return Concentration(status="nonpositive_or_nonfinite")
        if op and ">" in op:
            return Concentration(value, math.inf, math.nan, "right", "parsed")
        if op and "<" in op:
            return Concentration(0.0, value, math.nan, "left", "parsed")
        return Concentration(value, value, value, "exact", "parsed")
    match = re.fullmatch(rf"({NUMBER})-({NUMBER})", s)
    if match:
        lo, hi = map(float, match.groups())
        if lo <= 0 or hi <= 0:
            return Concentration(status="nonpositive_range")
        if lo > hi:
            return Concentration(status="reversed_range")
        return Concentration(lo, hi, lo if lo == hi else math.nan,
                             "exact" if lo == hi else "interval", "parsed")
    return Concentration()


def canonical_publication(article):
    pmid = text((article.get("pubmed") or {}).get("pubmedId"))
    if re.fullmatch(r"\d+", pmid):
        return "PMID:" + str(int(pmid)), "pmid"
    for key in ("doi", "DOI"):
        doi = text(article.get(key)).lower()
        if doi:
            doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi).strip()
            if re.fullmatch(r"10\.\d{4,9}/\S+", doi):
                return "DOI:" + doi, "doi"
    title = unicodedata.normalize("NFKC", text(article.get("title"))).casefold()
    title = re.sub(r"\W+", " ", title).strip()
    year = text(article.get("year"))
    # No 60-character truncation and no peptide-article join ID as publication.
    if title and year:
        return stable_hash([title, year], "TITLEYEAR:"), "title_year"
    return "", "unresolved"


def resolve_references(activity, articles, crosswalk=None):
    raw = text(activity.get("reference"))
    if not raw:
        indices = [1] if len(articles) == 1 else []
        rule = "only_article_when_reference_missing" if indices else "missing"
    elif re.fullmatch(r"\d+(?:\s*[,;]\s*\d+)*", raw):
        indices = [int(v) for v in re.split(r"[,;]", raw)]
        rule = "explicit_reference"
    else:
        indices, rule = [], "malformed_reference"
    if not indices or any(i < 1 or i > len(articles) for i in indices):
        return {"publication_id": "", "publication_ids": "[]", "publication_year": math.nan,
                "reference_resolution": rule if not indices else "out_of_bounds"}
    matched = [articles[i - 1] for i in indices]
    if crosswalk is not None:
        from .bibliography import article_key
        verified = [crosswalk[article_key(a)] for a in matched]
        ids = sorted({v["publication_id"] for v in verified})
        years = sorted({v["publication_year"] for v in verified if v["publication_year"] is not None})
        verification = "|".join(sorted({v["verification_status"] for v in verified}))
    else:
        ids = sorted(set(canonical_publication(a)[0] for a in matched))
        years = sorted({int(a["year"]) for a in matched if str(a.get("year", "")).isdigit()
                        and 1800 <= int(a["year"]) <= datetime.now(timezone.utc).year})
        verification = "not_externally_checked"
    resolved = len(ids) == 1 and bool(ids[0])
    return {
        "publication_id": ids[0] if resolved else "",
        "publication_ids": json.dumps(ids),
        "publication_year": years[0] if len(years) == 1 else math.nan,
        "reference_resolution": rule if resolved else "multiple_or_unresolved_publications",
        "bibliography_verification": verification,
    }


def bond_signature(bonds):
    keys = ("position1", "position2", "position", "type", "cycleType",
            "chainParticipating", "chain1", "chain2")
    projected = [{k: text(b[k]) if isinstance(b[k], dict) else b[k]
                  for k in keys if k in b} for b in (bonds or [])]
    return sorted(projected, key=lambda b: json.dumps(b, sort_keys=True))


def chemistry(card):
    seq = re.sub(r"\s+", "", text(card.get("sequence")))
    nterm = text(card.get("nTerminus") or card.get("NTerminus"))
    cterm = text(card.get("cTerminus") or card.get("CTerminus"))
    unusual = card.get("unusualAminoAcids") or []
    complexity = text(card.get("complexity"))
    signature = {
        "sequence_stereochemistry_preserved": seq, "nterm": nterm, "cterm": cterm,
        "intrachain": bond_signature(card.get("intrachainBonds")),
        "interchain": bond_signature(card.get("interchainBonds")),
        "coordination": bond_signature(card.get("coordinationBonds")),
        "unusual": [{k: v for k, v in u.items() if k != "id"} for u in unusual],
        "complexity": complexity,
    }
    weights, formulas, canonical_smiles = [], [], []
    invalid_smiles = 0
    for entry in card.get("smiles") or []:
        smi = text(entry.get("smiles")) if isinstance(entry, dict) else text(entry)
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            invalid_smiles += 1
            continue
        weights.append(Descriptors.MolWt(mol))
        formulas.append(rdMolDescriptors.CalcMolFormula(mol))
        canonical_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    mw, mass_rule = math.nan, "unresolved"
    if weights and max(weights) / min(weights) <= 1.001:
        mw, mass_rule = float(np.mean(weights)), "database_smiles_consensus"
    elif weights:
        mass_rule = "conflicting_smiles_masses"
    # Safe fallback only for explicitly monomeric, canonical, unmodified peptides.
    # Lowercase encodes D residues; chirality does not alter molecular mass.
    plain = (bool(seq) and set(seq.upper()) <= AA and complexity.lower() == "monomer"
             and not any([nterm, cterm, unusual, card.get("intrachainBonds"),
                          card.get("interchainBonds"), card.get("coordinationBonds")]))
    if not weights and plain:
        mol = Chem.MolFromSequence(seq.upper())
        if mol is not None:
            mw, mass_rule = Descriptors.MolWt(mol), "unmodified_sequence_formula"
    if not seq:
        signature["structure"] = sorted(set(canonical_smiles))
        if not canonical_smiles:
            signature["unresolved_card_id"] = card.get("id")
    backbone = seq.upper() if seq and set(seq.upper()) <= AA else ""
    valid_sequence = bool(backbone) and complexity.lower() == "monomer"
    return {
        "peptide_id": card["id"], "sequence_raw": seq, "backbone_sequence": backbone,
        "sequence_id": stable_hash(backbone, "SEQ:") if backbone else "",
        "chemical_id": stable_hash(signature, "CHEM:"),
        "n_terminus": nterm or "unmodified_or_unreported",
        "c_terminus": cterm or "unmodified_or_unreported",
        "n_intrachain_bonds": len(card.get("intrachainBonds") or []),
        "n_unusual_residues": len(unusual), "n_d_residues": sum(c.islower() for c in seq),
        "complexity": complexity, "valid_sequence": valid_sequence,
        "mw_da": mw, "mass_rule": mass_rule, "n_valid_smiles": len(weights),
        "n_invalid_smiles": invalid_smiles,
        "smiles_formulae": json.dumps(sorted(set(formulas))),
        "chemistry_signature": json.dumps(signature, sort_keys=True),
    }


def unit_factor(unit, mw):
    u = unicodedata.normalize("NFKC", text(unit)).replace("μ", "u").replace("µ", "u")
    u = re.sub(r"\s+", "", u).lower()
    molar = {"um": 1.0, "mm": 1000.0, "nm": .001, "m": 1e6}
    if u in molar:
        return molar[u], "native_molar"
    mass = {"ug/ml": 1000.0, "mg/l": 1000.0, "mg/ml": 1e6, "ug/l": 1.0}
    if u in mass:
        if math.isfinite(mw) and mw > 0:
            return mass[u] / mw, "mass_to_molar"
        return math.nan, "mass_unresolved"
    return math.nan, "unsupported_or_missing_unit"


def get_taxonomy(genera, cache_path, online=True):
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    session = requests.Session()
    for genus in sorted(genera):
        if genus in cache or not online:
            continue
        url = "https://www.ebi.ac.uk/ena/taxonomy/rest/scientific-name/" + quote(genus, safe="")
        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()
            candidates = response.json()
            exact = [r for r in candidates if r.get("scientificName", "").casefold() == genus.casefold()]
            realms = set()
            for r in exact:
                lineage = r.get("lineage", "")
                realms.add("Bacteria" if lineage.startswith("Bacteria;") else
                           "Archaea" if lineage.startswith("Archaea;") else
                           "Eukaryota" if lineage.startswith("Eukaryota;") else "Other")
            cache[genus] = {"kingdom": next(iter(realms)) if len(realms) == 1 else "Unknown",
                            "response": candidates, "url": url,
                            "retrieved_utc": datetime.now(timezone.utc).isoformat()}
            cache_path.write_text(json.dumps(cache, indent=2))
        except (requests.RequestException, ValueError) as exc:
            print(f"Unresolved taxonomy {genus}: {exc}", flush=True)
        time.sleep(.1)
    return cache


def species_name(target):
    tokens = re.sub(r"\([^)]*\)", "", target).split()
    if len(tokens) >= 2 and tokens[0][0:1].isupper() and tokens[1][0:1].islower():
        return " ".join(tokens[:2])
    return tokens[0] if tokens else "Unknown"


def context_values(act):
    result = {
        "medium": text(act.get("medium")) or "Unknown",
        "cfu_raw": text(act.get("cfu")),
        "salt_type_raw": text(act.get("saltType")),
        "ionic_strength_raw": text(act.get("ionicStrength")),
        "ph_raw": text(act.get("ph")),
        "note_raw": text(act.get("note")),
    }
    cfu = parse_concentration(result["cfu_raw"])
    result["log10_cfu"] = math.log10(cfu.point) if cfu.kind == "exact" else math.nan
    result["cfu_kind"] = cfu.kind
    ph = parse_concentration(result["ph_raw"])
    result["ph"] = ph.point if ph.kind == "exact" and 0 <= ph.point <= 14 else math.nan
    # Unknown absence is not the same as zero supplemental salt.
    salt = result["salt_type_raw"]
    result["supplement_category"] = salt if salt else "Unreported"
    ionic = parse_concentration(result["ionic_strength_raw"])
    result["nacl_reported_concentration"] = (
        ionic.point if salt.lower() == "nacl" and ionic.kind == "exact" else math.nan
    )
    # The API calls this ionicStrength. Retain its value and source without
    # asserting molarity until the units are verified against database guidance.
    result["context_reporting_count"] = sum(bool(result[k]) for k in
                                             ["cfu_raw", "salt_type_raw", "ph_raw"])
    return result


def card_paths(include_expansion):
    paths = list((ROOT / "data/raw/dbaasp_cards").glob("*.json"))
    if include_expansion:
        paths += [p for p in (ROOT / "data/raw/dbaasp_expansion_2026-10-20").glob("*.json")
                  if p.stem.isdigit()]
    return sorted(paths, key=lambda p: int(p.stem))


def build_review_queue(data):
    flags = pd.DataFrame({
        "concentration_unparsed_or_ambiguous": data.parse_status.ne("parsed"),
        "summary_mean_requires_replication_metadata": data.parse_kind.eq("summary_mean"),
        "conversion_unresolved": data.conversion_rule.isin(["mass_unresolved", "unsupported_or_missing_unit"]),
        "reference_ambiguous": data.reference_resolution.isin(
            ["out_of_bounds", "multiple_or_unresolved_publications", "missing", "malformed_reference"]),
        "bibliography_unverified": data.bibliography_verification.replace("", "unresolved").fillna(
            "unresolved").str.contains("unresolved"),
    })
    selected = flags.any(axis=1)
    queue = data.loc[selected].copy()
    queue["manual_review_reasons"] = [
        "|".join(flags.columns[row]) for row in flags.loc[selected].to_numpy(dtype=bool)]
    return queue


def amend_source_row(row, amendment):
    """Apply a reviewed medium correction without rewriting the source record."""
    if amendment["field"] != "medium":
        raise ValueError("Unsupported source amendment field")
    if row["record_id"] != amendment["record_id"]:
        raise ValueError("Source amendment record identity mismatch")
    for key, expected in amendment["expected"].items():
        if row[key] != expected:
            raise ValueError(f"Source amendment original value mismatch: {key}")
    if row["medium"] != amendment["original"]:
        raise ValueError("Source amendment medium mismatch")
    corrected = row.copy()
    corrected["medium"] = amendment["corrected"]
    return corrected


def curate(include_expansion=False, online_taxonomy=True, refresh_publications=False):
    output = ROOT / "data/curated" / ("expanded" if include_expansion else "sample")
    output.mkdir(parents=True, exist_ok=True)
    result_dir = ROOT / "results/reconstruction_2026-10-20"
    result_dir.mkdir(exist_ok=True)
    cards = []
    sources = []
    amendments_path = ROOT/"data/source_amendments_2026-10-20.json"
    amendment_document = json.loads(amendments_path.read_text()) if amendments_path.exists() else {"amendments": []}
    amendments = {r["record_id"]: r for r in amendment_document["amendments"]}
    if len(amendments) != len(amendment_document["amendments"]):
        raise ValueError("Conflicting duplicate source amendments")
    for amendment in amendments.values():
        if hashlib.sha256((ROOT/amendment["source_card"]).read_bytes()).hexdigest() != amendment["source_card_sha256"]:
            raise ValueError("Source card changed since amendment review")
        for evidence in amendment["evidence"]:
            if hashlib.sha256((ROOT/evidence["path"]).read_bytes()).hexdigest() != evidence["sha256"]:
                raise ValueError("Source amendment evidence changed")
    applied_amendments = []
    if include_expansion:
        index_ids = {int(row["id"]) for row in json.loads(
            (ROOT / "data/raw/dbaasp_peptide_index.json").read_text())}
        available = {int(p.stem) for p in card_paths(True)}
        missing = index_ids - available
        if missing:
            raise ValueError(f"Catalog expansion incomplete: {len(missing)} indexed cards missing. "
                             "Resume scripts/09_expand_dbaasp.py before a full-catalog analysis.")
    for path in card_paths(include_expansion):
        payload = path.read_bytes()
        obj = json.loads(payload)
        if not isinstance(obj, dict) or int(obj.get("id", -1)) != int(path.stem):
            raise ValueError(f"Invalid card: {path}")
        cards.append((obj, path))
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest()})
    frozen_crosswalk = output / "publication_crosswalk_used.json"
    crosswalk_path = (frozen_crosswalk if frozen_crosswalk.exists() and not refresh_publications else
                      ROOT / "data/raw/publication_crosswalk_2026-10-20.json")
    if not crosswalk_path.exists():
        raise FileNotFoundError("Run scripts/17_verify_publications.py before final curation.")
    crosswalk = json.loads(crosswalk_path.read_text())
    if crosswalk_path != frozen_crosswalk:
        frozen_crosswalk.write_bytes(crosswalk_path.read_bytes())
    crosswalk_path = frozen_crosswalk
    from .bibliography import article_key
    missing_articles = {article_key(a) for c, _ in cards for a in c.get("articles", [])
                        if article_key(a) not in crosswalk}
    if missing_articles:
        raise ValueError(f"{len(missing_articles)} source references have not been checked. "
                         "Run scripts/17_verify_publications.py with the matching scope.")
    genera = {species_name(text(a.get("targetSpecies"))).split()[0]
              for c, _ in cards for a in c.get("targetActivities", [])}
    taxonomy = get_taxonomy(genera, ROOT / "data/raw/taxonomy_ena_2026-10-20.json",
                            online=online_taxonomy)
    ambiguous_species = {
        species_name(text(a.get("targetSpecies")))
        for c, _ in cards for a in c.get("targetActivities", [])
        if taxonomy.get(species_name(text(a.get("targetSpecies"))).split()[0],
                        {}).get("kingdom", "Unknown") == "Unknown"
    }
    taxonomy = get_taxonomy(ambiguous_species, ROOT / "data/raw/taxonomy_ena_2026-10-20.json",
                            online=online_taxonomy)
    used_taxonomy = {key: taxonomy[key] for key in sorted(genera | ambiguous_species) if key in taxonomy}
    (output / "taxonomy_used.json").write_text(json.dumps(used_taxonomy, indent=2))
    rows, chem_rows = [], []
    for n, (card, path) in enumerate(cards):
        chem = chemistry(card)
        chem_rows.append(chem)
        for act in card.get("targetActivities") or []:
            target = text(act.get("targetSpecies"))
            species = species_name(target)
            conc = parse_concentration(act.get("concentration"))
            factor, conversion = unit_factor(act.get("unit"), chem["mw_da"])
            row = {
                **chem,
                "record_id": f"DBAASP:{card['id']}:{act.get('id')}",
                "activity_id": act.get("id"), "source_file": str(path.relative_to(ROOT)),
                "source_url": f"https://dbaasp.org/peptides/{card['id']}",
                "target_exact": target, "species": species,
                "kingdom": taxonomy.get(species, taxonomy.get(species.split()[0], {})).get("kingdom", "Unknown"),
                "activity_measure": text(act.get("activityMeasureValue")) or text(act.get("activityMeasureGroup")),
                "concentration_raw": text(act.get("concentration")),
                "unit_raw": text(act.get("unit")), "database_activity_ugml": act.get("activity"),
                "parse_kind": conc.kind, "parse_status": conc.status,
                "reported_error_raw": conc.reported_error,
                "conversion_rule": conversion, "conversion_factor": factor,
                "lower_uM": conc.lower * factor, "upper_uM": conc.upper * factor,
                "point_uM": conc.point * factor, "reference_raw": text(act.get("reference")),
                **resolve_references(act, card.get("articles") or [], crosswalk), **context_values(act),
            }
            if row["record_id"] in amendments:
                amendment = amendments[row["record_id"]]
                row = amend_source_row(row, amendment)
                applied_amendments.append(amendment)
            row["log2_lower"] = math.log2(row["lower_uM"]) if row["lower_uM"] > 0 else -math.inf
            row["log2_upper"] = math.log2(row["upper_uM"]) if row["upper_uM"] > 0 else math.nan
            row["log2_mic"] = math.log2(row["point_uM"]) if row["point_uM"] > 0 else math.nan
            reasons = []
            for valid, reason in [
                (row["kingdom"] == "Bacteria", "not_confirmed_bacteria"),
                (row["activity_measure"] == "MIC", "not_MIC"),
                (conc.status == "parsed", "unparsed_concentration"),
                (math.isfinite(factor), conversion),
                (bool(row["publication_id"]), "unresolved_publication"),
                (chem["valid_sequence"], "unsupported_sequence_or_complexity"),
                (row["medium"] != "Unknown", "medium_missing"),
                ("unresolved" not in row.get("bibliography_verification", "unresolved"),
                 "bibliography_unverified"),
            ]:
                if not valid:
                    reasons.append(reason)
            inferential_reasons = [r for r in reasons if r != "unsupported_sequence_or_complexity"]
            row["eligible_censored_analysis"] = not inferential_reasons and conc.kind != "summary_mean"
            row["eligible_exact_analysis"] = row["eligible_censored_analysis"] and conc.kind == "exact"
            row["eligible_benchmark"] = row["eligible_exact_analysis"] and chem["valid_sequence"]
            row["exclusion_reasons"] = "|".join(reasons)
            # Outcome is deliberately absent from cell identity.
            identity = {k: row[k] for k in [
                "chemical_id", "publication_ids", "target_exact", "activity_measure", "medium",
                "cfu_raw", "salt_type_raw", "ionic_strength_raw", "ph_raw", "note_raw",
            ]}
            row["assay_cell_id"] = stable_hash(identity, "CELL:")
            # A candidate is flagged, not automatically removed: equal MICs may
            # be genuine repeated dilution measurements.
            row["duplicate_candidate_key"] = stable_hash(
                [identity, row["concentration_raw"], row["unit_raw"]], "CAND:"
            )
            rows.append(row)
        if (n + 1) % 500 == 0:
            print(f"Curated {n + 1}/{len(cards)} cards", flush=True)
    data = pd.DataFrame(rows)
    # Only repeated database record identities can be deduplicated automatically.
    duplicated = data[data.record_id.duplicated(False)]
    if len(duplicated):
        varying = duplicated.groupby("record_id").nunique(dropna=False)
        if varying.gt(1).any(axis=1).any():
            raise ValueError("Conflicting repeated database record IDs")
    data = data.drop_duplicates("record_id")
    data["duplicate_candidate"] = data.duplicate_candidate_key.duplicated(False)
    data.to_csv(output / "activities.csv", index=False)
    (output/"source_amendments_applied.json").write_text(json.dumps(applied_amendments,indent=2))
    pd.DataFrame(chem_rows).drop_duplicates("peptide_id").to_csv(output / "peptides.csv", index=False)
    exact = data[data.eligible_exact_analysis].copy()
    metadata = [c for c in data.columns if c not in {
        "record_id", "activity_id", "concentration_raw", "unit_raw", "database_activity_ugml",
        "parse_kind", "parse_status", "reported_error_raw", "conversion_rule", "conversion_factor",
        "lower_uM", "upper_uM", "point_uM", "log2_lower", "log2_upper", "log2_mic",
        "source_file", "source_url", "reference_raw", "duplicate_candidate_key", "duplicate_candidate",
    }]
    cells = exact.groupby("assay_cell_id", sort=True, dropna=False)
    aggregated = cells[metadata].first()
    aggregated["log2_mic"] = cells.log2_mic.mean()
    aggregated["n_records"] = cells.size()
    aggregated["log2_sd"] = cells.log2_mic.std()
    aggregated["record_ids"] = cells.record_id.agg(lambda v: json.dumps(sorted(v)))
    aggregated = aggregated.reset_index(drop=True)
    aggregated.to_csv(output / "all_exact_assay_cells.csv", index=False)
    benchmark = aggregated[aggregated.eligible_benchmark].copy()
    benchmark.to_csv(output / "exact_assay_cells.csv", index=False)
    manual = build_review_queue(data)
    manual.to_csv(output / "manual_review_queue.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "expanded_catalog" if include_expansion else "original_2500_card_sample",
        "cards": len(cards), "records": len(data), "exact_assay_cells": len(aggregated),
        "benchmark_exact_assay_cells": len(benchmark),
        "eligible_exact_records": int(data.eligible_exact_analysis.sum()),
        "eligible_censored_records": int(data.eligible_censored_analysis.sum()),
        "parse_kinds": data.parse_kind.value_counts().to_dict(),
        "parse_status": data.parse_status.value_counts().to_dict(),
        "conversion_rules": data.conversion_rule.value_counts().to_dict(),
        "mass_rules_per_peptide": pd.DataFrame(chem_rows).mass_rule.value_counts().to_dict(),
        "reference_resolution": data.reference_resolution.value_counts().to_dict(),
        "kingdoms": data.kingdom.value_counts().to_dict(),
        "manual_review_records": len(manual),
        "candidate_duplicate_records_retained": int(data.duplicate_candidate.sum()),
        "exact_chemical_entities": aggregated.chemical_id.nunique(),
        "exact_backbone_sequences": aggregated.sequence_id.replace("", np.nan).nunique(),
        "exact_cells_without_backbone_id": int(aggregated.sequence_id.fillna("").eq("").sum()),
        "exact_publications": aggregated.publication_id.nunique(),
        "input_files": sources,
        "source_amendments_applied": len(applied_amendments),
        "source_amendments_sha256": hashlib.sha256(amendments_path.read_bytes()).hexdigest() if amendments_path.exists() else None,
        "publication_crosswalk_sha256": hashlib.sha256(crosswalk_path.read_bytes()).hexdigest(),
        "taxonomy_used_sha256": hashlib.sha256((output / "taxonomy_used.json").read_bytes()).hexdigest(),
        "rules": {
            "native_molar_values": "preserved; no mass round-trip",
            "mass_conversion": "average molecular weight from consistent source SMILES, or unmodified canonical monomer formula",
            "deduplication": "record ID only; identical outcomes are retained and flagged",
            "primary_exact_endpoint": "mean log2 MIC within the same reported assay cell",
            "censoring": "explicit inequalities and ranges retained; no assumed dilution intervals",
            "publication": "PMID, then explicit DOI, then full normalized title+year hash",
            "taxonomy": "cached ENA lineage; unrecognized taxa are Unknown, not assumed bacterial",
            "missing_context": "explicitly unreported; never assigned standard conditions",
            "source_amendments": "Reviewed corrections with record-level expected-value guards and primary-source hashes; original source cards are preserved.",
        },
    }
    (output / "curation_manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "input_files"}, indent=2), flush=True)
    return output
