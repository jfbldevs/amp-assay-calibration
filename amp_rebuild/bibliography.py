"""Verify source publication identifiers against PubMed, with raw-response caches."""
from datetime import datetime, timezone
from difflib import SequenceMatcher
import html
import json
from pathlib import Path
import re
import time
import unicodedata

import pandas as pd
import requests

from .curation import ROOT, canonical_publication, card_paths, stable_hash, text

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def normalize_title(value):
    value = html.unescape(re.sub(r"<[^>]+>", "", text(value))).casefold()
    for greek, latin in [("α", "alpha"), ("β", "beta"), ("γ", "gamma"), ("δ", "delta")]:
        value = value.replace(greek, latin)
    value = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", value)


def article_key(article):
    return stable_hash({
        "title": text(article.get("title")), "year": text(article.get("year")),
        "pmid": text((article.get("pubmed") or {}).get("pubmedId")),
        "doi": text(article.get("doi") or article.get("DOI")),
        "authors": [text(a) for a in article.get("authors") or []],
    }, "ARTICLE:")


def query(session, endpoint, params, cache):
    path = cache / (stable_hash([endpoint, params]) + ".json")
    if path.exists():
        return json.loads(path.read_text())["response"]
    last = None
    for attempt in range(4):
        try:
            time.sleep(.4)
            response = session.get(BASE+endpoint, params={**params, "tool": "amp_context_reconstruction"},
                                   timeout=45)
            response.raise_for_status()
            result = response.json()
            if "error" in result:
                raise ValueError(str(result["error"]))
            path.write_text(json.dumps({
                "url": response.url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "response": result,
            }, indent=2))
            return result
        except (requests.RequestException, ValueError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"PubMed query failed: {last}")


def fetch_metadata(pmids, session, cache, metadata):
    missing = sorted(set(pmids)-set(metadata))
    for start in range(0, len(missing), 150):
        batch = missing[start:start+150]
        response = query(session, "esummary.fcgi",
                         {"db": "pubmed", "id": ",".join(batch), "retmode": "json"}, cache)
        result = response.get("result", {})
        for pmid in batch:
            if pmid in result and "error" not in result[pmid]:
                metadata[pmid] = result[pmid]
        print(f"PubMed metadata {min(start+150, len(missing))}/{len(missing)}", flush=True)


def first_publication_year(meta):
    years = []
    for field in ["pubdate", "epubdate"]:
        match = re.search(r"\b(18|19|20)\d{2}\b", str(meta.get(field, "")))
        if match:
            years.append(int(match.group()))
    return min(years) if years else None


def verify_bibliography(include_expansion=False):
    cache = ROOT / "data/raw/pubmed_responses"
    cache.mkdir(exist_ok=True)
    metadata_path = ROOT / "data/raw/pubmed_metadata_2026-10-20.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    articles = {}
    for path in card_paths(include_expansion):
        for article in json.loads(path.read_text()).get("articles", []):
            articles[article_key(article)] = article
    pmids = {text((a.get("pubmed") or {}).get("pubmedId")) for a in articles.values()}
    pmids = {p for p in pmids if p.isdigit()}
    session = requests.Session()
    fetch_metadata(pmids, session, cache, metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    crosswalk, rows = {}, []
    # Title lookup can also resolve cards that omit PMID without assuming that
    # the first article or a truncated title identifies a publication.
    title_index = {}
    for pmid, meta in metadata.items():
        title_index.setdefault(normalize_title(meta.get("title")), set()).add(pmid)
    for number, (key, article) in enumerate(sorted(articles.items()), 1):
        title = normalize_title(article.get("title"))
        raw_pmid = text((article.get("pubmed") or {}).get("pubmedId"))
        meta = metadata.get(raw_pmid)
        similarity = SequenceMatcher(None, title, normalize_title(meta.get("title"))).ratio() if meta else 0
        resolved, status = raw_pmid, "verified_title"
        # A 0.90 match accommodates typography and short subtitle differences,
        # while every non-exact match remains visible in the audit table.
        if not meta or similarity < .90:
            candidates = title_index.get(title, set())
            if len(candidates) != 1 and title:
                query_title = re.sub(r"[\"<>]", " ", html.unescape(text(article.get("title"))))
                try:
                    search = query(session, "esearch.fcgi",
                                   {"db": "pubmed", "term": query_title+"[Title]",
                                    "retmode": "json", "retmax": 10}, cache)
                    ids = search.get("esearchresult", {}).get("idlist", [])
                    fetch_metadata(ids, session, cache, metadata)
                    candidates = {p for p in ids if normalize_title(metadata.get(p, {}).get("title")) == title}
                except RuntimeError as exc:
                    print(f"Unresolved title search: {exc}", flush=True)
            if len(candidates) == 1:
                resolved = next(iter(candidates))
                meta = metadata[resolved]
                status = "corrected_pmid_by_exact_title" if raw_pmid else "resolved_missing_pmid_by_exact_title"
            else:
                resolved = ""
                meta = None
                status = "unresolved_or_ambiguous_title"
        if resolved:
            publication_id = "PMID:"+resolved
            year = first_publication_year(meta)
            final_similarity = SequenceMatcher(None, title, normalize_title(meta.get("title"))).ratio()
        else:
            # Discard an inconsistent PMID rather than propagating it as identity.
            cleaned = {**article, "pubmed": None}
            publication_id = canonical_publication(cleaned)[0]
            raw_year = text(article.get("year"))
            year = int(raw_year) if raw_year.isdigit() and 1800 <= int(raw_year) <= 2026 else None
            final_similarity = None
        item = {
            "publication_id": publication_id, "publication_year": year,
            "verification_status": status, "raw_pmid": raw_pmid, "resolved_pmid": resolved,
            "title_similarity": final_similarity,
        }
        crosswalk[key] = item
        rows.append({
            "article_key": key, **item, "source_title": text(article.get("title")),
            "source_year": text(article.get("year")),
            "pubmed_title": meta.get("title") if meta else None,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{resolved}/" if resolved else None,
        })
        if number % 100 == 0:
            print(f"Verified references {number}/{len(articles)}", flush=True)
            metadata_path.write_text(json.dumps(metadata, indent=2))
    metadata_path.write_text(json.dumps(metadata, indent=2))
    (ROOT / "data/raw/publication_crosswalk_2026-10-20.json").write_text(json.dumps(crosswalk, indent=2))
    output = ROOT / "results/reconstruction_2026-10-20"
    table = pd.DataFrame(rows)
    table.to_csv(output / "publication_identity_audit.csv", index=False)
    summary = {
        "source_articles": len(articles),
        "unique_source_publication_identities": table.publication_id.nunique(),
        "unique_resolved_publications": table.loc[
            table.resolved_pmid.ne(""), "publication_id"].nunique(),
        "status_counts": table.verification_status.value_counts().to_dict(),
        "nonexact_verified_title_matches": int((table.title_similarity.notna() &
                                                table.title_similarity.lt(1)).sum()),
        "year_definition": "earliest PubMed print/electronic publication year where verified; otherwise source year",
        "threshold": "PMID title similarity >=0.90; changes to PMID require unique exact normalized title",
    }
    (output / "publication_identity_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
