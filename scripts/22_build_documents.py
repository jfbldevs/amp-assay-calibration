#!/usr/bin/env python3
"""Compile scope-specific manuscript and supplement without fabricated metadata."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def latex(value):
    text = re.sub(r"<[^>]+>", "", html.unescape(str(value)))
    text = re.sub(r"\s+", " ", text).strip()
    # Latin Modern's text font does not contain the Greek beta in source titles.
    # Preserve the character semantically using TeX's supported math glyph.
    return (text.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")
            .replace("β", r"{\(\beta\)}"))


def bibliography():
    entries = []
    for path in sorted((ROOT/"manuscript/reference_metadata").glob("*.json")):
        record = json.loads(path.read_text())
        authors = " and ".join(
            f"{a['family']}, {a.get('given', '')}" if "family" in a else "{"+a["name"]+"}"
            for a in record.get("author", []))
        date = record.get("published-print") or record.get("published-online") or record["published"]
        fields = {
            "author": latex(authors), "title": "{"+latex(record["title"][0])+"}",
            "journal": latex((record.get("container-title") or ["bioRxiv preprint"])[0]),
            "year": date["date-parts"][0][0], "doi": record["DOI"],
        }
        for source, key in [("volume", "volume"), ("issue", "number"), ("page", "pages")]:
            if record.get(source):
                fields[key] = record[source]
        if record.get("article-number") and not record.get("page"):
            fields["pages"] = record["article-number"]
        entries.append("@article{"+path.stem+",\n"+
                       ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items())+"\n}")
    required = set()
    source = (ROOT/"manuscript/manuscript.tex").read_text()
    for match in re.findall(r"\\cite\w*\{([^}]+)\}", source):
        required.update(match.split(","))
    available = {p.stem for p in (ROOT/"manuscript/reference_metadata").glob("*.json")}
    assert required <= available, f"Unverified references: {required-available}"
    (ROOT/"manuscript/references.bib").write_text("\n\n".join(entries)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["sample", "expanded"], default="sample")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    scope_path = ROOT/"data/study_scope_2026-10-20.json"
    study_scope = json.loads(scope_path.read_text())
    if study_scope.get("manuscript_authoring_required") is False:
        receipt = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "scope": args.scope,
            "status": "manuscript_and_journal_supplement_deferred_by_author",
            "products": [], "existing_drafts_preserved": True,
            "authority": str(scope_path.relative_to(ROOT)),
            "scope_sha256": hashlib.sha256(scope_path.read_bytes()).hexdigest(),
            "research_documentation": "scripts/24_build_execution_report.py",
        }
        (ROOT/"results/reconstruction_2026-10-20"/args.scope/"manuscript_deferral.json").write_text(
            json.dumps(receipt, indent=2)+"\n")
        print(json.dumps(receipt, indent=2), flush=True)
        return
    bibliography()
    output = ROOT/"output/pdf"
    output.mkdir(parents=True, exist_ok=True)
    products = []
    for short, template in [("manuscript", "manuscript.tex"), ("supplement", "supplementary_information.tex")]:
        stem = f"AMP_{short}_{args.scope}_2026-10-20"
        wrapper = ROOT/"manuscript"/(stem+".tex")
        wrapper.write_text(r"\def\StudyScope{"+args.scope+"}\n"+r"\input{"+template+"}\n")
        command = ["tectonic", "--keep-logs", "--keep-intermediates", "--outdir", str(output), str(wrapper)]
        result = subprocess.run(command, cwd=ROOT/"manuscript", capture_output=True, text=True)
        log = ROOT/"results/reconstruction_2026-10-20"/args.scope/(stem+"_build.log")
        log.write_text(result.stdout+"\n"+result.stderr)
        if result.returncode:
            print(result.stdout, result.stderr)
            raise SystemExit(result.returncode)
        pdf = output/(stem+".pdf")
        assert pdf.exists()
        pdf_text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
        if "??" in pdf_text:
            raise ValueError(f"Unresolved cross-reference in {pdf}")
        record = {"path": str(pdf.relative_to(ROOT)), "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
        if args.render:
            folder = ROOT/"tmp/pdfs"/stem
            folder.mkdir(parents=True, exist_ok=True)
            # Remove only pages generated for this document, so a shorter rebuild
            # cannot leave obsolete pages in the visual-review set.
            for old_page in folder.glob("page-*.png"):
                old_page.unlink()
            subprocess.run(["pdftoppm", "-r", "110", "-png", str(pdf), str(folder/"page")], check=True)
            record["rendered_pages"] = len(list(folder.glob("page-*.png")))
            record["visual_inspection"] = "pending"
        products.append(record)
        print(f"Built {pdf}", flush=True)
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": args.scope, "products": products}
    (ROOT/"results/reconstruction_2026-10-20"/args.scope/"document_build.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
