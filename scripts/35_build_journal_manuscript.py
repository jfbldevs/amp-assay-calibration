#!/usr/bin/env python3
"""Build English Analysis and SI entirely from audited, preserved sources."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from amp_rebuild.editorial_common import ROOT,OUT,utc,sha,read_json,write_json,verify_preservation
from amp_rebuild.journal_materials import (DOC,GEN,Materials,load_inputs,numeric_claims,further_numbers,
    make_figures,supplementary_figures,make_tables,escape)

PREFIXES={
    "pgain":"gain.temporal.publication_target.random.5.point_frozen.sequence_hgb.",
    "pcal":"cal.temporal.publication_target.random.5.point_frozen.sequence_hgb.empirical_bayes.",
    "pbase":"cal.temporal.publication_target.random.5.point_frozen.sequence_hgb.uncalibrated.",
    "pint":"cal.temporal.publication_target.random.5.interval_proper_fit.sequence_hgb.empirical_bayes.",
    "pintbase":"cal.temporal.publication_target.random.5.interval_proper_fit.sequence_hgb.uncalibrated.",
}
TITLE="Reference measurements improve antimicrobial peptide prediction beyond recorded assay metadata"
PREAMBLE=r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=24mm]{geometry}
\usepackage{fontspec}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,tabularx}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{microtype,xcolor,hyperref,fancyhdr,placeins,caption,needspace}
\setmainfont[BoldFont=lmroman10-bold.otf,ItalicFont=lmroman10-italic.otf,BoldItalicFont=lmroman10-bolditalic.otf]{lmroman10-regular.otf}
\hypersetup{colorlinks=true,linkcolor=blue!45!black,citecolor=blue!45!black,urlcolor=blue!45!black}
\newcommand{\doi}[1]{\href{https://doi.org/#1}{doi: \nolinkurl{#1}}}
\renewcommand{\bibfont}{\small}
\setlength{\bibsep}{3pt}
\captionsetup{font=small,labelfont=bf,skip=7pt}
\input{generated/numbers.tex}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small Reference calibration of peptide activity}
\fancyhead[R]{\small Analysis}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{15pt}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{2em}
\clubpenalty=10000\widowpenalty=10000
\setcounter{secnumdepth}{0}
"""


def expanded_numbers(text,numbers):
    for short,prefix in PREFIXES.items():
        text=re.sub(r"\\"+short+r"\{([^}]+)\}",lambda m:r"\val{"+prefix+m.group(1)+"}",text)
    used=set(re.findall(r"\\val\{([^}]+)\}",text))
    missing=used-set(numbers)
    assert not missing, f"Undefined numeric macros: {sorted(missing)}"
    return re.sub(r"\\val\{([^}]+)\}",lambda m:numbers[m.group(1)]["rendered"],text),used


def word_count(text):
    text=re.sub(r"\\cite\w*\{[^}]+\}","",text)
    text=re.sub(r"\\ref\{[^}]+\}","",text)
    text=re.sub(r"\\(?:subsection|section)\*?\{[^}]+\}","",text)
    text=re.sub(r"\\[A-Za-z]+", " ",text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[.,'-][A-Za-z0-9]+)*",text))


def bibliography(m):
    source=ROOT/"manuscript/references.bib"
    m.source(source)
    text=source.read_text()
    extra_path=DOC/"reference_metadata_extra.json"
    extra=m.json(extra_path)
    for key,row in extra.items():
        text+="\n@article{"+key+",\n"+",\n".join("  "+k+" = {"+escape(v)+"}" for k,v in
            [("author",row["authors"]),("title","{"+row["title"]+"}"),("journal",row["journal"]),("year",row["year"]),("doi",row["doi"])])+"\n}\n"
    (DOC/"references_editorial.bib").write_text(text)
    available=set(re.findall(r"@\w+\{([^,]+),",text))
    cited=set()
    for path in DOC.glob("*.tex"):
        for group in re.findall(r"\\cite\w*\{([^}]+)\}",path.read_text()):
            cited.update(group.split(","))
    assert cited<=available, cited-available
    return {"cited_keys":sorted(cited),"available_keys":sorted(available),"bibliography_sha256":sha(DOC/"references_editorial.bib")}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--reuse-displays",action="store_true")
    args=parser.parse_args()
    assert read_json(ROOT/"data/study_scope_editorial_2026-10-22.json")["manuscript_authoring_required"]
    verify_preservation()
    m=Materials();d=load_inputs(m)
    numeric_claims(m,d);further_numbers(m,d)
    if args.reuse_displays:
        prior=read_json(GEN/"display_provenance.json")
        m.figures=prior["figures"]
        for record in m.figures.values():
            for name,digest in record["files_sha256"].items(): assert sha(GEN/name)==digest
            for source in record["sources"]:m.source(ROOT/source)
    else:
        make_figures(m,d);supplementary_figures(m,d)
    make_tables(m,d)
    bibliography_review=bibliography(m)
    m.save()
    used=set()
    for path in list(DOC.glob("*.tex"))+list(GEN.glob("table_*.tex")):
        # Wrappers contain parameterized \val calls in definitions; inspect authored fragments and tables.
        if path.name.startswith("AMP_"):continue
        _,values=expanded_numbers(path.read_text(),m.numbers);used.update(values)
    abstract,_=expanded_numbers((DOC/"abstract.tex").read_text(),m.numbers)
    body,_=expanded_numbers((DOC/"main_text.tex").read_text(),m.numbers)
    limits=d["journal_sources"]["journal_limits"]
    counts={"abstract":word_count(abstract),"main_text_excluding_methods_references_legends":word_count(body)}
    assert limits["abstract_min_words"]<=counts["abstract"]<=limits["abstract_max_words"],counts
    assert counts["main_text_excluding_methods_references_legends"]<=limits["main_text_max_words"],counts
    assert "\\cite" not in abstract
    main_text=(DOC/"main_text.tex").read_text()
    result_section=main_text.split(r"\section*{Results}")[1].split(r"\section*{Discussion}")[0]
    assert result_section.count(r"\subsection*")==6
    displays=(DOC/"display_items.tex").read_text()
    display_count=displays.count(r"\begin{figure}")+displays.count(r"\input{generated/table_")
    assert display_count<=limits["main_display_items_max"]
    definitions="\n".join("\\newcommand{\\"+key+r"}[1]{\val{"+prefix+"#1}}" for key,prefix in PREFIXES.items())+"\n"
    main_wrapper=PREAMBLE+definitions+r"\begin{document}"+"\n"+r"{\LARGE\bfseries\raggedright\hyphenpenalty=10000 "+TITLE+r"\par}"+"\n"+r"\vspace{7pt}{\small Analysis | Retrospective in silico study}\par\vspace{10pt}"+"\n"+r"\begin{abstract}\input{abstract.tex}\end{abstract}"+"\n"+r"\input{main_text.tex}\input{methods.tex}"+"\n"+r"\bibliographystyle{unsrtnat}\bibliography{references_editorial}"+"\n"+r"\input{display_items.tex}\end{document}"+"\n"
    supplement_wrapper=PREAMBLE+definitions+r"\renewcommand{\thetable}{S\arabic{table}}\renewcommand{\thefigure}{S\arabic{figure}}"+"\n"+r"\begin{document}{\LARGE\bfseries Supplementary Information\par}\vspace{6pt}"+"\n"+r"{\large "+TITLE+r"\par}\vspace{12pt}"+"\n"+r"\input{supplement_text.tex}\input{supplement_displays.tex}\end{document}"+"\n"
    output=ROOT/"output/pdf";output.mkdir(parents=True,exist_ok=True)
    build=OUT/"manuscript_build";build.mkdir(parents=True,exist_ok=True)
    products=[]
    for short,content in [("manuscript",main_wrapper),("supplement",supplement_wrapper)]:
        stem=f"AMP_{short}_expanded_2026-10-20"
        wrapper=DOC/(stem+".tex");wrapper.write_text(content)
        work=build/short;work.mkdir(parents=True,exist_ok=True)
        result=subprocess.run(["tectonic","--keep-logs","--keep-intermediates","--outdir",str(work),str(wrapper)],cwd=DOC,text=True,capture_output=True)
        (work/"build_stdout.log").write_text(result.stdout+"\n"+result.stderr)
        if result.returncode:raise RuntimeError(result.stdout+"\n"+result.stderr)
        pdf=work/(stem+".pdf")
        text=subprocess.check_output(["pdftotext",str(pdf),"-"],text=True)
        assert "??" not in text and "[?]" not in text,"Unresolved reference in PDF"
        target=output/pdf.name;shutil.copyfile(pdf,target)
        (work/"extracted_text.txt").write_text(text)
        product={"path":str(target.relative_to(ROOT)),"sha256":sha(target),"bytes":target.stat().st_size}
        if args.render:
            render=ROOT/"tmp/pdfs/editorial_expanded_20261022"/short
            render.mkdir(parents=True,exist_ok=True)
            subprocess.run(["pdftoppm","-r","115","-png",str(target),str(render/"page")],check=True)
            from pypdf import PdfReader
            pages=len(PdfReader(target).pages)
            product.update({"pages":pages,"render_directory":str(render.relative_to(ROOT)),"visual_inspection":"pending"})
        products.append(product)
        print("Built "+str(target),flush=True)
    receipt={"created_utc":utc(),"products":products,"word_counts":counts,"main_result_sections":6,"main_display_items":display_count,
             "numeric_macros_defined":len(m.numbers),"numeric_macros_referenced_in_authored_fragments":len(used),
             "all_numeric_macros_resolved":True,"bibliography":bibliography_review,
             "numeric_provenance_sha256":sha(GEN/"numeric_provenance.json"),"display_provenance_sha256":sha(GEN/"display_provenance.json"),
             "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in DOC.glob("*.tex")},
             "interpretation":"English scientific manuscript and SI. Authorship, affiliations, declarations and public archive accession are not invented and remain submission metadata for the study team."}
    write_json(build/"build_receipt.json",receipt)


if __name__ == "__main__":main()
