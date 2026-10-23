"""Generate every manuscript number and display from audited data sources."""
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd

from .editorial_common import ROOT, BASE, OUT, sha, utc, read_json, write_json

DOC = ROOT/"manuscript/editorial_expanded_20261022"
GEN = DOC/"generated"
PROTOCOLS = ["random","sequence","homology80","homology50","publication","temporal"]
PROTOCOL_LABELS = ["Random","Sequence","Edit 0.80","Edit 0.50","Publication","Temporal"]
MODEL_LABELS = {"sequence_hgb":"Descriptor HGB", "sequence_context_hgb":"Descriptor HGB + context",
    "esm_hgb":"ESM HGB", "esm_context_hgb":"ESM HGB + context", "chem_hgb":"Morgan HGB", "chem_context_hgb":"Morgan HGB + context",
    "multimodal_mlp":"Multimodal MLP", "multimodal_film_context":"Multimodal FiLM",
    "target_mean":"Target mean", "context_ridge":"Context ridge", "sequence_ridge":"Descriptor ridge",
    "sequence_context_ridge":"Descriptor ridge + context", "esm_ridge":"ESM ridge", "esm_context_ridge":"ESM ridge + context",
    "chem_ridge":"Morgan ridge", "chem_context_ridge":"Morgan ridge + context"}
COMPONENTS = ["chemical_entity","publication","exact_target","chemical_by_target","chemical_by_medium","residual"]
COMP_LABELS = ["Chemical entity","Publication","Exact target","Chemical x target","Chemical x medium","Residual"]
FIT_LABELS = {"all_prior1.5":"All records", "all_prior3":"Broader prior", "exact_prior1.5":"Exact only",
    "multi_publication_prior1.5":"Shared chemicals", "no_reported_ranges_prior1.5":"No reported ranges"}
FIT_ORDER = ["all_prior1.5","exact_prior1.5","multi_publication_prior1.5","no_reported_ranges_prior1.5","all_prior3"]
PALETTE = ["#176B87","#C65C35","#6A52A3","#2B8876","#A78125","#87939C"]


def escape(text):
    return str(text).replace("&",r"\&").replace("%",r"\%").replace("_",r"\_").replace("#",r"\#")


class Materials:
    def __init__(self):
        self.sources, self.numbers, self.figures, self.tables = {}, {}, {}, {}
        self.preserved = read_json(OUT/"preservation_manifest.json")["files"]

    def source(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT/path
        relative = str(path.relative_to(ROOT))
        if relative not in self.sources:
            digest = sha(path)
            if relative in self.preserved:
                assert digest == self.preserved[relative]["sha256"]
            self.sources[relative] = digest
        return relative

    def json(self, path):
        self.source(path)
        return read_json(path)

    def csv(self, path, **kwargs):
        self.source(path)
        return pd.read_csv(path, low_memory=False, **kwargs)

    def number(self, key, value, sources, derivation, digits=3, integer=False, group_digits=True):
        paths = [self.source(s) for s in sources]
        if value is None or (isinstance(value, (float,np.floating)) and np.isnan(value)):
            rendered = "--"
            value = None
        elif np.isinf(float(value)):
            rendered = r"$\infty$"
            value = "infinity"
        else:
            rendered = (f"{int(value):,}" if group_digits else str(int(value))) if integer else f"{float(value):.{digits}f}"
            value = int(value) if integer else float(value)
        entry = {"value":value,"rendered":rendered,"sources":paths,"derivation":derivation}
        if key in self.numbers:
            assert self.numbers[key] == entry
        self.numbers[key] = entry
        return r"\val{"+key+"}"

    def text(self,key,value,sources,derivation):
        self.numbers[key]={"value":value,"rendered":escape(value),"sources":[self.source(p) for p in sources],"derivation":derivation}
        return r"\val{"+key+"}"

    def save(self):
        GEN.mkdir(parents=True, exist_ok=True)
        lines = [r"\newcommand{\val}[1]{\csname num@#1\endcsname}"]
        lines.extend(r"\expandafter\def\csname num@"+key+r"\endcsname{"+record["rendered"]+"}" for key,record in sorted(self.numbers.items()))
        (GEN/"numbers.tex").write_text("\n".join(lines)+"\n")
        write_json(GEN/"numeric_provenance.json", {"created_utc":utc(),"numbers":self.numbers,"source_sha256":self.sources})
        write_json(GEN/"display_provenance.json", {"figures":self.figures,"tables":self.tables})


def one(frame, **selectors):
    mask = np.ones(len(frame),bool)
    for column,value in selectors.items():
        mask &= frame[column].eq(value).to_numpy()
    result = frame.loc[mask]
    assert len(result) == 1, (selectors,len(result))
    return result.iloc[0]


def load_inputs(m):
    for analysis in ["assay_calibration","deep_contextual","posterior_tail_gaps"]:
        assert m.json(OUT/analysis/"completion.json")["passed"]
        assert m.json(OUT/analysis/"independent_audit.json")["passed"]
    d = {}
    d["manifest"] = m.json(ROOT/"data/curated/expanded/curation_manifest.json")
    d["support"] = m.json(BASE/"statistics/design_support.json")
    d["benchmark"] = m.csv(ROOT/"data/curated/expanded/benchmark_cells.csv",keep_default_na=False,na_values=[""])
    d["activities"] = m.csv(ROOT/"data/curated/expanded/activities.csv",keep_default_na=False,na_values=[""])
    d["exact_cells"] = m.csv(ROOT/"data/curated/expanded/all_exact_assay_cells.csv",keep_default_na=False,na_values=[""])
    d["base_metrics"] = m.csv(BASE/"benchmark/metrics.csv")
    d["base_pairs"] = m.csv(BASE/"benchmark/paired_context_improvements.csv")
    d["deep_metrics"] = m.csv(OUT/"deep_contextual/metrics.csv")
    d["deep_pairs"] = m.csv(OUT/"deep_contextual/paired_improvements.csv")
    d["cal_metrics"] = m.csv(OUT/"assay_calibration/metrics.csv")
    d["cal_pairs"] = m.csv(OUT/"assay_calibration/paired_improvements.csv")
    d["cal_support"] = m.csv(OUT/"assay_calibration/support_and_reference_availability.csv")
    d["tail"] = m.csv(OUT/"posterior_tail_gaps/tail_gap_summary.csv")
    d["tail_completion"] = m.json(OUT/"posterior_tail_gaps/completion.json")
    d["sum"] = m.json(BASE/"posterior_component_sum/exploratory_component_sum.json")
    d["sensitivity"] = m.json(BASE/"final_sensitivity_review_20261022.json")
    d["fits"], d["fit_paths"] = {}, {}
    for path in sorted((BASE/"censored").glob("*/*/result.json")):
        d["fits"][path.parent.parent.name] = m.json(path)
        d["fit_paths"][path.parent.parent.name] = path
    d["deep_config"] = m.json(OUT/"deep_contextual/locked_configuration.json")
    d["protocol"] = m.json(OUT/"locked_editorial_protocol.json")
    d["journal_sources"] = m.json(OUT/"method_and_journal_sources.json")
    return d


def numeric_claims(m,d):
    source = ROOT/"data/curated/expanded/curation_manifest.json"
    for field in ["cards","records","exact_assay_cells","benchmark_exact_assay_cells","eligible_exact_records","eligible_censored_records","exact_chemical_entities","exact_publications","exact_cells_without_backbone_id"]:
        m.number("catalog."+field,d["manifest"][field],[source],field,integer=True)
    source = BASE/"statistics/design_support.json"
    for field in ["bipartite_components","chemical_entities","publications","multi_publication_entities"]:
        m.number("support."+field,d["support"][field],[source],field,integer=True)
    m.number("support.single_pct",100*d["support"]["single_publication_fraction"],[source],"100 * single_publication_fraction",digits=1)
    fields = {"medium":"medium","inoculum":"cfu_raw","supplements":"salt_type_raw","ph":"ph_raw","ionic":"ionic_strength_raw"}
    missing = {}
    for label,field in fields.items():
        values=d["activities"][field].fillna("").astype(str).str.strip().str.casefold()
        missing[label]=100*values.isin(["","unreported","unknown"]).mean()
        m.number("missing."+label,missing[label],[ROOT/"data/curated/expanded/activities.csv"],f"Percent of all activity records with {field} empty, unreported or unknown",digits=1)
    d["missing"] = missing
    for fit,result in d["fits"].items():
        path=d["fit_paths"][fit]
        m.number("fit."+fit+".n",result["n_records"],[path],"n_records",integer=True)
        for kind in ["exact","right","interval","left"]:
            value=result["kind_counts"].get(kind,0)
            m.number("fit."+fit+"."+kind,value,[path],"kind_counts."+kind,integer=True)
        for component,values in result["conditional_variance_fractions"].items():
            for statistic,value in [("median",values["median"]),("low",values["ci95"][0]),("high",values["ci95"][1])]:
                m.number(f"variance.{fit}.{component}.{statistic}",100*value,[path],f"100 * conditional_variance_fractions.{component}.{statistic}",digits=1)
    rows=d["sum"]["specifications"]
    # The joint quantity is extracted from original draws, not a sum of marginal medians.
    joint=[]
    for row in rows:
        candidate = row["chemical_entity_plus_publication"]
        joint.append(candidate["median"])
    source=BASE/"posterior_component_sum/exploratory_component_sum.json"
    m.number("sum.minimum",100*min(joint),[source],"Minimum across five draw-wise sum posterior medians",digits=1)
    m.number("sum.maximum",100*max(joint),[source],"Maximum across five draw-wise sum posterior medians",digits=1)
    primary=d["fits"]["all_prior1.5"]["conditional_variance_fractions"]
    broader=d["fits"]["all_prior3"]["conditional_variance_fractions"]
    prior_change=max(abs(primary[c]["median"]-broader[c]["median"])*100 for c in COMPONENTS)
    m.number("prior.max_change",prior_change,[d["fit_paths"]["all_prior1.5"],d["fit_paths"]["all_prior3"]],"Maximum absolute median change in percentage points",digits=3)
    for which,table,path in [("base",d["base_metrics"],BASE/"benchmark/metrics.csv"),("deep",d["deep_metrics"],OUT/"deep_contextual/metrics.csv")]:
        for _,row in table.iterrows():
            prefix=f"{which}.{row.protocol}.{row.model}.{row.stratum}"
            for column in ["n","mae_log2","rmse_log2","r2","within_one_dilution","calibration_bias_log2"]:
                if column in row:
                    value=row[column]*100 if column == "within_one_dilution" else row[column]
                    m.number(prefix+"."+column,value,[path],f"{prefix}: {column}"+(" * 100" if column == "within_one_dilution" else ""),digits=1 if column=="within_one_dilution" else 3,integer=column=="n")
    original_gain=one(d["base_pairs"],protocol="temporal",stratum="all",base="sequence_hgb",alternative="sequence_context_hgb")
    for field in ["mae_reduction_log2","ci_low","ci_high"]:
        m.number("metadata.temporal."+field,original_gain[field],[BASE/"benchmark/paired_context_improvements.csv"],"Temporal descriptor context comparison: "+field)
    for _,row in d["cal_metrics"].iterrows():
        if row.model not in ["sequence_hgb","sequence_context_hgb","esm_hgb","esm_context_hgb"]:
            continue
        prefix=f"cal.{row.protocol}.{row.grouping}.{row.strategy}.{int(row.k)}.{row.branch}.{row.model}.{row.method}"
        for column in ["mae_log2","rmse_log2","r2","within_one_dilution","calibration_bias_log2","coverage90","coverage95","width90_log2","width95_log2","publication_coverage90","publication_coverage95","publications","anchor_measurements","eligible_blocks","expected_query_evaluations_per_draw"]:
            if column in row and pd.notna(row[column]):
                percent=column.startswith("coverage") or column.startswith("publication_coverage") or column=="within_one_dilution"
                m.number(prefix+"."+column,row[column]*(100 if percent else 1),[OUT/"assay_calibration/metrics.csv"],prefix+": "+column+(" * 100" if percent else ""),digits=1 if percent or column=="expected_query_evaluations_per_draw" else 3,integer=column in ["publications","anchor_measurements","eligible_blocks"])
    for _,row in d["cal_pairs"].iterrows():
        if row.calibration_method != "empirical_bayes":
            continue
        prefix=f"gain.{row.protocol}.{row.grouping}.{row.strategy}.{int(row.k)}.{row.branch}.{row.base_model}"
        for column in [c for c in row.index if c.endswith(("_log2","_ci_low","_ci_high"))]:
            m.number(prefix+"."+column,row[column],[OUT/"assay_calibration/paired_improvements.csv"],prefix+": "+column)
    temporal = d["benchmark"][d["benchmark"].publication_year.ge(2022)]
    split = m.csv(BASE/"benchmark/split_temporal_20261020_0.csv.gz")
    temporal = d["benchmark"].loc[split.role.eq("test").to_numpy()]
    for key,value in [("test_cells",len(temporal)),("test_publications",temporal.publication_id.nunique()),("start_year",temporal.publication_year.min()),("end_year",temporal.publication_year.max())]:
        m.number("temporal."+key,value,[ROOT/"data/curated/expanded/benchmark_cells.csv",BASE/"benchmark/split_temporal_20261020_0.csv.gz"],"Temporal test: "+key,integer=True,group_digits=not key.endswith("year"))
    cfg=OUT/"deep_contextual/locked_configuration.json"
    for key in ["max_epochs","patience","batch_size","learning_rate","weight_decay","dropout","validation_publication_fraction","esm_projection","morgan_projection","descriptor_projection","target_embedding","medium_embedding","hidden","hidden_layers","bootstrap_repeats"]:
        value=d["deep_config"][key]
        m.number("nn."+key,value,[cfg],key,digits=4 if value<.001 else 3,integer=isinstance(value,int))
    proto=OUT/"locked_editorial_protocol.json"
    for key,value in [("seed",d["protocol"]["few_shot"]["seed"]),("draws",d["protocol"]["few_shot"]["anchor_draws"]),("minimum_entities",d["protocol"]["few_shot"]["minimum_distinct_chemical_entities"]),("bootstrap",d["protocol"]["few_shot"]["bootstrap_publications"])]:
        m.number("fewshot."+key,value,[proto],"few_shot."+key,integer=True)
    d["joint_medians"] = joint


def further_numbers(m,d):
    bootstrap_source=ROOT/"amp_rebuild/assay_calibration.py"
    bounds=re.search(r"np\.quantile\(sampled, \[([^,]+),([^\]]+)\]\)",bootstrap_source.read_text())
    lower,upper=map(float,bounds.groups())
    m.number("bootstrap.confidence_pct",100*(upper-lower),[bootstrap_source],"100 times the difference between the paired-bootstrap percentile bounds",integer=True)
    source=OUT/"posterior_tail_gaps/tail_gap_summary.csv"
    for _,row in d["tail"].iterrows():
        for column in ["observed_q05","observed_q95","q05_median","q05_ci_low","q05_ci_high","q95_median","q95_ci_low","q95_ci_high",
                       "q95_gap_median_log2","q95_reduction_from_unselected_median_log2","fraction_positive_gap_removed_median",
                       "fraction_positive_gap_removed_ci_low","fraction_positive_gap_removed_ci_high","finite_limit_exact_records","exact_records_contradicted"]:
            if pd.notna(row[column]):
                percent=column.startswith("fraction_")
                m.number(f"tail.{row.fit}.{row.scenario}.{column}",row[column]*(100 if percent else 1),[source],f"{row.fit}, {row.scenario}: {column}"+(" * 100" if percent else ""),digits=1 if percent else 3,integer=column.endswith("records") or column=="exact_records_contradicted")
    for key in ["window_blocks","blocks_with_right_records","blocks_with_exact_above_maximum_right_limit","exact_records_in_contradicted_blocks","posterior_draws_per_fit"]:
        m.number("tail."+key,d["tail_completion"][key],[OUT/"posterior_tail_gaps/completion.json"],key,integer=True)
    for _,row in d["deep_pairs"].iterrows():
        if row.base != "multimodal_mlp":continue
        for column in ["mae_reduction_log2","ci_low","ci_high"]:
            m.number(f"film.{row.protocol}.{row.stratum}.{column}",row[column],[OUT/"deep_contextual/paired_improvements.csv"],f"FiLM minus MLP paired improvement, {row.protocol}/{row.stratum}: {column}")
    protocol=d["protocol"]["few_shot"]
    for k in protocol["k"]:m.number(f"anchors.{k}",k,[OUT/"locked_editorial_protocol.json"],"Locked anchor budget",integer=True)
    selected=one(d["cal_metrics"],protocol="temporal",grouping="publication_target",strategy="random",k=5,branch="point_frozen",model="sequence_hgb",method="empirical_bayes")
    baseline=one(d["cal_metrics"],protocol="temporal",grouping="publication_target",strategy="random",k=5,branch="point_frozen",model="sequence_hgb",method="uncalibrated")
    m.number("primary.relative_mae_reduction_pct",100*(baseline.mae_log2-selected.mae_log2)/baseline.mae_log2,[OUT/"assay_calibration/metrics.csv"],"100 * (uncalibrated MAE - EB MAE) / uncalibrated MAE on identical k=5 random queries",digits=1)
    support=one(d["cal_support"],protocol="temporal",fold=0,grouping="publication_target",strategy="random",k=5)
    for key in ["all_test_cells","all_test_publications","eligible_cells_before_anchors","eligible_publications","eligible_blocks","anchor_measurements"]:
        m.number("primary.support."+key,support[key],[OUT/"assay_calibration/support_and_reference_availability.csv"],"Temporal publication-target blocks, k=5 random: "+key,integer=True)
    m.number("primary.query_fraction_pct",100*selected.expected_query_evaluations_per_draw/support.all_test_cells,[OUT/"assay_calibration/metrics.csv",OUT/"assay_calibration/support_and_reference_availability.csv"],"Expected remaining queries / all temporal test cells * 100",digits=1)
    m.number("primary.eligible_fraction_pct",100*support.eligible_cells_before_anchors/support.all_test_cells,[OUT/"assay_calibration/support_and_reference_availability.csv"],"Eligible cells before anchor exclusion / all temporal test cells * 100",digits=1)
    m.number("primary.mean_queries",selected.expected_query_evaluations_per_draw,[OUT/"assay_calibration/metrics.csv"],"Expected remaining temporal queries per random draw",digits=1)
    intervals=one(d["cal_metrics"],protocol="temporal",grouping="publication_target",strategy="random",k=5,branch="interval_proper_fit",model="sequence_hgb",method="empirical_bayes")
    baseline_i=one(d["cal_metrics"],protocol="temporal",grouping="publication_target",strategy="random",k=5,branch="interval_proper_fit",model="sequence_hgb",method="uncalibrated")
    m.number("primary.interval_width_reduction_pct",100*(baseline_i.width90_log2-intervals.width90_log2)/baseline_i.width90_log2,[OUT/"assay_calibration/metrics.csv"],"90% full interval width reduction on the proper-fit branch, k=5 random",digits=1)
    m.number("primary.interval_halfwidth_fold",2**(intervals.width90_log2/2),[OUT/"assay_calibration/metrics.csv"],"2 raised to half the mean log2 interval width; geometric mean multiplicative half-width",digits=1)
    clusters=d["benchmark"].cluster_50.value_counts()
    m.number("homology.largest_component_cells",clusters.max(),[ROOT/"data/curated/expanded/benchmark_cells.csv"],"Largest EDIT50 transitive component in benchmark cells",integer=True)
    m.number("homology.largest_component_pct",100*clusters.max()/len(d["benchmark"]),[ROOT/"data/curated/expanded/benchmark_cells.csv"],"Largest EDIT50 component / benchmark cells * 100",digits=1)
    nnreceipt=m.json(OUT/"deep_contextual/completion.json")
    for key in ["splits","prediction_files","raw_predictions","unique_benchmark_cells"]:
        m.number("nn."+key,nnreceipt[key],[OUT/"deep_contextual/completion.json"],key,integer=True)
    base_receipt=m.json(BASE/"benchmark/summary_manifest.json")
    m.number("base.fits",base_receipt["prediction_files"],[BASE/"benchmark/summary_manifest.json"],"prediction_files",integer=True)
    m.number("base.configurations",d["base_metrics"].model.nunique(),[BASE/"benchmark/metrics.csv"],"Unique model configurations",integer=True)
    for key in ["chains","warmup_per_chain","draws_per_chain","prior_scale_log2"]:
        m.number("stan."+key,d["fits"]["all_prior1.5"][key],[d["fit_paths"]["all_prior1.5"]],key,integer=key!="prior_scale_log2",digits=1)
    m.number("stan.prior_broad",d["fits"]["all_prior3"]["prior_scale_log2"],[d["fit_paths"]["all_prior3"]],"prior_scale_log2",digits=1)
    m.number("stan.fits",len(d["fits"]),[*d["fit_paths"].values()],"Completed fitted specifications",integer=True)
    for name,pattern in [("alpha_mean",r"alpha ~ normal\(([^,]+),"),("alpha_sd",r"alpha ~ normal\([^,]+, ([^)]+)"),("sigma_sd",r"sigma ~ normal\(0, ([^)]+)")]:
        source=ROOT/"models/context_censored_sparse.stan";value=float(re.search(pattern,source.read_text()).group(1))
        m.number("stan."+name,value,[source],"Parsed Stan prior constant: "+name,integer=True)
    config_source=OUT/"deep_contextual/features/manifest.json";features=m.json(config_source)
    m.number("nn.descriptor_count",len(features["descriptor_names"]),[config_source],"descriptor_names length",integer=True)
    esm_source=ROOT/"data/features/esm2_t33_650M_UR50D/manifest.json"
    esm=m.json(esm_source)
    m.number("nn.esm_parameters_millions",int(re.search(r"_(\d+)M_",esm["model"]).group(1)),[esm_source],"Parameter scale parsed from frozen ESM model identifier",integer=True)
    # Coverage levels are recovered from verified interval output field names.
    for coverage in [90,95]:
        assert f"coverage{coverage}" in d["cal_metrics"]
        m.number(f"coverage.{coverage}",coverage,[OUT/"assay_calibration/metrics.csv"],"Nominal coverage encoded in audited interval column names",integer=True)
    environment=m.json(OUT/"deep_contextual/environment.json")
    for key in ["python","numpy","pandas","torch","platform","machine","device","workers","threads_per_worker"]:
        m.text("software."+key,str(environment[key]),[OUT/"deep_contextual/environment.json"],key)
    m.text("fewshot.seed",str(d["protocol"]["few_shot"]["seed"]),[OUT/"locked_editorial_protocol.json"],"Random-anchor master seed, ungrouped digits")
    m.text("nn.seeds",", ".join(map(str,d["deep_config"]["seeds"])),[OUT/"deep_contextual/locked_configuration.json"],"Frozen split seeds")
    m.text("snapshot.date",d["manifest"]["created_utc"][:10],[ROOT/"data/curated/expanded/curation_manifest.json"],"Date of curated expanded snapshot")
    raw=m.json(OUT/"deep_contextual/features/manifest.json")
    from .deep_contextual import ESM_END,MORGAN_END
    m.number("nn.esm_dimensions",ESM_END,[ROOT/"amp_rebuild/deep_contextual.py"],"ESM_END constant used by verified training code",integer=True)
    m.number("nn.morgan_dimensions",MORGAN_END-ESM_END,[ROOT/"amp_rebuild/deep_contextual.py"],"MORGAN_END minus ESM_END in verified training code",integer=True)
    review=m.json(OUT/"assay_calibration/independent_audit.json")
    m.number("audit.recomputed_publication_rows",sum(r["independent_publication_statistic_rows_recomputed"] for r in review["splits"]),[OUT/"assay_calibration/independent_audit.json"],"Sum of independently recomputed publication-statistic rows",integer=True)
    m.number("audit.recomputed_conformal_radii",sum(r["independent_conformal_radii_recomputed"] for r in review["splits"]),[OUT/"assay_calibration/independent_audit.json"],"Sum of independently recomputed conformal quantiles",integer=True)


def make_longtable(m,stem,headers,rows,caption,sources,widths=None):
    align="l"+"r"*(len(headers)-1) if widths is None else "".join(r">{\raggedright\arraybackslash}p{"+w+"}" for w in widths)
    head=" & ".join(headers)+r" \\"
    text=([r"\Needspace{14\baselineskip}"] if len(rows)<=8 else [])+[r"\begingroup\small\setlength{\tabcolsep}{4pt}",r"\begin{longtable}{"+align+"}",
          r"\caption{"+caption+r"}\label{tab:"+stem+r"}\\",r"\toprule",head,r"\midrule\endfirsthead",
          r"\multicolumn{"+str(len(headers))+r"}{l}{\small\itshape Continued from previous page}\\",r"\toprule",head,r"\midrule\endhead",
          r"\midrule\multicolumn{"+str(len(headers))+r"}{r}{\small\itshape Continued on next page}\\\endfoot",r"\bottomrule\endlastfoot"]
    text.extend(" & ".join(map(str,row))+r" \\" for row in rows)
    text.extend([r"\end{longtable}\endgroup",""])
    (GEN/f"{stem}.tex").write_text("\n".join(text))
    m.tables[stem]={"rows":len(rows),"sources":[m.source(s) for s in sources],"caption":caption,"sha256":sha(GEN/f"{stem}.tex")}


def make_tables(m,d):
    GEN.mkdir(parents=True,exist_ok=True)
    rows=[]
    for protocol in ["publication","temporal"]:
        for grouping in ["publication","publication_target"]:
            for k in [1,5]:
                prefix=f"cal.{protocol}.{grouping}.random.{k}.point_frozen.sequence_hgb."
                gain=f"gain.{protocol}.{grouping}.random.{k}.point_frozen.sequence_hgb."
                rows.append([("Publication" if protocol=="publication" else "Temporal")+r" /\newline "+("study" if grouping=="publication" else "study-target"),
                    r"\val{anchors."+str(k)+"}",r"\val{"+prefix+r"empirical_bayes.expected_query_evaluations_per_draw}",
                    r"\val{"+prefix+r"uncalibrated.mae_log2} $\to$ \val{"+prefix+r"empirical_bayes.mae_log2}",
                    r"\val{"+gain+r"calibration_gain_log2} [\val{"+gain+r"calibration_gain_ci_low}, \val{"+gain+r"calibration_gain_ci_high}]",
                    r"\val{"+prefix+r"empirical_bayes.anchor_measurements}"])
    make_longtable(m,"table_main_calibration",["Holdout / block","$k$","Mean queries","MAE before / after","Reduction [CI]","Anchors"],rows,
        "Calibration benefit and measurement budget. Random anchors, descriptor HGB and empirical Bayes offsets; paired queries exclude every cell of revealed chemical entities. Intervals are publication-bootstrap confidence intervals, not prediction intervals.",
        [OUT/"assay_calibration/metrics.csv",OUT/"assay_calibration/paired_improvements.csv"],widths=["3.1cm",".35cm","1.7cm","2.8cm","3.5cm","1.4cm"])
    rows=[]
    for fit in FIT_ORDER:
        result=d["fits"][fit];prefix="fit."+fit+"."
        rows.append([FIT_LABELS[fit],r"\val{"+prefix+"n}",*[r"\val{"+prefix+k+"}" for k in ["exact","right","interval","left"]]])
    make_longtable(m,"table_s_cohorts",["Posterior specification","Records","Exact","Right","Range","Left"],rows,
        "Observation cohorts in the five retained posterior fits. Repeated records are not collapsed for the censored likelihood.",list(d["fit_paths"].values()))
    rows=[]
    for fit in FIT_ORDER:
        for component,label in zip(COMPONENTS,COMP_LABELS):
            prefix=f"variance.{fit}.{component}."
            rows.append([FIT_LABELS[fit],label,r"\val{"+prefix+"median}",r"[\val{"+prefix+r"low}, \val{"+prefix+"high}]"])
    make_longtable(m,"table_s_variance",["Specification","Component","Median (\%)","Posterior interval (\%)"],rows,
        "Conditional variance fractions. The denominator contains random-effect and residual variances and excludes fixed-effect variation.",list(d["fit_paths"].values()))
    rows=[]
    for fit in FIT_ORDER:
        r=d["fits"][fit];path=d["fit_paths"][fit];precision=m.json(path.parent/"sampling_precision.json")
        vals=[]
        for key,value,integer in [("rhat",r["diagnostics"]["max_rhat_all_parameters"],False),("bulk",r["diagnostics"]["min_bulk_ess_scientific_parameters"],False),
            ("tail",precision["min_tail_ess_scientific_parameters"],False),("divergences",r["diagnostics"]["divergences"],True),("bfmi",min(precision["e_bfmi_by_chain"]),False)]:
            vals.append(m.number("diagnostic."+fit+"."+key,value,[path,path.parent/"sampling_precision.json"],key,digits=3,integer=integer))
        rows.append([FIT_LABELS[fit],*vals])
    make_longtable(m,"table_s_diagnostics",["Specification","Max. $\widehat R$","Min. bulk ESS","Min. tail ESS","Divergences","Min. BFMI"],rows,
        "Sampling diagnostics. Numerical convergence does not establish observation-model adequacy.",list(d["fit_paths"].values()),widths=["3.3cm","1.8cm","2.0cm","2.0cm","1.8cm","1.7cm"])
    rows=[]
    for protocol,label in zip(PROTOCOLS,PROTOCOL_LABELS):
        for which,frame in [("base",d["base_metrics"]),("deep",d["deep_metrics"])]:
            for _,row in frame[(frame.protocol==protocol)&(frame.stratum=="all")].sort_values("model").iterrows():
                prefix=f"{which}.{protocol}.{row.model}.all."
                rows.append([label,MODEL_LABELS[row.model],*[r"\val{"+prefix+c+"}" for c in ["n","mae_log2","rmse_log2","r2","within_one_dilution"]]])
    make_longtable(m,"table_s_models",["Protocol","Model","Cells","MAE","RMSE","$R^2$","Within one (\%)"],rows,
        "Complete aggregate prediction results. Repeated held-out losses are averaged within cells; joint strata are reported separately in the machine-readable tables.",[BASE/"benchmark/metrics.csv",OUT/"deep_contextual/metrics.csv"],widths=["1.8cm","4.1cm","1.2cm","1.2cm","1.2cm","1.1cm","2.0cm"])
    rows=[]
    for _,row in d["deep_pairs"].query("base == 'multimodal_mlp'").iterrows():
        prefix=f"film.{row.protocol}.{row.stratum}."
        stratum = row.stratum.replace("__"," / ").replace("_"," ").replace("sequence","seq.").replace("publication","pub.")
        rows.append([row.protocol,escape(stratum),r"\val{"+prefix+r"mae_reduction_log2}",r"[\val{"+prefix+r"ci_low}, \val{"+prefix+r"ci_high}]"])
    make_longtable(m,"table_s_film",["Protocol","Test stratum","MAE reduction","Paired CI"],rows,
        "Context modulation versus the same multimodal backbone. Positive differences favor FiLM. Intervals use crossed publication and sequence-alias bootstrap weights.",[OUT/"deep_contextual/paired_improvements.csv"],widths=["2.0cm","6.3cm","2.1cm","3.4cm"])
    rows=[]
    frame=d["cal_pairs"].query("branch == 'point_frozen' and calibration_method == 'empirical_bayes'")
    for _,row in frame.iterrows():
        prefix=f"gain.{row.protocol}.{row.grouping}.{row.strategy}.{int(row.k)}.point_frozen.{row.base_model}."
        rows.append(["Pub." if row.protocol=="publication" else "Temp.","Study" if row.grouping=="publication" else "Study-target",row.strategy,MODEL_LABELS[row.base_model],r"\val{anchors."+str(int(row.k))+"}",
            r"\val{"+prefix+r"calibration_gain_log2} [\val{"+prefix+r"calibration_gain_ci_low}, \val{"+prefix+r"calibration_gain_ci_high}]",
            r"\val{"+prefix+r"metadata_gain_same_queries_log2}"])
    make_longtable(m,"table_s_calibration",["Protocol","Block","Anchors","Predictor","$k$","EB gain [CI]","Metadata gain"],rows,
        "All prespecified point-accuracy comparisons for empirical Bayes calibration. Pub.: publication holdout; Temp.: temporal holdout. Metadata gains use the identical unexposed queries within each setting.",[OUT/"assay_calibration/paired_improvements.csv"],widths=["1.5cm","1.6cm","1.5cm","2.2cm",".35cm","3.4cm","1.8cm"])
    rows=[]
    frame=d["cal_metrics"].query("branch == 'interval_proper_fit' and strategy == 'random' and model == 'sequence_hgb' and method != 'unshrunk_anchor_mean'")
    for _,row in frame.iterrows():
        prefix=f"cal.{row.protocol}.{row.grouping}.{row.strategy}.{int(row.k)}.{row.branch}.{row.model}.{row.method}."
        rows.append([row.protocol,"Study" if row.grouping=="publication" else "Study-target",r"\val{anchors."+str(int(row.k))+"}","EB" if row.method=="empirical_bayes" else "Base",
                     *[r"\val{"+prefix+c+"}" for c in ["width90_log2","coverage90","publication_coverage90","width95_log2","coverage95","publication_coverage95"]]])
    make_longtable(m,"table_s_intervals",["Protocol","Block","$k$","Method",r"Width \val{coverage.90}",r"Cell cov. \val{coverage.90}",r"Pub. cov. \val{coverage.90}",r"Width \val{coverage.95}",r"Cell cov. \val{coverage.95}",r"Pub. cov. \val{coverage.95}"],rows,
        "Conformal publication-maximum intervals on the separate proper-fit branch. Widths are full log-scale widths; coverages are percentages. Publication coverage requires every remaining query in that publication to be covered.",[OUT/"assay_calibration/metrics.csv"],widths=["1.9cm","1.8cm",".4cm","1.3cm","1.25cm","1.25cm","1.25cm","1.25cm","1.25cm","1.25cm"])
    rows=[]
    names={"gaussian_unselected":"Gaussian, unselected","unique_right_limit_compatible":"Unique compatible limit","maximum_right_limit_compatible":"Maximum compatible limit",**{f"compatible_support_envelope_plus_{i}":f"Support envelope + {i}" for i in [0,1,2]},**{f"counterfactual_common_right_q{i}":f"Common limit, right q{i}" for i in [50,75,90]},**{f"student_t_{i}_variance_matched_unselected":f"Unrefitted t residual, df={i}" for i in [3,5,10]}}
    for fit in FIT_ORDER:
        for _,row in d["tail"][d["tail"].fit==fit].iterrows():
            prefix=f"tail.{fit}.{row.scenario}."
            share=r"\val{"+prefix+"fraction_positive_gap_removed_median}" if row.positive_gap_fraction_defined else "Undefined"
            rows.append([FIT_LABELS[fit],names[row.scenario],r"\val{"+prefix+r"q95_median} [\val{"+prefix+r"q95_ci_low}, \val{"+prefix+r"q95_ci_high}]",share])
    make_longtable(m,"table_s_tails",["Fit","Sensitivity scenario","Replicated q95 [interval]","Gap removed (\%)"],rows,
        "Assumption-dependent tail sensitivities. Percentages are unbounded fractions of a positive discrepancy removed, not estimated causal contributions. Fractions are undefined for the negative exact-only discrepancy.",[OUT/"posterior_tail_gaps/tail_gap_summary.csv"],widths=["3.1cm","4.0cm","3.8cm","2.3cm"])
    rows=[]
    for fit in FIT_ORDER:
        for scenario in ["gaussian_unselected","unique_right_limit_compatible","maximum_right_limit_compatible"]:
            prefix=f"tail.{fit}.{scenario}."
            compact_name=names[scenario].replace(" compatible limit"," compatible")
            rows.append([FIT_LABELS[fit],compact_name,r"\val{"+prefix+r"observed_q05}",r"\val{"+prefix+r"q05_median} [\val{"+prefix+r"q05_ci_low}, \val{"+prefix+r"q05_ci_high}]"])
    make_longtable(m,"table_s_lower_tails",["Fit","Sensitivity scenario","Observed q05","Replicated q05 [interval]"],rows,
        "Lower-tail checks under the same upper-window assumptions. These scenarios do not model lower observation limits and do not establish adequate lower-tail fit; all original adequacy flags are retained.",[OUT/"posterior_tail_gaps/tail_gap_summary.csv"],widths=["3.1cm","4.0cm","2.3cm","3.8cm"])


def save_figure(m,fig,stem,frame,sources,interpretation):
    GEN.mkdir(parents=True,exist_ok=True)
    for suffix in ["pdf","png"]:
        options={"metadata":{"CreationDate":None,"ModDate":None}} if suffix == "pdf" else {"dpi":300}
        fig.savefig(GEN/f"{stem}.{suffix}",bbox_inches="tight",**options)
    plt.close(fig)
    frame.to_csv(GEN/f"{stem}_data.csv",index=False)
    m.figures[stem]={"sources":[m.source(p) for p in sources],"interpretation":interpretation,
        "files_sha256":{f"{stem}.{suffix}":sha(GEN/f"{stem}.{suffix}") for suffix in ["pdf","png"]},
        "plotted_data_sha256":sha(GEN/f"{stem}_data.csv")}


def make_figures(m,d):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"axes.titlesize":10,"legend.fontsize":7.5})
    fig,axes=plt.subplots(1,3,figsize=(10.6,3.25))
    kinds=["exact","right","interval","left","summary_mean","unparsed"]
    counts=[d["manifest"]["parse_kinds"][k] for k in kinds]
    axes[0].barh(range(len(kinds)),counts,color=PALETTE[0]);axes[0].invert_yaxis()
    axes[0].set_yticks(range(len(kinds)),["Exact","Right-censored","Reported range","Left-censored","Mean with error","Unparsed"])
    axes[0].set_xlabel("Original activity records");axes[0].set_title("a  Observation formats",loc="left")
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=3,integer=True))
    axes[0].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    labels=list(d["missing"]);values=list(d["missing"].values())
    axes[1].barh(range(len(labels)),values,color=PALETTE[1]);axes[1].invert_yaxis()
    axes[1].set_yticks(range(len(labels)),["Medium","Inoculum","Supplements","pH","Ionic quantity"])
    axes[1].set_xlim(0,115);axes[1].set_xlabel("Unreported records (%)");axes[1].set_title("b  Recorded context",loc="left")
    for i,v in enumerate(values): axes[1].text(v+1,i,f"{v:.1f}",va="center",fontsize=8)
    degrees=d["exact_cells"].groupby("chemical_id").publication_id.nunique()
    bins=[int((degrees==1).sum()),int((degrees==2).sum()),int((degrees>=3).sum())]
    axes[2].bar(["One","Two","Three or more"],bins,color=[PALETTE[1],PALETTE[0],PALETTE[3]])
    axes[2].set_xlabel("Publications per chemical entity");axes[2].set_ylabel("Chemical entities")
    axes[2].set_title("c  Cross-publication support",loc="left");axes[2].set_ylim(0,max(bins)*1.16)
    for i,v in enumerate(bins): axes[2].text(i,v+max(bins)*.025,f"{v:,}",ha="center",fontsize=8)
    fig.tight_layout(w_pad=2.0)
    frame=pd.DataFrame([{"panel":"formats","category":k,"value":v} for k,v in zip(kinds,counts)]+
        [{"panel":"missingness","category":k,"value":v} for k,v in d["missing"].items()]+
        [{"panel":"publications_per_chemical","category":str(i),"value":v} for i,v in zip([1,2,3],bins)])
    save_figure(m,fig,"figure_1_catalog",frame,[ROOT/"data/curated/expanded/curation_manifest.json",ROOT/"data/curated/expanded/activities.csv",ROOT/"data/curated/expanded/all_exact_assay_cells.csv"],"Formats and missingness use all source records; connectivity uses the exact-label inferential cohort.")

    fig,axes=plt.subplots(1,2,figsize=(9.4,3.35),gridspec_kw={"width_ratios":[1.4,1]})
    records=[]
    for i,fit in enumerate(FIT_ORDER):
        for offset,c,color in [(-.12,"chemical_entity",PALETTE[0]),(.12,"publication",PALETTE[1])]:
            r=d["fits"][fit]["conditional_variance_fractions"][c];lo,hi=np.array(r["ci95"])*100;mid=r["median"]*100
            axes[0].errorbar(mid,i+offset,xerr=[[mid-lo],[hi-mid]],fmt="o",color=color,ms=4,capsize=2,label=COMP_LABELS[COMPONENTS.index(c)] if i==0 else None)
            records.append({"fit":fit,"component":c,"median":mid,"low":lo,"high":hi})
        result=d["fits"][fit]
        sr=next(r for r in d["sum"]["specifications"] if r["subset"]==result["subset"] and r["prior_scale"]==result["prior_scale_log2"])["chemical_entity_plus_publication"]
        mid=sr["median"]*100;lo,hi=np.array(sr["ci95"])*100
        axes[1].errorbar(mid,i,xerr=[[mid-lo],[hi-mid]],fmt="o",color=PALETTE[2],ms=4,capsize=2)
        records.append({"fit":fit,"component":"chemical_plus_publication","median":mid,"low":lo,"high":hi})
    axes[0].set_yticks(range(len(FIT_ORDER)),[FIT_LABELS[f] for f in FIT_ORDER]);axes[0].invert_yaxis()
    axes[1].set_yticks(range(len(FIT_ORDER)),[]);axes[1].invert_yaxis()
    axes[0].set_title("a  Allocation changes with support",loc="left");axes[1].set_title("b  Exploratory joint-draw sum",loc="left")
    axes[0].legend(loc="lower right",frameon=False)
    for ax in axes: ax.set_xlabel("Conditional variance fraction (%)");ax.grid(axis="x",alpha=.15)
    fig.tight_layout(w_pad=2)
    save_figure(m,fig,"figure_2_variance",pd.DataFrame(records),[*d["fit_paths"].values(),BASE/"posterior_component_sum/exploratory_component_sum.json"],"Posterior medians and 95% intervals. Fractions exclude fixed effects; cohort and likelihood change across fits. The sum is exploratory, not a causal invariance.")

    all_metrics=pd.concat([d["base_metrics"],d["deep_metrics"]],ignore_index=True)
    selected=["sequence_hgb","sequence_context_hgb","esm_context_hgb","multimodal_mlp","multimodal_film_context"]
    fig,axes=plt.subplots(1,2,figsize=(10,4.1))
    for name,color in zip(selected,["#939CA5",PALETTE[1],PALETTE[3],PALETTE[0],PALETTE[2]]):
        rows=all_metrics[(all_metrics.model==name)&(all_metrics.stratum=="all")].set_index("protocol").loc[PROTOCOLS]
        for ax,col in zip(axes,["mae_log2","r2"]):
            ax.plot(range(len(PROTOCOLS)),rows[col],marker="o",ms=4,lw=1.3,color=color,label=MODEL_LABELS[name])
            ax.set_xticks(range(len(PROTOCOLS)),PROTOCOL_LABELS,rotation=28,ha="right");ax.grid(axis="y",alpha=.15)
    axes[0].set_title("a  Prediction error",loc="left");axes[0].set_ylabel("MAE (log2 MIC)")
    axes[1].set_title("b  Generalization across settings",loc="left");axes[1].set_ylabel("R squared");axes[1].axhline(0,color="#888",lw=.7)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,-.015))
    fig.tight_layout(rect=[0,.13,1,1])
    save_figure(m,fig,"figure_3_generalization",all_metrics[(all_metrics.model.isin(selected))&(all_metrics.stratum=="all")],[BASE/"benchmark/metrics.csv",OUT/"deep_contextual/metrics.csv"],"Mean held-out losses, not errors of averaged predictions. Cross-protocol differences are descriptive; training composition and temporal test population differ.")

    fig,axes=plt.subplots(2,2,figsize=(9.5,6.2),sharex=True,sharey=True)
    chart=d["cal_pairs"].query("branch == 'point_frozen' and base_model == 'sequence_hgb' and calibration_method == 'empirical_bayes'")
    for r,grouping in enumerate(["publication","publication_target"]):
        for c,protocol in enumerate(["publication","temporal"]):
            ax=axes[r,c]
            for strategy,color,marker in [("random",PALETTE[0],"o"),("standard",PALETTE[1],"s")]:
                q=chart[(chart.grouping==grouping)&(chart.protocol==protocol)&(chart.strategy==strategy)].sort_values("k")
                mid=q.calibration_gain_log2.to_numpy();lo=q.calibration_gain_ci_low.to_numpy();hi=q.calibration_gain_ci_high.to_numpy()
                ax.errorbar(q.k,mid,yerr=[mid-lo,hi-mid],fmt=marker+"-",color=color,ms=4,capsize=3,label=strategy.capitalize()+" anchors")
                if strategy=="random":
                    ax.plot(q.k,q.metadata_gain_same_queries_log2,"^--",color="#7B848C",ms=4,label="Recorded metadata, same queries")
            ax.axhline(0,color="#888",lw=.7);ax.set_xticks([1,2,3,5]);ax.grid(axis="y",alpha=.15)
            ax.set_title(("abcd"[r*2+c])+"  "+("Publication holdout" if protocol=="publication" else "Temporal holdout")+"\n"+("Publication blocks" if grouping=="publication" else "Publication x target blocks"),loc="left",fontsize=9)
            if c==0:ax.set_ylabel("Paired MAE reduction (log2)")
            if r==1:ax.set_xlabel("Reference chemical entities per block")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,-.015))
    fig.tight_layout(rect=[0,.06,1,1],h_pad=2)
    save_figure(m,fig,"figure_4_calibration",chart,[OUT/"assay_calibration/paired_improvements.csv"],"EB offsets for descriptor HGB. Confidence intervals cluster by publication. Metadata gains use exactly the same query cells; random and standard policies can leave different query sets.")

    fig,axes=plt.subplots(1,2,figsize=(10.3,3.8),gridspec_kw={"width_ratios":[1.4,1]})
    scenarios=["gaussian_unselected","unique_right_limit_compatible","maximum_right_limit_compatible","compatible_support_envelope_plus_1","student_t_3_variance_matched_unselected"]
    labels=["Unselected Gaussian","Compatible unique right limit","Compatible maximum right limit","Support envelope + one dilution","Student-t residual (df = "+scenarios[-1].split("_")[2]+")"]
    primary=d["tail"].query("fit == 'all_prior1.5'").set_index("scenario").loc[scenarios]
    mid=primary.q95_median.to_numpy();lo=primary.q95_ci_low.to_numpy();hi=primary.q95_ci_high.to_numpy()
    axes[0].errorbar(mid,range(len(mid)),xerr=[mid-lo,hi-mid],fmt="o",color=PALETTE[0],capsize=3)
    axes[0].axvline(primary.observed_q95.iloc[0],color=PALETTE[1],ls="--",label="Observed exact-record q95")
    axes[0].set_yticks(range(len(mid)),labels);axes[0].invert_yaxis();axes[0].set_xlabel("Replicated q95 (log2 MIC)")
    axes[0].set_title("a  Assumption-dependent tail checks",loc="left");axes[0].legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.19),fontsize=7)
    selected_scenarios=scenarios[:3]
    for offset,scenario,color in zip([-.17,0,.17],selected_scenarios,[PALETTE[1],PALETTE[0],PALETTE[3]]):
        q=d["tail"][d["tail"].scenario==scenario].set_index("fit").loc[FIT_ORDER]
        axes[1].scatter(q.q95_gap_median_log2,np.arange(len(FIT_ORDER))+offset,s=20,color=color,
            label={"gaussian_unselected":"Unselected","unique_right_limit_compatible":"Unique limit","maximum_right_limit_compatible":"Maximum limit"}[scenario])
    axes[1].axvline(0,color="#888",lw=.8);axes[1].set_yticks(range(len(FIT_ORDER)),[FIT_LABELS[f] for f in FIT_ORDER]);axes[1].invert_yaxis()
    axes[1].set_xlabel("Median q95 gap from observed (log2)");axes[1].set_title("b  Dependence on posterior fit",loc="left")
    axes[1].legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.19),fontsize=7,ncol=3)
    fig.tight_layout(w_pad=2.5)
    save_figure(m,fig,"figure_5_observation_windows",d["tail"],[OUT/"posterior_tail_gaps/tail_gap_summary.csv"],"Conditional sensitivity, not an identified causal allocation. Red/orange: unselected; blue: unique compatible limit; green: maximum compatible limit. The Student-t perturbation is not refitted.")


def supplementary_figures(m,d):
    all_metrics=pd.concat([d["base_metrics"],d["deep_metrics"]],ignore_index=True)
    matrix=all_metrics[all_metrics.stratum.eq("all")].pivot(index="model",columns="protocol",values="mae_log2").loc[list(MODEL_LABELS),PROTOCOLS]
    fig,ax=plt.subplots(figsize=(8.6,6.8));im=ax.imshow(matrix.to_numpy(),aspect="auto",cmap="viridis_r")
    ax.set_yticks(range(len(matrix)),[MODEL_LABELS[n] for n in matrix.index]);ax.set_xticks(range(len(PROTOCOLS)),PROTOCOL_LABELS,rotation=25,ha="right")
    for i in range(len(matrix)):
        for j in range(len(PROTOCOLS)):
            value=matrix.iloc[i,j]
            rgb=np.array(im.cmap(im.norm(value))[:3]);linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
            luminance=linear@np.array([.2126,.7152,.0722])
            ax.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=7,color="white" if luminance<.179 else "#15232D")
    fig.colorbar(im,ax=ax,label="MAE (log2 MIC)",fraction=.045,pad=.03);fig.tight_layout()
    save_figure(m,fig,"figure_s1_all_models",matrix.reset_index(),[BASE/"benchmark/metrics.csv",OUT/"deep_contextual/metrics.csv"],"All sixteen model configurations on the same saved protocol assignments.")
    selected=["sequence_hgb","esm_context_hgb","multimodal_mlp","multimodal_film_context"]
    joint=all_metrics[(all_metrics.protocol=="joint") & all_metrics.model.isin(selected)]
    strata=sorted(joint.stratum.unique())
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.3))
    for j,name in enumerate(selected):
        q=joint[joint.model==name].set_index("stratum").loc[strata]
        for ax,col in zip(axes,["mae_log2","r2"]):
            colors={"sequence_hgb":"#939CA5","esm_context_hgb":PALETTE[3],"multimodal_mlp":PALETTE[0],"multimodal_film_context":PALETTE[2]}
            ax.plot(np.arange(len(strata)),q[col],"o-",ms=4,color=colors[name],label=MODEL_LABELS[name])
            labels=[s.replace("known_sequence","Known seq.").replace("new_sequence","New seq.").replace("known_publication","Known pub.").replace("new_publication","New pub.").replace("_","\n") for s in strata]
            ax.set_xticks(range(len(strata)),labels,rotation=0,ha="center",fontsize=8)
            ax.grid(axis="y",alpha=.15)
    axes[0].set_ylabel("MAE (log2 MIC)");axes[1].set_ylabel("R squared")
    axes[0].set_title("a  Joint sequence and publication transfer",loc="left");axes[1].set_title("b  Explained variation",loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,-.025));fig.tight_layout(rect=[0,.18,1,1])
    save_figure(m,fig,"figure_s2_joint",joint,[BASE/"benchmark/metrics.csv",OUT/"deep_contextual/metrics.csv"],"Joint strata retain separate training-exposure targets and are not pooled.")
    frame=d["cal_pairs"].query("branch == 'point_frozen' and strategy == 'random' and calibration_method == 'empirical_bayes'")
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.7),sharey=True)
    for ax,protocol in zip(axes,["publication","temporal"]):
        for j,(model,grouping) in enumerate([(a,b) for a in ["sequence_hgb","esm_hgb"] for b in ["publication","publication_target"]]):
            q=frame[(frame.protocol==protocol)&(frame.base_model==model)&(frame.grouping==grouping)].sort_values("k")
            y=q.calibration_gain_log2.to_numpy();lo=q.calibration_gain_ci_low.to_numpy();hi=q.calibration_gain_ci_high.to_numpy()
            ax.errorbar(q.k,y,yerr=[y-lo,hi-y],fmt="o-",ms=4,capsize=2,color=PALETTE[j],label=MODEL_LABELS[model]+(" / study" if grouping=="publication" else " / study-target"))
        ax.set_title(protocol.capitalize()+" holdout",loc="left");ax.set_xlabel("Reference entities per block");ax.set_xticks([1,2,3,5]);ax.axhline(0,color="#888",lw=.7)
    axes[0].set_ylabel("Paired MAE reduction (log2)");fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,-.025));fig.tight_layout(rect=[0,.19,1,1])
    save_figure(m,fig,"figure_s3_embedding_calibration",frame,[OUT/"assay_calibration/paired_improvements.csv"],"Replication of calibration gains with frozen ESM-based HGB predictions; pointwise publication bootstrap intervals.")
    frame=d["cal_metrics"].query("branch == 'interval_proper_fit' and strategy == 'random' and model == 'sequence_hgb' and grouping == 'publication_target' and method != 'unshrunk_anchor_mean'")
    fig,axes=plt.subplots(2,2,figsize=(9.4,5.8),sharex=True)
    for c,protocol in enumerate(["publication","temporal"]):
        for method,color,ls in [("uncalibrated",PALETTE[1],"--"),("empirical_bayes",PALETTE[0],"-")]:
            q=frame[(frame.protocol==protocol)&(frame.method==method)].sort_values("k")
            axes[0,c].plot(q.k,q.width90_log2,"o"+ls,color=color,label=method.replace("_"," "))
            axes[1,c].plot(q.k,q.publication_coverage90*100,"o"+ls,color=color,label=method.replace("_"," "))
        axes[0,c].set_title(protocol.capitalize()+" holdout",loc="left");axes[1,c].axhline(90,color="#777",ls=":",lw=.9)
        axes[1,c].set_xlabel("Reference entities per study-target block")
        for r in range(2):axes[r,c].set_xticks([1,2,3,5]);axes[r,c].grid(axis="y",alpha=.15)
    axes[0,0].set_ylabel("90% full interval width (log2)");axes[1,0].set_ylabel("Whole-publication coverage (%)")
    axes[0,0].legend(frameon=False);fig.tight_layout()
    save_figure(m,fig,"figure_s4_interval_tradeoff",frame,[OUT/"assay_calibration/metrics.csv"],"Publication maxima give conservative cell coverage and wide intervals. Width and coverage use the same proper-fit predictor; the dotted line is nominal publication coverage.")
    frame=d["cal_metrics"].query("branch == 'point_frozen' and strategy == 'random' and model == 'sequence_hgb' and grouping == 'publication_target'")
    fig,axes=plt.subplots(1,2,figsize=(9.0,3.5),sharey=True)
    for ax,protocol in zip(axes,["publication","temporal"]):
        for method,color in zip(["uncalibrated","unshrunk_anchor_mean","empirical_bayes"],["#8A939A",PALETTE[1],PALETTE[0]]):
            q=frame[(frame.protocol==protocol)&(frame.method==method)].sort_values("k")
            ax.plot(q.k,q.mae_log2,"o-",ms=4,color=color,label=method.replace("_"," "))
        ax.set_xticks([1,2,3,5]);ax.set_xlabel("Reference entities per study-target block");ax.set_title(protocol.capitalize()+" holdout",loc="left");ax.grid(axis="y",alpha=.15)
    axes[0].set_ylabel("MAE on remaining queries (log2)");fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,-.025));fig.tight_layout(rect=[0,.13,1,1])
    save_figure(m,fig,"figure_s5_shrinkage",frame,[OUT/"assay_calibration/metrics.csv"],"Unshrunk offsets can amplify reference noise when only one entity is observed. Query composition changes with k; within-k method differences remain paired.")
    records=[]
    for receipt in sorted((OUT/"deep_contextual/fits").glob("*/*.json")):
        if receipt.name not in ["multimodal_mlp.json","multimodal_film_context.json"]:continue
        r=m.json(receipt);records.append({k:r[k] for k in ["protocol","seed","fold","model","selected_epochs","early_stopping_epochs_run"]})
    frame=pd.DataFrame(records)
    fig,ax=plt.subplots(figsize=(9.4,3.6))
    protocols=[*PROTOCOLS[:-1],"joint","temporal"]
    for j,model in enumerate(["multimodal_mlp","multimodal_film_context"]):
        boxes=[frame[(frame.model==model)&(frame.protocol==p)].selected_epochs.to_numpy() for p in protocols]
        bp=ax.boxplot(boxes,positions=np.arange(len(protocols))+(j-.5)*.28,widths=.23,patch_artist=True,manage_ticks=False)
        for box in bp["boxes"]:box.set_facecolor(PALETTE[j*2]);box.set_alpha(.75)
        ax.scatter([],[],color=PALETTE[j*2],label=MODEL_LABELS[model])
    ax.set_xticks(range(len(protocols)),[p.capitalize() for p in protocols]);ax.set_ylabel("Epochs selected by inner publication validation")
    ax.legend(frameon=False);ax.grid(axis="y",alpha=.15);fig.tight_layout()
    save_figure(m,fig,"figure_s6_training",frame,list((OUT/"deep_contextual/fits").glob("*/multimodal_mlp.json"))+list((OUT/"deep_contextual/fits").glob("*/multimodal_film_context.json")),"Epoch selection uses inner training publications only. The complete model is then refitted for the selected number of epochs.")
