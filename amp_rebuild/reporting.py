"""Generate research tables and figures from verified results, respecting author scope."""
from datetime import datetime, timezone
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd

from .curation import ROOT

PROTOCOLS = ["random", "sequence", "homology80", "homology50", "publication", "temporal"]
PROTOCOL_LABELS = ["Random", "Sequence", "Edit 0.80", "Edit 0.50", "Publication", "Temporal"]
BASES = ["sequence_ridge", "esm_ridge", "chem_ridge", "sequence_hgb", "esm_hgb", "chem_hgb"]
MODEL_LABELS = {
    "target_mean": "Target mean", "context_ridge": "Context ridge",
    "sequence_ridge": "Descriptors / ridge", "sequence_context_ridge": "Descriptors + context / ridge",
    "esm_ridge": "ESM-2 / ridge", "esm_context_ridge": "ESM-2 + context / ridge",
    "chem_ridge": "Morgan / ridge", "chem_context_ridge": "Morgan + context / ridge",
    "sequence_hgb": "Descriptors / boosting", "sequence_context_hgb": "Descriptors + context / boosting",
    "esm_hgb": "ESM-2 / boosting", "esm_context_hgb": "ESM-2 + context / boosting",
    "chem_hgb": "Morgan / boosting", "chem_context_hgb": "Morgan + context / boosting",
}
COLORS = ["#236B8E", "#2B8A78", "#AB5C3C", "#7D67A5", "#B69139", "#A4ADB4"]
COMPONENTS = ["chemical_entity", "publication", "exact_target",
              "chemical_by_target", "chemical_by_medium", "residual"]
COMP_LABELS = ["Chemical identity", "Publication", "Exact target",
               "Chemical x target", "Chemical x medium", "Residual"]
REML_NAMES = {"chemical_id": "chemical_entity", "publication_id": "publication",
              "target_exact": "exact_target", "chemical_id:target_exact": "chemical_by_target",
              "chemical_id:medium": "chemical_by_medium", "Residual": "residual"}


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)


def tex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "_": r"\_", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(c, c) for c in str(value))


def fmt(value, digits=3):
    return f"{float(value):.{digits}f}" if np.isfinite(float(value)) else "--"


def count(value):
    return f"{int(value):,}"


def save_figure(fig, path):
    # Keep unchanged figures byte-stable across report builds and visual review.
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def posterior_results(scope):
    from .censored import active_stan_path, preparation_fingerprint, prepare_stan_data, select_analysis_rows
    source = ROOT / "data/curated" / scope / "activities.csv"
    input_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    stan_hash = hashlib.sha256(active_stan_path(scope).read_bytes()).hexdigest()
    preparation_hash = preparation_fingerprint()
    legacy_source = None
    found = {}
    for path in (ROOT/"results/reconstruction_2026-10-20"/scope/"censored").glob("*/*/result.json"):
        result = read_json(path)
        if result["input_sha256"] != input_hash or result["stan_source_sha256"] != stan_hash:
            continue
        recorded_preparation = result.get("preparation_source_sha256")
        if recorded_preparation and recorded_preparation != preparation_hash:
            continue
        if not recorded_preparation:
            if legacy_source is None:
                legacy_source = read_csv(source)
            prepared, _ = prepare_stan_data(select_analysis_rows(legacy_source, result["subset"]),
                                           result["prior_scale_log2"])
            if prepared != read_json(path.parent/"stan_data.json"):
                continue
        key = (result["subset"], result["prior_scale_log2"])
        score = (result["diagnostics"]["passed"], result["draws_per_chain"])
        if key not in found or score > found[key][0]:
            found[key] = (score, result, path)
    return {key: (value[1], value[2]) for key, value in found.items()}


def posterior_cohort_counts(result, result_path):
    """Describe the observations and all grouping levels of a saved fit."""
    levels = read_json(result_path.parent/"design.json")["levels"]
    kinds = result["kind_counts"]
    if sum(kinds.values()) != result["n_records"]:
        raise ValueError(f"Posterior cohort counts disagree: {result_path}")
    return {
        "records": result["n_records"], "chemical_entities": len(levels["chem"]),
        "publications": len(levels["pub"]), "exact_targets": len(levels["target_id"]),
        **{kind: kinds.get(kind, 0) for kind in ["exact", "left", "right", "interval"]},
    }


def load_covariance_design_audit(scope):
    """Require a current design diagnostic, without requiring a favorable rank."""
    from .censored import active_stan_path, preparation_fingerprint
    path = ROOT/"results/reconstruction_2026-10-20"/scope/"statistics/covariance_design_audit.json"
    audit = read_json(path)
    sources = {
        "source_sha256": ROOT/"data/curated"/scope/"activities.csv",
        "script_sha256": ROOT/"scripts/29_audit_covariance_design.py",
        "stan_source_sha256": active_stan_path(scope),
    }
    if (audit.get("scope") != scope or
            any(audit.get(key) != hashlib.sha256(source.read_bytes()).hexdigest()
                for key, source in sources.items()) or
            audit.get("preparation_source_sha256") != preparation_fingerprint() or
            audit.get("components") != COMPONENTS or
            set(audit.get("cohorts", {})) != {"all", "exact", "multi_publication", "no_reported_ranges"} or
            not audit.get("verification", {}).get("dense_counting_identity_passed")):
        raise ValueError(f"Stale or incomplete covariance design audit: {path}")
    return audit


def load_component_sum_review(scope, posterior):
    """Use the exploratory joint-draw summaries only with current reviewed sources."""
    folder = ROOT/"results/reconstruction_2026-10-20"/scope/"posterior_component_sum"
    path = folder/"exploratory_component_sum.json"
    if not path.exists():
        return None
    analysis = read_json(path)
    review = read_json(folder/"review_20261022.json")
    rows = analysis["specifications"]
    keys = {(r["subset"], r["prior_scale"]) for r in rows}
    passing = {key for key, (result, _) in posterior.items() if result["diagnostics"]["passed"]}
    if (analysis.get("analysis_type") != "exploratory_post_inspection" or
            not review.get("passed") or len(keys) != len(rows) or keys != passing or
            review.get("completed_fits") != len(rows)):
        raise ValueError(f"Incomplete or incompatible component-sum review: {path}")
    sources = dict(analysis["source_sha256"])
    sources.update(review["source_sha256"])
    for row in rows:
        result, result_path = posterior[(row["subset"], row["prior_scale"])]
        if (row["run_id"] != result_path.parent.name or
                row["records"] != result["n_records"]):
            raise ValueError(f"Component-sum cohort differs: {result_path}")
        for component, summary in row["component_summaries_reconciled"].items():
            original = result["conditional_variance_fractions"][component]
            if not np.allclose([summary["median"], *summary["ci95"]],
                               [original["median"], *original["ci95"]], rtol=0, atol=1e-12):
                raise ValueError(f"Component summary differs: {result_path} / {component}")
        sources.update(row["source_sha256"])
    for relative, expected_hash in sources.items():
        source = ROOT/relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Stale component-sum source: {relative}")
    return analysis


def make_figures(scope, generated, data):
    manifest, activities, cells, metrics, pairs, support, reml = (
        data[k] for k in ["manifest", "activities", "cells", "metrics", "pairs", "support", "reml"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "axes.titlesize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    kinds = ["exact", "right", "left", "interval", "summary_mean", "unparsed"]
    values = [manifest["parse_kinds"].get(k, 0) for k in kinds]
    axes[0].barh(range(len(kinds)), values, color=COLORS[0])
    axes[0].set_yticks(range(len(kinds)), ["Exact", "Right limit", "Left limit", "Range", "Mean + error", "Unparsed"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Activity records")
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axes[0].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0].set_title("a  Original concentration formats", loc="left")
    fields = ["medium", "cfu_raw", "salt_type_raw", "ph_raw", "ionic_strength_raw"]
    fraction = [activities[f].fillna("").astype(str).str.strip().str.casefold().isin(
        ["", "unreported", "unknown"]).mean()*100 for f in fields]
    axes[1].barh(range(len(fields)), fraction, color=COLORS[2])
    axes[1].set_yticks(range(len(fields)), ["Medium", "Inoculum", "Supplements", "pH", "Ionic quantity"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 110)
    axes[1].set_xlabel("Unreported (%)")
    axes[1].set_title("b  Context availability", loc="left")
    for y, value in enumerate(fraction):
        axes[1].text(value+1, y, f"{value:.1f}", va="center", fontsize=8)
    pubcounts = data["all_cells"].groupby("chemical_id").publication_id.nunique()
    bins = [(pubcounts == 1).sum(), (pubcounts == 2).sum(), (pubcounts >= 3).sum()]
    axes[2].bar(["1", "2", "3+"], bins, color=[COLORS[2], COLORS[0], COLORS[1]])
    axes[2].set_xlabel("Publications per chemical entity")
    axes[2].set_ylabel("Chemical entities")
    axes[2].set_title("c  Cross-study support", loc="left")
    for x, n in enumerate(bins):
        axes[2].text(x, n + max(bins)*.015, count(n), ha="center", fontsize=8)
    axes[2].set_ylim(0, max(bins)*1.15)
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, generated/"fig_curation")

    selected = ["target_mean", "sequence_hgb", "sequence_context_hgb",
                "esm_hgb", "esm_context_hgb", "chem_hgb", "chem_context_hgb"]
    colors = ["#89929A", COLORS[0], COLORS[0], COLORS[1], COLORS[1], COLORS[2], COLORS[2]]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    for model, color in zip(selected, colors):
        subset = metrics[(metrics.model == model) & (metrics.stratum == "all")].set_index("protocol")
        context = "_context_" in model
        for ax, metric, title in zip(axes, ["mae_log2", "r2"], ["a  Absolute prediction error", "b  Explained variation"]):
            ax.plot(range(len(PROTOCOLS)), subset.loc[PROTOCOLS, metric],
                    marker="s" if context else "o", ms=4, lw=1.4,
                    ls="--" if context else "-", color=color, label=MODEL_LABELS[model])
            ax.set_xticks(range(len(PROTOCOLS)), PROTOCOL_LABELS, rotation=28, ha="right")
            ax.set_title(title, loc="left")
            ax.grid(axis="y", alpha=.17)
    axes[0].set_ylabel("Mean absolute error (log2 MIC)")
    axes[1].set_ylabel("R squared")
    axes[1].axhline(0, color="#777777", lw=.7)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(.5, -.075))
    fig.tight_layout(rect=[0, .12, 1, 1], w_pad=2)
    save_figure(fig, generated/"fig_generalization")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), sharex=True, sharey=True)
    for ax, protocol, title in zip(axes, ["publication", "temporal"], ["a  New publications", "b  Later publications"]):
        subset = pairs[(pairs.protocol == protocol) & (pairs.stratum == "all")].set_index("base")
        for y, base in enumerate(BASES):
            row = subset.loc[base]
            ax.errorbar(row.mae_reduction_log2, y,
                        xerr=[[row.mae_reduction_log2-row.ci_low], [row.ci_high-row.mae_reduction_log2]],
                        fmt="o", ms=5, color=COLORS[y % 3], capsize=3)
        ax.axvline(0, color="#666666", lw=1, ls="--")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Paired MAE reduction from adding context (log2)")
        ax.grid(axis="x", alpha=.16)
    axes[0].set_yticks(range(len(BASES)), [MODEL_LABELS[b] for b in BASES])
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=2)
    save_figure(fig, generated/"fig_context_gain")

    titles = {"all_additive": "All / additive", "all_interactions": "All / interactions",
              "multi_publication_additive": "Shared / additive",
              "multi_publication_interactions": "Shared / interactions",
              "cross_publication_same_exact_target": "Shared chemical + target"}
    rows, labels = [], []
    for key, result in reml.items():
        if "variance_components" not in result:
            continue
        d = {REML_NAMES[r["grp"]]: r["fraction"] for r in result["variance_components"]}
        rows.append([100*d.get(name, 0) for name in COMPONENTS])
        labels.append(f"{titles.get(key, key)} (n={count(result['n_used'])})" +
                      (" *" if result.get("singular") else ""))
    fig, ax = plt.subplots(figsize=(10.8, 3.9))
    cumulative = np.zeros(len(rows))
    array = np.asarray(rows)
    for i, name in enumerate(COMP_LABELS):
        ax.barh(range(len(rows)), array[:, i], left=cumulative, color=COLORS[i], label=name, height=.65)
        for y, value in enumerate(array[:, i]):
            if value >= 7:
                ax.text(cumulative[y]+value/2, y, f"{value:.1f}", ha="center", va="center",
                        color="white" if i < 4 else "#15252E", fontsize=8)
        cumulative += array[:, i]
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Conditional variance fraction (%)")
    ax.set_title("Specification and overlap sensitivity of descriptive REML components", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, generated/"fig_variance_sensitivity")

    all_models = list(MODEL_LABELS)
    matrix = metrics[metrics.stratum.eq("all")].pivot(index="model", columns="protocol", values="mae_log2")
    matrix = matrix.reindex(index=all_models, columns=PROTOCOLS)
    fig, ax = plt.subplots(figsize=(10.3, 6.5))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(PROTOCOLS)), PROTOCOL_LABELS)
    ax.set_yticks(range(len(all_models)), [MODEL_LABELS[x] for x in all_models])
    midpoint = (matrix.min().min()+matrix.max().max())/2
    for y in range(len(matrix)):
        for x in range(len(PROTOCOLS)):
            value = matrix.iloc[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > midpoint else "#16232C", fontsize=9)
    fig.colorbar(im, ax=ax, label="MAE (log2 MIC)", shrink=.75)
    ax.set_title("Held-out MAE across fourteen fixed model configurations", loc="left", pad=12)
    fig.tight_layout()
    save_figure(fig, generated/"fig_all_models")

    joint = metrics[(metrics.protocol == "joint") & metrics.model.isin(
        ["sequence_hgb", "sequence_context_hgb", "esm_hgb", "esm_context_hgb"])]
    strata = ["known_sequence__known_publication", "new_sequence__known_publication",
              "known_sequence__new_publication", "new_sequence__new_publication"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
    for i, model in enumerate(["sequence_hgb", "sequence_context_hgb", "esm_hgb", "esm_context_hgb"]):
        values = joint[joint.model.eq(model)].set_index("stratum").reindex(strata)
        axes[0].plot(range(4), values.mae_log2, marker="o", lw=1.4,
                     color=COLORS[i//2], ls="--" if i % 2 else "-", label=MODEL_LABELS[model])
    axes[0].set_xticks(range(4), ["Known seq.\nKnown pub.", "New seq.\nKnown pub.",
                                 "Known seq.\nNew pub.", "New seq.\nNew pub."])
    axes[0].set_ylabel("MAE (log2 MIC)")
    axes[0].set_title("a  Actual training exposure", loc="left")
    values = joint[joint.model.eq("sequence_hgb")].set_index("stratum").reindex(strata)
    axes[1].bar(range(4), values.n, color=COLORS[0])
    axes[1].set_xticks(range(4), ["K/K", "N/K", "K/N", "N/N"])
    axes[1].set_ylabel("Distinct assay cells per stratum")
    axes[1].set_title("b  Evaluation support", loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, fontsize=8, bbox_to_anchor=(.5, -.035))
    fig.tight_layout(rect=[0, .12, 1, 1])
    save_figure(fig, generated/"fig_joint")


def latex_table(path, headers, rows, specification=None, long=False):
    specification = specification or "l" + "r"*(len(headers)-1)
    env = "longtable" if long else "tabular"
    lines = [r"\begin{" + env + "}{" + specification + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    if long:
        lines += [r"\endfirsthead", r"\toprule", " & ".join(headers) + r" \\",
                  r"\midrule", r"\endhead"]
    lines.extend(" & ".join(str(x) for x in row) + r" \\" for row in rows)
    lines += [r"\bottomrule", r"\end{" + env + "}"]
    path.write_text("\n".join(lines) + "\n")


def write_table_guide(scope, generated, output, posterior):
    """Record the selection and full numerical sources behind every display table."""
    benchmark = output/"benchmark"
    statistics = output/"statistics"
    result_paths = [path for _, path in posterior.values()]
    posterior_files = lambda name: [path.parent/name for path in result_paths
                                   if (path.parent/name).exists()]
    definitions = [
        ("table_eligibility", "Motivos de exclusión", [output/"eligibility_exclusions.csv"],
         "Recuentos de banderas de exclusión; pueden coincidir varias en el mismo registro y no deben sumarse como categorías disjuntas."),
        ("table_main_models", "Comparación principal con y sin contexto",
         [benchmark/"metrics.csv", benchmark/"paired_context_improvements.csv"],
         "Tres representaciones con boosting, seis protocolos y estrato all. No incluye ridge ni los estratos de exposición conjunta."),
        ("table_all_metrics", "Métricas globales de los catorce modelos", [benchmark/"metrics.csv"],
         "Todas las configuraciones y protocolos disponibles, pero solo el estrato all. El CSV fuente conserva también los estratos específicos."),
        ("table_all_pairs", "Comparaciones pareadas globales", [benchmark/"paired_context_improvements.csv"],
         "Las seis parejas de representación y aprendizaje, en el estrato all. El CSV conserva todas las comparaciones por estrato; los intervalos son puntuales y condicionales a modelos ajustados."),
        ("table_covariance_design", "Diseño de covarianza", [statistics/"covariance_design_audit.json"],
         "Cuatro cohortes distintas. Rango y condición de la base de covarianzas latentes, antes de proyectar efectos fijos; no son diagnósticos de adecuación o causalidad."),
        ("table_posterior_cohorts", "Cohortes de los ajustes censurados", result_paths+posterior_files("design.json"),
         "Cinco especificaciones previstas, con marcadores pendientes si faltan ajustes. Cuenta registros y todos los niveles, incluidos los integrados analíticamente; no son celdas predictivas."),
        ("table_posterior_formats", "Formatos de concentración en cada ajuste", result_paths,
         "Exactos, límites izquierdos y derechos y rangos, dentro de cada cohorte censurada; no son los recuentos del catálogo sin filtrar."),
        ("table_posterior_diagnostics", "Diagnósticos computacionales", result_paths,
         "Disponibilidad, Rhat, ESS y divergencias de las cinco especificaciones previstas. Pasar estos criterios no demuestra adecuación científica."),
        ("table_sampling_precision", "Precisión de Monte Carlo", result_paths+posterior_files("sampling_precision.json"),
         "ESS de cola, E-BFMI y MCSE de las medias de fracciones. Las fuentes distinguen la regla computacional y la revisión adicional."),
        ("table_posterior_fractions", "Fracciones condicionales de varianza", result_paths,
         "Solo ajustes que superan los diagnósticos computacionales. Seis componentes por ajuste; la variación de efectos fijos queda fuera del denominador. Los intervalos completos se conservan en JSON."),
        ("table_posterior_checks", "Comprobaciones predictivas internas", posterior_files("posterior_predictive_checks.json"),
         "Cinco resúmenes del subconjunto exacto por ajuste comprobado. Se retienen todas las banderas. La tabla TeX expresa la cola superior como proporción; la memoria muestra el mismo valor como recuento de réplicas sobre el total. La selección de registros exactos limita la interpretación; no es validación externa."),
        ("table_curation_sensitivity", "Sensibilidades REML de curación", [statistics/"reml_curation_sensitivity.json"],
         "Seis restricciones exploratorias de la cohorte exacta. Cambian tanto composición como soporte; no estiman efectos causales de corregir datos."),
        ("table_prediction_intervals", "Intervalos predictivos temporales", [output/"prediction_intervals/coverage_by_split.csv"],
         "Cuatro modelos, coberturas nominales 90% y 95%, prueba temporal y puntuación publication_maximum. El CSV completo incluye también particiones por publicación y puntuaciones pooled_rows."),
        ("table_rank_thresholds", "Sensibilidad al umbral de inversión", [statistics/"rank_threshold_sensitivity.csv"],
         "Siete umbrales sobre los mismos pares. Tolerancia numérica 1e-12 log2 y remuestreo cruzado; los pares no son experimentos independientes."),
        ("table_screening_ties", "Sensibilidad a empates de selección", [output/"temporal_screening_tie_sensitivity.csv"],
         "Seis modelos seleccionados para mostrar el mecanismo, protocolo temporal y tolerancia 1e-12. La fuente conserva catorce modelos y ambos criterios de empate. No se infiere significación de las diferencias."),
    ]
    if (generated/"table_component_sum.tex").exists():
        definitions.append((
            "table_component_sum", "Suma exploratoria de química y publicación",
            [output/"posterior_component_sum"/name for name in
             ["exploratory_component_sum.json", "exploratory_component_sum.csv", "review_20261022.json"]],
            "Cinco especificaciones. Suma calculada dentro de cada muestra posterior conjunta, con cuantiles propios; no se suman medianas ni extremos marginales. Análisis elegido después de inspeccionar tres ajustes. Son fracciones normalizadas sin variación fija, no varianzas absolutas ni evidencia de equivalencia o causalidad."))
    if {stem for stem, *_ in definitions} != {p.stem for p in generated.glob("table_*.tex")}:
        raise ValueError("Every generated display table must have one source-guide entry.")
    entries = []
    for stem, title, sources, selection in definitions:
        table_path = generated/(stem+".tex")
        entries.append({
            "stem": stem, "title": title, "selection_and_limits": selection,
            "table_path": str(table_path.relative_to(ROOT)),
            "table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(set(sources)) if p.is_file()},
            "unavailable_sources": [str(p.relative_to(ROOT)) for p in sorted(set(sources))
                                    if not p.is_file()],
        })
    guide = {
        "scope": scope,
        "note": "Las tablas de presentación están redondeadas. Para cálculos y reutilización se deben consultar los CSV o JSON completos enlazados, conservando sus unidades, estratos y limitaciones. La generación de tablas no implica redacción de un manuscrito.",
        "tables": entries,
    }
    (generated/"table_guide.json").write_text(json.dumps(guide, indent=2, ensure_ascii=False)+"\n")


def generate_manuscript_inputs(scope):
    source = ROOT/"data/curated"/scope
    output = ROOT/"results/reconstruction_2026-10-20"/scope
    benchmark = output/"benchmark"
    audit = read_json(benchmark/"completion_audit.json")
    if not audit["passed"]:
        raise ValueError("Do not generate a result manuscript from an incomplete benchmark.")
    if hashlib.sha256((source/"benchmark_cells.csv").read_bytes()).hexdigest() != audit["input_sha256"]:
        raise ValueError("The verified benchmark is stale.")
    summary_audit = read_json(benchmark/"summary_completion_audit.json")
    if (not summary_audit["passed"] or summary_audit["input_sha256"] != audit["input_sha256"]
            or any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != digest
                   for p,digest in summary_audit["files"].items())):
        raise ValueError("The primary metric summaries have not passed the current independent audit.")
    generated = ROOT/"manuscript/generated"/scope
    generated.mkdir(parents=True, exist_ok=True)
    data = {
        "manifest": read_json(source/"curation_manifest.json"),
        "activities": read_csv(source/"activities.csv"), "cells": read_csv(source/"benchmark_cells.csv"),
        "all_cells": read_csv(source/"all_exact_assay_cells.csv"),
        "metrics": read_csv(benchmark/"metrics.csv"),
        "pairs": read_csv(benchmark/"paired_context_improvements.csv"),
        "screening": read_csv(benchmark/"screening_summary.csv"),
        "support": read_json(output/"statistics/design_support.json"),
        "covariance_design": load_covariance_design_audit(scope),
        "rank": read_json(output/"statistics/matched_rank_summary.json"),
        "rank_thresholds": read_csv(output/"statistics/rank_threshold_sensitivity.csv"),
        "reml": read_json(output/"statistics/reml_sensitivity.json"),
        "clusters": read_json(source/"cluster_manifest.json"),
    }
    rank_sensitivity = read_json(output/"statistics/rank_threshold_sensitivity.json")
    rank_source_hash = hashlib.sha256((output/"statistics/matched_rank_pairs.csv").read_bytes()).hexdigest()
    expected_thresholds = pd.DataFrame(rank_sensitivity["results"])
    if (not rank_sensitivity["primary_result_reproduced"] or
            rank_sensitivity["input_sha256"] != rank_source_hash or
            list(data["rank_thresholds"].columns) != list(expected_thresholds.columns) or
            data["rank_thresholds"].shape != expected_thresholds.shape or
            not np.allclose(data["rank_thresholds"], expected_thresholds, rtol=0, atol=1e-12)):
        raise ValueError("Rank-threshold sensitivity is stale or inconsistent.")
    make_figures(scope, generated, data)
    m, s, cells = data["manifest"], data["support"], data["cells"]
    metrics, pairs = data["metrics"], data["pairs"]
    reasons = data["activities"].exclusion_reasons.fillna("").str.split("|").explode()
    exclusions = reasons[reasons.ne("")].value_counts().rename_axis("exclusion_reason").reset_index(name="records")
    exclusions.to_csv(output/"eligibility_exclusions.csv", index=False)
    latex_table(generated/"table_eligibility.tex", ["Exclusion reason", "Flagged records"],
                [[tex(row.exclusion_reason), count(row.records)] for row in exclusions.itertuples()],
                "lr")
    split_audits = read_json(benchmark/"verified_split_audits.json")
    temporal = [a for a in split_audits if a["protocol"] == "temporal"][0]
    homology50_splits = [a for a in split_audits if a["protocol"] == "homology50"]
    figure_guide = [
        {"stem": "fig_curation", "title": "Curación y soporte de comparación",
         "description": f"Los paneles a y b describen los {m['records']:,} registros de actividad antes de los filtros analíticos: formatos de concentración y porcentaje de campos no reportados. El panel c usa las entidades químicas de la cohorte exacta para inferencia y cuenta en cuántas publicaciones aparecen. Estos denominadores son distintos; publicaciones no equivalen a laboratorios independientes.",
         "sources": [str((source/name).relative_to(ROOT)) for name in ["curation_manifest.json", "activities.csv", "all_exact_assay_cells.csv"]]},
        {"stem": "fig_generalization", "title": "Error y rendimiento según la partición",
         "description": f"MAE log2 y R² de tres representaciones con boosting, con y sin contexto, y del comparador de media por objetivo. Los protocolos de validación cruzada evalúan {len(cells):,} celdas; la prueba temporal evalúa {temporal['n_test']:,} celdas de publicaciones posteriores. Las líneas conectan categorías, no una trayectoria temporal ni un efecto causal. Edit 0.80 y Edit 0.50 representan umbrales de similitud de edición global, no identidad BLAST.",
         "sources": [str((benchmark/"metrics.csv").relative_to(ROOT)), str((benchmark/"verified_split_audits.json").relative_to(ROOT))]},
        {"stem": "fig_context_gain", "title": "Mejora pareada al añadir contexto",
         "description": "Los puntos indican la reducción pareada de MAE log2; las barras son intervalos puntuales del 95% obtenidos con 2.000 remuestreos cruzados por publicación y secuencia. Los valores positivos favorecen añadir contexto. Cada par compara los mismos casos. Los intervalos son exploratorios y condicionales a modelos ya ajustados; no incluyen repetir el entrenamiento completo ni un ajuste por comparaciones múltiples.",
         "sources": [str((benchmark/"paired_context_improvements.csv").relative_to(ROOT))]},
        {"stem": "fig_variance_sensitivity", "title": "Componentes REML y sensibilidad al soporte",
         "description": "Fracciones descriptivas de varianza de modelos REML con etiquetas exactas, bajo especificaciones aditivas, interacciones y subconjuntos compartidos. Los tamaños de cohorte aparecen en cada barra. La fracción atribuida a publicación depende del modelo y del subconjunto; no identifica un efecto causal de laboratorio. Esta figura corresponde a REML y es distinta de los ajustes bayesianos censurados.",
         "sources": [str((output/"statistics/reml_sensitivity.json").relative_to(ROOT))]},
        {"stem": "fig_all_models", "title": "Las catorce configuraciones predictivas",
         "description": "MAE log2 de todas las configuraciones fijas. Dentro de cada columna los modelos evalúan los mismos casos; la población temporal difiere de la validación cruzada. Los números son estimaciones puntuales y el color representa su magnitud. La tabla no reproduce todas las herramientas especializadas disponibles ni demuestra superioridad frente al estado del arte.",
         "sources": [str((benchmark/"metrics.csv").relative_to(ROOT)), str((benchmark/"analysis_config.json").relative_to(ROOT))]},
        {"stem": "fig_joint", "title": "Exposición conjunta a secuencias y publicaciones",
         "description": "El panel a compara el error según la exposición conocida o nueva observada en el entrenamiento. K significa conocida y N nueva; el orden es secuencia/publicación. El panel b cuenta celdas distintas dentro de cada estrato. Una misma celda puede contribuir a varios estratos en repeticiones diferentes, por lo que las barras no deben sumarse. Las poblaciones de los estratos difieren: sus contrastes no son efectos causales emparejados.",
         "sources": [str((benchmark/"metrics.csv").relative_to(ROOT)), str((benchmark/"verified_split_audits.json").relative_to(ROOT))]},
    ]
    (generated/"figure_guide.json").write_text(json.dumps(
        {"scope": scope, "figures": figure_guide}, indent=2, ensure_ascii=False)+"\n")
    mode = "Expanded catalog" if scope == "expanded" else "Initial 2,500-card sample"
    status = ("Retrospective in silico study"
              if scope == "expanded" else "Provisional sample analysis; expanded-catalog analysis pending")
    values = {
        "ScopeName": mode, "StudyStatus": status, "CardCount": count(m["cards"]),
        "RecordCount": count(m["records"]), "ExactCount": count(m["exact_assay_cells"]),
        "BenchmarkCount": count(len(cells)), "ChemicalCount": count(s["chemical_entities"]),
        "PublicationCount": count(s["publications"]), "BenchmarkPublications": count(cells.publication_id.nunique()),
        "SequenceCount": count(cells.sequence_id.nunique()), "CensoredCount": count(m["eligible_censored_records"]),
        "SinglePublicationPct": f"{100*s['single_publication_fraction']:.1f}",
        "SharedChemicalCount": count(s["multi_publication_entities"]),
        "TemporalCutoff": str(int(temporal["train_year_max"])),
        "TemporalStart": str(int(temporal["test_year_min"])),
        "TemporalTestCount": count(temporal["n_test"]),
        "FitCount": count(audit["verified_fits"]),
    }
    for protocol, prefix in [("random", "Random"), ("publication", "Publication"), ("temporal", "Temporal")]:
        row = metrics[(metrics.protocol == protocol) & (metrics.model == "sequence_hgb") & (metrics.stratum == "all")].iloc[0]
        values[prefix+"Rtwo"] = fmt(row.r2)
        values[prefix+"MAE"] = fmt(row.mae_log2)
    temporal_gain = pairs[(pairs.protocol == "temporal") & pairs.base.eq("sequence_hgb") &
                          pairs.stratum.eq("all")].iloc[0]
    values.update({"TemporalContextGain": fmt(temporal_gain.mae_reduction_log2),
                   "TemporalContextLow": fmt(temporal_gain.ci_low, 4),
                   "TemporalContextHigh": fmt(temporal_gain.ci_high, 4)})
    (generated/"results_macros.tex").write_text("\n".join(
        r"\newcommand{\%s}{%s}" % (key, tex(value)) for key, value in values.items())+"\n")
    rows = []
    for protocol, label in zip(PROTOCOLS, PROTOCOL_LABELS):
        for base in ["sequence_hgb", "esm_hgb", "chem_hgb"]:
            alt = base.replace("_hgb", "_context_hgb")
            a = metrics[(metrics.protocol == protocol) & metrics.model.eq(base) & metrics.stratum.eq("all")].iloc[0]
            b = metrics[(metrics.protocol == protocol) & metrics.model.eq(alt) & metrics.stratum.eq("all")].iloc[0]
            gain = pairs[(pairs.protocol == protocol) & pairs.base.eq(base) & pairs.stratum.eq("all")].iloc[0]
            rows.append([label, MODEL_LABELS[base].split(" / ")[0], fmt(a.mae_log2), fmt(b.mae_log2),
                         fmt(gain.mae_reduction_log2), f"[{fmt(gain.ci_low, 4)}, {fmt(gain.ci_high, 4)}]"])
    latex_table(generated/"table_main_models.tex",
                ["Partition", "Representation", "Base MAE", "+ Context", r"$\Delta$ MAE", "95\\% interval"],
                rows, "llrrrr")
    rows = []
    for _, row in metrics[metrics.stratum.eq("all")].sort_values(["protocol", "model"]).iterrows():
        rows.append([tex(row.protocol), tex(MODEL_LABELS[row.model]), count(row.n), fmt(row.mae_log2),
                     fmt(row.r2), fmt(row.within_one_dilution)])
    latex_table(generated/"table_all_metrics.tex",
                ["Partition", "Model", "Cells", "MAE", r"$R^2$", r"$|\mathrm{error}|\leq1$"],
                rows, "llrrrr", long=True)
    rows = []
    for _, row in pairs[pairs.stratum.eq("all") & pairs.base.isin(BASES)].sort_values(["protocol", "base"]).iterrows():
        rows.append([tex(row.protocol), tex(MODEL_LABELS[row.base]), fmt(row.mae_reduction_log2),
                     f"[{fmt(row.ci_low, 4)}, {fmt(row.ci_high, 4)}]", count(row.publications), count(row.sequences)])
    latex_table(generated/"table_all_pairs.tex",
                ["Partition", "Base model", r"$\Delta$ MAE", "95\\% interval", "Papers", "Seq."],
                rows, "llrrrr", long=True)
    posterior = posterior_results(scope)
    covariance_rows = [
        [tex(subset), count(row["records"]), f"{row['rank']}/{row['component_count']}",
         fmt(row["normalized_gram_eigenvalues"][0], 4),
         fmt(row["normalized_basis_condition_number"]) if row["normalized_basis_condition_number"] is not None else "--"]
        for subset, row in data["covariance_design"]["cohorts"].items()
    ]
    latex_table(generated/"table_covariance_design.tex",
                ["Cohort", "Records", "Rank", r"$\lambda_{\min}$", r"$\kappa_{\mathrm{basis}}$"],
                covariance_rows, "lrrrr")
    expected = [("all", 1.5), ("exact", 1.5), ("multi_publication", 1.5),
                ("no_reported_ranges", 1.5), ("all", 3.0)]
    posterior_rows, diagnostics_rows, check_rows, precision_rows = [], [], [], []
    cohort_rows, format_rows = [], []
    posterior_sources = [benchmark/"summary_completion_audit.json"]
    checked_fits = flagged_fits = 0
    for key in expected:
        label = tex(f"{key[0]} / prior {key[1]:g}")
        if key not in posterior:
            diagnostics_rows.append([label, "--", "--", "--", "--", "Pending"])
            precision_rows.append([label, "--", "--", "--", "Pending"])
            cohort_rows.append([label, "--", "--", "--", "--"])
            format_rows.append([label, "--", "--", "--", "--"])
            continue
        result, path = posterior[key]
        posterior_sources.append(path)
        posterior_sources.append(path.parent/"design.json")
        cohort = posterior_cohort_counts(result, path)
        cohort_rows.append([label, *[count(cohort[name]) for name in
                                    ["records", "chemical_entities", "publications", "exact_targets"]]])
        format_rows.append([label, *[count(cohort[name]) for name in
                                    ["exact", "left", "right", "interval"]]])
        if result.get("reuse_provenance"):
            posterior_sources.append(path.parent/"posterior_reuse.json")
        d = result["diagnostics"]
        diagnostics_rows.append([label, count(result["n_records"]), fmt(d["max_rhat_all_parameters"]),
                                 f"{d['min_bulk_ess_scientific_parameters']:.0f}", str(d["divergences"]),
                                 "Pass" if d["passed"] else "Failed"])
        if d["passed"]:
            precision_path = path.parent/"sampling_precision.json"
            if precision_path.exists():
                precision = read_json(precision_path)
                if any(hashlib.sha256((path.parent/name).read_bytes()).hexdigest() != digest
                       for name, digest in precision["source_sha256"].items()):
                    raise ValueError(f"Stale sampling precision checks: {precision_path}")
                posterior_sources.append(precision_path)
                precision_rows.append([label, f"{precision['min_tail_ess_scientific_parameters']:.0f}",
                                       fmt(min(precision['e_bfmi_by_chain'])),
                                       f"{precision['max_mcse_fraction_mean']:.5f}",
                                       "Pass" if precision["passed"] else "Flag"])
            else:
                precision_rows.append([label, "--", "--", "--", "Pending"])
            for component, component_label in zip(COMPONENTS, COMP_LABELS):
                row = result["conditional_variance_fractions"][component]
                posterior_rows.append([label, tex(component_label), fmt(row["median"]),
                                       f"[{fmt(row['ci95'][0])}, {fmt(row['ci95'][1])}]"])
            check_path = path.parent/"posterior_predictive_checks.json"
            if check_path.exists():
                checks = read_json(check_path)
                data_signature = hashlib.sha256((path.parent/"stan_data.json").read_bytes()).hexdigest()
                if checks["stan_data_sha256"] != data_signature:
                    raise ValueError(f"Stale posterior checks: {check_path}")
                checked_fits += 1
                flagged_fits += int(checks["any_extreme_tail_flag"])
                posterior_sources.append(check_path)
                posterior_sources.append(path.parent/"conditional_residual_groups.csv")
                for statistic, check in checks["exact_subset_checks"].items():
                    interval = check["replicated_interval95"]
                    check_rows.append([
                        label, tex(statistic), fmt(check["observed"]),
                        f"[{fmt(interval[0])}, {fmt(interval[1])}]",
                        fmt(check["posterior_predictive_upper_tail"]),
                        "Flag" if check["extreme_tail_flag"] else "--",
                    ])
    latex_table(generated/"table_posterior_cohorts.tex",
                ["Subset / prior scale", "Records", "Chemicals", "Papers", "Targets"],
                cohort_rows, "lrrrr")
    latex_table(generated/"table_posterior_formats.tex",
                ["Subset / prior scale", "Exact", "Left limit", "Right limit", "Range"],
                format_rows, "lrrrr")
    latex_table(generated/"table_posterior_diagnostics.tex",
                ["Subset / prior scale", "Records", r"Max $\widehat R$", "Min ESS", "Diverg.", "Gate"],
                diagnostics_rows, "lrrrrl")
    latex_table(generated/"table_sampling_precision.tex",
                ["Subset / prior scale", "Min tail ESS", "Min E-BFMI", "Max fraction MCSE", "Review"],
                precision_rows, "lrrrl")
    latex_table(generated/"table_posterior_fractions.tex",
                ["Subset / prior scale", "Component", "Median", "95\\% interval"],
                posterior_rows or [["Pending", "No passing posterior available", "--", "--"]], "llrr", long=True)
    latex_table(generated/"table_posterior_checks.tex",
                ["Subset / prior scale", "Statistic", "Observed", "Replicated 95\\%", "Upper tail", "Flag"],
                check_rows or [["Pending", "No checked posterior", "--", "--", "--", "--"]],
                "llrrrl", long=True)
    component_sum = load_component_sum_review(scope, posterior)
    if component_sum:
        sum_rows = []
        for row in component_sum["specifications"]:
            value = row["chemical_entity_plus_publication"]
            diagnostic = value["sampling_diagnostics"]
            sum_rows.append([
                tex(f"{row['subset']} / prior {row['prior_scale']:g}"),
                fmt(value["median"], 4),
                f"[{fmt(value['ci95'][0], 4)}, {fmt(value['ci95'][1], 4)}]",
                fmt(diagnostic["R_hat"], 4), fmt(diagnostic["ESS_bulk"], 0),
                fmt(diagnostic["ESS_tail"], 0), fmt(diagnostic["MCSE"], 6),
            ])
            posterior_sources.extend(ROOT/relative for relative in row["source_sha256"])
        posterior_sources.extend(ROOT/relative for relative in component_sum["source_sha256"])
        latex_table(generated/"table_component_sum.tex",
                    ["Subset / prior", "Median", "95\\% interval", r"$\widehat R$",
                     "Bulk ESS", "Tail ESS", "Mean MCSE"], sum_rows, "lrrrrrr")
    passed = sum(key in posterior and posterior[key][0]["diagnostics"]["passed"] for key in expected)
    sensitivity_path = output/"statistics/reml_curation_sensitivity.json"
    sensitivity_rows = []
    if sensitivity_path.exists():
        for key, result in read_json(sensitivity_path).items():
            labels = {
                "native_molar_only": "Native molar units",
                "no_candidate_duplicates": "No flagged repeat candidates",
                "central_98pct_outcomes": "Central 98\\% of outcomes",
                "at_least_two_context_fields": "At least 2 context fields",
                "largest_connected_component": "Largest connected component",
                "excluding_endpoint_ambiguity_example": "Without endpoint-ambiguity example",
            }
            fractions = {row["grp"]: row["fraction"] for row in result.get("variance_components", [])}
            sensitivity_rows.append([
                labels.get(key, tex(key)), count(result.get("n_used", result.get("n_requested", 0))),
                *[fmt(fractions.get(group, np.nan)) for group in
                  ["chemical_id", "publication_id", "target_exact", "Residual"]],
                "Failed" if "error" in result else ("Yes" if result["singular"] else "No"),
            ])
        posterior_sources.extend([sensitivity_path, output/"statistics/curation_sensitivity_design.json"])
    source_review = ROOT/"results/reconstruction_2026-10-20/source_review/targeted_source_review.json"
    if source_review.exists():
        posterior_sources.append(source_review)
    for path in [ROOT/"data/source_amendments_2026-10-20.json",
                 source/"source_amendments_applied.json",
                 ROOT/"results/reconstruction_2026-10-20/source_review/supplement_download_manifest.json",
                 ROOT/"results/reconstruction_2026-10-20/source_review/randall2023_MIC_checks.json",
                 ROOT/"results/reconstruction_2026-10-20/source_review/jaskiewicz2017_MIC_checks.json",
                 ROOT/"results/reconstruction_2026-10-20/source_review/editorial_positioning_20261021.json",
                 ROOT/"results/reconstruction_2026-10-20/source_review/editorial_positioning_update_20261021.json",
                 ROOT/"results/reconstruction_2026-10-20/source_review/editorial_positioning_complete_predictions_20261021.json",
                 ROOT/"results/reconstruction_2026-10-20/source_review/editorial_positioning_after_secondary_audits_20261021.json",
                 ROOT/"results/reconstruction_2026-10-20/source_review/editorial_positioning_two_posteriors_20261021T090747Z.json",
                 ROOT/"results/reconstruction_2026-10-20/source_review/editorial_positioning_three_posteriors_20261021T133902Z.json",
                 ROOT/"results/reconstruction_2026-10-20/source_review/editorial_positioning_final_analysis_20261022.json",
                 ROOT/"results/reconstruction_2026-10-20/source_review/verify_all_posterior_inputs_20261021.py",
                 ROOT/"results/reconstruction_2026-10-20/source_review/verify_pending_sensitivity_inputs_20261021.py",
                 ROOT/"results/reconstruction_2026-10-20/source_review/check_component_sum_20261021.py",
                 ROOT/"results/reconstruction_2026-10-20/source_review/review_final_sensitivities_20261022.py",
                 ROOT/"results/reconstruction_2026-10-20/cohort_metadata_correction.json",
                 output/"predictive_audit_review.json",
                 output/"interim_to_full_metric_reconciliation.json",
                 output/"paired_uncertainty_review.json",
                 output/"exact_posterior_review.json",
                 output/"all_posterior_input_review.json",
                 output/"all_posterior_review.json",
                 output/"pending_sensitivity_input_review_20261021.json",
                 output/"no_reported_ranges_posterior_review.json",
                 output/"broader_prior_posterior_review.json",
                 output/"final_sensitivity_review_20261022.json",
                 output/"posterior_component_sum/exploratory_component_sum.json",
                 output/"posterior_component_sum/exploratory_component_sum.csv",
                 output/"posterior_component_sum/review_20261022.json",
                 output/"screening_candidate_identity_review.json",
                 output/"screening_alias_cell_graph_links.csv",
                 output/"secondary_evaluation_review.json",
                 output/"temporal_screening_tie_sensitivity.csv",
                 output/"prediction_interval_review.json",
                 output/"parallel_posterior_scheduling_verification.json"]:
        if path.exists():
            posterior_sources.append(path)
    latex_table(generated/"table_curation_sensitivity.tex",
                ["Subset", "Cells", "Chemical", "Paper", "Target", "Residual", "Singular"],
                sensitivity_rows or [["Pending", "--", "--", "--", "--", "--", "--"]],
                "lrrrrrl")
    interval_path = output/"prediction_intervals/coverage_by_split.csv"
    interval_rows = []
    if interval_path.exists():
        interval_data = read_csv(interval_path)
        selected = interval_data[(interval_data.protocol == "temporal") &
                                 (interval_data.score_kind == "publication_maximum")]
        for r in selected.itertuples():
            interval_rows.append([
                tex(MODEL_LABELS[r.model]), f"{r.nominal_coverage:.0%}".replace("%", r"\%"),
                fmt(r.width_log2), fmt(r.cell_weighted_coverage),
                fmt(r.equal_publication_mean_cell_coverage), fmt(r.whole_publication_coverage),
            ])
        posterior_sources.extend([interval_path, output/"prediction_intervals/config.json",
                                  output/"prediction_intervals/completion_audit.json"])
    latex_table(generated/"table_prediction_intervals.tex",
                ["Model", "Nominal", "Width", "Cells", "Paper-balanced", "Whole paper"],
                interval_rows or [["Pending", "--", "--", "--", "--", "--"]],
                "lrrrrr")
    rank = data["rank"]["all_matching_reported_fields"]
    threshold_rows = data["rank_thresholds"]
    near_threshold = threshold_rows[np.isclose(threshold_rows.threshold_log2, 1.01)].iloc[0]
    latex_table(generated/"table_rank_thresholds.tex",
                ["Threshold (log2)", "Inversions", "Proportion", "95\\% interval", "Papers with inversions"],
                [[fmt(r.threshold_log2, 2), count(r.inversions), fmt(r.rate),
                  f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]", count(r.publications_with_inversions)]
                 for r in threshold_rows.itertuples()], "rrrrr")
    posterior_sources.extend([output/"statistics/rank_threshold_sensitivity.csv",
                              output/"statistics/rank_threshold_sensitivity.json",
                              output/"statistics/matched_rank_pairs.csv",
                              ROOT/"results/reconstruction_2026-10-20/source_review/rank_source_review.json"])
    grampa_path = ROOT/"results/reconstruction_2026-10-20/sample/statistics/grampa_reliability.json"
    grampa = read_json(grampa_path)
    shared = data["reml"]["multi_publication_additive"]
    def fraction(result, group):
        return next(r["fraction"] for r in result["variance_components"] if r["grp"] == group)
    p_all = fraction(data["reml"]["all_additive"], "publication_id")
    p_shared = fraction(shared, "publication_id")
    shared_target = data["reml"]["cross_publication_same_exact_target"]
    p_shared_target = fraction(shared_target, "publication_id")
    context_summary = {}
    for protocol in ["publication", "temporal"]:
        subset = pairs[pairs.protocol.eq(protocol) & pairs.stratum.eq("all") & pairs.base.isin(BASES)]
        base_mae = metrics[metrics.protocol.eq(protocol) & metrics.stratum.eq("all")].set_index("model").mae_log2
        relative = 100*subset.mae_reduction_log2.to_numpy()/base_mae.loc[subset.base].to_numpy()
        context_summary[protocol] = {
            "comparisons": len(subset),
            "minimum_point_gain_log2": float(subset.mae_reduction_log2.min()),
            "maximum_point_gain_log2": float(subset.mae_reduction_log2.max()),
            "pointwise_intervals_wholly_above_zero": int((subset.ci_low > 0).sum()),
            "minimum_relative_point_gain_percent": float(relative.min()),
            "maximum_relative_point_gain_percent": float(relative.max()),
        }
    publication_context, temporal_context = context_summary["publication"], context_summary["temporal"]
    temporal_screening = data["screening"][data["screening"].protocol.eq("temporal")].set_index("model")
    screen_base, screen_context = (temporal_screening.loc[name] for name in
                                   ["sequence_hgb", "sequence_context_hgb"])
    secondary_path = output/"secondary_evaluation_review.json"
    tie_rows, tie_result_text = [], ""
    if secondary_path.exists():
        secondary_review = read_json(secondary_path)
        if not secondary_review.get("passed") or any(
                not (ROOT/path).exists() or hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected
                for path, expected in secondary_review["source_sha256"].items()):
            raise ValueError("Secondary evaluation review no longer matches its inputs.")
        tie_data = read_csv(output/"temporal_screening_tie_sensitivity.csv")
        tie_data = tie_data[tie_data.tie_tolerance_log2.eq(1e-12)].set_index("model")
        for name in ["sequence_hgb", "sequence_context_hgb", "esm_hgb", "esm_context_hgb",
                     "target_mean", "context_ridge"]:
            row = tie_data.loc[name]
            tie_rows.append([tex(MODEL_LABELS[name]), fmt(row.deterministic_gain_log2),
                             fmt(row.expected_tie_gain_log2), count(row.blocks_with_cutoff_ties),
                             count(row.block_evaluations)])
        tie_base, tie_context = (tie_data.loc[name] for name in
                                 ["sequence_hgb", "sequence_context_hgb"])
        tie_result_text = (
            "An exploratory sensitivity averages uniformly over candidate choices tied at "
            "the selection cutoff, using an absolute numerical tolerance of $10^{-12}$ log2. "
            f"The corresponding descriptor gains are {fmt(tie_base.expected_tie_gain_log2)} "
            f"and {fmt(tie_context.expected_tie_gain_log2)} with context "
            "(Supplementary Table~\\ref{tab:screeningties}). The primary deterministic "
            "selection and trained models are unchanged.\n")
    latex_table(generated/"table_screening_ties.tex",
                ["Model", "ID tie-break", "Tie-averaged", "Tied blocks", "Blocks"],
                tie_rows or [["Pending", "--", "--", "--", "--"]], "lrrrr")
    result_text = rf"""
\subsection{{Auditable curation and a common prediction cohort}}
The {tex(mode.lower())} contains {count(m['cards'])} peptide cards and
{count(m['records'])} activity records. The concentration parser identifies
{count(m['parse_kinds'].get('exact', 0))} exact values,
{count(m['parse_kinds'].get('right', 0))} right limits,
{count(m['parse_kinds'].get('left', 0))} left limits and
{count(m['parse_kinds'].get('interval', 0))} numeric ranges before endpoint,
unit, taxonomic and publication filters. It preserves
{count(m['conversion_rules'].get('native_molar', 0))} native molar records
without a molecular-mass round trip. {count(m['manual_review_records'])}
records enter a review queue; this is an exclusion and review record, not a claim
that their source measurements are incorrect.
The exact-label inferential cohort comprises {count(m['exact_assay_cells'])}
reported assay cells from {count(s['chemical_entities'])} chemical entities and
{count(s['publications'])} verified publications. Restricting to the common
sequence-eligible prediction cohort gives {count(len(cells))} cells,
{count(cells.chemical_id.nunique())} chemical entities,
{count(cells.sequence_id.nunique())} canonical backbones and
{count(cells.publication_id.nunique())} publications (Fig.~\ref{{fig:curation}}).
Chemical identifiers retain reported stereochemistry and modifications.

\subsection{{Generalization depends on the evaluation target}}
All {count(audit['verified_fits'])} locked fits have independently verified
prediction files, identical test cases across model comparisons, and valid split
membership. For descriptor boosting, random cross-validation gives
$R^2={values['RandomRtwo']}$ and MAE {values['RandomMAE']} log2 units;
publication-grouped evaluation gives $R^2={values['PublicationRtwo']}$
and MAE {values['PublicationMAE']}.
The chronological partition trains through {values['TemporalCutoff']} and tests
{values['TemporalTestCount']} cells from {values['TemporalStart']} onward,
without a shared publication. Its descriptor-boosting result is
$R^2={values['TemporalRtwo']}$ and MAE {values['TemporalMAE']}.
These partitions address different deployment settings and are not interchangeable
estimates of one population quantity (Fig.~\ref{{fig:generalization}}).
The 0.50 edit-similarity graph contains a largest component of
{count(data['clusters']['clusters']['cluster_50']['largest_sequence_component'])}
backbones and {count(data['clusters']['clusters']['cluster_50']['largest_assay_component'])}
cells. This creates an uneven extrapolation test whose training sizes are reported
with the saved split audits: training folds range from
{count(min(a['n_train'] for a in homology50_splits))} to
{count(max(a['n_train'] for a in homology50_splits))} cells.
Consequently, its performance combines reduced training support and a change in
chemical distribution; it does not isolate the effect of sequence distance.

\subsection{{Incremental predictive value of recorded context}}
Table~\ref{{tab:models}} reports each boosting comparison on paired cells;
Fig.~\ref{{fig:gain}} includes both ridge and boosting families under publication
and temporal transfer. Positive differences mean lower absolute error with context.
For temporal descriptor boosting, the reduction is
{fmt(pairs[(pairs.protocol=='temporal') & (pairs.base=='sequence_hgb') & (pairs.stratum=='all')].iloc[0].mae_reduction_log2)}
log2 units, with a crossed-bootstrap interval
[{fmt(pairs[(pairs.protocol=='temporal') & (pairs.base=='sequence_hgb') & (pairs.stratum=='all')].iloc[0].ci_low)},
{fmt(pairs[(pairs.protocol=='temporal') & (pairs.base=='sequence_hgb') & (pairs.stratum=='all')].iloc[0].ci_high)}].
These are pointwise, exploratory intervals conditional on fitted models.
The complete comparison set, including unfavorable differences, is retained.
Across the six representation--learner pairs, point reductions range from
{publication_context['minimum_point_gain_log2']:.3f} to
{publication_context['maximum_point_gain_log2']:.3f} log2 under publication transfer
and from {temporal_context['minimum_point_gain_log2']:.3f} to
{temporal_context['maximum_point_gain_log2']:.3f} under temporal transfer.
Respectively, {publication_context['pointwise_intervals_wholly_above_zero']} and
{temporal_context['pointwise_intervals_wholly_above_zero']} of six pointwise intervals
lie wholly above zero; this count is descriptive and is not a multiplicity-adjusted test.
The temporal point reductions amount to
{temporal_context['minimum_relative_point_gain_percent']:.2f}--{temporal_context['maximum_relative_point_gain_percent']:.2f}\%
of each corresponding baseline MAE. Their experimental usefulness is not established.
The joint design additionally separates actual known/new sequence and publication
exposure (Supplementary Fig.~S2). A cell can contribute to different exposure
strata, but repeated predictions are collapsed within each cell and stratum.

\subsection{{Within-assay candidate selection}}
The separate temporal screening diagnostic contains
{count(screen_base.publications)} publications with evaluable assay blocks.
Selecting the 20\% of candidates with the lowest predicted MICs in each block
(rounded up to a whole candidate) gives a publication-balanced
observed gain of {fmt(screen_base.mean_screening_gain_log2)} log2 relative to the block
mean for descriptor boosting and {fmt(screen_context.mean_screening_gain_log2)} with context.
These are descriptive selection outcomes on the available retrospective blocks,
without an inferential claim about their difference. They measure a different
decision from absolute-error reduction and do not establish prospective utility.
{tie_result_text}

\subsection{{Cross-publication support and variance sensitivity}}
In the exact-label inferential cohort, {100*s['single_publication_fraction']:.1f}\% of chemical entities occur in one
publication; only {count(s['multi_publication_entities'])} appear in more than one.
The chemical--publication graph has {count(s['bipartite_components'])} connected
components. In the additive exact-label REML model, the conditional publication
fraction is {100*p_all:.1f}\%; restricting to shared chemical entities yields
{100*p_shared:.1f}\% across {count(shared['n_used'])} cells.
Requiring a shared chemical--exact-target pair gives {100*p_shared_target:.1f}\%
across {count(shared_target['n_used'])} cells.
The cohorts differ, so this contrast is a sensitivity analysis, not an estimate
of a causal effect (Fig.~\ref{{fig:variance}}).
Of five planned censored-model specifications, {passed} currently pass the
computational diagnostic gate. Only passing fits appear in the posterior table;
unavailable or failing fits remain explicitly marked in the supplement.
An exploratory design audit finds linearly independent component matrices in
{sum(r['rank'] == r['component_count'] for r in data['covariance_design']['cohorts'].values())}
of four latent Gaussian covariance designs (Supplementary Information).
Sparse cross-publication overlap therefore does not by itself establish algebraic
nonidentifiability. This design check and the computational diagnostics do not
validate the model assumptions or confer causal meaning on the components.
Conditional posterior predictive checks have been performed for {checked_fits}
passing specifications; {flagged_fits} have at least one extreme-tail flag among
the five exact-record summaries. The supplement reports those diagnostics
without treating them as external validation or as calibrated significance tests.

\subsection{{Matched rank contrasts and record agreement}}
Matching publication, exact target and the other recorded context fields leaves
{count(rank['pairs'])} peptide-pair contrasts in {count(rank.get('blocks', 0))}
blocks from {count(rank.get('publications', 0))} publications.
The strict-inversion count is {count(rank.get('inversions', 0))}, a proportion of
{fmt(rank.get('rate', np.nan))}.
The crossed-bootstrap interval for that proportion is
[{fmt(rank.get('crossed_bootstrap_ci95', [np.nan, np.nan])[0])},
{fmt(rank.get('crossed_bootstrap_ci95', [np.nan, np.nan])[1])}].
Matching missing fields does not establish identical protocols.
An exploratory threshold sensitivity reduces the inversion count to
{count(near_threshold.inversions)} when both absolute differences must exceed
1.01 rather than 1.00 log2 units. This sensitivity was added after inspecting a
near-boundary mass-to-molar conversion. The primary threshold is retained;
more than one log2 unit in molar concentration need not mean more than one
original experimental dilution step (Supplementary Information).
The subset with all four matching metadata fields reported contains
{count(data['rank']['all_four_fields_reported']['pairs'])} contrasts.
Separately, the GRAMPA grouping with a reported strain contains
{count(grampa['reported_strain_only']['groups'])} repeat groups and gives a
descriptive ICC of {fmt(grampa['reported_strain_only']['icc'])}
[{fmt(grampa['reported_strain_only']['icc_ci95'][0])},
{fmt(grampa['reported_strain_only']['icc_ci95'][1])}].
No same-source, reported-strain replicate group is available in that analysis.
These records therefore cannot identify a universal measurement-noise ceiling.
"""
    if interval_path.exists():
        interval_example = interval_data[
            (interval_data.protocol == "temporal") &
            (interval_data.score_kind == "publication_maximum") &
            (interval_data.model == "sequence_hgb") &
            np.isclose(interval_data.nominal_coverage, .9)
        ].iloc[0]
        result_text += rf"""
\subsection{{Predictive coverage and resolution are distinct}}
The exploratory calibration analysis uses separate fitting, calibration and test
publications. For temporal descriptor boosting, the nominal 90\% interval based
on publication-maximum residuals covers
{100*interval_example.cell_weighted_coverage:.1f}\% of test cells and every cell in
{100*interval_example.whole_publication_coverage:.1f}\% of test publications.
Its half-width is {interval_example.radius_log2:.2f} log2 units, corresponding
to division and multiplication by {2**interval_example.radius_log2:.1f} around
the central MIC prediction. High empirical coverage therefore coexists with
limited resolution for distinguishing candidates with similar potency.
All four model comparisons, both nominal levels, and the pooled-row alternative
are retained in the saved outputs. No distribution-free guarantee is asserted
under the observed study and temporal shifts.
"""
    study_scope = read_json(ROOT/"data/study_scope_2026-10-20.json")
    if study_scope.get("manuscript_authoring_required", True):
        (generated/"results_summary.tex").write_text(result_text)
    write_table_guide(scope, generated, output, posterior)
    trace = {
        "scope": scope, "status": status, "created_utc": datetime.now(timezone.utc).isoformat(),
        "passing_censored_specifications": passed, "planned_censored_specifications": len(expected),
        "posterior_specifications_checked": checked_fits,
        "posterior_specifications_with_extreme_tail_flag": flagged_fits,
        "context_effect_summary": context_summary,
        "study_type": "strictly_in_silico",
        "experimental_validation_required": False,
        "prospective_measurements_performed": False,
        "manuscript_authoring_required": study_scope.get("manuscript_authoring_required", True),
        "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [source/"curation_manifest.json", source/"benchmark_cells.csv",
                            source/"all_exact_assay_cells.csv", source/"activities.csv",
                            benchmark/"completion_audit.json", benchmark/"metrics.csv",
                            benchmark/"verified_split_audits.json",
                            benchmark/"paired_context_improvements.csv",
                            benchmark/"screening_summary.csv", benchmark/"calibration_diagnostics.csv",
                            output/"statistics/design_support.json", output/"statistics/reml_sensitivity.json",
                            output/"statistics/covariance_design_audit.json",
                            output/"statistics/matched_rank_summary.json", grampa_path,
                            source/"cluster_manifest.json", source/"structural_alias_audit.json",
                            ROOT/"results/reconstruction_2026-10-20/literature_review.json",
                            *sorted((ROOT/"manuscript/reference_metadata").glob("*.json")),
                            ROOT/"data/study_scope_2026-10-20.json",
                            generated/"figure_guide.json",
                            generated/"table_guide.json",
                            *sorted(generated.glob("table_*.tex")),
                            *posterior_sources]},
    }
    (generated/"report_provenance.json").write_text(json.dumps(trace, indent=2))
    print(json.dumps({k: v for k, v in trace.items() if k != "files"}, indent=2), flush=True)
    return data
