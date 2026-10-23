"""Additive final documentation and closure of the audited editorial extension."""
from pathlib import Path
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

import pandas as pd
from pypdf import PdfReader

from .editorial_common import ROOT, BASE, OUT, sha, utc, read_json, write_json, verify_preservation, rerun_original_audit
from .journal_materials import DOC, GEN, FIT_ORDER, FIT_LABELS, PROTOCOLS, PROTOCOL_LABELS

NOTE = ROOT/"notes/01_memoria_consolidacion_editorial_2026-10-22.md"
GAP_SOURCE = ROOT/"results/reconstruction_2026-10-20/source_review/editorial_positioning_final_analysis_20261022.json"


def document():
    registry = read_json(GEN/"numeric_provenance.json")
    numbers = registry["numbers"]
    n = lambda key: numbers[key]["rendered"]
    build = read_json(OUT/"manuscript_build/build_receipt.json")
    source = read_json(GAP_SOURCE)
    paths = lambda *names: [str((OUT/name).relative_to(ROOT)) for name in names]
    gaps = [
        {"original_requirement":source["remaining_for_a_stronger_editorial_case"][0],
         "status":"addressed_with_retrospective_operational_evidence",
         "resolution":"Training-only empirical Bayes reference calibration improves unrevealed query accuracy beyond recorded metadata on identical support, with explicit reference cost, eligibility and publication-bootstrap uncertainty.",
         "evidence":paths("assay_calibration/metrics.csv","assay_calibration/paired_improvements.csv","assay_calibration/support_and_reference_availability.csv","assay_calibration/independent_audit.json"),
         "limits":"No prospective measurement or universal reference panel; query composition changes with anchor budget; uncertainty conditions on fitted models and variance estimates."},
        {"original_requirement":source["remaining_for_a_stronger_editorial_case"][1],
         "status":"conditional_requirement_not_invoked_no_specialized_tool_superiority_claim",
         "resolution":"Two deterministic multimodal neural comparators evaluated on all original assignments strengthen the capacity comparison. The manuscript does not claim superiority over contemporary specialized AMP tools.",
         "evidence":paths("deep_contextual/completion.json","deep_contextual/paired_improvements.csv","deep_contextual/independent_audit.json"),
         "limits":"A full equivalent-task benchmark of current specialized tools was not performed; such a superiority claim would require that additional evidence."},
        {"original_requirement":source["remaining_for_a_stronger_editorial_case"][2],
         "status":"addressed_by_explicit_selection_sensitivity_and_restricted_interpretation",
         "resolution":"All retained posteriors have reproducible upper/lower-tail sensitivities with original-unit window compatibility. The unknown exact-observation windows prevent identification of a unique selection-versus-residual causal share.",
         "evidence":paths("posterior_tail_gaps/tail_gap_summary.csv","posterior_tail_gaps/source_unit_window_compatibility.csv.gz","posterior_tail_gaps/independent_audit.json"),
         "limits":"Original model-adequacy flags remain. No mechanistic laboratory effect, conservation law or universal variance partition is asserted."},
    ]
    write_json(OUT/"editorial_gap_resolution.json",{
        "created_utc":utc(),"source":str(GAP_SOURCE.relative_to(ROOT)),"source_sha256":sha(GAP_SOURCE),
        "requirements":gaps,"all_requested_analyses_executed":True,"all_scientific_uncertainties_eliminated":False,
        "editorial_assessment":{"preferred_target":"Nature Computational Science, Analysis",
            "assessment":"Ambitious but more defensible after the operational reference-budget comparison. This is an editorial judgment, not an acceptance prediction.",
            "alternative":"Nature Machine Intelligence is a less direct fit: FiLM and empirical Bayes are established methods; the contribution is the computational analysis and evaluation design, not a new general learning algorithm.",
            "guideline_sources":str((OUT/"method_and_journal_sources.json").relative_to(ROOT))},
        "submission_metadata_remaining":["Author-approved author list, affiliations, contributions, funding and competing-interest statements","Public data/code archive accession and applicable redistribution terms"],
        "submission_or_public_release_performed":False})
    cm = pd.read_csv(OUT/"assay_calibration/metrics.csv")
    cp = pd.read_csv(OUT/"assay_calibration/paired_improvements.csv")
    dm = pd.read_csv(OUT/"deep_contextual/metrics.csv")
    dp = pd.read_csv(OUT/"deep_contextual/paired_improvements.csv")
    tails = pd.read_csv(OUT/"posterior_tail_gaps/tail_gap_summary.csv")
    p = "cal.temporal.publication_target.random.5.point_frozen.sequence_hgb."
    gp = "gain.temporal.publication_target.random.5.point_frozen.sequence_hgb."
    lines = ["# Memoria técnica: consolidación editorial expanded", "",
        "Extensión aditiva de la reconstrucción cerrada. Este archivo es documentación del análisis; no reemplaza la memoria histórica ni sus auditorías.", "",
        "## Alcance y preservación", "",
        f"El nuevo ámbito `data/study_scope_editorial_2026-10-22.json` registra la autorización posterior para redactar el manuscrito. El ámbito histórico que difería esa redacción permanece intacto. El estudio es estrictamente retrospectivo e in silico; no se generaron mediciones experimentales nuevas. La fase se almacena en `{OUT.relative_to(ROOT)}`.", "",
        "`preservation_manifest.json` contiene los hashes SHA-256 de todos los artefactos cerrados. `final_preservation_review.json` registra la comprobación completa al cierre. Se preservan el catálogo, las cadenas y resultados de los cinco modelos, las particiones y predicciones clásicas y el `completion_audit.json` original. El script original de auditoría se ejecuta sin modificar su lógica: únicamente sus tres escrituras se redirigen a `final_original_audit/`, con recibo explícito. No ejecutar el script 25 directamente sobre los resultados cerrados.", "",
        "## Lectura de los resultados", "",
        f"La cohorte predictiva contiene {n('nn.unique_benchmark_cells')} celdas; la prueba temporal contiene {n('temporal.test_cells')} de publicaciones {n('temporal.start_year')}–{n('temporal.end_year')}. Los recuentos de registros originales, registros inferenciales y celdas colapsadas no son intercambiables. El manuscrito y la tabla S1 explican los denominadores.", "",
        f"La mejora práctica principal es {n(gp+'calibration_gain_log2')} log2 MIC, IC [{n(gp+'calibration_gain_ci_low')}, {n(gp+'calibration_gain_ci_high')}], con cinco referencias aleatorias por bloque publicación × patógeno. El MAE pasa de {n(p+'uncalibrated.mae_log2')} a {n(p+'empirical_bayes.mae_log2')} ({n('primary.relative_mae_reduction_pct')}%). La comparación de metadatos en esas mismas consultas mejora {n(gp+'metadata_gain_same_queries_log2')}; la ventaja adicional de calibrar es {n(gp+'gain_over_metadata_log2')} si ese campo aparece en la tabla de comparaciones." if gp+'gain_over_metadata_log2' in numbers else f"La mejora práctica principal es {n(gp+'calibration_gain_log2')} log2 MIC, IC [{n(gp+'calibration_gain_ci_low')}, {n(gp+'calibration_gain_ci_high')}], con cinco referencias aleatorias por bloque publicación × patógeno. El MAE pasa de {n(p+'uncalibrated.mae_log2')} a {n(p+'empirical_bayes.mae_log2')} ({n('primary.relative_mae_reduction_pct')}%). Los metadatos en esas mismas consultas mejoran {n(gp+'metadata_gain_same_queries_log2')}.", "",
        "### Protocolo de calibración", "",
        f"Se conservan las predicciones fuera de muestra de descriptor HGB y ESM HGB, con y sin contexto, para temporal y los cinco folds por publicación de semilla {n('fewshot.seed')}. Los bloques usan publicación o publicación × target_exact. Se exige un mínimo común de {n('fewshot.minimum_entities')} entidades químicas distintas. Los presupuestos son 1, 2, 3 y 5 referencias, con {n('fewshot.draws')} sorteos aleatorios anidados por bloque; la política estándar es determinista y prioriza frecuencia en el subconjunto de ajuste de entrenamiento.", "",
        "Cada ancla revela una celda de una entidad. Todas las celdas de esa entidad se excluyen de las consultas de ese bloque. La selección no usa MIC ni residuos de prueba. Entre bloques de distintos patógenos puede repetirse una entidad, pero cada ajuste usa exclusivamente sus propias anclas. La política estándar requiere que el candidato exista en el bloque retrospectivo; se registran candidatos sin antecedentes de entrenamiento y no se presenta un panel universal.", "",
        "Las varianzas entre bloques y dentro de bloques se estiman mediante momentos ANOVA no balanceados, únicamente con residuos fuera de ajuste de publicaciones de entrenamiento. La varianza entre bloques se trunca a cero en la frontera. El desplazamiento es k σ_b²/(k σ_b²+σ_e²) por la media residual de las anclas. No se añade la media global residual como corrección oculta.", "",
        "Las publicaciones previamente reservadas al calibrador del modelo clásico se dividen en dos conjuntos disjuntos: A estima varianzas; B estima radios conformales. Ambos son ajenos al ajuste propio y a la prueba exterior. `point_frozen` corrige las predicciones originales del modelo entrenado con todo el fold; usa la varianza del modelo de ajuste menor como aproximación declarada. `interval_proper_fit` corrige el predictor propio guardado y calcula sus intervalos con B. No se transfieren intervalos entre predictores distintos.", "",
        "Los radios conformales utilizan el máximo error absoluto por publicación de conformidad, con corrección finita de rango e infinito si no hay publicaciones suficientes. Se registran coberturas por celda y de todas las consultas de una publicación. La garantía requiere intercambiabilidad de publicaciones completas, con tamaño, composición y mecanismo de selección; no se presupone para literatura posterior.", "",
        f"Los errores se promedian sobre sorteos mediante estadísticos suficientes y luego se agrupan por publicación. Hay {n('fewshot.bootstrap')} réplicas bootstrap pareadas; no se cuentan sorteos como experimentos adicionales. El análisis principal pondera celdas y conserva MAE con igual peso por publicación como contraste secundario. Los intervalos de confianza son condicionales al ajuste, varianzas, curación y planes; no incluyen reentrenamiento.", "",
        "| k | MAE sin calibrar | MAE EB | Ganancia [IC] | Metadatos, mismas consultas |",
        "|---:|---:|---:|---:|---:|"]
    for k in [1,2,3,5]:
        a=cm.query("protocol=='temporal' and grouping=='publication_target' and strategy=='random' and branch=='point_frozen' and model=='sequence_hgb' and k==@k").set_index("method")
        b=cp.query("protocol=='temporal' and grouping=='publication_target' and strategy=='random' and branch=='point_frozen' and base_model=='sequence_hgb' and calibration_method=='empirical_bayes' and k==@k").iloc[0]
        lines.append(f"| {k} | {a.loc['uncalibrated','mae_log2']:.6f} | {a.loc['empirical_bayes','mae_log2']:.6f} | {b.calibration_gain_log2:.6f} [{b.calibration_gain_ci_low:.6f}, {b.calibration_gain_ci_high:.6f}] | {b.metadata_gain_same_queries_log2:.6f} |")
    lines += ["",f"El escenario temporal principal abarca {n('primary.support.eligible_publications')} publicaciones y {n('primary.support.eligible_blocks')} bloques. Antes de revelar referencias incluye {n('primary.eligible_fraction_pct')}% de la prueba temporal. Con cinco referencias se consumen {n('primary.support.anchor_measurements')} mediciones por sorteo y quedan {n('primary.mean_queries')} consultas promedio, {n('primary.query_fraction_pct')}% de toda la prueba temporal. Los presupuestos cambian la población remanente; no se interpreta la curva como comparación sobre consultas invariantes.", "",
        f"El ancho completo del intervalo temporal al 90% se reduce {n('primary.interval_width_reduction_pct')}%, pero conserva una semianchura multiplicativa geométrica de {n('primary.interval_halfwidth_fold')}. La cobertura de publicación es {n('cal.temporal.publication_target.random.5.interval_proper_fit.sequence_hgb.empirical_bayes.publication_coverage90')}%; por publicación retenida es {n('cal.publication.publication_target.random.5.interval_proper_fit.sequence_hgb.empirical_bayes.publication_coverage90')}%, inferior al nominal. Mejorar MAE no equivale a resolver la precisión o cobertura en todos los contextos.", "",
        "### Comparador neuronal", "",
        f"Se completaron {n('nn.prediction_files')} ajustes en {n('nn.splits')} particiones originales y {n('nn.raw_predictions')} predicciones individuales. Los siete protocolos son random, sequence, homology80, homology50, publication, joint y temporal. Los estratos joint se mantienen separados. Las pérdidas se promedian por celda y estrato antes del bootstrap cruzado publicación × alias de secuencia, con {n('nn.bootstrap_repeats')} réplicas.", "",
        f"Las entradas son ESM congelado ({n('nn.esm_dimensions')} coordenadas), Morgan quiral ({n('nn.morgan_dimensions')}) y {n('nn.descriptor_count')} escalares; sus proyecciones son {n('nn.esm_projection')}, {n('nn.morgan_projection')} y {n('nn.descriptor_projection')}, con embedding target_exact de {n('nn.target_embedding')}. Dos capas ocultas de ancho {n('nn.hidden')} usan LayerNorm, SiLU y dropout {n('nn.dropout')}. FiLM recibe medium, log10_cfu, ph y nacl_reported_concentration, con indicadores de ausencias. No se interpreta NaCl registrado como fuerza iónica armonizada. Las rutas contextuales clásicas además incluyen otras categorías; su contraste con las redes no aísla solamente arquitectura.", "",
        f"La validación interna reserva una fracción {n('nn.validation_publication_fraction')} de publicaciones de entrenamiento. Imputación, escalado y vocabularios se ajustan dentro del subconjunto correspondiente. AdamW usa tasa {n('nn.learning_rate')}, decaimiento {n('nn.weight_decay')}, lotes {n('nn.batch_size')}, hasta {n('nn.max_epochs')} épocas y paciencia {n('nn.patience')}. Se escoge la época por MAE interno; después se reinicializa y reajusta con todo el entrenamiento exterior. No se usa el fold de prueba para elegir épocas. FiLM comienza con modulación identidad y comparte la inicialización de la red sin contexto.", "",
        "Se verificaron todas las decisiones de early stopping, los hashes de pesos y la ausencia de fuga; se reprodujeron las dos predicciones temporales completas desde sus checkpoints. El determinismo es específico del entorno guardado, no una garantía entre plataformas. El posible solapamiento del preentrenamiento de ESM con las secuencias del catálogo no se ha descartado.", "",
        "| Protocolo | MAE MLP | MAE FiLM | R² FiLM | Ganancia FiLM [IC] |",
        "|---|---:|---:|---:|---:|"]
    for protocol,label in zip(PROTOCOLS,PROTOCOL_LABELS):
        a=dm.query("protocol==@protocol and stratum=='all'").set_index("model")
        b=dp.query("protocol==@protocol and stratum=='all' and base=='multimodal_mlp'").iloc[0]
        lines.append(f"| {label} | {a.loc['multimodal_mlp','mae_log2']:.6f} | {a.loc['multimodal_film_context','mae_log2']:.6f} | {a.loc['multimodal_film_context','r2']:.6f} | {b.mae_reduction_log2:.6f} [{b.ci_low:.6f}, {b.ci_high:.6f}] |")
    lines += ["",f"La componente mayor de cluster_50 contiene {n('homology.largest_component_cells')} celdas ({n('homology.largest_component_pct')}%). La cifra se recalcula directamente en benchmark_cells.csv. Retenerla cambia a la vez novedad y soporte de entrenamiento. La semejanza es distancia global de edición normalizada con cierre transitivo, no identidad BLAST ni una intervención causal sobre homología.", "",
        "### Diagnóstico de colas", "",
        "Se extrajeron los mismos 200 draws de cada uno de los cinco ajustes posteriores, combinando sus cuatro cadenas y se reconciliaron 20 archivos de cadenas con el manifiesto original. Se reprodujeron 25 resúmenes predictivos originales. No se volvió a ajustar Stan. Los escenarios de ventanas comparten uniformes para reducir ruido Monte Carlo.", "",
        "Matemáticamente, f_exact(y|x) es proporcional a f_latent(y|x) π_exact(y|x). Si la observación exacta requiere Y≤U, su cuantil normal seleccionado es μ+s Φ⁻¹[p Φ((U−μ)/s)]. Sin U observado para los registros exactos, la factorización no identifica una contribución causal única de selección y forma residual.", "",
        "Las ventanas locales agrupan publicación × target_exact × unidad original. Un límite de masa se convierte por peso molecular para cada registro, evitando aplicar un mismo umbral molar a péptidos con masas distintas. Los escenarios de límite único y máximo solo se aplican a bloques compatibles; los demás registros quedan sin truncar. Envolventes de soporte se derivan de resultados observados y son sensibilidades, no validación. Los umbrales molares comunes pueden contradecir exactos y se conservan esos recuentos.", "",
        "| Escenario, modelo censurado completo | q95 replicado | Fracción de brecha positiva retirada | Exactos con límite finito | Exactos contradichos |",
        "|---|---:|---:|---:|---:|"]
    for _,r in tails.query("fit=='all_prior1.5'").iterrows():
        lines.append(f"| {r.scenario} | {r.q95_median:.6f} | {100*r.fraction_positive_gap_removed_median:.2f}% | {r.finite_limit_exact_records:.0f} | {r.exact_records_contradicted:.0f} |")
    lines += ["",f"Hay {n('tail.blocks_with_exact_above_maximum_right_limit')} bloques incompatibles con un único límite superior compartido, que contienen {n('tail.exact_records_in_contradicted_blocks')} registros exactos. El modelo exact-only ya tiene brecha superior negativa; la fracción de una brecha positiva es indefinida. Student-t modifica solo la distribución residual con varianza equiparada, conservando parámetros posteriores Gaussianos y contribuciones singleton Gaussianas: no es un modelo alternativo reajustado. La tabla S9 muestra que las sensibilidades del extremo superior no reparan automáticamente q05.", "",
        "## Balance editorial y límites de las afirmaciones", "",
        "`editorial_gap_resolution.json` reproduce exactamente las tres brechas del dictamen original y enlaza su respuesta. La conclusión diferenciadora es comparar información de referencias nuevas con metadatos ya registrados, a un coste y soporte explícitos. No se reivindica inventar FiLM, Empirical Bayes, la dependencia contextual ni los problemas de generalización. El requisito de superioridad frente a herramientas especializadas no se activa porque no se afirma esa superioridad; la comparación exhaustiva equivalente no se realizó.", "",
        "Nature Computational Science, Analysis, es el destino prioritario ambicioso por el resultado operativo y el análisis del proceso de observación. Nature Machine Intelligence tiene un encaje menos directo al faltar una innovación general en aprendizaje. Son juicios editoriales fundamentados en los resultados y las guías consultadas, no estimaciones de aceptación. La ejecución in silico está completa bajo este alcance; los límites de identificación y transporte se documentan como resultados científicos, no se ocultan para simular un cierre más fuerte.", "",
        "Las fracciones químicas y de publicación son condicionales a la estructura aleatoria y al residuo, excluyen variación de efectos fijos y no establecen causalidad de laboratorio. El cambio exact-only no demuestra eliminación de la mitad de una señal biológica porque también cambia la cohorte y la verosimilitud. La suma química + publicación es exploratoria posterior, no una ley de conservación ni una prueba de equivalencia preespecificada.", "",
        "## Documentos, datos y regeneración", "",
        f"Los PDF tienen {build['products'][0]['pages']} y {build['products'][1]['pages']} páginas. El abstract tiene {build['word_counts']['abstract']} palabras y el cuerpo principal {build['word_counts']['main_text_excluding_methods_references_legends']}, excluyendo Methods, referencias y leyendas. Hay seis secciones de Results, cinco figuras y una tabla principales, seis figuras y nueve tablas suplementarias. Los valores empíricos se insertan mediante macros de un registro con fuente, derivación y SHA-256; las constantes matemáticas, nombres de modelos e identificadores de figuras no son estimaciones empíricas.", "",
        "| Artefacto | Ubicación |",
        "|---|---|",
        "| Manuscrito y suplemento | `output/pdf/AMP_manuscript_expanded_2026-10-20.pdf`, `output/pdf/AMP_supplement_expanded_2026-10-20.pdf` |",
        "| Fuentes LaTeX, bibliografía, macros y figuras vectoriales/PNG | `manuscript/editorial_expanded_20261022/` |",
        "| Métricas completas, sesgo, intervalos, errores y comparaciones | `assay_calibration/{metrics,paired_improvements}.csv` y `deep_contextual/{metrics,paired_improvements}.csv`, dentro de la fase |",
        "| Anclas, varianzas, particiones A/B y estadísticos por publicación | `assay_calibration/`, dentro de la fase |",
        "| Pesos, transformaciones, curvas y predicciones por split | `deep_contextual/`, dentro de la fase |",
        "| Ventanas, q05/q95 por draw, resúmenes y extractos posteriores | `posterior_tail_gaps/`, dentro de la fase |",
        "| Procedencia de números y figuras | `manuscript/editorial_expanded_20261022/generated/{numeric_provenance,display_provenance}.json` |",
        "| Inspección visual de cada página y figura | `manuscript_build/visual_review.json`, dentro de la fase |",
        "| Tests, auditoría original y cierre aditivo | `final_tests.xml`, `final_original_audit/`, `completion_audit_editorial.json`, dentro de la fase |",
        "| Hashes de todos los artefactos nuevos | `artifact_manifest.json`, dentro de la fase |", "",
        "Comandos desde la raíz, usando el entorno conservado. Los scripts 32–34 reusan los artefactos válidos y las configuraciones bloqueadas. Una modificación de los protocolos exige una nueva versión; no editar el módulo neuronal bajo checkpoints existentes.", "",
        "```sh\nvenv/bin/python scripts/32_evaluate_assay_calibration.py\nvenv/bin/python scripts/33_benchmark_deep_contextual.py --workers 2\nvenv/bin/python scripts/33_benchmark_deep_contextual.py --summarize\nvenv/bin/python scripts/34_analyze_posterior_tail_gaps.py\nvenv/bin/python scripts/36_audit_editorial_consolidation.py --part all\nvenv/bin/python scripts/35_build_journal_manuscript.py --render\n# Revisar visualmente todas las páginas de los PDF recién generados.\n# Registrar los hashes de la revisión antes de cerrar.\nvenv/bin/python scripts/37_close_editorial_phase.py\n```", "",
        "El script 37 exige que la inspección visual corresponda exactamente a los PDF y PNG actuales; no genera automáticamente una aprobación visual. Ejecuta los tests originales y adicionales, la auditoría original redirigida, la comprobación SHA-256 completa y el cierre documental. El manifiesto nuevo excluye sus propios bytes y el recibo final para evitar referencias hash circulares. No se ha publicado un repositorio ni se ha asignado un DOI de datos; las rutas son locales. Los metadatos de autoría, afiliaciones, financiación y conflictos y el acceso público corresponden a la preparación administrativa de una futura entrega.", "",
        "## Referencias, fuentes y límites de acceso", "",
        "La revisión previa está en `results/reconstruction_2026-10-20/literature_review.json` y los dictámenes/fuentes primarias en `source_review/`. La bibliografía se conserva en `manuscript/editorial_expanded_20261022/references_editorial.bib`; las dos incorporaciones bibliográficas tienen metadatos propios. Se diferencia expresamente lectura del texto completo, abstract de autor y repositorio. El trabajo de Yue se revisó a nivel de abstract y BATTLE-AMP además mediante repositorio oficial fechado; no se afirma haber reproducido esas herramientas.", ""]
    for s in read_json(OUT/"method_and_journal_sources.json")["sources"]:
        lines.append(f"- [{s['key']}]({s['url']}): {s['supports']} Acceso: {s['access']}.")
    lines += ["", "La evidencia de pruebas, preservación y cierre se consulta en los recibos enlazados; este documento no sustituye los resultados verificables ni convierte las limitaciones científicas en garantías.", ""]
    NOTE.parent.mkdir(parents=True,exist_ok=True)
    NOTE.write_text("\n".join(lines))


def close():
    build = read_json(OUT/"manuscript_build/build_receipt.json")
    visual = read_json(OUT/"manuscript_build/visual_review.json")
    assert visual["passed"] and visual["all_pages_actually_inspected"]
    visual_products={r["path"]:r for r in visual["products"]}
    for p in build["products"]:
        v=visual_products[p["path"]]
        assert sha(ROOT/p["path"]) == p["sha256"] == v["pdf_sha256"]
        assert len(PdfReader(ROOT/p["path"]).pages)==p["pages"]==len(v["pages"])
        for row in v["pages"]:
            assert row["inspected"] and sha(ROOT/row["path"])==row["sha256"]
    for row in visual["figures"]:
        for path,digest in row["files_sha256"].items():assert sha(ROOT/path)==digest
        assert row["inspected"]
    numeric=read_json(GEN/"numeric_provenance.json")
    assert sha(GEN/"numeric_provenance.json")==build["numeric_provenance_sha256"]
    assert sha(GEN/"display_provenance.json")==build["display_provenance_sha256"]
    for path,digest in {**numeric["source_sha256"],**build["source_sha256"]}.items():
        assert sha(ROOT/path)==digest, path
    displays=read_json(GEN/"display_provenance.json")
    for name,r in displays["figures"].items():
        for f,digest in r["files_sha256"].items():assert sha(GEN/f)==digest
        assert sha(GEN/(name+"_data.csv"))==r["plotted_data_sha256"]
    for name,r in displays["tables"].items():assert sha(GEN/(name+".tex"))==r["sha256"]
    summaries={}
    for name in ["deep_contextual","assay_calibration","posterior_tail_gaps"]:
        a=read_json(OUT/name/"completion.json"); b=read_json(OUT/name/"independent_audit.json")
        assert a["passed"] and b["passed"]
        if name=="posterior_tail_gaps":assert sha(OUT/name/"tail_gap_summary.csv")==a["summary_sha256"]==b["summary_sha256"]
        else:
            assert sha(OUT/name/"metrics.csv")==a["metrics_sha256"]
            assert sha(OUT/name/"paired_improvements.csv")==a["comparisons_sha256"]
        summaries[name]={"completion_sha256":sha(OUT/name/"completion.json"),"independent_audit_sha256":sha(OUT/name/"independent_audit.json")}
    print("Run original and extension tests",flush=True)
    command=[sys.executable,"-m","pytest","-q","tests/test_reconstruction.py","tests/test_editorial_extension.py","--junitxml="+str(OUT/"final_tests.xml")]
    with (OUT/"final_tests.log").open("w") as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    cases=list(ET.parse(OUT/"final_tests.xml").iter("testcase"))
    assert not any(list(c) for c in cases)
    original=sum(c.attrib.get("classname","").endswith("test_reconstruction") for c in cases)
    assert original==42 and len(cases)>=55
    print("Run unchanged original 19-gate audit with protected write routing",flush=True)
    old=rerun_original_audit(OUT/"final_original_audit")
    print("Rehash all protected original files",flush=True)
    preservation=verify_preservation(full=True)
    write_json(OUT/"final_preservation_review.json",preservation)
    document()
    sources=[ROOT/"amp_rebuild"/f for f in ["editorial_common.py","assay_calibration.py","deep_contextual.py","posterior_tail_gaps.py","editorial_audit.py","journal_materials.py","editorial_release.py"]]
    sources += [p for p in (ROOT/"scripts").glob("*.py") if p.name[:2] in ["32","33","34","35","36","37"]]
    sources += [ROOT/"tests/test_editorial_extension.py",ROOT/"data/study_scope_editorial_2026-10-22.json",NOTE]
    sources += [ROOT/p["path"] for p in build["products"]]
    sources += [ROOT/r["path"] for p in visual["products"] for r in p["pages"]]
    sources += [p for folder in [OUT,DOC] for p in folder.rglob("*") if p.is_file()]
    excluded={OUT/"artifact_manifest.json",OUT/"completion_audit_editorial.json"}
    files={}
    for path in sorted(set(sources)-excluded):
        if "__pycache__" in path.parts or path.name.endswith(".partial"):continue
        files[str(path.relative_to(ROOT))]={"sha256":sha(path),"bytes":path.stat().st_size}
    write_json(OUT/"artifact_manifest.json",{"created_utc":utc(),"files":files,
        "exclusions":[str(p.relative_to(ROOT)) for p in sorted(excluded)],
        "interpretation":"New artifacts only, with original-input hashes separately preserved; manifest and closing receipt excluded to prevent self-referential hashes."})
    checks={
        "original_artifacts_byte_identical":preservation["passed"] and preservation["full_sha256_recheck"],
        "original_42_tests_pass":original==42,"extension_tests_pass":len(cases)-original>=13,
        "original_19_gates_pass":old["study_complete"] and len(old["requirements"])==19,
        "few_shot_complete_and_independently_audited":True,"deep_complete_and_independently_audited":True,
        "tail_complete_and_independently_audited":True,
        "numeric_and_display_source_hashes_match":True,"all_referenced_macros_resolved":build["all_numeric_macros_resolved"],
        "six_results_sections":build["main_result_sections"]==6,
        "analysis_word_limits":100<=build["word_counts"]["abstract"]<=150 and build["word_counts"]["main_text_excluding_methods_references_legends"]<=3500,
        "main_display_limit":build["main_display_items"]<=6,
        "bibliography_keys_resolved":set(build["bibliography"]["cited_keys"])<=set(build["bibliography"]["available_keys"]),
        "all_pdf_pages_actually_inspected":True,"all_additional_figures_inspected":len(visual["figures"])==11,
        "technical_memory_and_exact_gap_mapping_written":NOTE.is_file() and (OUT/"editorial_gap_resolution.json").is_file(),
        "new_artifacts_sha256_manifest_written":True,
        "original_adequacy_limits_and_unidentified_causal_share_retained":not read_json(OUT/"posterior_tail_gaps/completion.json")["causal_fraction_identified"],
    }
    assert all(checks.values())
    receipt={"created_utc":utc(),"scientific_phase_complete":True,"passed":True,"requirements":checks,
        "tests":{"original":original,"extension":len(cases)-original,"total":len(cases),"junit_sha256":sha(OUT/"final_tests.xml")},
        "original_closure_gates":len(old["requirements"]),"protected_files":preservation["protected_files"],
        "analysis_receipts":summaries,"products":build["products"],
        "visual_review_sha256":sha(OUT/"manuscript_build/visual_review.json"),
        "artifact_manifest_sha256":sha(OUT/"artifact_manifest.json"),"new_manifest_files":len(files),
        "original_preservation_manifest_sha256":sha(OUT/"preservation_manifest.json"),
        "scientific_limits":read_json(OUT/"editorial_gap_resolution.json"),
        "interpretation":"Requested in-silico editorial analysis and manuscript deliverables complete. No journal acceptance, prospective effectiveness or identified causal selection share is claimed. Submission metadata and public archive accession are explicitly unassigned."}
    # The original build receipt remains evidence of the build; visual status is supplied by the separate signed-by-hash review.
    for p in receipt["products"]:p["visual_inspection"]="passed; see visual_review.json"
    write_json(OUT/"completion_audit_editorial.json",receipt)
    print(json.dumps({"passed":True,"checks":len(checks),"tests":len(cases),"original_gates":19,"preserved_files":preservation["protected_files"],"new_files":len(files)}),flush=True)

