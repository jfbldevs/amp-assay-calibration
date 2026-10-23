"""Detailed Spanish research record appended to the execution report."""
import hashlib
import html
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib.units import mm

from .reporting import (ROOT, MODEL_LABELS, posterior_results, posterior_cohort_counts,
                        load_covariance_design_audit, load_component_sum_review, read_csv, read_json)


def append_research_record(scope, p, heading, table, page):
    """Use the report's flowables so every page receives the same visual treatment."""
    curated = ROOT/"data/curated"/scope
    out = ROOT/"results/reconstruction_2026-10-20"/scope
    m = read_json(curated/"curation_manifest.json")
    support = read_json(out/"statistics/design_support.json")
    metrics = read_csv(out/"benchmark/metrics.csv")
    pairs = read_csv(out/"benchmark/paired_context_improvements.csv")
    splits = read_json(out/"benchmark/verified_split_audits.json")
    models = posterior_results(scope)
    def section(title):
        page()
        p("MEMORIA DETALLADA DE INVESTIGACIÓN", "KickerAMP")
        heading(title)
    def small(text):
        p(text, "SmallAMP")
    def linked(title, url):
        small(f'<link href="{escape(url)}" color="#216F79">{escape(title)}</link>')
    def number(value):
        return f"{value:,.0f}"

    section("A. Preguntas, hipótesis y unidades de análisis")
    p("El estudio es <b>estrictamente in silico</b>, según la instrucción reafirmada por el "
      "autor el 20 de octubre de 2026. Analiza mediciones publicadas y resultados computacionales. "
      "La validación experimental nueva queda fuera del alcance y de los criterios de finalización.")
    p("El autor indicó después que la redacción del manuscrito se realizará en otra etapa. "
      "Esta entrega documenta fuentes, referencias, proceso, metodología, resultados, "
      "tablas, figuras y limitaciones; no constituye un artículo preparado para envío.")
    p("El objetivo científico es determinar cómo la definición del ensayo y de la transferencia "
      "modifica la evaluación de predictores de actividad antimicrobiana. La reconstrucción no "
      "presupone que el contexto domine a la química ni que añadir metadatos mejore siempre las predicciones.")
    table(["Pregunta", "Cantidad que se estima", "Límite de interpretación"], [
        ("Curación", "Registros, formatos, exclusiones, identidades y procedencia recuperables.",
         "Cobertura de la base de datos no equivale a exactitud de todas sus anotaciones."),
        ("Predicción", "Error al predecir la media de log2 MIC de una celda de ensayo reportada.",
         "No es una estimación directa de actividad biológica latente."),
        ("Contexto", "Diferencia pareada de MAE entre modelos comparables con y sin contexto.",
         "Es una comparación predictiva; no identifica efectos causales de medio o laboratorio."),
        ("Heterogeneidad", "Componentes condicionales de modelos con efectos cruzados.",
         "Sus estimaciones e interpretación dependen del diseño, la especificación y la selección de observaciones."),
        ("Ranking", "Inversiones entre dos péptidos bajo dos medios con variables registradas emparejadas.",
         "No controla variables ausentes ni convierte todos los pares en réplicas independientes."),
        ("Transferencia", "Rendimiento según qué secuencias, publicaciones y años quedan fuera del entrenamiento.",
         "Un cambio de partición puede cambiar también la población de prueba."),
    ], [28*mm, 68*mm, 74*mm])
    p("La unidad predictiva combina entidad química, publicación, objetivo exacto y contexto reportado. "
      "Si varios registros pertenecen a la misma combinación, se promedia log2 MIC y se conservan "
      "sus identificadores. Esta media corresponde a una media geométrica en la escala original. "
      "No se añade una réplica independiente por cada fila repetida.")
    p("El modelo censurado utiliza los registros elegibles y sus límites originales. El análisis "
      "de concordancia de GRAMPA tiene otra agrupación y otra escala suministrada por la fuente. "
      "La memoria mantiene separados esos estimandos.")

    section("B. Adquisición, cobertura y trazabilidad")
    table(["Objeto", "Recuento o regla"], [
        ("Fichas DBAASP", number(m["cards"])),
        ("Registros de actividad, antes de filtros", number(m["records"])),
        ("Registros exactos elegibles para inferencia", number(m["eligible_exact_records"])),
        ("Registros elegibles para la verosimilitud con censura", number(m["eligible_censored_records"])),
        ("Celdas exactas para inferencia", number(m["exact_assay_cells"])),
        ("Celdas de la comparación predictiva común", number(m["benchmark_exact_assay_cells"])),
        ("Cola de revisión", number(m["manual_review_records"])),
        ("Registros candidatos a repetición, conservados", number(m["candidate_duplicate_records_retained"])),
    ], [115*mm, 55*mm])
    predictive_cells = read_csv(curated/"benchmark_cells.csv")
    p(f"La cohorte exacta para inferencia contiene {number(m['exact_chemical_entities'])} "
      f"entidades químicas y {number(m['exact_publications'])} publicaciones. La cohorte "
      f"predictiva, más restringida, contiene {number(predictive_cells.chemical_id.nunique())} "
      f"entidades químicas, {number(predictive_cells.sequence_id.nunique())} backbones "
      f"canónicos y {number(predictive_cells.publication_id.nunique())} publicaciones. "
      "Un identificador de backbone ausente no cuenta como una secuencia adicional.")
    p("La ampliación parte del índice de 25.542 identificadores. La descarga es reanudable y "
      "conserva las respuestas originales, fecha y URL de adquisición. Cada fichero entra en un "
      "manifiesto SHA-256. El flujo rechaza una ampliación con identificadores ausentes o una "
      "ficha cuyo contenido no corresponde a su nombre. La muestra de 2.500 fichas permanece "
      "como alcance separado, sin combinar sus predicciones con las de la ampliación.")
    p("La fecha identifica una instantánea del catálogo, no una garantía de que represente toda "
      "la literatura de péptidos. La disponibilidad en la base, la selección editorial de resultados "
      "y la falta de metadatos pueden introducir sesgos de selección.")
    p("GRAMPA se utiliza para una descripción complementaria de concordancia de registros. "
      "No se presenta como un conjunto prospectivo independiente ni como una validación externa "
      "libre de solapamiento. Los datos de QMAP consultados en la revisión tampoco se consideran "
      "automáticamente independientes de DBAASP.")
    small("Rutas: data/raw/; data/curated/"+scope+"/curation_manifest.json; "
          "results/reconstruction_2026-10-20/"+scope+"/benchmark/input_manifest.json.")

    section("C. Publicaciones, taxonomía y contexto ausente")
    crosswalk = read_json(curated/"publication_crosswalk_used.json")
    statuses = {}
    verified_ids = set()
    for row in crosswalk.values():
        key = row["verification_status"]
        statuses[key] = statuses.get(key, 0)+1
        if row.get("resolved_pmid"):
            verified_ids.add(row["publication_id"])
    table(["Verificación en el cruce bibliográfico congelado", "Entradas"], [
        ("Título compatible con el PMID original", statuses.get("verified_title", 0)),
        ("PMID ausente resuelto por título exacto único", statuses.get("resolved_missing_pmid_by_exact_title", 0)),
        ("PMID contradictorio corregido por título exacto único", statuses.get("corrected_pmid_by_exact_title", 0)),
        ("Identidad no resuelta o ambigua", statuses.get("unresolved_or_ambiguous_title", 0)),
        ("Identidades PubMed verificadas distintas", len(verified_ids)),
    ], [135*mm, 35*mm])
    p("La comparación de títulos normalizados admite similitud ≥0,90 para variaciones de "
      "tipografía o subtítulos. Cambiar un PMID exige una coincidencia exacta y única del título "
      "normalizado. Las coincidencias no exactas quedan visibles. Este procedimiento verifica "
      "identidad bibliográfica; no demuestra que cada MIC haya sido extraída correctamente.")
    p("El año temporal se obtiene del primer año electrónico o impreso informado por PubMed. "
      "Una publicación no puede aparecer simultáneamente antes y después del corte. Las "
      "referencias múltiples o fuera de rango no se sustituyen por el primer artículo de la ficha.")
    p("La clasificación de organismos se apoya en linajes de ENA almacenados con la curación. "
      "Un nombre no reconocido permanece desconocido. Solo el grupo bacteriano elegible entra "
      "en la comparación principal. La etiqueta exacta del objetivo conserva la información "
      "de cepa tal como aparece en la fuente, con las limitaciones de su normalización.")
    p("Se exige un medio reportado para la cohorte principal. Inóculo, suplemento y pH pueden "
      "estar ausentes; la ausencia no se convierte en un protocolo estándar. La etiqueta NA "
      "de medio se conserva como agar nutritivo. Una cantidad llamada ionicStrength en la API "
      "se retiene sin inventar una unidad molar.")

    section("D. Identidad química y tratamiento de concentraciones")
    aliases = read_json(curated/"structural_alias_audit.json")
    p("En este estudio, molécula o entidad química designa un péptido antimicrobiano "
      "registrado en DBAASP, incluidas sus variantes químicamente modificadas. La identidad "
      "se establece a partir de las anotaciones disponibles: por ejemplo, una versión amidada "
      "y una versión con extremo ácido libre se distinguen cuando esa diferencia está registrada. "
      "El subconjunto compartido exige mediciones elegibles de la misma entidad química en "
      "al menos dos publicaciones distintas dentro de la cohorte analizada. Se reutilizan "
      "mediciones existentes mediante análisis computacional.")
    p("La entidad química conserva la secuencia con su uso de mayúsculas y minúsculas, "
      "extremos, residuos inusuales, enlaces y complejidad. Para controlar solapamiento se usa "
      "además una identidad de esqueleto en mayúsculas. Esa identidad de partición no convierte "
      "enantiómeros y modificaciones en la misma sustancia.")
    p(f"La auditoría estructural encontró <b>{aliases['graphs_with_different_backbones']}</b> "
      f"grafos moleculares con representaciones de esqueleto distintas y "
      f"<b>{aliases['distinct_backbone_alias_edges']}</b> conexiones de alias. Los grupos de "
      "partición unen esos alias, incluidas rotaciones de secuencias cíclicas. La información "
      "química de origen se conserva para inspeccionar posibles contradicciones.")
    table(["Formato original", "Registros", "Tratamiento"], [
        ("Exacto", m["parse_kinds"].get("exact", 0), "Punto observado; agregación solo en la misma celda reportada."),
        ("Límite derecho", m["parse_kinds"].get("right", 0), "Probabilidad por encima del límite."),
        ("Límite izquierdo", m["parse_kinds"].get("left", 0), "Probabilidad por debajo del límite."),
        ("Rango numérico", m["parse_kinds"].get("interval", 0), "Probabilidad entre límites; sensibilidad excluyendo rangos."),
        ("Media con error", m["parse_kinds"].get("summary_mean", 0), "Revisión de semántica de réplicas; no tratarla como un punto individual."),
        ("No resuelto", m["parse_kinds"].get("unparsed", 0), "Conservar expresión original y razón de exclusión."),
    ], [33*mm, 25*mm, 112*mm])
    p("Los valores molares nativos se transforman directamente a micromolar. Para concentraciones "
      "de masa se usa la masa molecular media derivada de SMILES válidos y consistentes. "
      "La fórmula a partir de secuencia solo sirve de alternativa en monómeros canónicos "
      "explícitamente no modificados. La fórmula dimensional para µg/ml es "
      "µM = 1.000 × (µg/ml) / masa molecular en g/mol.")
    p("No se sustituyen desigualdades por umbrales exactos ni rangos por sus puntos medios. "
      "Tampoco se impone una cuadrícula de diluciones que la fuente no permita recuperar. "
      "Una expresión de intervalo puede resumir varias observaciones; su uso como censura "
      "de una sola cantidad latente es una suposición, no un dato comprobado.")

    section("E. Revisión dirigida de fuentes originales")
    p("La revisión manual asistida se dirigió a valores extremos y candidatos a repetición "
      "de la muestra inicial. No constituye una muestra aleatoria del catálogo ni una extracción "
      "completa por dos revisores independientes.")
    review = read_json(ROOT/"results/reconstruction_2026-10-20/source_review/targeted_source_review.json")
    descriptions = [
        ("Enterocina O16", "0,5 ng/ml en la tabla equivale a 0,0005 µg/ml en la base. "
         "La concentración coincide, pero la definición de inhibición del 50% exige cautela sobre el endpoint."),
        ("Chol-37(F34-R)", "La tabla original informa 0,0078 µg/ml y distingue la modificación con colesterol. "
         "El valor pequeño no es, por sí solo, un error de extracción."),
        ("Péptidos NPRRRERR y KDENK", "La fuente presenta 12 y 9 mM, compatibles con 12.000 y 9.000 µM. "
         "Suprimirlos por magnitud alteraría datos que sí están documentados."),
        ("Dos celdas candidatas a repetición", "La coincidencia de filas no resolvió la procedencia experimental "
         "primaria. Se retienen los registros, se agrega cada celda exacta y se analiza su exclusión conservadora."),
    ]
    table(["Caso", "Resultado y decisión"], descriptions, [44*mm, 126*mm])
    for case in review["cases"]:
        linked(case["publication_id"]+" - "+case["location"], case["source_url"])
    p("Las incidencias conservan identificadores de registro, localización en la fuente, dato "
      "original, decisión y estado. Un error HTTP al recuperar texto completo no se interpreta "
      "como evidencia de que una tabla no exista. Los materiales suplementarios que no pudieron "
      "verificarse permanecen pendientes.")
    p("La cohorte primaria sigue las anotaciones MIC de la base de datos. La equivalencia "
      "de todos esos endpoints con una definición microbiológica única no está establecida. "
      "La sensibilidad que excluye el ejemplo de enterocina describe su impacto, pero no corrige "
      "por sí misma la posible heterogeneidad del resto del catálogo.")

    section("F. Representaciones, comparadores y entrenamiento")
    config = read_json(out/"benchmark/analysis_config.json")
    table(["Familia", "Información y aprendizaje"], [
        ("Media por objetivo", "Media de entrenamiento por especie, contraída hacia la media general con peso 10."),
        ("Solo contexto", "Ridge con objetivo y campos de contexto; comparador de referencia, no ablación causal."),
        ("Descriptores", "Composición, longitud, carga aproximada, hidrofobicidad, momento hidrofóbico y fracción aromática."),
        ("ESM-2", "Representación congelada de 650 millones de parámetros; promedio de residuos de la capa final."),
        ("Huella molecular", "2.048 bits Morgan de radio 2 y quiralidad, promedio de SMILES válidos, descriptores y ausencia."),
        ("Información común", "Objetivo y anotaciones químicas explícitas: extremos, residuos D e inusuales y enlaces."),
        ("Contexto adicional", "Medio, suplementos, tipo y cantidad de inóculo, pH y cantidad de NaCl reportada."),
    ], [38*mm, 132*mm])
    p("Cada representación molecular se combina con ridge y boosting, y cada combinación se "
      "entrena con y sin contexto: doce configuraciones, más los dos comparadores. Todas usan "
      "las mismas celdas elegibles. No se utiliza el identificador de publicación como predictor.")
    p(f"Ridge fija α={config['ridge_alpha']:g}. Boosting usa 200 iteraciones, tasa 0,06, "
      "hasta 15 hojas, profundidad 6, al menos 20 observaciones por hoja y penalización L2=10, "
      "sin parada temprana. Las representaciones vectoriales se reducen a 32 componentes "
      "para boosting; las huellas se estandarizan antes de PCA.")
    p("Imputación, escalado, categorías y PCA se ajustan exclusivamente con entrenamiento. "
      "La codificación categórica agrupa niveles infrecuentes usando un mínimo de tres casos "
      "y hasta cien categorías por campo. El test no decide hiperparámetros.")
    p("La configuración fue bloqueada localmente antes de los ajustes correspondientes, con "
      "semillas y huellas de entrada. No es un prerregistro público. Estos comparadores "
      "representan alternativas reproducibles, pero no una búsqueda exhaustiva del estado del arte. "
      "No se ha auditado el solapamiento con el corpus de preentrenamiento de ESM-2.")

    section("G. Particiones, solapamiento y dependencia")
    split_rows = []
    for protocol in ["random", "sequence", "homology80", "homology50", "publication", "joint", "temporal"]:
        rows = [r for r in splits if r["protocol"] == protocol]
        split_rows.append([protocol, len(rows),
                           f"{min(r['n_train'] for r in rows):,}-{max(r['n_train'] for r in rows):,}",
                           f"{min(r['n_test'] for r in rows):,}-{max(r['n_test'] for r in rows):,}"])
    table(["Partición", "Divisiones", "Tamaño de entrenamiento", "Tamaño de prueba"],
          split_rows, [42*mm, 25*mm, 55*mm, 48*mm])
    p("Las particiones aleatoria, de secuencia, de similitud y de publicación usan cinco folds "
      "y tres semillas. La partición temporal es determinista y se evalúa una vez. La "
      "partición conjunta reserva publicaciones, componentes de similitud y filas adicionales; "
      "la exposición conocida o nueva se etiqueta según el entrenamiento realmente observado.")
    clusters = read_json(curated/"cluster_manifest.json")["clusters"]
    largest = clusters["cluster_50"]
    p(f"La similitud utiliza distancia de edición global normalizada, con umbrales 0,80 y "
      f"0,50; no equivale a identidad BLAST. El componente más grande a 0,50 contiene "
      f"<b>{largest['largest_sequence_component']:,}</b> esqueletos y "
      f"<b>{largest['largest_assay_component']:,}</b> celdas. La clausura transitiva impide "
      "separar miembros conectados, pero puede producir entrenamiento muy desigual.")
    p("Se guardan todas las asignaciones y se reconstruyen en una auditoría separada. Se "
      "comprueba que los grupos prohibidos no crucen entrenamiento y prueba, que las etiquetas "
      "correspondan a la curación y que las métricas se reproduzcan desde las predicciones.")
    p("Una segunda implementación acumula errores por celda directamente desde los archivos de "
      "predicciones y verifica MAE, RMSE, R², acuerdo dentro de una dilución, sesgo y diferencias "
      "pareadas. Es una comprobación del cálculo de las tablas; no demuestra por sí sola "
      "la validez estadística del remuestreo ni vuelve a ajustar los modelos.")
    p("Las repeticiones se resumen promediando pérdidas por celda y estrato. Calcular el error "
      "de una predicción promediada respondería a la precisión de otro predictor, un ensemble. "
      "La incertidumbre pareada utiliza pesos Poisson cruzados por publicación y secuencia "
      "o grupo de alias, con 2.000 remuestreos. Los intervalos son puntuales y exploratorios; "
      "no vuelven a entrenar todos los modelos ni garantizan cobertura simultánea.")

    for protocol, title in [("publication", "H. Resultados al reservar publicaciones"),
                            ("temporal", "I. Resultados en publicaciones posteriores")]:
        section(title)
        rows = metrics[metrics.protocol.eq(protocol)&metrics.stratum.eq("all")].sort_values("model")
        table(["Modelo", "MAE log2", "R²", "Error ≤1 log2"], [
            [MODEL_LABELS[r.model], f"{r.mae_log2:.3f}", f"{r.r2:.3f}", f"{r.within_one_dilution:.1%}"]
            for r in rows.itertuples()
        ], [88*mm, 28*mm, 25*mm, 29*mm])
        selected = pairs[pairs.protocol.eq(protocol)&pairs.stratum.eq("all")&
                         pairs.base.isin(["sequence_ridge","esm_ridge","chem_ridge",
                                         "sequence_hgb","esm_hgb","chem_hgb"])]
        table(["Modelo base", "Reducción de MAE", "Intervalo 95%"], [
            [MODEL_LABELS[r.base], f"{r.mae_reduction_log2:.3f}", f"[{r.ci_low:.4f}, {r.ci_high:.4f}]"]
            for r in selected.itertuples()
        ], [88*mm, 39*mm, 43*mm])
        small("Un valor positivo favorece añadir contexto. Las seis comparaciones se conservan "
              "sin elegir retrospectivamente una familia por la significación de su intervalo. "
              "MAE mide error en escala log2; un error de una unidad corresponde a un factor dos.")

    section("J. Soporte de comparación y sensibilidades de curación")
    p(f"En la cohorte de celdas con MIC exacta, el grafo entre química y publicación "
      f"contiene <b>{support['bipartite_components']:,}</b> componentes. "
      f"<b>{support['single_publication_fraction']:.1%}</b> de las entidades aparece en un solo "
      f"artículo y <b>{support['multi_publication_entities']:,}</b> se observa en varios. "
      f"Solo <b>{support['chemical_exact_target_pairs_in_multiple_publications']:,}</b> combinaciones "
      "de entidad y objetivo exacto se repiten entre publicaciones.")
    p("Esa estructura limita las comparaciones directas de la misma entidad entre trabajos; "
      "no demuestra por sí sola que los componentes de varianza sean algebraicamente indistinguibles. "
      "Un efecto aleatorio de publicación puede absorber protocolos, series químicas y variables "
      "no registradas. La publicación no identifica por sí sola un laboratorio.")
    path = out/"statistics/reml_curation_sensitivity.json"
    sensitivity = read_json(path) if path.exists() else {}
    labels = {
        "native_molar_only": "Solo unidades molares",
        "no_candidate_duplicates": "Sin celdas con repetición candidata",
        "central_98pct_outcomes": "98% central de resultados",
        "at_least_two_context_fields": "Al menos dos campos de contexto",
        "largest_connected_component": "Mayor componente conectado",
        "excluding_endpoint_ambiguity_example": "Sin ejemplo de endpoint ambiguo",
    }
    table(["Restricción", "Celdas", "Fracción publicación", "Singular"], [
        [labels.get(key,key), r.get("n_used",r.get("n_requested",0)),
         next((f"{x['fraction']:.3f}" for x in r.get("variance_components",[]) if x["grp"]=="publication_id"), "No estimable"),
         "Error" if "error" in r else ("Sí" if r["singular"] else "No")]
        for key,r in sensitivity.items()
    ] or [["Pendiente","-","-","-"]], [78*mm, 25*mm, 42*mm, 25*mm])
    p("Estas restricciones se especificaron después de inspeccionar la muestra inicial y son "
      "exploratorias. El análisis de unidades molares exige que todos los registros de una celda "
      "sean molares. La sensibilidad de repetición retira toda la celda candidata; no afirma "
      "que la coincidencia pruebe duplicación. El recorte de colas cambia deliberadamente "
      "la distribución de resultados.")
    p("El subconjunto de metadatos exige al menos dos entre inóculo, suplemento y pH. "
      "El componente conectado se selecciona por número de nodos, sin usar la respuesta. "
      "En todos los casos cambian composición y soporte; sus diferencias no pueden leerse "
      "como un experimento sobre el efecto de la curación.")

    section("K. Modelo censurado y cohortes")
    p("<b>Qué significa bayesiano.</b> Se combinan supuestos explícitos sobre valores "
      "plausibles de los parámetros, llamados priors, con la información de las MIC "
      "publicadas, expresada mediante una verosimilitud. El resultado es una distribución "
      "posterior: un conjunto de valores plausibles después de incorporar los datos. "
      "En este estudio permite estimar diferencias asociadas con el péptido, la publicación, "
      "el objetivo y el contexto del ensayo, junto con su incertidumbre.")
    p("Un intervalo posterior central del 95% contiene ese porcentaje de la distribución "
      "posterior del parámetro, condicionado a los datos, el modelo y sus priors. Su amplitud "
      "no incluye automáticamente errores de anotación ni sesgos por variables ausentes. "
      "Una MIC reportada como mayor que un umbral aporta información de un límite; las "
      "muestras computacionales exploran parámetros compatibles con el modelo. Las cuatro "
      "cadenas son recorridos del algoritmo y no representan cuatro experimentos ni nuevos "
      "datos biológicos. Las sensibilidades posteriores evalúan cuánto dependen las "
      "estimaciones de la cohorte y de los supuestos seleccionados.")
    p("La variable latente es log2 MIC. La media incluye contexto fijo y efectos gaussianos "
      "para química, publicación, objetivo exacto, química×objetivo y química×medio. "
      "Los registros exactos aportan densidades; los límites y rangos aportan probabilidades "
      "acumuladas o complementarias evaluadas de forma numéricamente estable.")
    p("El intercepto tiene prior normal con media 3 y desviación 5; la escala residual, "
      "half-normal de escala 3. Coeficientes y escalas de efectos usan escala 1,5; se contrasta "
      "escala 3. Se integran analíticamente los niveles de efecto presentes en una sola "
      "observación, sumando su varianza a la de esa observación. Esta transformación conserva "
      "la verosimilitud marginal y los priors; no aporta identificación nueva.")
    expected = [("all",1.5),("exact",1.5),("multi_publication",1.5),("no_reported_ranges",1.5),("all",3.0)]
    cohort_rows, format_rows = [], []
    for key in expected:
        result, path = models.get(key, ({},None))
        label = f"{key[0]} / {key[1]:g}"
        if path is None:
            cohort_rows.append([label, "-", "-", "-", "-"])
            format_rows.append([label, "-", "-", "-", "-"])
            continue
        cohort = posterior_cohort_counts(result, path)
        cohort_rows.append([label, *[number(cohort[name]) for name in
                                    ["records", "chemical_entities", "publications", "exact_targets"]]])
        format_rows.append([label, *[number(cohort[name]) for name in
                                    ["exact", "left", "right", "interval"]]])
    table(["Especificación / prior", "Registros", "Entidades", "Publicaciones", "Objetivos"],
          cohort_rows, [66*mm,26*mm,26*mm,26*mm,26*mm], padding=4)
    table(["Especificación / prior", "Exactos", "Límite izq.", "Límite der.", "Rangos"],
          format_rows, [70*mm,25*mm,25*mm,25*mm,25*mm], padding=4)
    small("Los niveles incluyen los efectos presentes una sola vez e integrados analíticamente. "
          "Estos son registros del modelo censurado; no son las celdas agregadas usadas por REML "
          "ni la cohorte predictiva. Un guion indica que aún no hay un ajuste guardado para esta especificación.")
    section("K.1. Diseño de covarianza e interpretación")
    covariance = load_covariance_design_audit(scope)
    p("Se añadió una comprobación exploratoria después de revisar el primer ajuste posterior "
      "del catálogo ampliado. Distingue la independencia algebraica de las matrices de "
      "covarianza, la precisión de las estimaciones y la atribución biológica. Son preguntas distintas.")
    p("Cada uno de los cinco bloques de efectos aleatorios define una matriz que vale uno "
      "cuando dos registros comparten nivel, y cero en otro caso. La matriz identidad representa "
      "el residuo. La covarianza gaussiana marginal es una combinación de esas seis matrices, "
      "ponderadas por las varianzas correspondientes. Se usa la formulación de efectos "
      "gaussianos de Bates et al. (2015, sección 1.1); el diagnóstico de conteos se deriva en este estudio.")
    p("El producto interno de dos matrices se calcula sumando los cuadrados de los conteos "
      "en sus niveles conjuntos. Su producto con la identidad es el número de registros. "
      "Se normaliza cada matriz a norma de Frobenius uno y se examinan los autovalores de "
      "la matriz de Gram. El rango indica cuántos componentes son linealmente independientes.")
    cohort_labels = {"all": "Todos", "exact": "Exactos", "multi_publication": "Química compartida",
                     "no_reported_ranges": "Sin rangos"}
    table(["Cohorte", "Registros", "Rango", "Autovalor mínimo", "Condición de la base"], [
        [cohort_labels[key], number(row["records"]), f"{row['rank']}/{row['component_count']}",
         f"{row['normalized_gram_eigenvalues'][0]:.4f}",
         f"{row['normalized_basis_condition_number']:.3f}" if row["normalized_basis_condition_number"] is not None else "No finita"]
        for key, row in covariance["cohorts"].items()
    ], [52*mm,25*mm,20*mm,35*mm,38*mm])
    small("La condición de la base es la raíz del cociente entre autovalores máximo y mínimo "
          "del Gram normalizado; no es la condición de la información de Fisher ni de la distribución posterior. "
          "Los dos ajustes con distinta escala de prior usan el mismo diseño de la cohorte completa.")
    p("La verificación incluye un cálculo independiente con matrices densas pequeñas, un ejemplo "
      "anidado de rango completo sin ninguna entidad compartida entre artículos y un ejemplo "
      "de niveles únicos donde los componentes sí coinciden. Esto corrige la interpretación "
      "demasiado fuerte que asociaba automáticamente escaso solapamiento con falta de identificación.")
    p("El diagnóstico corresponde a la covarianza latente antes de proyectar los efectos fijos; "
      "no audita la información REML. La censura, el tamaño de muestra y los valores de los parámetros "
      "pueden limitar la precisión. El rango completo tampoco valida intercambiabilidad, supuestos "
      "de distribución o atribución causal. Se conservan por separado las sensibilidades a priors, "
      "cohortes y adecuación del modelo.")
    small("Reproducción: scripts/29_audit_covariance_design.py; matrices, espectros, tolerancia "
          "numérica y huellas de datos y código en statistics/covariance_design_audit.json.")
    linked("Bates et al. (2015): formulación del modelo gaussiano", "https://doi.org/10.18637/jss.v067.i01")

    section("K.2. Diagnósticos y comprobaciones posteriores")
    rows = []
    for key in expected:
        result, _ = models.get(key, ({},None))
        d = result.get("diagnostics",{})
        rows.append([f"{key[0]} / {key[1]:g}", result.get("n_records","-"),
                     f"{d['max_rhat_all_parameters']:.4f}" if d else "-",
                     f"{d['min_bulk_ess_scientific_parameters']:.0f}" if d else "-",
                     "Pasa" if d.get("passed") else ("Fallido" if d else "Pendiente")])
    table(["Especificación / prior", "Registros", "Rhat máx.", "ESS mín.", "Estado"],
          rows, [70*mm, 28*mm, 27*mm, 23*mm, 22*mm])
    p("Cada ajuste usa cuatro cadenas. La regla exige cero divergencias, cero saturaciones de "
      "profundidad, Rhat máximo inferior a 1,01 y ESS bulk mínimo de 400 en los parámetros "
      "científicos. Los ajustes fallidos se conservan como desarrollo y no aportan cifras "
      "interpretables a las tablas posteriores.")
    precision_rows = []
    for key in expected:
        _, path = models.get(key, ({},None))
        precision_path = path.parent/"sampling_precision.json" if path else None
        if precision_path and precision_path.exists():
            precision = read_json(precision_path)
            precision_rows.append([f"{key[0]} / {key[1]:g}",
                                   f"{precision['min_tail_ess_scientific_parameters']:.0f}",
                                   f"{min(precision['e_bfmi_by_chain']):.3f}",
                                   f"{precision['max_mcse_fraction_mean']:.5f}",
                                   "Pasa" if precision["passed"] else "Bandera"])
    if precision_rows:
        table(["Especificación / prior", "ESS cola", "E-BFMI mín.", "MCSE fracc.", "Revisión"],
              precision_rows, [70*mm,25*mm,25*mm,25*mm,25*mm])
        p("La revisión adicional conserva banderas para ESS de cola inferior a 400, E-BFMI "
          "inferior a 0,3 en alguna cadena o diagnósticos no finitos. El MCSE corresponde a la "
          "media posterior de las fracciones. El ESS de cola se refiere a cuantiles 5% y 95%; "
          "no es una estimación directa del error Monte Carlo de los extremos 2,5% y 97,5%. "
          "Superar estos controles no prueba adecuación del modelo ni interpretación causal.")
    p("Para cada ajuste que pasa se comparan cinco resúmenes de observaciones exactas con "
      "200 réplicas posteriores condicionales. Se retienen colas extremas, probabilidades de "
      "eventos censurados y residuos por grupo. Estas comprobaciones son internas y pueden "
      "estar afectadas por la selección del subconjunto exacto. No sustituyen validación externa.")
    check_path = out/"posterior_check_summary.json"
    checks = read_json(check_path) if check_path.exists() else []
    small(f"Comprobaciones disponibles: {len(checks)}. Especificaciones con alguna bandera de cola: "
          f"{sum(bool(r['extreme_tail_flag']) for r in checks)}. Los detalles numéricos completos "
          "se incluyen en las tablas de diagnóstico y en posterior_predictive_checks.json de cada ajuste.")

    passing = {key: value[0] for key, value in models.items() if value[0]["diagnostics"]["passed"]}
    if passing:
        section("K.3. Resultados y sensibilidad posterior")
        p("Las fracciones siguientes normalizan las varianzas de efectos aleatorios y el "
          "residuo por su suma. La variación explicada por los efectos fijos no entra en el "
          "denominador. Se muestran únicamente ajustes que superan la regla computacional; "
          "un intervalo posterior estrecho sigue condicionado a los supuestos del modelo "
          "y no establece atribución causal.")
        labels = {("all",1.5): "Todos / escala 1,5", ("exact",1.5): "Exactos / escala 1,5",
                  ("multi_publication",1.5): "Química compartida / 1,5",
                  ("no_reported_ranges",1.5): "Sin rangos / escala 1,5",
                  ("all",3.0): "Todos / escala 3"}
        rows = []
        for key in expected:
            if key not in passing:
                continue
            result = passing[key]
            fractions = result["conditional_variance_fractions"]
            publication = fractions["publication"]
            rows.append([labels[key], f"{result['n_records']:,}",
                         f"{fractions['chemical_entity']['median']:.1%}",
                         f"{publication['median']:.1%} [{publication['ci95'][0]:.1%}, {publication['ci95'][1]:.1%}]",
                         f"{fractions['residual']['median']:.1%}"])
        table(["Especificación", "Registros", "Química", "Publicación [95%]", "Residuo"], rows,
              [47*mm, 23*mm, 25*mm, 51*mm, 24*mm])
        small("Química y residuo: mediana posterior. Publicación: mediana e intervalo central "
              "del 95%. Las otras fracciones corresponden a objetivo exacto, química×objetivo y "
              "química×medio; sus intervalos completos aparecen en las tablas y los archivos JSON.")
        if ("all",1.5) in passing and ("all",3.0) in passing:
            primary_fractions = passing[("all",1.5)]["conditional_variance_fractions"]
            wide_fractions = passing[("all",3.0)]["conditional_variance_fractions"]
            a, b = primary_fractions["publication"], wide_fractions["publication"]
            max_change = 100*max(abs(wide_fractions[k]["median"]-v["median"])
                                 for k, v in primary_fractions.items())
            p(f"Manteniendo los mismos registros, ampliar la escala de los priors de 1,5 a 3 "
              f"cambia la mediana de la fracción de publicación de <b>{a['median']:.1%}</b> "
              f"a <b>{b['median']:.1%}</b>. El mayor cambio absoluto entre las seis medianas "
              f"es <b>{max_change:.3f} puntos porcentuales</b>. La verificación del diseño, "
              "registros y datos preparados confirma que solo cambia prior_scale. Esta "
              "comparación altera priors de coeficientes y escalas aleatorias; su estabilidad "
              "se limita a las dos escalas probadas y no cubre otras estructuras del modelo.")
        if ("all",1.5) in passing and ("no_reported_ranges",1.5) in passing:
            primary = passing[("all",1.5)]
            no_ranges = passing[("no_reported_ranges",1.5)]
            delta = 100*(no_ranges["conditional_variance_fractions"]["publication"]["median"]-
                         primary["conditional_variance_fractions"]["publication"]["median"])
            p(f"Excluir los <b>{primary['n_records']-no_ranges['n_records']:,}</b> rangos "
              f"reportados deja <b>{no_ranges['n_records']:,}</b> registros y aumenta la "
              f"mediana de publicación en <b>{delta:.2f} puntos porcentuales</b>. Los "
              "límites izquierdos y derechos permanecen incluidos. La selección se "
              "contrastó con todas las columnas y los índices de censura de la fuente.")
        p("Restringir a registros exactos, eliminar rangos o exigir química compartida modifica "
          "la población analizada además de la información disponible. Las diferencias entre esas "
          "filas son sensibilidades y no efectos causales atribuibles a la censura o a la curación.")
        flagged = sum(bool(r["extreme_tail_flag"]) for r in checks)
        p(f"Las comprobaciones internas disponibles señalan alguna discrepancia extrema en "
          f"<b>{flagged} de {len(checks)}</b> especificaciones. Deben leerse junto con la selección "
          "de registros exactos y la ausencia de un mecanismo completo de observación. La "
          "convergencia permite describir el posterior ajustado; no basta para validar sus "
          "supuestos como explicación biológica.")
        small("Evidencia: pending_sensitivity_input_review_20261021.json conserva la revisión "
              "de entradas realizada antes de finalizar los ajustes; "
              "final_sensitivity_review_20261022.json documenta su revisión terminada.")

        component_sum = load_component_sum_review(scope, models)
        if component_sum:
            section("K.3.1. Suma exploratoria de química y publicación")
            p("Este análisis se propuso <b>después de inspeccionar los primeros tres ajustes</b>, "
              "al observar que cambiaba más la asignación individual que su suma aproximada. "
              "Se informa como exploratorio y se incluyen las cinco especificaciones previstas. "
              "No constituye una hipótesis prerregistrada ni una prueba confirmatoria.")
            p("Para cada muestra posterior conjunta se sumaron las fracciones de los efectos "
              "principales de identidad química y publicación. Cada ajuste aporta cuatro cadenas "
              "cerradas con 3.000 muestras posteriores al calentamiento. Los cuantiles 2,5%, 50% "
              "y 97,5% se calcularon sobre esas 12.000 sumas, preservando la dependencia entre "
              "componentes. No se sumaron medianas o extremos de intervalos marginales ni se "
              "emparejaron muestras entre ajustes.")
            sum_rows = []
            medians, diagnostics = [], []
            for row in component_sum["specifications"]:
                value = row["chemical_entity_plus_publication"]
                diagnostic = value["sampling_diagnostics"]
                medians.append(value["median"])
                diagnostics.append(diagnostic)
                sum_rows.append([
                    labels[(row["subset"], row["prior_scale"])],
                    f"{value['median']:.2%} [{value['ci95'][0]:.2%}, {value['ci95'][1]:.2%}]",
                    f"{diagnostic['R_hat']:.4f}", f"{diagnostic['ESS_bulk']:,.0f}",
                ])
            table(["Especificación", "Suma: mediana [95%]", "Rhat", "ESS central"],
                  sum_rows, [54*mm, 66*mm, 23*mm, 27*mm])
            p(f"Las medianas se sitúan entre <b>{min(medians):.1%} y {max(medians):.1%}</b>. "
              "El denominador es la suma de cinco varianzas aleatorias y la residual; excluye "
              "la variación de efectos fijos. Por tanto, esta cercanía describe fracciones "
              "normalizadas y no demuestra igualdad de varianzas absolutas, equivalencia "
              "entre cohortes, identificación causal ni un límite universal de predicción.")
            p("Los seis resúmenes originales de componentes se reconciliaron con los archivos "
              "de resultados con tolerancia absoluta 10⁻¹². La extracción conserva huellas "
              "de las cadenas originales y evidencia de cierre. CmdStan 2.40 calculó "
              "diagnósticos específicos de la suma: Rhat, ESS central y de cola y error de "
              "Monte Carlo de la media. Estos son controles de precisión computacional; "
              "las discrepancias predictivas del modelo siguen vigentes.")
            small(f"En los cinco ajustes, ESS de cola mínimo: "
                  f"{min(d['ESS_tail'] for d in diagnostics):,.0f}; MCSE máximo de la media: "
                  f"{max(d['MCSE'] for d in diagnostics):.6f}. El ESS de cola de stansummary "
                  "usa cuantiles 5% y 95%; no es el error de Monte Carlo de los límites "
                  "2,5% y 97,5% del intervalo.")
            small("Fuentes y reproducción: posterior_component_sum/exploratory_component_sum.json, "
                  "su CSV, review_20261022.json y los archivos extraction.json y derived_summary.csv "
                  "de cada ajuste. Script: source_review/check_component_sum_20261021.py. "
                  "La tabla completa es table_component_sum.tex.")
            linked("CmdStan: resumen y diagnósticos de cadenas (documentación consultada el 21/10/2026)",
                   "https://mc-stan.org/docs/cmdstan-guide/stansummary.html")

        predictive_rows = []
        exact_checks = None
        statistic_labels = {"mean": "Media", "sd": "Desv. estándar", "q05": "Percentil 5",
                            "median": "Mediana", "q95": "Percentil 95"}
        for key in expected:
            if key not in passing:
                continue
            check_file = models[key][1].parent/"posterior_predictive_checks.json"
            if not check_file.exists():
                continue
            diagnostic = read_json(check_file)
            draws_checked = diagnostic["posterior_draws_checked"]
            if key == ("exact", 1.5):
                exact_checks = diagnostic["exact_subset_checks"]
            for statistic, label in statistic_labels.items():
                check = diagnostic["exact_subset_checks"][statistic]
                low, high = check["replicated_interval95"]
                upper_count = round(check["posterior_predictive_upper_tail"]*draws_checked)
                predictive_rows.append([
                    labels[key], label, f"{check['observed']:.3f}",
                    f"[{low:.3f}, {high:.3f}]", f"{upper_count}/{draws_checked}",
                    "Sí" if check["extreme_tail_flag"] else "No",
                ])
        if predictive_rows:
            section("K.4. Distribución observada y réplicas del modelo")
            p("Cada fila compara un resumen observado con el intervalo central del 95% "
              "de sus réplicas posteriores condicionales. La columna de recuento indica "
              "cuántas réplicas igualan o superan el resumen observado. Cero réplicas no "
              "significa probabilidad teórica cero. Las banderas usan la regla fijada de "
              "cola inferior a 0,01 o superior a 0,99; no son valores p frecuentistas.")
            table(["Especificación / prior", "Resumen", "Observado", "Réplicas [95%]",
                   "Réplicas ≥ obs.", "Bandera"], predictive_rows,
                  [38*mm, 22*mm, 22*mm, 46*mm, 22*mm, 20*mm], padding=4)
            if exact_checks:
                inside = [statistic_labels[k].lower() for k, v in exact_checks.items()
                          if v["replicated_interval95"][0] <= v["observed"] <= v["replicated_interval95"][1]]
                flagged_exact = [k for k, v in exact_checks.items() if v["extreme_tail_flag"]]
                p("En el ajuste de registros exactos, los resúmenes observados dentro del "
                  "intervalo replicado son: <b>"+escape(", ".join(inside) or "ninguno")+
                  f"</b>. La regla de cola marca <b>{len(flagged_exact)} de cinco</b> resúmenes.")
                if flagged_exact:
                    gaps = [max(v["replicated_interval95"][0]-v["observed"],
                                v["observed"]-v["replicated_interval95"][1], 0)
                            for k, v in exact_checks.items() if k in flagged_exact]
                    p(f"En los resúmenes marcados, la distancia del valor observado al borde "
                      f"más próximo del intervalo replicado va de <b>{min(gaps):.3f}</b> a "
                      f"<b>{max(gaps):.3f}</b> unidades log2. La magnitud debe examinarse junto "
                      "con la bandera: una discrepancia diagnóstica en una cohorte grande no "
                      "establece por sí sola importancia práctica. Estas comprobaciones "
                      "tampoco demuestran adecuación de toda la distribución ni transferencia externa.")
            if ("all",1.5) in passing:
                primary_check_path = models[("all",1.5)][1].parent/"posterior_predictive_checks.json"
                if primary_check_path.exists():
                    primary_check = read_json(primary_check_path)
                    primary_summaries = primary_check["exact_subset_checks"]
                    primary_flagged = [v for v in primary_summaries.values()
                                       if v["extreme_tail_flag"]]
                    p(f"En el ajuste principal de todos los registros, la comprobación usa "
                      f"<b>{primary_check['exact_observations_checked']:,}</b> observaciones "
                      f"exactas y marca <b>{len(primary_flagged)} de cinco</b> resúmenes. "
                      "La cohorte del ajuste también incluye observaciones censuradas.")
                    if primary_flagged:
                        primary_gaps = [max(v["replicated_interval95"][0]-v["observed"],
                                            v["observed"]-v["replicated_interval95"][1], 0)
                                        for v in primary_flagged]
                        p(f"Las distancias al borde más próximo del intervalo replicado "
                          f"van de <b>{min(primary_gaps):.3f}</b> a "
                          f"<b>{max(primary_gaps):.3f}</b> unidades log2. La selección de "
                          "registros reportados como exactos puede depender de rangos de ensayo "
                          "no registrados. Estos diagnósticos condicionados no aíslan una causa "
                          "de discrepancia ni acreditan calibración externa.")
                    event_labels = {"left": "izquierda", "right": "derecha", "interval": "por rango"}
                    event_parts = []
                    for kind, value in primary_check.get("mean_probability_of_recorded_censoring_event", {}).items():
                        event_parts.append(f"{event_labels[kind]}: {value['median']:.3f} "
                                           f"[{value['interval95'][0]:.3f}, {value['interval95'][1]:.3f}]")
                    if event_parts:
                        p("Probabilidad media asignada al evento de censura registrado "
                          "(mediana e intervalo posterior del 95%): "+"; ".join(event_parts)+
                          ". Son resúmenes internos del modelo, no tasas de acierto ni cobertura externa.")
                    input_review_path = out/"all_posterior_input_review.json"
                    if input_review_path.exists():
                        input_review = read_json(input_review_path)
                        if (input_review.get("passed") and
                            input_review.get("input_sha256") == passing[("all",1.5)]["input_sha256"] and
                            input_review.get("run_id") == models[("all",1.5)][1].parent.name):
                            p("La selección del ajuste principal se contrastó registro por registro "
                              "con la tabla de origen; se verificaron tipos de censura, límites y "
                              "cuatro archivos cerrados de 3.000 muestras. La regeneración del "
                              "diseño emplea la misma función de preparación y comprueba su "
                              "reproducibilidad. La evidencia y el verificador se conservan en "
                              "all_posterior_input_review.json y source_review/verify_all_posterior_inputs_20261021.py.")
            small("Fuentes: posterior_predictive_checks.json de cada ajuste; tabla reproducible "
                  "table_posterior_checks.tex y su selección en table_guide.json.")
            if (out/"all_posterior_review.json").exists():
                small("La revisión de procedencia, precisión y limitaciones del ajuste principal "
                      "se conserva en all_posterior_review.json.")
            if (out/"exact_posterior_review.json").exists():
                small("La revisión de procedencia, precisión y limitaciones del ajuste exacto "
                      "se conserva en exact_posterior_review.json.")
            if all(key in passing for key in [("no_reported_ranges",1.5),("all",3.0)]):
                exact_n = passing[("no_reported_ranges",1.5)]["kind_counts"]["exact"]
                p("Las dos sensibilidades finales conservan banderas en los cinco resúmenes "
                  f"de las mismas {exact_n:,} observaciones exactas. Ni excluir rangos ni ampliar "
                  "el prior resuelve estas discrepancias. La estabilidad de las fracciones "
                  "entre priors debe leerse junto con este límite de adecuación.")
                small("Revisiones finales: no_reported_ranges_posterior_review.json, "
                      "broader_prior_posterior_review.json y final_sensitivity_review_20261022.json. "
                      "Verificador reproducible: source_review/review_final_sensitivities_20261022.py.")

    section("L. Concordancia, ranking y utilidad de selección")
    grampa = read_json(ROOT/"results/reconstruction_2026-10-20/sample/statistics/grampa_reliability.json")
    g = grampa["reported_strain_only"]
    p(f"GRAMPA aporta <b>{grampa['rows']:,}</b> registros para su análisis descriptivo. "
      f"Con cepa reportada se identifican <b>{g['groups']:,}</b> grupos y un ICC de "
      f"<b>{g['icc']:.3f}</b>, con intervalo [{g['icc_ci95'][0]:.3f}, {g['icc_ci95'][1]:.3f}]. "
      "No hay grupos verificables de réplica que combinen cepa reportada y la misma fuente. "
      "Por ello el ICC se interpreta como concordancia de registros bajo esa agrupación.")
    p("El estimador de ICC incorpora la corrección por tamaño desigual de grupo y su "
      "incertidumbre se remuestrea por secuencia. Los valores iguales se conservan. "
      "La discordancia binaria distingue la existencia de ambas etiquetas en un grupo "
      "de la probabilidad de desacuerdo entre dos registros.")
    rank = read_json(out/"statistics/matched_rank_summary.json")
    r = rank["all_matching_reported_fields"]
    p(f"Los contrastes de ranking incluyen <b>{r['pairs']:,}</b> pares, "
      f"<b>{r.get('blocks',0):,}</b> bloques y <b>{r.get('publications',0):,}</b> publicaciones. "
      f"Hay <b>{r.get('inversions',0):,}</b> inversiones estrictas, definidas por signos "
      "opuestos y más de una unidad log2 de separación en ambos medios. "
      "Los pares con péptidos compartidos reciben pesos dependientes en el remuestreo.")
    rank_interval = r.get("crossed_bootstrap_ci95", [np.nan, np.nan])
    p(f"La proporción descriptiva es <b>{r.get('rate', np.nan):.1%}</b>, con intervalo "
      f"del 95% <b>[{rank_interval[0]:.1%}, {rank_interval[1]:.1%}]</b>, obtenido con "
      "5.000 remuestreos cruzados. No se interpreta como la probabilidad causal de que "
      "cambiar de medio invierta cualquier ranking futuro.")
    rank_pairs = read_csv(out/"statistics/matched_rank_pairs.csv")
    by_publication = rank_pairs.groupby("publication_id").strict_inversion.agg(["size", "sum"])
    p(f"Las inversiones aparecen en <b>{int((by_publication['sum'] > 0).sum())}</b> "
      f"publicaciones. La publicación con más contrastes aporta "
      f"<b>{by_publication['size'].max()/by_publication['size'].sum():.1%}</b> de todos los pares. "
      "Esta concentración debe acompañar al recuento total: 1.000 pares combinatorios no "
      "equivalen a 1.000 experimentos independientes.")
    p("Para selección de candidatos se exigen al menos cinco casos de prueba en el mismo "
      "artículo, objetivo y contexto reportado. Se selecciona el 20% con menor MIC "
      "predicha, redondeando el número de candidatos hacia arriba, y se compara con la media "
      "del bloque y con un oráculo de etiquetas observadas. "
      "Las repeticiones se promedian primero por publicación; ese oráculo no es la actividad "
      "biológica verdadera.")
    identity_review_path = out/"screening_candidate_identity_review.json"
    if identity_review_path.exists():
        identity_review = read_json(identity_review_path)
        for relative, expected in identity_review["source_sha256"].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Stale screening identity review: {relative}")
        p(f"La revisión de identidad abarca las {identity_review['benchmark_cells']:,} "
          "celdas predictivas. No encontró candidatos repetidos dentro del mismo bloque "
          "bajo el identificador químico declarado ni bajo los alias de grafo exacto "
          "conocidos. Para estos últimos se recuperaron todos los registros originales "
          "de cada celda. La relación entre fichas y grafos puede ser múltiple: un primer "
          "intento de unión uno-a-uno fue rechazado y se conservó la relación completa. "
          "Esta comprobación no descarta alias desconocidos ni demuestra que las "
          "condiciones no registradas fueran idénticas.")
        small("Evidencia: screening_candidate_identity_review.json y "
              "screening_alias_cell_graph_links.csv, dentro del directorio de resultados "
              "de este alcance. La comprobación no modificó las entradas ni los modelos.")
    screening = read_csv(out/"benchmark/screening_summary.csv")
    screen = screening[screening.protocol.isin(["publication","temporal"])&
                       screening.model.isin(["sequence_hgb","sequence_context_hgb",
                                             "esm_hgb","esm_context_hgb"])]
    table(["Partición / modelo", "Artículos", "Ganancia log2", "Regret log2"], [
        [r.protocol+" / "+MODEL_LABELS[r.model], r.publications,
         f"{r.mean_screening_gain_log2:.3f}", f"{r.mean_regret_log2:.3f}"]
        for r in screen.itertuples()
    ], [100*mm, 22*mm, 24*mm, 24*mm])
    small("Estas cifras describen los bloques disponibles, no una nueva selección experimental. "
          "La calibración se calcula en estratos de prueba con al menos diez casos sin corregir "
          "posteriormente las predicciones usando esas mismas etiquetas.")
    temporal_screen = screening[screening.protocol.eq("temporal")].set_index("model")
    base_screen, context_screen = (temporal_screen.loc[name] for name in
                                  ["sequence_hgb", "sequence_context_hgb"])
    small(f"Para descriptores con boosting, la ganancia temporal observada al seleccionar "
          f"candidatos cambia de {base_screen.mean_screening_gain_log2:.3f} a "
          f"{context_screen.mean_screening_gain_log2:.3f} log2 al añadir contexto, en "
          f"{int(base_screen.publications)} artículos. Es una comparación descriptiva sin "
          "inferencia sobre esa diferencia; una mejora de MAE no asegura una mejor selección.")
    secondary_path = out/"secondary_evaluation_review.json"
    if secondary_path.exists():
        secondary = read_json(secondary_path)
        p(f"Una revisión separada comprobó {secondary['screening_block_rows_checked']:,} "
          f"evaluaciones de bloques y recalculó las {secondary['screening_summary_rows_recomputed']} "
          "filas de resumen. Reconstruyó la selección temporal desde las predicciones "
          f"individuales de los {secondary['temporal_prediction_files_checked']} modelos "
          f"y sus {secondary['temporal_calibration_strata_recomputed']} estratos de calibración. "
          "Los modelos comparten candidatos dentro de cada protocolo; las publicaciones "
          "elegibles cambian entre protocolos. La selección conjunta agrupa todos los "
          "candidatos disponibles del bloque, sin restringirse al estrato de secuencia "
          "y publicación nuevas.")
        ties = read_csv(out/"temporal_screening_tie_sensitivity.csv")
        ties = ties[ties.tie_tolerance_log2.eq(1e-12)].set_index("model")
        p("La selección principal desempata por identificador de celda. Se añadió una "
          "sensibilidad exploratoria que promedia las elecciones posibles cuando hay "
          "empates en el corte, con igualdad exacta y tolerancia numérica de "
          "10<super>-12</super> log2. Esta tolerancia no define equivalencia biológica. "
          "Un predictor constante puede mostrar una ganancia distinta de cero por el "
          "orden arbitrario de los identificadores; al promediar uniformemente los "
          "empates su ganancia esperada es cero.")
        table(["Modelo", "Ganancia original", "Promedio de empates"], [
            [MODEL_LABELS[name], f"{ties.loc[name, 'deterministic_gain_log2']:.3f}",
             f"{ties.loc[name, 'expected_tie_gain_log2']:.3f}"]
            for name in ["sequence_hgb", "sequence_context_hgb", "esm_hgb", "esm_context_hgb",
                         "target_mean", "context_ridge"]
        ], [96*mm, 37*mm, 37*mm])
        small("Evidencia reproducible: scripts/30_audit_secondary_evaluation.py, "
              "secondary_evaluation_review.json y temporal_screening_tie_sensitivity.csv. "
              "La sensibilidad conserva los resultados principales y no aporta un "
              "intervalo de incertidumbre para la diferencia de selección.")

    section("L.2. Umbral de inversión y fuentes dominantes")
    sensitivity = read_csv(out/"statistics/rank_threshold_sensitivity.csv")
    table(["Umbral log2", "Inversiones", "Proporción", "Intervalo 95%", "Artículos con inversiones"], [
        [f"{t.threshold_log2:.2f}", t.inversions, f"{t.rate:.2%}",
         f"[{t.ci_low:.2%}, {t.ci_high:.2%}]", t.publications_with_inversions]
        for t in sensitivity.itertuples()
    ], [30*mm, 27*mm, 30*mm, 45*mm, 38*mm])
    p("Se mantiene el mismo denominador y se exige superar el umbral en ambos medios, con "
      "signos opuestos. La sensibilidad es exploratoria: se añadió después de identificar un "
      "contraste próximo al límite. Los 5.000 remuestreos cruzados son comunes a todos los "
      "umbrales; el resultado principal de 1,00 log2 no se sustituye.")
    p("La implementación aplica una tolerancia numérica de <b>10<super>-12</super> log2</b>. "
      "Antes de esta corrección, ocho diferencias iguales al umbral se contaban como "
      "superaciones por redondeo de coma flotante. En la cohorte anterior a la corrección "
      "del medio, el recuento pasó de 41/1.380 a 33/1.380. La corrección del medio añadió "
      "35 comparaciones y dejó 33/1.415 en el catálogo ampliado. Los resultados previos "
      "quedan archivados como evidencia del desarrollo, no como alternativas para seleccionar.")
    source_review = read_json(ROOT/"results/reconstruction_2026-10-20/source_review/rank_source_review.json")
    p(f"En el catálogo ampliado, los dos artículos revisados aportan "
      f"<b>{source_review['top_two_inversions']} de {source_review['primary_inversions']}</b> "
      "inversiones principales. Los métodos originales confirman comparaciones de medios y, "
      "en el estudio de lipopéptidos, distinguen disolventes y formas lineales o cíclicas. "
      "El análisis conserva las notas del disolvente en el emparejamiento.")
    p("Un ejemplo de la base compara formas lineal y cíclica del mismo péptido: una razón "
      "de MIC de 8/16 en unidades de masa pasa a una diferencia de <b>-1,0029 log2</b> "
      "en unidades molares, por la diferencia de masa molecular. Supera el umbral primario "
      "aunque la diferencia original sea exactamente un paso de dilución de factor dos. "
      "Esto muestra sensibilidad de la definición; no demuestra que la conversión sea errónea.")
    p("Las tablas S1 y S2 del suplemento original de Jaśkiewicz et al. (2017) confirman "
      "las cuatro MIC del ejemplo: lineal/cíclico 8/16 en MH y 64/8 en BHI, en µg/ml "
      "y con PBS como disolvente. La tabla S7 informa masas de 1.002,44 y 1.000,43 Da; "
      "son próximas a las calculadas desde la estructura, 1.002,452 y 1.000,436 Da, "
      "pero no se presentan como coincidencias exactas. El cotejo numérico ampliado "
      "se describe más adelante; no valida todos los metadatos del artículo.")
    for source in source_review["sources"]:
        linked(source["title"], source["url"])

    section("L.3. Verificación de suplementos y corrección del medio")
    p("Se recuperaron los suplementos originales desde el depósito público de PMC, "
      "incluidos el PDF y el archivo Excel Data S1 de Randall et al. (2023). Los intentos "
      "anteriores de descarga fallidos se conservan en el registro, pero ya no constituyen "
      "una limitación de acceso a esas tablas. Los archivos originales no se modificaron.")
    table(["Cotejo dirigido de E. coli W3110", "Registros", "Evidencia"], [
        ("Coincidencia directa con Data S1", 74, "Hoja SySA; secuencia, medio y concentración."),
        ("SySA-66, ausente de esa hoja", 2, "PDF, página 10, tabla S2: MH 128 y RPMI 32 µg/ml."),
        ("SySA-5.17", 1, "PDF, página 12, tabla S4: W3110, MH 4 µg/ml."),
        ("Medio mal asignado en la base", 1, "SySA-73: PDF S2 y Excel I72/J72 coinciden."),
    ], [80*mm, 20*mm, 70*mm])
    p("La ficha DBAASP:21860:172770 contiene 8 µg/ml asignados a MHB; las fuentes "
      "primarias los asignan a RPMI-1640. El registro compañero 172769 ya contiene "
      "32 µg/ml en MHB. Se corrigió <b>el medio, conservando el valor 8</b>. Antes, "
      "ambos registros se agregaban en una sola celda con media geométrica de 16 µg/ml; "
      "después corresponden a dos celdas, cada una con su medio y concentración originales.")
    p("En la ampliación, las celdas exactas pasan de 85.057 a 85.058 y las predictivas de "
      "66.876 a 66.877. El número de registros con censura elegibles permanece en 117.286. "
      "Se archivaron los resultados derivados anteriores y se repitieron los análisis "
      "afectados. El registro de enmiendas conserva valores esperados, localizadores de "
      "evidencia y hashes; una auditoría verifica su propagación a las celdas analíticas.")
    small("Alcance: revisión dirigida por el asistente, seleccionada tras observar los "
          "resultados. No equivale a una extracción independiente por dos investigadores "
          "ni permite estimar la frecuencia de errores de toda la base. Rutas: "
          "data/source_amendments_2026-10-20.json; source_review/randall2023_MIC_checks.json "
          "y source_review/supplement_download_manifest.json dentro de los resultados de reconstrucción.")

    section("L.4. Agregados y anotaciones que siguen sin resolver")
    p("La extracción de las tablas S1-S3 de Jaśkiewicz et al. (2017) recuperó 540 valores "
      "MIC de medio, compuesto, disolvente y cepa. Se cotejaron sus 253 registros en la "
      "base: <b>251 concordaron numéricamente</b>, incluidos los extremos de algunos "
      "agregados. Un registro puede representar ambos disolventes o varios aislados; "
      "por eso los recuentos de tabla y base no son intercambiables. Se comprobó la "
      "alineación visual de las tres tablas, sin una segunda extracción humana independiente.")
    table(["Registro", "Hallazgo", "Tratamiento"], [
        ("10676:80842", "MIC 16 µg/ml sin medio en la base.", "Excluido por medio ausente; asignación a tabla pendiente de adjudicación."),
        ("10676:80872", "La base resume 16-32 µg/ml; los valores de cinco aislados en ambos disolventes abarcan 8-32 en TSB.", "Agregado discordante señalado; no entra en predicción o ranking de valores exactos. La sensibilidad sin rangos lo excluye."),
    ], [32*mm, 70*mm, 68*mm])
    p("Los registros que sustentan las seis inversiones fuertes de este artículo "
      "concuerdan con las tablas. Los dos casos pendientes no intervienen en ellas. "
      "La discrepancia del rango requiere reconstruir los ensayos individuales y su "
      "agregación; sustituir un extremo no resuelve la semántica de observación. "
      "Se conserva el dato agregado de la base con su discrepancia explícita.")
    p("Este hallazgo limita el modelo con todos los rangos: hay intervalos que resumen "
      "aislados diferentes, no censura de una única MIC latente. La convergencia no "
      "valida esa interpretación. El ajuste sin rangos examina la sensibilidad conjunta "
      "a este formato y no identifica el efecto aislado de un registro. Las conclusiones "
      "deben distinguir esos ajustes y sus poblaciones.")
    small("Reproducción y detalle por registro: source_review/check_jaskiewicz2017.py y "
          "source_review/jaskiewicz2017_MIC_checks.json, dentro de results/reconstruction_2026-10-20/.")

    section("M. Decisiones, resultados descartados y límites")
    table(["Decisión de reconstrucción", "Motivo y consecuencia"], [
        ("Retirar el techo universal de predicción", "La concordancia observada no identifica un límite común para otras tareas, poblaciones o modelos."),
        ("Retirar el reparto fijo del error de QMAP", "El porcentaje estaba fijado en código y no provenía de una estimación identificable."),
        ("Cambiar la interpretación de publicación", "Es un indicador de procedencia que mezcla química, selección y protocolos; no un efecto causal de laboratorio."),
        ("Conservar igualdad de MIC", "Dos observaciones pueden coincidir; la procedencia experimental debe sustentar una deduplicación."),
        ("Retirar el contraste de Fisher original", "El nulo agrupado no era intercambiable y los pares compartían observaciones."),
        ("Corregir la comparación numérica", "Ocho empates de umbral no son inversiones fuertes; se aplica una tolerancia de 10⁻¹² log2."),
        ("Corregir un medio según fuentes primarias", "El ensayo de 8 µg/ml de SySA-73 corresponde a RPMI-1640. Se preserva el valor y se recalculan las celdas y análisis afectados."),
        ("Mantener ajustes no convergentes", "Se archivan y se muestran sus diagnósticos; no se reutilizan sus estimaciones como resultados válidos."),
        ("Mostrar todas las comparaciones", "Evitar escoger una representación o partición solo porque su intervalo sea favorable."),
        ("Corregir el alcance del proyecto", "El autor reafirmó que será estrictamente in silico. La planificación experimental previa se conserva como historial y se retira de los requisitos activos."),
    ], [68*mm, 102*mm])
    p("Persisten límites materiales: anotaciones incompletas, verificación dirigida de fuentes, "
      "semántica de endpoints no armonizada exhaustivamente, selección de la cohorte predictiva "
      "por MIC exacta, medio reportado y secuencias compatibles, preentrenamiento no auditado y comparadores con "
      "hiperparámetros fijos. Los intervalos de confianza del efecto contextual son distintos "
      "de los intervalos predictivos evaluados con una cohorte de calibración separada.")
    p("No se ha realizado una reproducción por un investigador independiente. La evaluación "
      "de transferencia se limita a las particiones retrospectivas descritas. El estudio "
      "puede sustentar conclusiones sobre predicción y ranking de registros publicados; "
      "no permite afirmar beneficios experimentales de selección ni identificar efectos "
      "causales de laboratorio. Estas fronteras delimitan las afirmaciones del trabajo in silico.")

    section("N. Intervalos predictivos y cobertura observada")
    p("Esta extensión exploratoria usa cuatro configuraciones de boosting: descriptores y "
      "ESM-2, con y sin contexto. Se evalúan cinco particiones por publicación con una semilla "
      "y la partición temporal. Dentro del entrenamiento externo se reserva un 20% de "
      "publicaciones para calibrar; esas publicaciones no ajustan el predictor ni sus transformaciones.")
    p("El radio es la puntuación ordenada de posición k, donde k es el redondeo hacia "
      "arriba de (n+1)×c: n es el número de puntuaciones de calibración y c la cobertura "
      "nominal. Si k supera n, el radio es infinito. Se comparan máximos de error por "
      "publicación, para cobertura simultánea del artículo, y errores individuales "
      "agrupados, para cobertura marginal de una celda "
      '(<link href="https://doi.org/10.1080/01621459.2017.1307116" color="#216F79">Lei et al., 2018</link>).')
    interval_path = out/"prediction_intervals/coverage_by_split.csv"
    if interval_path.exists():
        intervals = read_csv(interval_path)
        selected = intervals[intervals.protocol.eq("temporal") &
                             intervals.score_kind.eq("publication_maximum")]
        p("<b>Prueba temporal: calibración con el error máximo de cada publicación.</b>",
          "SmallKeepAMP")
        table(["Modelo", "Nominal", "Anchura log2", "Cobertura celdas", "Artículos completos"], [
            [MODEL_LABELS[r.model], f"{r.nominal_coverage:.0%}", f"{r.width_log2:.2f}",
             f"{r.cell_weighted_coverage:.1%}", f"{r.whole_publication_coverage:.1%}"]
            for r in selected.itertuples()
        ], [69*mm, 20*mm, 26*mm, 27*mm, 28*mm], padding=6)
        example = selected[selected.model.eq("sequence_hgb") &
                           np.isclose(selected.nominal_coverage, .9)].iloc[0]
        p(f"En el modelo de descriptores sin contexto, el intervalo nominal del 90% tiene "
          f"radio <b>{example.radius_log2:.2f}</b> log2: alrededor de la predicción central, "
          f"sus extremos equivalen a dividir y multiplicar por <b>{2**example.radius_log2:.1f}</b>. "
          "Una cobertura elevada con esta amplitud ofrece poca resolución para ordenar "
          "candidatos cuyas MIC difieren solo por un factor de dos.")
        pooled = intervals[intervals.protocol.eq("temporal") &
                           intervals.model.eq("sequence_hgb") &
                           intervals.score_kind.eq("pooled_rows") &
                           intervals.nominal_coverage.eq(.9)].iloc[0]
        p(f"Al calibrar con residuos individuales agrupados, el intervalo nominal del "
          f"90% de ese modelo cubre <b>{pooled.cell_weighted_coverage:.1%}</b> de las "
          f"celdas y todos los casos de <b>{pooled.whole_publication_coverage:.1%}</b> "
          "de las publicaciones. Son cantidades distintas: una cobertura marginal "
          "cercana al valor nominal no exige cubrir simultáneamente todos los casos "
          "de un artículo. La comparación entre ambos procedimientos debe mantener "
          "explícita esa diferencia de objetivo.")
    else:
        p("Los resultados de esta extensión todavía no están disponibles en este alcance.")
    p("El porcentaje nominal no garantiza cobertura bajo cualquier cambio de distribución. "
      "La intercambiabilidad entre estudios puede fallar por selección de publicaciones, "
      "péptidos compartidos y cambio temporal. Por eso se informa la cobertura observada, "
      "la anchura y el porcentaje de publicaciones cuyos casos quedan todos cubiertos.")
    p("Son 24 ajustes adicionales con menos datos de entrenamiento que el benchmark "
      "principal. Se conservan asignaciones a entrenamiento, calibración y prueba, "
      "predicciones, residuos y límites. Ninguna etiqueta de prueba determina el radio.")

    section("O. Reproducibilidad y expediente de evidencia")
    p("La reconstrucción conserva tres niveles: datos originales con huellas, resultados "
      "computacionales auditables y documentos generados desde esas salidas. Los resultados "
      "anteriores a la reconstrucción y los ajustes de desarrollo permanecen separados en archive/. "
      "La existencia de un archivo histórico no lo convierte en resultado vigente.")
    table(["Contenido", "Ubicación en el proyecto"], [
        ("Entradas y metadatos congelados", "data/raw/; data/curated/"+scope+"/"),
        ("Modelos y particiones", "results/reconstruction_2026-10-20/"+scope+"/benchmark/"),
        ("Concordancia, ranking y REML", "results/reconstruction_2026-10-20/"+scope+"/statistics/"),
        ("Cadenas y comprobaciones posteriores", "results/reconstruction_2026-10-20/"+scope+"/censored/"),
        ("Suma posterior exploratoria", "results/reconstruction_2026-10-20/"+scope+"/posterior_component_sum/"),
        ("Revisión dirigida de fuentes", "results/reconstruction_2026-10-20/source_review/"),
        ("Alcance autorizado e historial", "data/study_scope_2026-10-20.json; archive/"),
        ("Referencias y antecedentes", "manuscript/reference_metadata/; results/reconstruction_2026-10-20/literature_review.json"),
        ("Tablas, figuras y procedencia del informe", "manuscript/generated/"+scope+"/"),
        ("Entorno Python fijado", "requirements-reconstruction.lock.txt"),
        ("Auditoría final", "results/reconstruction_2026-10-20/"+scope+"/completion_audit.json"),
    ], [61*mm, 109*mm], padding=5)
    p("El entorno utiliza Python 3.11, R con lme4/jsonlite y CmdStan 2.40.0; Tectonic y "
      "Poppler compilan y renderizan los documentos. Los paquetes Python y los detalles de R "
      "se conservan en archivos de entorno. Los pesos ESM se verifican mediante su manifiesto.")
    p("La secuencia reproducible es adquisición, bibliografía, curación, representaciones, "
      "particiones, pruebas, estadística, comparaciones predictivas, modelos censurados, "
      "diagnósticos y generación documental. El comando principal es "
      "<b>./run_pipeline.sh --scope expanded</b>. Una entrada o configuración incompatible "
      "impide reanudar silenciosamente resultados antiguos.")
    small("Las revisiones adicionales conservan verificadores separados: "
          "scripts/30_audit_secondary_evaluation.py y scripts/31_review_prediction_intervals.py; "
          "en source_review/, verify_all_posterior_inputs_20261021.py, "
          "verify_pending_sensitivity_inputs_20261021.py, check_component_sum_20261021.py "
          "y review_final_sensitivities_20261022.py. La generación documental usa "
          "scripts/21_generate_manuscript_inputs.py y scripts/24_build_execution_report.py "
          "con --scope expanded. El nombre histórico del primer script no activa la "
          "redacción del artículo: el manifiesto de alcance mantiene esa tarea diferida.")
    p("La revisión final comprueba integridad, cobertura de ajustes, diagnósticos, procedencia "
      "de las cifras y páginas renderizadas. La finalización del estudio exige completar "
      "el alcance in silico ampliado y verificar sus documentos. No se crea un identificador "
      "de repositorio, financiación o afiliación que no haya sido confirmado.")
    digest = hashlib.sha256((curated/"benchmark_cells.csv").read_bytes()).hexdigest()
    small("SHA-256 de la cohorte predictiva: "+digest[:32]+"<br/>"+digest[32:])

    literature = read_json(ROOT/"results/reconstruction_2026-10-20/literature_review.json")
    for group, title in [
        ("benchmark", "P. Novedad frente a benchmarks recientes"),
        ("measurement", "Q. Antecedentes de medición y contexto"),
    ]:
        section(title)
        for reference in [x for x in literature["sources"] if x["group"] == group]:
            p("<b>"+escape(reference["label"])+"</b><br/>"+
              escape(reference["established"])+" "+escape(reference["implication"]), "BodyKeepAMP")
            p("<b>Alcance de la revisión:</b> "+escape(reference["evidence_read"])+" "+
              escape(reference["not_claimed"]), "SmallKeepAMP")
            if reference.get("reviewed_code_url"):
                small(f'<link href="{escape(reference["url"])}" color="#216F79">Fuente primaria</link> | '
                      f'<link href="{escape(reference["reviewed_code_url"])}" color="#216F79">Repositorio revisado (versión fija)</link>')
            else:
                linked("Fuente primaria", reference["url"])
        small("Evaluación dirigida actualizada el 20 de octubre de 2026. Las diferencias "
              "de datos, particiones y escala impiden comparar directamente las métricas "
              "publicadas con las calculadas en esta reconstrucción.")

    section("R. Qué novedad está sustentada y qué falta")
    table(["Propuesta de contribución", "Valoración con la evidencia disponible"], [
        ("El contexto modifica la actividad",
         "Antecedente establecido. No debe presentarse como descubrimiento de este estudio."),
        ("Las particiones cambian el rendimiento",
         "Conocido en evaluación predictiva y ya abordado por QMAP. Nuestro análisis precisa varias poblaciones de transferencia."),
        ("Curación trazable y soporte de comparación",
         "Aportación implementada: conecta química, límites, procedencia y escaso cruce entre publicaciones. Su prioridad mundial no está demostrada."),
        ("Mejora predictiva por contexto",
         "Se estima en pares de modelos y casos idénticos. La significación, si aparece, no sustituye una magnitud útil ni la réplica externa."),
        ("Método o arquitectura nueva",
         "No se propone una arquitectura nueva. Los componentes estadísticos y predictivos utilizados tienen precedentes."),
        ("Utilidad de la evaluación computacional",
         "Se examinan error, calibración, cobertura y ranking retrospectivo. Su valor debe demostrarse mediante resultados robustos y consecuencias científicas concretas."),
    ], [61*mm, 109*mm])
    descriptor = metrics[metrics.model.eq("sequence_hgb") &
                         metrics.stratum.eq("all")].set_index("protocol")
    random_mae = descriptor.loc["random", "mae_log2"]
    homology_mae = descriptor.loc["homology50", "mae_log2"]
    p(f"Como ejemplo cuantitativo de este alcance, el MAE del modelo de descriptores "
      f"con boosting pasa de <b>{random_mae:.3f}</b> log2 en la partición aleatoria a "
      f"<b>{homology_mae:.3f}</b> en la separación por homología al 50%, un aumento "
      f"descriptivo del <b>{100*(homology_mae/random_mae-1):.1f}%</b>. La comparación "
      "utiliza la misma cohorte, con distintas asignaciones de entrenamiento y prueba; "
      "no estima un efecto causal ni un límite universal de precisión.")
    focus = pairs[pairs.base.eq("sequence_hgb") &
                  pairs.alternative.eq("sequence_context_hgb") &
                  (pairs.stratum.eq("all") |
                   (pairs.protocol.eq("joint") & pairs.stratum.eq("new_sequence__new_publication")))]
    zero_included = focus[(focus.ci_low <= 0) & (focus.ci_high >= 0)]
    if len(zero_included):
        labels = {"homology50": "homología al 50%", "homology80": "homología al 80%",
                  "temporal": "tiempo", "joint": "secuencia y publicación nuevas",
                  "random": "aleatoria", "sequence": "secuencia", "publication": "publicación"}
        p("Para ese par de modelos, el intervalo puntual del 95% de la mejora al añadir "
          "contexto incluye cero en: <b>"+
          escape(", ".join(labels.get(x, x) for x in zero_included.protocol))+"</b>. "
          "Esto es compatible con beneficios o perjuicios pequeños; no demuestra "
          "equivalencia ni irrelevancia biológica de las condiciones del ensayo. Los "
          "resultados de todas las familias y sus límites de inferencia aparecen en "
          "las secciones H e I.")
    p("Mi valoración es que la versión actual sostiene una reconstrucción metodológica y "
      "empírica, pero no demuestra aún una contribución del alcance necesario para recomendar "
      "un envío a Nature o Science. Aumentar el número de ajustes o de páginas no resuelve "
      "esa brecha. La aportación debe cambiar una conclusión científica con evidencia "
      "computacional robusta y distinguirse de los antecedentes. Esta valoración no impone "
      "validación experimental ni presupone que un estudio in silico sea inadecuado para esas revistas.")
    p("Nature Communications y Science Advances siguen siendo objetivos ambiciosos, "
      "condicionados a una conclusión sustancial y bien sustentada. Nature Computational Science "
      "ofrece un encaje temático si emerge una contribución computacional de suficiente alcance. "
      "El encaje temático no equivale a preparación para el envío ni permite estimar la aceptación.")
    if len(passing) == 5:
        p("La actualización del <b>22 de octubre de 2026</b> incorpora los cinco ajustes "
          "terminados y sus revisiones. La sensibilidad al prior y la suma exploratoria "
          "añaden información, pero no resuelven las discrepancias predictivas ni establecen "
          "novedad frente a herramientas recientes. Por encaje, <b>Nature Computational "
          "Science, formato Analysis</b>, es mi primera opción ambiciosa.")
    p("Nature Computational Science contempla el formato <b>Analysis</b> para análisis "
      "de datos existentes o comparaciones que produzcan conclusiones nuevas de interés "
      "para una audiencia amplia. Por tanto, una arquitectura nueva no es la única vía "
      "editorial. Communications Biology contempla análisis de datos secundarios y exige "
      "un avance con nueva comprensión biológica para un área especializada. Es una opción "
      "potencialmente más plausible dentro de Nature Portfolio si se demuestra esa "
      "conclusión; la curación y la ejecución correcta de modelos no la establecen por sí solas.")
    linked("Nature Computational Science: tipos de artículos, incluido Analysis",
           "https://www.nature.com/natcomputsci/content")
    p("El editorial de formatos de Nature Computational Science de diciembre de 2023 "
      "describe Analysis como una comparación sistemática con conclusiones prácticas "
      "importantes y una evaluación de los métodos en sus mejores condiciones. Las 14 "
      "configuraciones controladas de esta reconstrucción no constituyen por sí solas "
      "una comparación exhaustiva de las herramientas especializadas actuales. Este "
      "criterio orienta la valoración editorial; no convierte una arquitectura nueva "
      "ni experimentos de laboratorio en requisitos de esta investigación.")
    linked("Nature Computational Science: Formats for reporting primary research (2023)",
           "https://www.nature.com/articles/s43588-023-00588-y")
    p("El editorial de abril de 2026 sobre observaciones frecuentes de revisores refuerza "
      "la necesidad de justificar los métodos de comparación y su configuración, considerar "
      "herramientas recientes, examinar posibles fugas de información y explicar la utilidad "
      "práctica y las limitaciones. Se consultó el texto oficial indexado. Es contexto "
      "editorial para valorar esta investigación, no evidencia de novedad ni una decisión "
      "de la revista sobre estos resultados.")
    linked("Nature Computational Science: What reviewers request the most (2026)",
           "https://www.nature.com/articles/s43588-026-00989-9")
    linked("Communications Biology: alcance y criterios de publicación",
           "https://www.nature.com/commsbio/aims")
    small("Las valoraciones intermedias se conservan en los registros editorial_positioning* "
          "de source_review/, con su evidencia y nivel de acceso. El editorial de formatos "
          "de 2023 se leyó como contenido oficial indexado, sin acceso autenticado. "
          "Estos criterios no constituyen una probabilidad de aceptación.")
    small("La síntesis editorial tras los cinco ajustes figura en "
          "source_review/editorial_positioning_final_analysis_20261022.json. Reutiliza "
          "fuentes editoriales consultadas el 21/10/2026 y conserva sus límites de acceso; "
          "la fecha de actualización del juicio sobre el estudio no implica una nueva "
          "consulta de esas páginas. Las valoraciones intermedias se preservan como historial.")
    p("Antes de sostener una superioridad metodológica faltan comparaciones ejecutables "
      "con herramientas especializadas recientes en una tarea común. Las 14 configuraciones "
      "de esta reconstrucción son comparadores controlados; no equivalen a reproducir todas "
      "las herramientas de los benchmarks citados.")
    p("La búsqueda fue dirigida, no sistemática: no se aplicó un protocolo exhaustivo de "
      "selección ni se demuestra ausencia de antecedentes. El registro conserva consultas, "
      "fuentes, fecha, nivel de lectura y limitaciones de acceso. Para Yue se revisó el "
      "resumen; para BATTLE-AMP, el resumen y código oficial posterior al preprint; para "
      "STAMP, el registro institucional. Esa cobertura limita cualquier afirmación de "
      "diferencia exclusiva.")
    p("La prueba temporal ordena publicaciones de ensayos, no las fechas de creación de "
      "los modelos ni de su preentrenamiento. Es una evaluación retrospectiva con herramientas "
      "actuales; no reproduce las predicciones que habrían estado disponibles históricamente "
      "en el año de corte.")
    small("Registro: results/reconstruction_2026-10-20/literature_review.json. "
          "La revisión inicial se conserva en output/pdf/Revision_critica_AMP_novedad_2026-10-20.pdf.")

    section("S. Referencias bibliográficas y procedencia")
    p("Las fichas bibliográficas conservan autores, título, año, revista y DOI. Esta lista "
      "sirve para localizar las fuentes; no implica que se haya leído íntegramente cada "
      "artículo. Las secciones P y Q y el registro de revisión indican qué contenido se "
      "consultó y sus limitaciones de acceso. Las fuentes originales y sus suplementos "
      "se documentan además mediante ubicaciones, fechas y huellas de archivo.")
    for path in sorted((ROOT/"manuscript/reference_metadata").glob("*.json")):
        reference = read_json(path)
        authors = reference.get("author", [])
        names = [a.get("family", a.get("name", "")) for a in authors[:3]]
        author_text = ", ".join(names)+(" et al." if len(authors) > 3 else "")
        date = reference.get("published-print") or reference.get("published-online") or reference.get("published", {})
        year = str(date.get("date-parts", [[""]])[0][0])
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(
            (reference.get("title") or [path.stem])[0]))).strip()
        venue = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(
            (reference.get("container-title") or [""])[0]))).strip()
        citation = ". ".join(part.strip().rstrip(".") for part in
                             [f"{author_text} ({year})", title, venue] if part.strip())+"."
        p(escape(citation), "BodyKeepAMP")
        doi = reference.get("DOI")
        if doi:
            linked("DOI: "+doi, "https://doi.org/"+doi)
        small("Ficha bibliográfica: manuscript/reference_metadata/"+path.name)
    p("Los antecedentes sin ficha DOI completa, los preprints y los repositorios de "
      "código se conservan también en literature_review.json, con enlace primario, "
      "versión consultada y alcance de lectura. Los criterios editoriales son fuentes "
      "de orientación; no se confunden con resultados científicos del estudio.")

    section("T. Guía de las figuras y tablas")
    guide_path = ROOT/"manuscript/generated"/scope/"figure_guide.json"
    if guide_path.exists():
        guide = read_json(guide_path)
        p("Cada figura se conserva en PNG y PDF vectorial. El directorio "
          f"manuscript/generated/{scope}/ mantiene su nombre histórico como ubicación "
          "de los materiales numéricos y gráficos; no implica que se esté redactando "
          "un manuscrito. Esta guía identifica el contenido y sus fuentes.")
        for item in guide["figures"]:
            p("<b>"+escape(item["title"])+"</b><br/>"+escape(item["description"]), "BodyKeepAMP")
            small("Archivos: "+escape(item["stem"])+".png / .pdf<br/>Fuentes: "+
                  escape("; ".join(item["sources"])))
    p("Las tablas de métricas completas, comparaciones pareadas, selección y calibración "
      "se conservan en CSV bajo benchmark/. La cobertura y amplitud de intervalos están "
      "en prediction_intervals/coverage_by_split.csv. Los diagnósticos y sensibilidades "
      "se conservan en statistics/, censored/ y los registros de revisión. Las tablas "
      "presentadas en esta memoria son resúmenes; los archivos conservan todas las "
      "configuraciones y estratos, incluidas estimaciones desfavorables o inciertas.")
    table_guide_path = ROOT/"manuscript/generated"/scope/"table_guide.json"
    if table_guide_path.exists():
        table_guide = read_json(table_guide_path)
        p(f"El índice <b>table_guide.json</b> documenta las {len(table_guide['tables'])} "
          "tablas de presentación: qué filas o modelos seleccionan, sus límites de "
          "interpretación y las fuentes numéricas completas con huellas SHA-256. "
          "Las tablas de presentación están redondeadas; los cálculos posteriores "
          "deben utilizar los CSV o JSON originales enlazados.")
