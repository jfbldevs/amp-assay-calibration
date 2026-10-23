#!/usr/bin/env python3
"""Detailed Spanish research memory; study numbers come from versioned outputs."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_rebuild.reporting import ROOT, posterior_results, read_csv, read_json
from amp_rebuild.dossier import append_research_record
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

INK = colors.HexColor("#163144")
TEAL = colors.HexColor("#216F79")
LIGHT = colors.HexColor("#EEF4F6")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["sample", "expanded"], default="sample")
    args = parser.parse_args()
    scope = args.scope
    out = ROOT/"results/reconstruction_2026-10-20"/scope
    curated = ROOT/"data/curated"/scope
    generated = ROOT/"manuscript/generated"/scope
    manifest = read_json(curated/"curation_manifest.json")
    support = read_json(out/"statistics/design_support.json")
    metrics = read_csv(out/"benchmark/metrics.csv")
    pairs = read_csv(out/"benchmark/paired_context_improvements.csv")
    audit = read_json(out/"benchmark/completion_audit.json")
    ranks = read_json(out/"statistics/matched_rank_summary.json")["all_matching_reported_fields"]
    reml = read_json(out/"statistics/reml_sensitivity.json")
    posteriors = posterior_results(scope)
    passed = sum(r["diagnostics"]["passed"] for r, _ in posteriors.values())
    report_provenance = read_json(generated/"report_provenance.json")
    data_scope = "catálogo ampliado" if scope == "expanded" else "muestra inicial de 2.500 fichas"
    # Use open fonts shipped with matplotlib, keeping the build portable.
    import matplotlib
    fonts = Path(matplotlib.get_data_path())/"fonts/ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(fonts/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuBold", str(fonts/"DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVuBold",
                                italic="DejaVu", boldItalic="DejaVuBold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyAMP", fontName="DejaVu", fontSize=9.4, leading=14.3,
                             textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle("TitleAMP", fontName="DejaVuBold", fontSize=25, leading=31,
                             textColor=INK, spaceAfter=15))
    styles.add(ParagraphStyle("HeadingAMP", fontName="DejaVuBold", fontSize=15, leading=20,
                             textColor=TEAL, spaceAfter=11))
    styles.add(ParagraphStyle("SmallAMP", fontName="DejaVu", fontSize=7.8, leading=11,
                             textColor=INK, spaceAfter=6))
    styles.add(ParagraphStyle("KickerAMP", fontName="DejaVuBold", fontSize=9.2, leading=13,
                             textColor=TEAL, spaceAfter=14))
    styles.add(ParagraphStyle("BodyKeepAMP", parent=styles["BodyAMP"], keepWithNext=True))
    styles.add(ParagraphStyle("SmallKeepAMP", parent=styles["SmallAMP"], keepWithNext=True))
    story = []
    def p(text, style="BodyAMP"):
        story.append(Paragraph(text, styles[style]))
    def heading(text):
        p(text, "HeadingAMP")
    def table(headers, rows, widths, padding=7):
        cells = [[Paragraph(escape(str(x)), styles["SmallAMP"]) for x in headers]]
        cells += [[Paragraph(escape(str(x)), styles["SmallAMP"]) for x in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), LIGHT), ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LINEBELOW", (0,0), (-1,0), .8, TEAL),
            ("LINEBELOW", (0,1), (-1,-1), .3, colors.HexColor("#D9E2E6")),
            ("TOPPADDING", (0,0), (-1,-1), padding), ("BOTTOMPADDING", (0,0), (-1,-1), padding),
        ]))
        story.extend([t, Spacer(1, 10)])
    def fig(path, width=170*mm):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            height = width*im.height/im.width
        story.extend([Image(str(path), width=width, height=height), Spacer(1, 7)])
    def page():
        story.append(PageBreak())
    def metric(protocol, model="sequence_hgb"):
        return metrics[metrics.protocol.eq(protocol)&metrics.model.eq(model)&metrics.stratum.eq("all")].iloc[0]
    def component(key, name):
        return next(r["fraction"] for r in reml[key]["variance_components"] if r["grp"] == name)

    p("MEMORIA DE INVESTIGACIÓN · 20 OCTUBRE 2026", "KickerAMP")
    p("Actualización documental: "+datetime.now().strftime("%d/%m/%Y"), "SmallAMP")
    p("Reconstrucción científica<br/>del estudio de actividad AMP", "TitleAMP")
    p(f"Alcance de este informe: <b>{data_scope}</b>. Las cifras de esta versión proceden "
      "de resultados calculados y verificados. El estudio es estrictamente in silico y "
      "la validación experimental nueva queda fuera de su alcance.")
    if passed < 5:
        p(f"<b>Documentación provisional:</b> están disponibles {passed} de cinco "
          "especificaciones censuradas con diagnóstico computacional satisfactorio. "
          "La síntesis final de sensibilidad permanece pendiente.")
    heading("Dictamen operativo")
    p("La reconstrucción convierte la propuesta inicial en un análisis trazable de curación, "
      "generalización y contexto del ensayo. Retira las afirmaciones que los datos no identifican: "
      "un efecto causal de laboratorio, un techo universal de precisión y una descomposición "
      "del error de modelos publicados que no procedía de una estimación reproducible.")
    p(f"El benchmark contiene <b>{audit['verified_fits']:,} ajustes verificados</b> sobre "
      f"<b>{manifest['benchmark_exact_assay_cells']:,} celdas de ensayo</b>. "
      "El contraste entre particiones permite evaluar transferencia. El valor adicional del contexto "
      "se informa con comparaciones pareadas completas, incluidas mejoras pequeñas y resultados desfavorables.")
    table(["Componente", "Estado verificable"], [
        ("Curación", f"{manifest['cards']:,} fichas; {manifest['records']:,} registros; límites y unidades conservados."),
        ("Benchmark", "14 configuraciones; particiones por secuencia, similitud, publicación, exposición conjunta y tiempo."),
        ("Inferencia", f"{passed}/5 especificaciones censuradas con diagnóstico computacional satisfactorio en este alcance."),
        ("Alcance", "Análisis retrospectivo in silico de mediciones existentes; sin fase experimental pendiente."),
        ("Redacción del artículo", "Diferida por instrucción del autor; la entrega actual es la documentación científica de la investigación."),
    ], [39*mm, 131*mm])
    p("El objetivo editorial debe ajustarse a la contribución demostrada. Un análisis retrospectivo "
      "reproducible puede constituir un recurso útil; una aspiración a revistas de máximo impacto "
      "requiere una contribución nueva y sustancial, evidencia robusta y consecuencias científicas claras.", "SmallAMP")
    page()

    heading("1. Qué dicen las comparaciones predictivas")
    rows = []
    for proto, label in [("random","Aleatoria"),("sequence","Nueva secuencia"),
                         ("homology80","Similitud 0,80"),("homology50","Similitud 0,50"),
                         ("publication","Nueva publicación"),("temporal","Publicaciones posteriores")]:
        r = metric(proto)
        rows.append((label, f"{r.r2:.3f}", f"{r.mae_log2:.3f}", f"{r.within_one_dilution:.1%}"))
    table(["Partición", "R²", "MAE log2", "Error ≤ 1 log2"], rows, [70*mm, 28*mm, 35*mm, 37*mm])
    p("La tabla muestra el modelo de boosting con descriptores para mantener una comparación "
      "interpretable. Las tablas completas conservan también ridge, ESM-2, huellas moleculares y sus pares "
      "con contexto. El rendimiento temporal corresponde a otro conjunto de publicaciones y "
      "no debe tratarse como una diferencia causal frente a la partición aleatoria.")
    fig(generated/"fig_context_gain.png")
    temporal = pairs[pairs.protocol.eq("temporal")&pairs.base.eq("sequence_hgb")&pairs.stratum.eq("all")].iloc[0]
    p(f"Al añadir contexto al boosting con descriptores en la evaluación temporal, la reducción "
      f"de MAE es <b>{temporal.mae_reduction_log2:.3f}</b> log2, con intervalo "
      f"<b>[{temporal.ci_low:.4f}, {temporal.ci_high:.4f}]</b>. "
      "Los intervalos son exploratorios, puntuales y condicionados a los modelos ajustados; "
      "no incorporan un nuevo entrenamiento en cada remuestreo.")
    context_summary = report_provenance.get("context_effect_summary", {}).get("temporal")
    if context_summary:
        p(f"En las seis parejas de representación y aprendizaje, las reducciones temporales "
          f"puntuales van de <b>{context_summary['minimum_point_gain_log2']:.3f} a "
          f"{context_summary['maximum_point_gain_log2']:.3f}</b> log2: representan entre "
          f"<b>{context_summary['minimum_relative_point_gain_percent']:.2f}% y "
          f"{context_summary['maximum_relative_point_gain_percent']:.2f}%</b> del MAE de cada "
          "comparador. Es una descripción de magnitud; no acredita una ventaja útil para seleccionar péptidos.")
    p("Una ganancia reducida del metadato disponible no demuestra que el contexto experimental "
      "sea irrelevante. Campos ausentes, categorías poco precisas y unidades incompletas limitan "
      "la información que el modelo puede utilizar.")
    page()

    heading("2. Curación: qué cambió y qué queda sin resolver")
    fig(generated/"fig_curation.png")
    table(["Decisión", "Consecuencia científica"], [
        ("Concentraciones", "Los valores molares originales se preservan. Masa a molar exige una masa verificable."),
        ("Censura", "Se distinguen valores exactos, límites, rangos y medias con error; los límites no se convierten en puntos."),
        ("Química", "Se conservan estereoquímica, extremos y modificaciones. Los alias estructurales se agrupan para evitar fuga."),
        ("Publicaciones", "Se verifica identidad bibliográfica y año; los casos ambiguos no entran en la evaluación principal."),
        ("Duplicados", "Dos MIC iguales no bastan para eliminar un registro. Las coincidencias dudosas quedan señaladas."),
        ("Metadatos", "Lo no informado permanece ausente. La etiqueta de medio NA se conserva como agar nutritivo."),
    ], [38*mm, 132*mm])
    p(f"La cola de revisión conserva <b>{manifest['manual_review_records']:,} registros</b>. "
      "Su existencia hace visible la incertidumbre de extracción; no implica que todas esas "
      "mediciones sean erróneas. La verificación bibliográfica tampoco sustituye una revisión "
      "del valor original en la tabla, figura o método del artículo.")
    p("La muestra original y los análisis de desarrollo se preservan en archivos separados. "
      "Las predicciones no se mezclan entre versiones de datos. El informe incorpora "
      "comprobaciones de identidad molecular, etiquetas, particiones y métricas recalculadas.")
    page()

    heading("3. Límites de la interpretación de las varianzas")
    p(f"En la cohorte de celdas con MIC exacta, <b>{100*support['single_publication_fraction']:.1f}%</b> de las "
      f"entidades químicas aparece en una sola publicación. Solo <b>{support['multi_publication_entities']:,}</b> "
      "se repiten entre publicaciones. Ese diseño observacional permite que un efecto de publicación "
      "absorba selección de péptidos, protocolos, objetivos y otras variables no registradas.")
    p("El escaso solapamiento limita las comparaciones directas entre artículos, pero no "
      "demuestra por sí solo falta de identificación algebraica. La memoria detalla una "
      "comprobación separada de las matrices de covarianza; su independencia no establece "
      "una interpretación causal de los componentes.")
    fig(generated/"fig_variance_sensitivity.png")
    p(f"La fracción descriptiva atribuida a publicación en el REML aditivo cambia de "
      f"<b>{100*component('all_additive','publication_id'):.1f}%</b> en el conjunto completo "
      f"a <b>{100*component('multi_publication_additive','publication_id'):.1f}%</b> al restringir "
      "a entidades compartidas. Cambian tanto el conjunto como el soporte del diseño: esta "
      "comparación es una sensibilidad, no una estimación causal.")
    p(f"El análisis de ranking conserva <b>{ranks['pairs']:,}</b> contrastes pareados de "
      f"<b>{ranks.get('publications',0):,}</b> publicaciones, con "
      f"un recuento de inversiones estrictas de <b>{ranks.get('inversions',0):,}</b>. "
      "La dependencia entre pares que comparten péptidos se mantiene en el remuestreo. "
      "Campos ausentes iguales no prueban que dos protocolos sean equivalentes.")
    remaining = (" Las restantes permanecen pendientes o fallidas en las tablas de diagnóstico."
                 if passed < 5 else "")
    p(f"Se han obtenido <b>{passed} de cinco</b> especificaciones censuradas que pasan los "
      "umbrales computacionales en este alcance."+remaining+
      " Convergencia numérica y adecuación científica son preguntas distintas.")
    checked = report_provenance.get("posterior_specifications_checked", 0)
    flagged = report_provenance.get("posterior_specifications_with_extreme_tail_flag", 0)
    if checked:
        p(f"Las comprobaciones predictivas internas abarcan <b>{checked}</b> especificaciones; "
          f"<b>{flagged}</b> presentan alguna discrepancia extrema en los resúmenes de valores exactos. "
          "La memoria las muestra junto con sus límites: la selección de registros exactos y "
          "el carácter interno de esta comprobación impiden tratarla como validación externa.")
    p("El ICC de GRAMPA se presenta como acuerdo entre registros bajo una agrupación concreta. "
      "No se transforma en una cota universal de R² ni se usa para repartir el error de QMAP.")
    page()

    heading("4. Alcance y validación estrictamente in silico")
    p("El autor reafirmó que no habrá validación experimental. El estudio analiza "
      "mediciones existentes y su procedencia; su finalización depende de los análisis "
      "computacionales y de la documentación verificada. La decisión se registra en "
      "<b>data/study_scope_2026-10-20.json</b>.")
    table(["Evidencia", "Qué permite evaluar"], [
        ("Particiones retenidas", "Transferencia a secuencias, grupos de similitud, publicaciones y periodos definidos."),
        ("Comparaciones pareadas", "Valor adicional del contexto sobre los mismos casos de prueba y modelos comparables."),
        ("Calibración y ranking", "Cobertura, amplitud y selección retrospectiva de candidatos en las poblaciones evaluadas."),
        ("Sensibilidades", "Dependencia de las conclusiones respecto de curación, límites, cohortes y supuestos del modelo."),
        ("Trazabilidad", "Correspondencia entre fuentes, datos derivados, predicciones, métricas, tablas y figuras."),
    ], [50*mm, 120*mm])
    p("La separación por publicaciones no acredita independencia entre instituciones. "
      "La prueba temporal ordena artículos de ensayos y no reproduce una predicción "
      "prospectiva histórica. La revisión de fuentes es dirigida y no valida cada "
      "anotación del catálogo. Estas limitaciones deben acompañar las conclusiones.")
    p("La planificación de laboratorio preparada previamente se conserva como historial "
      "fuera del alcance activo. No se incluyen sus escenarios de potencia como resultados "
      "del estudio ni se mantienen solicitudes de colaboradores, presupuesto o mediciones.")
    p("Un resultado nulo o negativo puede ser una contribución computacional si responde "
      "una pregunta nueva, sobrevive a controles pertinentes y modifica una conclusión "
      "científica. Completar más ajustes, por sí solo, no establece esa novedad.")
    page()

    heading("5. Posicionamiento editorial y próximos criterios")
    p("La novedad no puede descansar solamente en que el ensayo importa o en que una partición "
      "aleatoria produce mejores métricas. QMAP ya establece un benchmark cuantitativo y el "
      "trabajo de Yue y colaboradores añade una evaluación retrospectiva reciente. La aportación "
      "diferenciadora debe demostrar qué decisiones científicas cambian gracias a esta reconstrucción.")
    table(["Objetivo", "Evidencia que todavía debe consolidarse"], [
        ("Nature / Science", "Para esta propuesta: un hallazgo de alcance amplio, respaldo computacional convincente y una consecuencia científica que exceda una auditoría del catálogo."),
        ("Nature Computational Science", "Posible encaje como Analysis si aporta conclusiones nuevas y prácticas de interés amplio. El editorial de formatos pide comparar métodos en sus mejores condiciones; las 14 configuraciones actuales no equivalen a evaluar exhaustivamente las herramientas especializadas."),
        ("Nature Communications / Science Advances", "Para esta propuesta: una conclusión sustancial sobre transferencia o medición, con validación computacional suficiente para sostenerla."),
        ("Communications Biology", "Opción dentro de Nature Portfolio si se consolida una conclusión biológica nueva para una comunidad especializada. Su alcance incluye análisis de datos secundarios."),
    ], [54*mm, 116*mm])
    p("Este cuadro es una valoración editorial del estudio, no una predicción de aceptación "
      "ni una regla universal que exija experimentos de laboratorio en esas revistas. Un resultado "
      "nulo bien sustentado puede ser valioso, pero necesita una pregunta relevante y evidencia "
      "suficiente para descartar explicaciones triviales como metadatos insuficientes o modelos poco competitivos.")
    heading("Entregables reproducibles")
    p("El expediente contiene la memoria de investigación, las figuras y tablas generadas, "
      "las predicciones individuales, las auditorías de particiones, el historial de curación y "
      "la decisión documentada de alcance in silico. Por instrucción expresa del autor, "
      "la redacción del manuscrito y del suplemento para una revista queda para una etapa "
      "posterior. Los borradores de desarrollo previos se conservan como material histórico.")
    p("<b>Reproducción:</b> usar el entorno fijado en "
      "<font face='DejaVuBold'>requirements-reconstruction.lock.txt</font>, CmdStan 2.40.0 "
      "y el flujo <font face='DejaVuBold'>./run_pipeline.sh --scope expanded</font>. "
      "El flujo rechaza un catálogo incompleto y una mezcla de entradas o configuraciones distintas. "
      "El cálculo bayesiano y las representaciones requieren recursos apreciables.")
    heading("Fuentes de posicionamiento")
    sources = [
        ("QMAP, Scientific Reports (2026)", "https://doi.org/10.1038/s41598-026-56004-8"),
        ("Yue et al., Small Methods (2026)", "https://doi.org/10.1002/smtd.71137"),
        ("Loffredo et al., PNAS (2021)", "https://doi.org/10.1073/pnas.2014364118"),
        ("Kolmar y Grulke, Journal of Cheminformatics (2021)", "https://doi.org/10.1186/s13321-021-00571-7"),
        ("Nature: criterios y proceso editorial", "https://www.nature.com/nature/for-referees/policies-and-processes"),
        ("Nature Computational Science: alcance editorial", "https://www.nature.com/natcomputsci/aims"),
        ("Nature Computational Science: formato Analysis", "https://www.nature.com/natcomputsci/content"),
        ("Nature Computational Science: editorial sobre formatos (2023)", "https://www.nature.com/articles/s43588-023-00588-y"),
        ("Nature Communications: alcance editorial", "https://www.nature.com/ncomms/ncomms/aims"),
        ("Communications Biology: alcance y criterios", "https://www.nature.com/commsbio/aims"),
    ]
    source_links = [Paragraph(
        f'<link href="{url}" color="#216F79">{escape(title)}</link>', styles["SmallAMP"])
        for title, url in sources]
    source_table = Table([[source_links[i], source_links[i+5]] for i in range(5)],
                         colWidths=[85*mm, 85*mm], hAlign="LEFT")
    source_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(source_table)
    append_research_record(scope, p, heading, table, page)
    destination = ROOT/"output/pdf"/f"Ejecucion_reconstruccion_AMP_{scope}_2026-10-20.pdf"
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(TEAL)
        canvas.drawString(20*mm, 13*mm, f"Reconstrucción AMP | {data_scope}")
        canvas.drawRightString(190*mm, 13*mm, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(destination), pagesize=(210*mm,297*mm),
                            leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=23*mm)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(destination)
    receipt = {"path": str(destination.relative_to(ROOT)),
               "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
               "scope": scope, "visual_review": "pending"}
    (out/"execution_report_build.json").write_text(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
