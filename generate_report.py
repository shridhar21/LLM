"""Select a completed batch folder and create its report.pdf locally."""
import csv
import json
import math
import statistics
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent / "emissions_reports"
ANSWER_FILES = ("answers.csv", "exp1_llm_only_answers_per_query.csv",
                "exp2_rag_answers_per_query.csv", "live_answers.csv")


def answers_file(folder):
    return next((folder / name for name in ANSWER_FILES if (folder / name).is_file()), None)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return [0] + [None] * 8
    pos = (len(values) - 1) * .95
    low = int(pos)
    p95 = values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)
    return [len(values), statistics.mean(values), statistics.median(values),
            statistics.variance(values) if len(values) > 1 else None,
            statistics.stdev(values) if len(values) > 1 else None,
            min(values), max(values), p95, sum(values)]


def fmt(value):
    return "N/A" if value is None else f"{value:.5g}"


def generate(folder):
    # Lazy imports let the selector explain missing optional dependencies.
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.graphics.shapes import Drawing, Line, Circle, String, Rect

    source = answers_file(folder)
    if source is None:
        raise ValueError("No per-question CSV found.")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError("The answers CSV has no question rows.")
    rag = "total_latency_s" in fields
    definitions = [("Latency (s)", "total_latency_s" if rag else "latency_s", 1),
                   ("Energy (Wh)", "energy_kwh", 1000),
                   ("Emissions (g CO2e)", "total_emissions_kg" if rag else "emissions_kg", 1000)]
    if rag:
        definitions += [(f"{phase.title()} {label}", f"{phase}_{key}", scale)
                        for phase in ("retrieval", "generation")
                        for label, key, scale in (("latency (s)", "latency_s", 1),
                                                  ("energy (Wh)", "energy_kwh", 1000),
                                                  ("emissions (g CO2e)", "emissions_kg", 1000))]
    definitions += [(f"{part.upper()} energy (Wh)", f"{part}_energy_kwh", 1000)
                    for part in ("cpu", "gpu", "ram")]
    def series(key, scale):
        return [None if (v := number(row.get(key))) is None else v * scale for row in rows]
    data = [(label, series(key, scale)) for label, key, scale in definitions]
    success = [row.get("status") == "ok" for row in rows]
    styles = getSampleStyleSheet()
    story = []
    def paragraph(text, style="BodyText"):
        story.append(Paragraph(escape(str(text)), styles[style]))
        story.append(Spacer(1, 4))
    def table(content, widths=None):
        obj = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
        obj.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14394b")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                                ("TOPPADDING", (0, 0), (-1, -1), 3),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white])]))
        story.append(obj)
        story.append(Spacer(1, 6))

    paragraph("Batch emissions report", "Title")
    paragraph(folder.name, "Heading2")
    paragraph("Models: " + ", ".join(sorted({r.get("model_name", "Unknown") for r in rows})))
    order_path = folder / 'question_order.json'
    try:
        order = json.loads(order_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        order = {}
    paragraph('Question ordering: ' + str(order.get('description', rows[0].get('ordering_mode', 'Not recorded'))))
    paragraph('Shuffle seed: ' + str(order.get('seed', rows[0].get('shuffle_seed', 'Not recorded'))))
    paragraph(f"Pipeline: {'RAG' if rag else 'LLM only'} | Questions: {len(rows)} | Successful: {sum(success)} | Failed/unknown: {len(rows) - sum(success)}")
    summary = next((folder / name for name in ("summary.csv", "exp1_llm_only_summary_per_query.csv", "exp2_rag_summary_per_query.csv") if (folder / name).exists()), None)
    runtime = None
    if summary:
        with summary.open(encoding="utf-8-sig", newline="") as handle:
            runtime = number(next(csv.DictReader(handle), {}).get("total_runtime_s"))
    paragraph(f"Batch runtime: {fmt(runtime)} s | Successful requests/min: {fmt(sum(success) * 60 / runtime if runtime and runtime > 0 else None)}")
    def cost(vals):
        valid = [v for v in vals if v is not None]
        return (fmt(sum(valid)) + (" (partial)" if len(valid) < len(vals) else "")) if valid else "N/A"
    table([["Measured cost (all requests)", "Observed sum", "Valid / all"]] +
          [[label, cost(vals),
            f"{sum(v is not None for v in vals)} / {len(rows)}"] for label, vals in data[1:3]], [290, 95, 110])
    paragraph("Method and data quality", "Heading2")
    paragraph("Statistics and distributions use successful requests with finite measurements. Cost totals include measured failed requests. Missing/non-finite values are excluded, never replaced with zero. Partial totals are not estimates of unmeasured cost.")
    paragraph("Variance and standard deviation use the sample definition (n - 1); both are N/A for fewer than two observations. Variance has squared units. P95 uses linear interpolation. Small values use significant digits or scientific notation.")
    paragraph("Historical pipelines may already have stored missing telemetry as zero. This report cannot distinguish those defaults from measured zero. RAM fields describe estimated energy, not memory usage. Per-question charts use CSV row order; red points mark failed/unknown requests.")
    table([["Metric", "Missing", "Zero"]] + [[label, str(sum(v is None for v in vals)), str(sum(v == 0 for v in vals))] for label, vals in data], [330, 80, 85])
    story.append(PageBreak())
    paragraph("Measurement scope and reliability", "Title")
    metadata_path = folder / "measurement_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metadata = {}
    sessions = metadata.get("sessions", [])
    if not sessions:
        paragraph("UNVERIFIED: This batch has no runtime measurement metadata. Its historical hardware coverage and backend methods cannot be established. Earlier zero defaults cannot be recovered.")
    else:
        paragraph(metadata.get("scope_note", "Measurement scope not documented."))
        paragraph(f"Tracking sessions recorded: {len(sessions)}. Expected for this CSV: {len(rows) * (2 if rag else 1)}. Metadata describes configuration and backend evidence, not independent meter validation.")
        for key in ("codecarbon_version", "os", "effective_scope", "measure_power_secs", "pue", "country_iso_code"):
            paragraph(key + ": " + ", ".join(sorted({str(s.get(key, 'unknown')) for s in sessions})))
        for component in ("cpu", "gpu", "ram"):
            paragraph(component.upper() + " backend: " + "; ".join(sorted({str(s.get('backends', {}).get(component, 'unknown')) for s in sessions})))
        paragraph("Hardware: " + "; ".join(sorted({str(s.get('hardware', {})) for s in sessions})))
        paragraph("Backend objects/modes: " + "; ".join(sorted({str(s.get('hardware_evidence', [])) for s in sessions})))
        paragraph("Sensor-backed versus estimated: backend names containing constant, TDP or CPU load indicate estimation; RAPL, NVML and power metrics indicate hardware telemetry. Unknown modes remain unverified. Emissions are calculated estimates in all cases.")
        failures = [s for s in sessions if s.get('stop_error')]
        paragraph(f"Tracker stop failures: {len(failures)}")
        for error in sorted({str(s['stop_error']) for s in failures}):
            paragraph(error)
        if any(s.get('effective_scope') != 'machine' for s in sessions):
            paragraph("WARNING: Machine scope was not confirmed for every tracking session.")
    paragraph("A partial observed sum excludes unavailable readings. A complete batch cost cannot be claimed when coverage is incomplete. Zero readings are retained; unknown hardware coverage must not be interpreted as zero consumption.")
    story.append(PageBreak())
    paragraph("Statistical summary", "Title")
    for label, vals in data:
        s = stats([v for v, ok in zip(vals, success) if ok])
        paragraph(label, "Heading3")
        table([["Valid n", "Mean", "Median", "Variance", "Std dev", "Min", "Max", "P95"],
               [str(s[0])] + [fmt(v) for v in s[1:8]]], [45] + [64] * 7)
    story.append(PageBreak())
    from report_charts import per_question, histogram, breakdown
    paragraph("Which questions cost the most?", "Title")
    paragraph("Bars show individual requests; large batches use points to avoid overcrowding. Missing measurements are marked X, not shown as zero.")
    for label, vals in data[:3]:
        story.append(per_question(label, vals, success))
    story.append(PageBreak())
    paragraph("How much do requests vary?", "Title")
    paragraph("Successful requests only. Bin boundaries show measurement values; bar heights and labels show request counts.")
    for label, vals in data[:3]:
        story.append(histogram(label, [v for v, ok in zip(vals, success) if ok and v is not None]))
    story.append(PageBreak())
    paragraph("Where is energy spent?", "Title")
    paragraph("Compare observed component costs below. Amber bars indicate incomplete coverage; comparisons with different coverage may be misleading. These are tracking estimates, not isolated model or wall-socket measurements.")
    story.append(breakdown("Hardware energy (Wh)", [(label.split()[0], vals) for label, vals in data[-3:]]))
    if rag:
        paragraph("Retrieval versus generation", "Heading2")
        for offset, title in enumerate(("Time (s)", "Energy (Wh)", "Emissions (g CO2e)")):
            story.append(breakdown(title, [("Retrieval", data[3+offset][1]), ("Generation", data[6+offset][1])]))
    if 'topic_sheet' in fields and any(row.get('topic_sheet') for row in rows):
        story.append(PageBreak())
        paragraph('Question execution mapping', 'Title')
        paragraph('Chart question rows correspond to this execution sequence. Original serial numbers are local to each topic sheet.')
        mapping = [['Position', 'Topic sheet', 'Serial', 'Question type']]
        for i, row in enumerate(rows, 1):
            mapping.append([str(row.get('execution_position') or i),
                            Paragraph(escape(row.get('topic_sheet', '')), styles['BodyText']),
                            str(row.get('question_number', '')),
                            Paragraph(escape(row.get('question_type', '')), styles['BodyText'])])
        table(mapping, [50, 150, 50, 245])
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(42, 22, "Local benchmark report | " + source.name)
        canvas.drawRightString(553, 22, f"Page {doc.page}")
    output = folder / "report.pdf"
    temporary = folder / "report.pdf.tmp"
    try:
        SimpleDocTemplate(str(temporary), pagesize=(595, 842), rightMargin=50, leftMargin=50,
                          topMargin=40, bottomMargin=42).build(story, onFirstPage=footer, onLaterPages=footer)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    folders = {p.name: p for p in sorted(ROOT.iterdir())
               if p.is_dir() and p.name.startswith(("exp1_llm_only_", "exp2_rag_")) and answers_file(p)} if ROOT.exists() else {}
    if not folders:
        print("No batch reports found in", ROOT)
        return
    print("Available batch reports:\n" + "\n".join(folders))
    while True:
        name = input("\nType the exact folder name (or q to quit): ").strip()
        if name.lower() == "q":
            return
        if name not in folders:
            print("Choose a folder from the list above.")
            continue
        folder = folders[name]
        if (folder / "report.pdf").exists() and input("Replace existing report.pdf? [y/N]: ").strip().lower() != "y":
            continue
        try:
            print("Saved:", generate(folder))
        except ImportError:
            print("PDF support requires ReportLab. Install with: python -m pip install -r requirements-report.txt")
        except (OSError, ValueError) as exc:
            print("Could not generate PDF:", exc)
        return


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
