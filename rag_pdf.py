"""RAG-only PDF sections, using saved decisions rather than inferring fallback."""
import json
import statistics
from xml.sax.saxutils import escape

from rag_reporting import read_details, numeric, TIMING_KEYS


def matching_details(rows, details):
    recorded = {q['execution_position']: q for q in (details or {}).get('queries', [])}
    matched = []
    for i, row in enumerate(rows, 1):
        position = int(row.get('execution_position') or i)
        q = recorded.get(position)
        matched.append(q if q and q.get('question') == row.get('question') else None)
    return matched


def stacked_chart(title, labels, series, colors, unit, modes=None):
    from reportlab.graphics.shapes import Drawing, Rect, Line, String
    from reportlab.lib import colors as palette
    height = 95 + 27 * len(labels)
    d = Drawing(495, height)
    d.add(String(0, height - 16, title, fontSize=11))
    legend_x = 0
    for name, color in zip(series, colors):
        d.add(Rect(legend_x, height - 36, 9, 9, fillColor=color, strokeColor=None))
        d.add(String(legend_x + 13, height - 35, name, fontSize=8))
        legend_x += 98
    totals = [sum(values[i] for values in series.values()) if all(values[i] is not None for values in series.values())
              else None for i in range(len(labels))]
    maximum = max((v for v in totals if v is not None), default=0) or 1
    for i, label in enumerate(labels):
        y = height - 66 - i * 27
        d.add(String(0, y + 4, str(label), fontSize=8))
        if modes:
            d.add(String(32, y + 4, modes[i], fontSize=7, fillColor=palette.HexColor('#64748b')))
        if totals[i] is None:
            d.add(String(110, y + 4, 'N/A - incomplete readings', fontSize=8))
            continue
        x = 110
        for values, color in zip(series.values(), colors):
            width = 285 * values[i] / maximum
            d.add(Rect(x, y, width, 17, fillColor=color, strokeColor=None))
            x += width
        d.add(String(x + 5, y + 4, f'{totals[i]:.4g}', fontSize=8))
    d.add(Line(110, 30, 395, 30, strokeColor=palette.HexColor('#94a3b8')))
    for fraction in (0, .5, 1):
        x = 110 + 285 * fraction
        d.add(String(x - 3, 16, f'{maximum * fraction:.3g}', fontSize=7))
    d.add(String(220, 2, unit, fontSize=8))
    return d


def append_sections(story, folder, rows, styles):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import ParagraphStyle
    small = ParagraphStyle('RagTable', parent=styles['BodyText'], fontSize=8, leading=10)
    def p(text, style='BodyText'):
        story.append(Paragraph(escape(str(text)), styles[style]))
        story.append(Spacer(1, 5))
    def table(content, widths):
        header_style = ParagraphStyle('RagHeader', parent=small, textColor=colors.white)
        wrapped = [[Paragraph(escape(str(v)).replace('\n', '<br/>'), header_style if i == 0 else small)
                    for v in row] for i, row in enumerate(content)]
        obj = Table(wrapped, colWidths=widths, repeatRows=1, hAlign='LEFT', splitByRow=1, splitInRow=1)
        obj.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#14394b')),
                                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                ('TOPPADDING', (0, 0), (-1, -1), 5),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
        story.append(obj)
        story.append(Spacer(1, 8))
    def value(v):
        return 'N/A' if v is None else f'{v:.5g}'
    try:
        details = read_details(folder)
    except (OSError, ValueError) as exc:
        details = None
        detail_error = str(exc)
    else:
        detail_error = None
    matched = matching_details(rows, details)
    retrievals = [(q.get('retrieval') or {}) if q else {} for q in matched]
    modes = [r.get('response_mode', 'unrecorded') for r in retrievals]

    story.append(PageBreak())
    p('RAG evidence and fallback overview', 'Title')
    counts = {mode: modes.count(mode) for mode in ('rag', 'base_model_fallback', 'unrecorded')}
    table([['Evidence-backed', 'Base-model fallback', 'Not recorded'],
           [counts['rag'], counts['base_model_fallback'], counts['unrecorded']]], [165] * 3)
    p('Counts cover attempted question rows, including failed or cancelled attempts. Evidence acceptance is a retrieval heuristic, not an answer-quality score. Unknown decisions are never inferred from historical chunks.')
    reasons = {}
    for r in retrievals:
        if r.get('fallback_used'):
            reason = r.get('reason', 'Not recorded')
            reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        table([['Fallback reason', 'Questions']] + [[reason.replace('_', ' '), count] for reason, count in sorted(reasons.items())], [395, 100])
    if not details or detail_error:
        p('Detailed RAG decisions and timings were not recorded for this run.' + (f' Details: {detail_error}' if detail_error else ''))

    for start in range(0, len(rows), 12):
        batch = rows[start:start + 12]
        labels = [str(row.get('execution_position') or start + i + 1) for i, row in enumerate(batch)]
        legend_modes = [{'rag': 'RAG', 'base_model_fallback': 'Fallback'}.get(m, 'Unknown') for m in modes[start:start + 12]]
        legend_modes = [mode + (f" / {row['status']}" if row.get('status') != 'ok' else '')
                        for mode, row in zip(legend_modes, batch)]
        story.append(PageBreak())
        p('Per-question retrieval and generation emissions', 'Title')
        p('Horizontal stacks show retrieval plus generation for each execution position. Failed and cancelled attempts may include costs; missing phase readings are marked N/A. Corpus costs are excluded.')
        series = {phase.title(): [None if numeric(row.get(f'{phase}_emissions_kg')) is None else
                                  numeric(row[f'{phase}_emissions_kg']) * 1000 for row in batch]
                  for phase in ('retrieval', 'generation')}
        story.append(stacked_chart('Emissions by execution position', labels, series,
                                  [colors.HexColor('#168577'), colors.HexColor('#346ab0')], 'g CO2e', legend_modes))
        if any(r.get('timings') for r in retrievals[start:start + 12]):
            story.append(PageBreak())
            p('Retrieval-step timings', 'Title')
            names = ('Embedding', 'FAISS', 'BM25', 'Fusion/MMR', 'Gate/parents')
            series = {name: [numeric(r.get('timings', {}).get(key)) for r in retrievals[start:start + 12]]
                      for name, key in zip(names, TIMING_KEYS)}
            story.append(stacked_chart('Retrieval suboperation time', labels, series,
                                      [colors.HexColor(c) for c in ('#168577', '#346ab0', '#d18b28', '#8759a8', '#64748b')],
                                      'seconds'))
            p('These steps share one retrieval emissions measurement. Timer sums need not equal tracker duration because setup, context formatting and other overhead are excluded.')

    story.append(PageBreak())
    p('Evidence selection by question', 'Title')
    p('Positions follow execution order. Child candidates are expanded into unique parent sources. Source details refer to supplied evidence, not verified citations in the generated answer.')
    evidence_rows = [['Position', 'Mode / status', 'Children / parents', 'Best cosine', 'Gate reason / source files']]
    for i, (row, r) in enumerate(zip(rows, retrievals), 1):
        files = ', '.join(dict.fromkeys(s['filename'] for s in r.get('sources', []))) or 'No sources recorded'
        evidence_rows.append([row.get('execution_position') or i,
                              f"{ {'rag': 'Evidence-backed', 'base_model_fallback': 'Fallback'}.get(r.get('response_mode'), 'Not recorded')} / {row.get('status', 'unknown')}",
                              f"{r.get('selected_child_count', 'N/A')} / {r.get('parent_count', 'N/A')}",
                              value(r.get('best_dense_similarity')),
                              r.get('reason', 'Not recorded').replace('_', ' ') + '\n' + files])
    table(evidence_rows, [45, 110, 65, 60, 215])

    story.append(PageBreak())
    p('Evidence-backed versus fallback costs', 'Title')
    p('Successful requests with recorded decisions only. Each metric uses available finite readings and shows valid n. Different question difficulty can explain cost differences; this is not a causal efficiency ranking.')
    summary = [['Response mode', 'Successful n', 'Mean emissions (g) / n', 'Mean energy (Wh) / n', 'Mean latency (s) / n']]
    for mode in ('rag', 'base_model_fallback'):
        selected_rows = [row for row, actual in zip(rows, modes) if actual == mode and row.get('status') == 'ok']
        entry = ['Evidence-backed' if mode == 'rag' else 'Fallback', len(selected_rows)]
        for key, scale in [('total_emissions_kg', 1000), ('energy_kwh', 1000), ('total_latency_s', 1)]:
            values = [numeric(row.get(key)) for row in selected_rows]
            valid = [v * scale for v in values if v is not None]
            entry.append(f"{value(statistics.mean(valid) if valid else None)} / {len(valid)}")
        summary.append(entry)
    table(summary, [105, 65, 110, 110, 105])

    p('Recorded retrieval configuration', 'Heading2')
    settings = (details or {}).get('settings', {})
    config_rows = [['Setting', 'Value'], ['Index generation', (details or {}).get('index_generation') or 'Not recorded']]
    keys = ['embedding_model', 'index_type', 'normalize_embeddings', 'dense_candidates', 'sparse_candidates',
            'rrf_constant', 'selected_children', 'mmr_lambda', 'evidence_similarity_threshold',
            'context_characters', 'child_target_tokens', 'parent_target_tokens', 'sentence_overlap']
    config_rows.extend([[key.replace('_', ' ').title(), settings.get(key, 'Not recorded')] for key in keys])
    table(config_rows, [245, 250])
    p('Settings are the run snapshot, not the current index configuration. The context budget covers evidence text only. Corpus preparation costs remain in the indexing report and are not added to these question totals.')
