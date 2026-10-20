"""Read-only experiment analysis; generate CSV statistics, Python figures and a LaTeX section.

python analyse_llama_machines.py emissions_reports/SV emissions_reports/SN emissions_reports/SC
Dependencies: requirements-report.txt (no inference, server or internet required).
"""
import argparse
import csv
import hashlib
import io
import json
import math
import platform
import statistics as st
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from question_order import BANK, load_bank

ROOT = Path(__file__).resolve().parent
SCHEMES = {
    'original': ('Original', '#0072B2'),
    'global': ('Mix all', '#D55E00'),
    'question_types_questions': ('Shuffle groups + questions', '#009E73'),
    'topics': ('Shuffle topics only', '#CC79A7'),
    'within_topics': ('Mix within topics', '#8A7700'),
}
POLICIES = {
    'original': 'Preserve topic, question-type and question order.',
    'global': 'Mix across all topics and question types; preserve each follow-up chain\'s turn order.',
    'question_types_questions': 'Keep topic order; shuffle question-type groups and their questions, except follow-up turns.',
    'topics': 'Shuffle whole topics; preserve the internal order of every topic.',
    'within_topics': 'Keep topic order; mix questions across question types within each topic; preserve follow-up turn order.',
}
SOURCES = {}


def machine_label(machine):
    """Publication labels; source identities remain available in audit exports."""
    return {'SV': 'Windows1', 'SN': 'Windows2', 'SC': 'MacBook'}.get(machine, machine)


def source_bytes(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if str(path) in SOURCES and SOURCES[str(path)] != digest:
        raise ValueError(f'Source changed during analysis: {path}')
    SOURCES[str(path)] = digest
    return data


def read_json(path):
    return json.loads(source_bytes(path).decode('utf-8-sig'))


def read_rows(path):
    reader = csv.DictReader(io.StringIO(source_bytes(path).decode('utf-8-sig'), newline=''))
    fields = reader.fieldnames or []
    rows = list(reader)
    if len(fields) != len(set(fields)) or any(None in r or None in r.values() for r in rows):
        raise ValueError(f'CSV column alignment error: {path}')
    return rows


def number(value):
    if value is None or str(value).strip().lower() in ('', 'nan', 'none', 'null'):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'Invalid non-negative measurement: {value!r}')
    return result


def tokens(value):
    result = number(value)
    if result is not None and not result.is_integer():
        raise ValueError('Token counts must be whole numbers')
    return int(result) if result is not None else None


def describe(values):
    v = [x for x in values if x is not None]
    mean = st.mean(v) if v else None
    sd = st.stdev(v) if len(v) > 1 else None
    return dict(n=len(v), mean=mean, variance=st.variance(v) if len(v) > 1 else None,
                sd=sd, minimum=min(v) if v else None, maximum=max(v) if v else None,
                cv=sd / mean if sd is not None and mean else None)


def ranks(values):
    """Average ranks for ties, rather than the no-ties Spearman shortcut."""
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in order[i:j]:
            result[k] = (i + 1 + j) / 2
        i = j
    return result


def relationship(pairs):
    """Descriptive within-question/machine fits; no p values or causal interpretation."""
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    x, y = ([v[i] for v in pairs] for i in (0, 1))
    ratios = describe([1000 * b / a for a, b in pairs if a > 0])
    result = dict(n_pairs=len(pairs), distinct_token_counts=len(set(x)),
                  distinct_emission_values=len(set(y)), pearson_r=None, spearman_rho=None,
                  slope_g_per_token=None, intercept_g=None, r_squared=None,
                  zero_intercept_slope_g_per_token=None, zero_intercept_rmse_g=None,
                  ratio_n=ratios['n'], mean_g_per_1000_tokens=ratios['mean'],
                  ratio_cv=ratios['cv'], status='fewer_than_3_pairs')
    if len(pairs) < 3:
        return result
    if len(set(x)) < 2 or len(set(y)) < 2:
        result['status'] = 'constant_tokens_or_emissions'
        return result
    pearson = st.correlation(x, y)
    slope = st.covariance(x, y) / st.variance(x)
    intercept = st.mean(y) - slope * st.mean(x)
    origin_slope = sum(a * b for a, b in pairs) / sum(a * a for a in x)
    result.update(pearson_r=pearson, spearman_rho=st.correlation(ranks(x), ranks(y)),
                  slope_g_per_token=slope, intercept_g=intercept, r_squared=pearson ** 2,
                  zero_intercept_slope_g_per_token=origin_slope,
                  zero_intercept_rmse_g=math.sqrt(st.mean([(b - origin_slope * a) ** 2 for a, b in pairs])),
                  status='descriptive_fit')
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def raw_evidence(folder, answer_rows, warnings):
    """Only aligned raw rows supply metadata. Energy/emission prefix survives old shifts."""
    by_id, aligned = {}, []
    for path in sorted(folder.glob('*.csv')):
        if not path.name.startswith('emissions'):
            continue
        reader = csv.reader(io.StringIO(source_bytes(path).decode('utf-8-sig'), newline=''))
        header = next(reader)
        required = ['project_name', 'duration', 'emissions', 'cpu_energy', 'gpu_energy', 'ram_energy', 'energy_consumed']
        # Old files only lose fields AFTER energy_consumed, not inside this prefix.
        if header.index('energy_consumed') != 13 or header.index('emissions') != 5:
            raise ValueError(f'Unrecognized raw prefix: {path}')
        short = 0
        for row in reader:
            if len(row) < 14:
                raise ValueError(f'Unusable raw row: {path}')
            prefix = {k: row[header.index(k)] for k in required}
            if prefix['project_name'] in by_id:
                raise ValueError(f'Duplicate raw project name: {path}')
            by_id[prefix['project_name']] = prefix
            if len(row) == len(header):
                aligned.append(dict(zip(header, row)))
            else:
                short += 1
        if short:
            warnings.append(f'{folder.name}: {short} raw rows have shortened metadata; hardware comes from measurement_metadata.json and aligned rows only.')
    for row in answer_rows:
        qid = row['question_id']
        candidates = [r for name, r in by_id.items() if name.startswith('LLM_only_q' + qid + '_')]
        if len(candidates) != 1:
            raise ValueError(f'Cannot match raw measurement uniquely: {folder}/{qid}')
        raw = candidates[0]
        for a, b in [('emissions_kg', 'emissions'), ('energy_kwh', 'energy_consumed'),
                     ('cpu_energy_kwh', 'cpu_energy'), ('gpu_energy_kwh', 'gpu_energy'), ('ram_energy_kwh', 'ram_energy')]:
            x, y = number(row.get(a)), number(raw.get(b))
            if x is not None and y is not None and not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-14):
                raise ValueError(f'Raw/answer measurement mismatch: {folder}/{qid}/{a}')
    return aligned


def collect(folders, model, bank):
    expected = {r['Question']: (int(i) + 1, r) for i, r in bank.iterrows()}
    records, runs, inventory, warnings = [], [], [], []
    hardware = defaultdict(lambda: defaultdict(set))
    contexts, token_assets = [], []
    for parent in folders:
        machine = parent.name
        for path in sorted(parent.rglob('answers.csv')):
            rows = read_rows(path)
            models = sorted({r.get('model_name', '') for r in rows})
            order_path = path.parent / 'question_order.json'
            order = read_json(order_path) if order_path.exists() else {}
            inventory.append(dict(machine=machine, run=path.parent.name, models='; '.join(models),
                                  question_count=len(rows), scheme=order.get('mode', 'unrecorded')))
            if model not in models:
                continue
            if models != [model] or any('retrieval_latency_s' in r for r in rows):
                raise ValueError(f'Comparison requires one model and Normal LLM mode: {path}')
            mapped = {r['question']: r for r in rows}
            if len(mapped) != len(rows) or set(mapped) != set(expected):
                raise ValueError(f'Question texts do not match the questionbank exactly: {path}')
            if sorted(int(r['execution_position']) for r in rows) != list(range(1, len(rows) + 1)):
                raise ValueError(f'Invalid execution positions: {path}')
            seen = set()
            for row in sorted(rows, key=lambda r: int(r['execution_position'])):
                b = expected[row['question']][1]
                if not set(b['conversation_prerequisites']) <= seen:
                    raise ValueError(f'Follow-up order violated: {path}')
                seen.add(str(b['Question ID']))
            scheme = order.get('mode')
            if scheme not in SCHEMES:
                raise ValueError(f'Unsupported scheme {scheme}: {path}')
            if any(r.get('ordering_mode') != scheme for r in rows):
                raise ValueError(f'Scheme mismatch between answers and order manifest: {path}')
            run_records = []
            for row in rows:
                q, b = expected[row['question']]
                e = number(row.get('emissions_kg'))
                rec = dict(machine=machine, run=path.parent.name, scheme=scheme, seed=order.get('seed'),
                           original_question=q, question=row['question'], topic=b['topic_sheet'],
                           question_type=b['question_type'], execution_position=int(row['execution_position']),
                           status=row['status'], emissions_g=e * 1000 if e is not None else None,
                           energy_kwh=number(row.get('energy_kwh')), latency_s=number(row.get('latency_s')),
                           actual_output_tokens=tokens(row.get('output_tokens')),
                           reconstructed_output_tokens=tokens(row.get('reconstructed_output_tokens')))
                records.append(rec)
                run_records.append(rec)
            aligned = raw_evidence(path.parent, rows, warnings)
            sessions = read_json(path.parent / 'measurement_metadata.json')['sessions']
            for session in sessions:
                for k, v in session.get('hardware', {}).items():
                    hardware[machine][k].add(str(v))
                for k in ('codecarbon_version', 'os', 'effective_scope', 'measure_power_secs', 'pue', 'country_iso_code'):
                    hardware[machine][k].add(str(session.get(k, 'unrecorded')))
                for item in session.get('hardware_evidence', []):
                    hardware[machine][item['class'] + '_backend'].add(item.get('mode', 'unknown'))
            for row in aligned:
                for k in ('cpu_count', 'python_version'):
                    if row.get(k):
                        hardware[machine][k].add(row[k])
            context_path = path.parent / 'conversation_context.json'
            if context_path.exists():
                context = read_json(context_path)
                contexts.append(dict(machine=machine, run=path.parent.name,
                                     effective_options=context.get('effective_options'), settings=context.get('settings')))
            token_path = path.parent / 'token_reconstruction.json'
            if any(r['reconstructed_output_tokens'] is not None for r in run_records):
                if not token_path.exists():
                    raise ValueError(f'Reconstructed counts lack provenance: {path}')
                provenance = read_json(token_path)
                token_assets.append(dict(machine=machine, run=path.parent.name,
                                         models=provenance['models'], limitations=provenance['limitations']))
            valid = [r for r in run_records if r['status'] == 'ok' and r['emissions_g'] is not None]
            runs.append(dict(machine=machine, run=path.parent.name, scheme=scheme, seed=order.get('seed'),
                             n=len(rows), n_valid=len(valid),
                             total_emissions_g=sum(r['emissions_g'] for r in valid) if len(valid) == len(rows) else None,
                             observed_emissions_g=sum(r['emissions_g'] for r in valid),
                             folder=str(path.parent.resolve())))
    for machine in [p.name for p in folders]:
        counts = Counter(r['scheme'] for r in runs if r['machine'] == machine)
        if counts != Counter({s: 1 for s in SCHEMES}):
            raise ValueError(f'{machine}: expected exactly one run for each of the five schemes, got {dict(counts)}. Select folders without duplicate repeats.')
    metadata = {m: {k: '; '.join(sorted(v)) for k, v in h.items()} for m, h in hardware.items()}
    return records, runs, inventory, metadata, contexts, token_assets, warnings


def statistics_tables(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['machine'], row['original_question']].append(row)
    stats, relations = [], []
    for (machine, q), rows in sorted(groups.items()):
        valid = [r for r in rows if r['status'] == 'ok']
        stat = dict(machine=machine, original_question=q, topic=rows[0]['topic'], question_type=rows[0]['question_type'])
        stat.update({'emissions_' + k: v for k, v in describe([r['emissions_g'] for r in valid]).items()})
        stat['n_selected_runs'] = len(rows)
        stats.append(stat)
        for basis, key in [('actual', 'actual_output_tokens'), ('reconstructed', 'reconstructed_output_tokens')]:
            rel = dict(machine=machine, original_question=q, topic=rows[0]['topic'], token_basis=basis)
            rel.update({'tokens_' + k: v for k, v in describe([r[key] for r in valid]).items()})
            rel.update(relationship([(r[key], r['emissions_g']) for r in valid]))
            relations.append(rel)
    cross = []
    for q in sorted({r['original_question'] for r in records}):
        rows = [r for r in stats if r['original_question'] == q]
        # Only a complete five-scheme mean enters the three-machine summary.
        means = [r['emissions_mean'] if r['emissions_n'] == 5 else None for r in rows]
        result = dict(original_question=q, topic=rows[0]['topic'])
        result.update({'machine_means_' + k: v for k, v in describe(means).items()})
        for row in rows:
            result[row['machine'] + '_mean_g'] = row['emissions_mean'] if row['emissions_n'] == 5 else None
        cross.append(result)
    return stats, relations, cross


def unified_graph(records, machines, questions, output, xmax, tick_step):
    selected = [r for r in records if r['original_question'] in questions]
    fig, ax = plt.subplots(figsize=(11.3, 7.25 if len(questions) <= 12 else max(7.25, len(questions) * .40 + 1.3)))
    ticks, labels = [], []
    for qi, q in enumerate(questions):
        for mi, machine in enumerate(machines):
            y = qi * (len(machines) + 1) + mi
            rows = [r for r in selected if r['original_question'] == q and r['machine'] == machine]
            values = [r['emissions_g'] for r in rows if r['status'] == 'ok' and r['emissions_g'] is not None]
            ticks.append(y)
            labels.append((f'Q{q}  {rows[0]["topic"]} | ' if mi == 0 else '') + machine_label(machine))
            if values:
                ax.hlines(y, min(values), max(values), color='#a0a9b1', lw=1.1, zorder=2)
            for si, scheme in enumerate(SCHEMES):
                row = next(r for r in rows if r['scheme'] == scheme)
                if row['status'] == 'ok' and row['emissions_g'] is not None:
                    ax.scatter(row['emissions_g'], y + (si - 2) * .065, s=13,
                               color=SCHEMES[scheme][1], edgecolors='none', zorder=3)
        if qi < len(questions) - 1:
            ax.axhline(qi * (len(machines) + 1) + len(machines), color='#e0e5e9', lw=.5)
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.set_ylim(ticks[-1] + .75, -.75)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.tick_params(axis='x', labelsize=6.5, rotation=45)
    ax.grid(axis='x', color='#e4e8ed', lw=.5)
    ax.set_xlabel('Per-question emissions (g CO2eq)', fontsize=9)
    ax.set_title(f'Llama 3.2 3B | Q{questions[0]}-Q{questions[-1]} | three machines', loc='left', fontsize=12, fontweight='bold', pad=35)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, markersize=4, label=name) for name, c in SCHEMES.values()]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.005), ncol=3, frameon=False, fontsize=8)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.subplots_adjust(left=.29, right=.985, top=.85, bottom=.13)
    fig.savefig(output)
    plt.close(fig)


def scatter_graph(records, machines, q, basis, output):
    key = 'actual_output_tokens' if basis == 'actual' else 'reconstructed_output_tokens'
    subset = [r for r in records if r['original_question'] == q and r['status'] == 'ok' and r[key] is not None and r['emissions_g'] is not None]
    if not subset:
        return False
    fig, axes = plt.subplots(len(machines), 1, figsize=(8.6, 8.2), sharex=True, squeeze=False)
    for machine, ax in zip(machines, axes[:, 0]):
        rows = [r for r in subset if r['machine'] == machine]
        for row in rows:
            ax.scatter(row[key], row['emissions_g'], color=SCHEMES[row['scheme']][1], s=48, zorder=3)
        fit = relationship([(r[key], r['emissions_g']) for r in rows])
        if fit['slope_g_per_token'] is not None:
            x = [min(r[key] for r in rows), max(r[key] for r in rows)]
            ax.plot(x, [fit['intercept_g'] + fit['slope_g_per_token'] * v for v in x], color='#515e69', lw=1)
        ax.set_title(machine_label(machine) + f' | {fit["n_pairs"]} runs', fontsize=10, loc='left', fontweight='bold')
        ax.grid(color='#e5e9ee', lw=.5)
        ax.set_ylabel('Emissions (g CO2eq)', fontsize=9)
        ax.tick_params(labelsize=9)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[-1][0].set_xlabel('Answer length (output tokens)' if basis == 'actual' else 'Answer length (tokens recounted from saved text)', fontsize=10)
    fig.suptitle(f'One example: answer length and emissions for Q{q}', fontsize=13, fontweight='bold', y=.98)
    fig.text(.5, .945, 'Each dot is one run. The grey line shows the fitted trend. Emissions scales differ by machine.', ha='center', fontsize=9)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=name) for name, c in SCHEMES.values()]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(top=.88, bottom=.17, left=.13, right=.97, hspace=.42)
    fig.savefig(output)
    plt.close(fig)
    return True


def token_overview_graph(relations, records, machines, output):
    """Count positive within-question fits without treating pooled questions as one fit."""
    counts, totals = [], []
    for machine in machines:
        rows = [r for r in relations if r['machine'] == machine and r['token_basis'] == 'reconstructed']
        counts.append(sum(r['pearson_r'] is not None and r['pearson_r'] > 0 for r in rows))
        totals.append(len({r['original_question'] for r in records if r['machine'] == machine}))
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    y = list(range(len(machines)))
    ax.barh(y, totals, color='#edf1f4', height=.48)
    ax.barh(y, counts, color='#168577', height=.48)
    for i, (count, total) in enumerate(zip(counts, totals)):
        ax.text(count + 1.5, i, f'{count}/{total}', va='center', fontsize=11, fontweight='bold')
    ax.set_yticks(y, [machine_label(m) for m in machines], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.12)
    ax.set_xticks([0, 40, 80, 120, 160] if max(totals) == 160 else range(0, max(totals) + 1, max(1, max(totals) // 4)))
    ax.set_xlabel('Number of questions with a positive answer-length / emissions relationship', fontsize=9)
    ax.set_title('For most questions, longer answers came with higher emissions', loc='left', fontsize=12, fontweight='bold', pad=16)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e6ebf0')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.subplots_adjust(left=.16, right=.97, bottom=.24, top=.80)
    fig.savefig(output)
    plt.close(fig)


def tex(value):
    chars = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
             '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(chars.get(c, c) for c in str(value))


def fmt(value, digits=3):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def update_report_section(content):
    """Keep the generated section inline, preserving the rest of the user's report."""
    path = ROOT / 'ShivanshShridhar_MidSemester_Updated_Report.tex'
    original = path.read_text(encoding='utf-8')
    begin, end = '% BEGIN LLAMA MACHINE SECTION', '% END LLAMA MACHINE SECTION'
    block = begin + '\n' + content.rstrip() + '\n' + end
    include = r'\input{report_assets/llama_comparison/section.tex}'
    if original.count(begin) == original.count(end) == 1:
        start, stop = original.index(begin), original.index(end) + len(end)
        if stop <= start:
            raise ValueError('Invalid generated-section boundaries in main report')
        updated = original[:start] + block + original[stop:]
    elif begin not in original and end not in original and original.count(include) == 1:
        updated = original.replace(include, block, 1)
    else:
        raise ValueError('Cannot locate one unambiguous Llama section in the main report')
    path.write_text(updated, encoding='utf-8')


def table(headers, rows, widths=None):
    spec = widths or ('Y' + 'r' * (len(headers) - 1))
    lines = [r'\begin{table}[H]\centering\small', r'\begin{tabularx}{\linewidth}{@{}' + spec + '@{}}', r'\toprule',
             ' & '.join(r'\textbf{' + tex(h) + '}' for h in headers) + r'\\\midrule']
    lines += [' & '.join(tex(v) for v in row) + r'\\' for row in rows]
    return '\n'.join(lines + [r'\bottomrule\end{tabularx}\end{table}'])


def write_section(out, machines, model, inventory, meta, contexts, assets, model_info, runs, records, stats, relations, figures, scatters, warnings):
    relative = out.relative_to(ROOT).as_posix()
    lines = [r'% Generated by analyse_llama_machines.py; regenerate rather than hand-edit this file.',
             r'\section{Updated Normal LLM Experiments: Llama Shuffling Across Three Machines}',
             r'\label{sec:llama-machines}', r'\subsection{Updated experiment scope and model selection}',
             'This section extends the earlier ten-question architectural comparison. The earlier Llama and Granite results remain a separate historical study; they are not pooled with these newer experiments. The current code lists installed local Ollama models and records the exact user-selected tag for each run. The supplied machine collections contain the following Normal LLM runs:',
             table(['Machine', 'Model tag', 'Runs'], [[machine_label(m), tag, sum(r['machine'] == m and tag in r['models'].split('; ') for r in inventory)] for m in machines for tag in sorted({r['models'] for r in inventory if r['machine'] == m})], 'lYr'),
             f'The analysis below uses only {tex(model)}: five schemes on each of three machines, 160 identical questions per run, and {len(records):,} recorded question executions. Questions are matched by exact text, then labelled using their original questionbank serials. Model tags alone do not prove that historical model binaries were identical.',
             r'\subsection{Llama model metadata and inference settings}',
             table(['Property', 'Recorded or locally inspected value'], [
                 ['Recorded run tag', model], ['Current local model', model_info.get('name', 'unrecorded')],
                 ['Current parameter-size label', model_info.get('model_type', 'unrecorded')],
                 ['Current quantization', model_info.get('file_type', 'unrecorded')],
                 ['Model family / format', model_info.get('model_family', 'unrecorded') + ' / ' + model_info.get('model_format', 'unrecorded')],
                 ['Reconstruction vocabulary size', ', '.join(sorted({str(m['vocabulary_size']) for a in assets for m in a['models']}))],
                 ['Follow-up context / output limit', 'num_ctx=32768; num_predict=2048'],
                 ['Follow-up history policy', 'Successful earlier turns within the same chain; no summarization'],
                 ['Independent-question limits', 'Ollama defaults; not recorded as explicit per-request options'],
                 ['Temperature and model sampling seed', 'Not recorded; shuffle seeds are not inference seeds'],
                 ['Warm-up and keep-alive', 'Batch warm-up before measurement; keep_alive=15m']
             ], 'lY'),
             'The parameter and quantization labels describe the current local Ollama config inspected during analysis. Historical runs did not record per-machine model digests or Ollama versions, so historical binary identity and default sampling settings cannot be verified. Follow-up settings are read from saved conversation files and checked for consistency across the selected runs. The native tokenizer used for reconstruction records a vocabulary of 128,256 tokens. Model and template hashes are retained in the machine-readable provenance files, not treated as proof of the historical installations.',
             r'\subsection{Machine specifications and measurement configuration}']
    props = [('CPU', 'cpu_model'), ('Logical CPU count recorded', 'cpu_count'), ('RAM reported (GiB)', 'ram_total_size'),
             ('Operating system', 'os'), ('Python', 'python_version'), ('CodeCarbon', 'codecarbon_version'),
             ('CPU backend', 'CPU_backend'), ('GPU devices tracked', 'gpu_count'), ('Tracking scope', 'effective_scope'),
             ('Sampling interval (seconds)', 'measure_power_secs'), ('Country factor', 'country_iso_code'), ('PUE', 'pue')]
    hardware_rows = []
    for title, key in props:
        values = [meta[m].get(key, 'unrecorded') for m in machines]
        if key == 'ram_total_size':
            values = [fmt(float(v)) if ';' not in v and v != 'unrecorded' else v for v in values]
        hardware_rows.append([title, *values])
    lines += [table(['Property', *[machine_label(m) for m in machines]], hardware_rows, 'YYYY'),
              'CPU counts above are CodeCarbon logical-processor counts, not an inferred number of physical cores. A tracked GPU count of zero does not establish the absence of an integrated GPU or accelerator use by Ollama. MacBook reports an Apple M4 and a CPU-load estimation backend; the two Intel machines report Windows EMI. These backend differences, software versions, load and machine-level background activity limit direct hardware-efficiency claims. RAM energy is an estimate. Device brand/model, thermal state, power mode, disk specifications and historical Ollama versions were not recorded.',
              r'\subsection{Question set, shuffling schemes and follow-up context}',
              'The questionbank contains eight topic sheets with 20 questions each. Original serials 1--160 remain fixed identifiers even when execution positions change. Five selected schemes are listed below; each has one run per machine, rather than repeated independent trials per scheme.',
              table(['Scheme', 'What changes and what is preserved'], [[SCHEMES[s][0] + ' (' + s + ')', POLICIES[s]] for s in SCHEMES], r'p{0.28\linewidth}Y'),
              'Numbered follow-up turns preserve their original within-chain order in every scheme. Global and within-topic mixing may interleave unrelated questions between turns. Each chain maintains separate successful user/assistant history; unrelated questions receive no such history. The current follow-up request includes the earlier questions and answers from its own chain. Therefore, the same final question text does not imply the same full input prompt: earlier generated answers may differ between runs.',
              r'\subsubsection{How random shuffling works}',
              r'Before a shuffled run, the program creates a random starting number called a \textbf{seed}. It uses the operating system to create a 32-bit seed, then gives it to Python\textquotesingle s \texttt{random.Random(seed)}. This generator rearranges only the topic sheets, question-type groups or questions allowed by the chosen scheme. Every selected question still runs once.',
              'For free mixing, the program randomly picks the next available question. In a follow-up chain, only its next turn is available; later turns must wait. This allows unrelated questions between turns while preserving the chain order.',
              r'The same logic is used in Normal LLM and RAG modes. The seed and final execution order are saved in \texttt{question\_order.json}. Reusing the seed with the same questions, scheme and implementation reproduces the order. The seed controls question order only; it does not fix the model\textquotesingle s generated answers.',
              r'\subsection{Measurement scope}',
              'The analysis reads the recorded per-question emissions in kilograms and converts them to grams by multiplying by 1,000. It does not re-estimate emissions from tokens. The measured Normal LLM request includes conversation/prompt handling, the Ollama request and output collection. Batch warm-up, final report writing and work outside that measured interval are excluded. Machine scope includes activity from other processes.',
              'Unavailable or failed observations are omitted from token fits and graphs; missing observations are not assigned zero. No different models or machines are pooled for within-question token regressions.']
    lines += [r'\subsection{Answer length and emissions}',
              r"\textbf{Main finding: longer answers usually came with higher recorded emissions.} This was common on all three machines, but it did not happen for every question. It also does not mean that doubling an answer's length doubles its emissions.",
              'For each question, the analysis compares its five shuffling runs on the same machine. It repeats this for all 160 questions and all three machines. Different questions and machines are kept separate. Answer length is measured in tokens, the small pieces of text processed by the model.',
              'The historical runs did not save Ollama\'s original output-token counts. This analysis therefore recounts tokens from each saved answer. These reconstructed counts may miss removed text, hidden reasoning or end-of-answer tokens; they are labelled separately from actual Ollama counters.',
              r'\begin{figure}[H]\centering',
              rf'\includegraphics[width=\linewidth]{{{relative}/token_overview.pdf}}',
              r'\caption{Overall result across all 160 questions. Each green bar counts questions with a positive Pearson correlation between answer length and emissions across their five runs. The grey remainder contains other questions, including unavailable fits. A positive correlation describes a tendency, not an increase in every individual run.}',
              r'\end{figure}',
              r'\textbf{How to read the scores below.} Pearson measures how closely answer length and emissions follow a straight-line trend. Spearman checks whether longer answers generally have higher emissions, even when the trend is not a straight line. Both scores range from $-1$ to $+1$: near $+1$ means a strong upward relationship, near zero means little consistent relationship, and near $-1$ means a downward relationship.',
              r'$R^2$ describes how much of the observed emissions variation the fitted straight line captures: closer to 1 means a closer fit. Per-token variation checks how much emissions per 1,000 output tokens change across runs; a smaller percentage means a more stable cost per token. Each table value is the median (middle value) of the separate question-level results on that machine.']
    token_summary = []
    for m in machines:
        rels = [r for r in relations if r['machine'] == m and r['token_basis'] == 'reconstructed']
        eligible = [r for r in rels if r['status'] == 'descriptive_fit']
        med = lambda key: st.median([r[key] for r in eligible]) if eligible else None
        token_summary.append([machine_label(m), fmt(med('pearson_r')), fmt(med('spearman_rho')), fmt(med('r_squared')), fmt(med('ratio_cv') * 100, 1) + '%' if eligible else 'N/A'])
    positive = []
    for m in machines:
        rels = [r for r in relations if r['machine'] == m and r['token_basis'] == 'reconstructed' and r['pearson_r'] is not None]
        positive.append(machine_label(m) + ': ' + str(sum(r['pearson_r'] > 0 for r in rels)) + '/' + str(len(rels)))
    lines += [table(['Machine', 'Pearson', 'Spearman', 'R2', 'Per-token variation'], token_summary),
              'The upward relationship appears in ' + tex('; '.join(positive)) + ' question-level comparisons. Windows2 has the strongest typical relationship; MacBook has more variation in emissions per token. Answer length is therefore related to emissions, but does not fully explain them.',
              r"\textbf{What was calculated.} Each trend line has the form $e=\alpha+\beta t$, where $t$ is answer length, $e$ is emissions, $\alpha$ is the intercept and $\beta$ is the change in emissions per token. The exports retain every question's correlation scores, slope, intercept, $R^2$, and emissions per 1,000 tokens. They also include a line forced through zero and its typical prediction error (RMSE), to help examine proportionality. Per-token variation is the sample standard deviation of the emissions/token ratio divided by its mean (CV). Spearman uses average ranks when values tie.",
              'All 160 questions on each machine have different saved-answer lengths across the selected runs. Correlations need at least three usable runs and variation in both values; otherwise they are unavailable. Zero-token answers are excluded from per-token ratios. The full results are in token_relationships.csv, with actual and reconstructed counts kept separate.',
              r'\textbf{What this cannot prove.} There are only five runs per question, one for each scheme. These results do not prove that shuffling caused the difference. Longer input history, different earlier follow-up answers, generation randomness, other processes and measurement differences can also affect emissions. No significance tests or confidence intervals are claimed.']
    for q, basis, name in scatters:
        lines += [r'\begin{figure}[H]\centering', rf'\includegraphics[width=\linewidth]{{{relative}/{name}}}',
                  rf'\caption{{One example, Q{q}: dots show answer length and emissions for each shuffling run, using {tex(basis)} token counts. Colours identify schemes; the grey line shows the fitted trend. Each machine has its own emissions scale so its points remain visible. Compare the trend within each panel, not the vertical positions across panels.}}', r'\end{figure}',
                  f'Q{q} illustrates how the comparison works; it is not the overall result or a question chosen for a strong relationship. The bar chart above and exported token tables cover all 160 questions.']
    lines += [r'\subsection{Unified three-machine emissions graphs and interpretation}',
              'Questions stay in their original order. Each question has three rows: Windows1, Windows2 and MacBook. Each dot is one run; its colour identifies the shuffling scheme. The grey line joins the lowest and highest emissions on that row. It shows the observed range, not a confidence interval.',
              r'Further right means higher emissions. A longer grey line means more difference between that question\textquotesingle s runs; it does not explain the cause. All pages share the same scale, with ticks every ' + str(ARGS.tick_step) + r' g CO$_2$eq. Python generates the graphs from the recorded values. Questions are split into readable page ranges, with all three machines together.']
    for start, end, name in figures:
        lines += [r'\clearpage\begin{landscape}', r'\begin{figure}[H]\centering',
                  rf'\includegraphics[width=\linewidth,height=0.90\textheight,keepaspectratio]{{{relative}/{name}}}',
                  rf'\caption{{Llama emissions for original questions {start}--{end}: all three machines in one graph, one machine row per question, with shared scheme colours and emission scale.}}',
                  r'\end{figure}\end{landscape}']
    content = '\n\n'.join(lines) + '\n'
    for name in ('question_statistics.csv', 'cross_machine_questions.csv', 'token_relationships.csv', 'measurement_metadata.json', 'eval_count'):
        content = content.replace(name, r'\texttt{' + tex(name) + '}')
    update_report_section(content)


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='*', type=Path, default=[ROOT / 'emissions_reports' / m for m in ('SV', 'SN', 'SC')])
    parser.add_argument('--model', default='llama3.2:3b')
    parser.add_argument('--bank', type=Path, default=BANK)
    parser.add_argument('--output', type=Path, default=ROOT / 'report_assets' / 'llama_comparison')
    parser.add_argument('--tick-step', type=float, default=.025)
    parser.add_argument('--questions-per-figure', type=int, default=12)
    parser.add_argument('--scatter-questions', default='1', help='Comma-separated original serials; default Q1 avoids choosing by fit strength')
    parser.add_argument('--local-models', type=Path, default=Path.home() / '.ollama/models')
    ARGS = parser.parse_args()
    if len(ARGS.folders) != 3 or len({p.resolve() for p in ARGS.folders}) != 3 or len({p.name for p in ARGS.folders}) != 3:
        parser.error('Select three distinct machine folders with distinct names')
    if not math.isfinite(ARGS.tick_step) or ARGS.tick_step <= 0 or not 1 <= ARGS.questions_per_figure <= 16:
        parser.error('Positive finite tick step and 1--16 questions per figure required')
    if not ARGS.output.resolve().is_relative_to(ROOT):
        parser.error('Output must be inside this project for portable LaTeX references')
    if any(ARGS.output.resolve().is_relative_to(p.resolve()) for p in ARGS.folders):
        parser.error('Output must not be inside a source machine folder')
    start, stamp = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    for source in (ARGS.bank, Path(__file__), ROOT / 'question_order.py', ROOT / 'conversation.py'):
        source_bytes(source)
    bank = load_bank(ARGS.bank)
    records, runs, inventory, meta, contexts, assets, warnings = collect(ARGS.folders, ARGS.model, bank)
    settings = {json.dumps(c['effective_options'], sort_keys=True) for c in contexts}
    if settings != {json.dumps({'num_ctx': 32768, 'num_predict': 2048}, sort_keys=True)}:
        raise ValueError('Report settings text requires review: follow-up options differ or are unavailable')
    model_info = {}
    manifest_path = ARGS.local_models / 'manifests/registry.ollama.ai/library' / ARGS.model.replace(':', '/')
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        config = read_json(ARGS.local_models / 'blobs' / manifest['config']['digest'].replace(':', '-'))
        model_info.update({k: config.get(k, 'unrecorded') for k in ('model_type', 'file_type', 'model_family', 'model_format')})
        model_info['name'] = 'Llama 3.2 3B Instruct' if ARGS.model == 'llama3.2:3b' else ARGS.model
    stats, relations, cross = statistics_tables(records)
    out = ARGS.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    machines = [p.name for p in ARGS.folders]
    token_overview_graph(relations, records, machines, out / 'token_overview.pdf')
    for name, rows in [('observations', records), ('run_totals', runs), ('model_inventory', inventory),
                       ('question_statistics', stats), ('token_relationships', relations), ('cross_machine_questions', cross)]:
        write_csv(out / (name + '.csv'), rows)
    (out / 'metadata.json').write_text(json.dumps(dict(machines=meta, model=model_info, contexts=contexts,
                                                     reconstruction_assets=assets, warnings=warnings), indent=2), encoding='utf-8')
    questions = list(range(1, len(bank) + 1))
    scatter_questions = [int(q) for q in ARGS.scatter_questions.split(',')]
    if not set(scatter_questions) <= set(questions):
        raise ValueError('Scatter question outside questionbank')
    maximum = max(r['emissions_g'] for r in records if r['emissions_g'] is not None)
    xmax = math.ceil(maximum / ARGS.tick_step + 1) * ARGS.tick_step
    if xmax / ARGS.tick_step > 120:
        raise ValueError('Requested tick spacing would produce more than 120 labels; choose a larger --tick-step')
    figures, scatters = [], []
    for i in range(0, len(questions), ARGS.questions_per_figure):
        part = questions[i:i + ARGS.questions_per_figure]
        name = f'unified_q{part[0]:03d}_{part[-1]:03d}.pdf'
        unified_graph(records, machines, part, out / name, xmax, ARGS.tick_step)
        figures.append((part[0], part[-1], name))
    unified_graph(records, machines, questions, out / 'unified_all_questions.pdf', xmax, ARGS.tick_step)
    for q in scatter_questions:
        for basis in ('actual', 'reconstructed'):
            name = f'tokens_q{q:03d}_{basis}.pdf'
            if scatter_graph(records, machines, q, basis, out / name):
                scatters.append((q, basis, name))
    write_section(out, machines, ARGS.model, inventory, meta, contexts, assets, model_info, runs, records,
                  stats, relations, figures, scatters, warnings)
    for path, digest in SOURCES.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Source changed during generation: {path}')
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    outputs = [dict(path=p.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in sorted(out.iterdir()) if p.is_file() and p.name != 'computation_manifest.json']
    manifest = dict(schema_version=1, claim_id='llama-three-machine-descriptive-comparison',
                    repository=dict(commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain'))),
                    command=subprocess.list2cmdline([sys.executable, *sys.argv]),
                    environment=dict(software=[f'Python {platform.python_version()}', f'Matplotlib {matplotlib.__version__}'], hardware=platform.platform()),
                    mathematics=dict(assertion_tested='Descriptive question/machine emissions variation and token associations on the supplied matched runs',
                                     coefficient_domain='IEEE-754 double precision; source decimal measurements converted to float',
                                     conventions='sample variance ddof=1; average ranks for ties; within-question/machine OLS; actual/reconstructed token bases separate',
                                     inputs=[dict(path=p, sha256=h) for p, h in SOURCES.items()],
                                     bounds=dict(machines=3, runs=len(runs), questions=len(bank), observations=len(records)),
                                     non_claims=['No causal shuffle effect', 'No universal proportionality', 'No verified historical model-binary identity', 'No replicate-based uncertainty']),
                    randomness=dict(used=False, generator='none; deterministic descriptive analysis', seed=None),
                    run=dict(started_at=stamp, runtime_seconds=time.perf_counter() - start, exit_status=0),
                    outputs=outputs, checks=['Exact question sets', 'Follow-up prerequisite order', 'Raw emissions/energy prefix agrees',
                                            'One run per matched scheme/machine', 'Source hashes unchanged'],
                    result='Execution completed; numerical benchmark checks are documented in the project tests.',
                    residual_risks=warnings + ['Five non-replicated scheme observations per fit', 'Historical saved-answer reconstruction', 'Different hardware measurement backends'])
    (out / 'computation_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Generated {len(records)} observations, {len(stats)} question statistics and {len(relations)} token relationships.')
    print(f'{len(figures)} combined graph pages; {len(scatters)} scatter figures; {len(warnings)} raw metadata warnings.')
    print('Output:', out)


if __name__ == '__main__':
    main()
