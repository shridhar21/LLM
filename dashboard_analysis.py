"""Question-level dashboard selection and averaging, independent of chart rendering."""
import csv
import re
import statistics
from collections import defaultdict

from compare_reports import metric
from generate_report import answers_file


METRICS = {
    'emissions_g': 'Emissions (g CO2eq)',
    'energy_wh': 'Total energy (Wh)',
    'latency_s': 'Latency (s)',
    'input_tokens': 'Input tokens',
    'output_tokens': 'Output tokens',
    'reconstructed_input_tokens': 'Reconstructed input tokens',
    'reconstructed_output_tokens': 'Reconstructed saved-answer tokens',
    'cpu_energy_wh': 'CPU energy (Wh)',
    'gpu_energy_wh': 'GPU energy (Wh)',
    'ram_energy_wh': 'RAM energy (Wh)',
    'ane_energy_wh': 'ANE energy (Wh)',
    'ollama_load_duration_s': 'Ollama model load time (s)',
    'ollama_eval_duration_s': 'Ollama generation time (s)',
    'ollama_prompt_eval_duration_s': 'Ollama prompt processing time (s)',
    'retrieval_emissions_g': 'Retrieval emissions (g CO2eq)',
    'generation_emissions_g': 'Generation emissions (g CO2eq)',
    'retrieval_energy_wh': 'Retrieval energy (Wh)',
    'generation_energy_wh': 'Generation energy (Wh)',
    'retrieval_latency_s': 'Retrieval latency (s)',
    'generation_latency_s': 'Generation latency (s)',
}


def question_records(bank):
    return [dict(serial=int(index)+1, text=row['Question'], topic=row['topic_sheet'],
                 question_type=row['question_type'], chain=row['conversation_chain'],
                 turn=int(row['conversation_turn']) if row['conversation_chain'] else None)
            for index, row in bank.iterrows()]


def add_coverage(machines, catalogue, questions):
    by_text = {q['text']: q['serial'] for q in questions}
    for machine in machines:
        for entry in machine['runs']:
            coverage = defaultdict(list)
            for text, record in catalogue[entry['id']]['run']['questions'].items():
                if text in by_text:
                    coverage[record['model']].append(by_text[text])
            entry['coverage'] = dict(coverage)


def choose_questions(spec, available, questions):
    mode = spec.get('mode', 'all')
    candidates = [q for q in questions if q['serial'] in available]
    if mode == 'serials':
        wanted = set()
        text = str(spec.get('serials', '')).replace('–', '-').replace('—', '-')
        for token in text.split(','):
            match = re.fullmatch(r'\s*(\d+)\s*(?:-\s*(\d+)\s*)?', token)
            if not match:
                raise ValueError('Enter question serials such as 4, 13-17, 74')
            lo, hi = int(match[1]), int(match[2] or match[1])
            if lo < 1 or hi < lo or hi > len(questions):
                raise ValueError('Question serial range is outside the questionbank')
            wanted.update(range(lo, hi+1))
        if not wanted <= available:
            raise ValueError('Some requested question serials are unavailable on this machine')
        candidates = [q for q in candidates if q['serial'] in wanted]
    elif mode == 'sheets':
        candidates = [q for q in candidates if q['topic'] in spec.get('sheets', [])]
        subset = spec.get('subset', 'all')
        if subset == 'followups':
            candidates = [q for q in candidates if q['chain']]
        elif subset == 'types':
            candidates = [q for q in candidates if q['question_type'] in spec.get('types', [])]
        elif subset != 'all':
            raise ValueError('Unknown sheet question selection')
    elif mode == 'types':
        candidates = [q for q in candidates if q['question_type'] in spec.get('types', [])
                      or ('__followups__' in spec.get('types', []) and q['chain'])]
    elif mode != 'all':
        raise ValueError('Unknown question selection method')
    if not candidates:
        raise ValueError('Select at least one available question')
    return candidates


def complete_mean(values):
    return sum(values)/len(values) if values and all(v is not None for v in values) else None


def average_rows(rows):
    """Equal scheme weights; repeated runs are averaged inside their own scheme first."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['machine_index'], row['model'], row['pipeline'], row['question'])].append(row)
    result = []
    for group in groups.values():
        base = group[0]
        schemes = sorted({r['scheme'] for r in group})
        row = {key: base[key] for key in ('machine_index', 'machine', 'model', 'pipeline',
               'original_question', 'question', 'topic', 'question_type', 'chain', 'turn')}
        row.update(series_id='mean:'+base['model'], series=base['model']+' | mean across schemes',
                   measurement_basis=base.get('measurement_basis', 'codecarbon_machine'),
                   scheme=', '.join(schemes), run='Mean across selected schemes', position=None,
                   gap=None, scheme_count=len(schemes), run_count=len(group),
                   status='ok' if all(r['status']=='ok' for r in group) else 'Includes failed/unknown',
                   contributors='; '.join(r['run'] for r in group))
        for key in METRICS:
            row[key] = complete_mean([complete_mean([r[key] for r in group if r['scheme']==scheme])
                                      for scheme in schemes])
            row[key+'_valid_runs'] = sum(r[key] is not None for r in group)
        result.append(row)
    return result


def analyse_selection(payload, catalogue, machines, bank):
    if catalogue is None:
        raise ValueError('Load machine folders first')
    questions = question_records(bank)
    if len({q['text'] for q in questions}) != len(questions):
        raise ValueError('Questionbank contains ambiguous duplicate question text')
    mode = payload.get('averaging', 'none')
    if mode not in ('none', 'schemes'):
        raise ValueError('Choose no averaging or averaging across shuffling schemes')
    selections = payload.get('selections', [])
    if not selections or len({s['machine'] for s in selections}) != len(selections):
        raise ValueError('Select distinct machines to analyse')
    pipeline = payload.get('pipeline')
    if pipeline not in ('Normal LLM', 'RAG'):
        raise ValueError('Choose Normal LLM or RAG')
    raw, metadata, chosen, warnings = [], [], [], []
    by_chain_turn = {(q['chain'], q['turn']): q for q in questions if q['chain']}
    for spec in selections:
        machine = next((m for m in machines if m['index']==spec['machine']), None)
        if machine is None:
            raise ValueError('Unknown machine')
        if spec.get('averaging', mode) != mode:
            raise ValueError('All compared machines must use the same averaging mode')
        entries = [i for i in catalogue.values() if i['entry']['machine']==spec['machine']
                   and i['entry']['pipeline']==pipeline]
        available = {q['serial'] for q in questions if any(q['text'] in i['run']['questions'] for i in entries)}
        qs = choose_questions(spec.get('questions', {}), available, questions)
        models, schemes, ids = spec.get('models', []), spec.get('schemes', []), spec.get('runs', [])
        if not models or not schemes or not ids or len(ids) != len(set(ids)):
            raise ValueError(f'{machine["name"]}: choose models, schemes and distinct runs')
        if mode == 'schemes' and ('unknown' in schemes or 'Not recorded' in models):
            raise ValueError('Averaging requires recorded model and shuffling scheme identities')
        eligible = {i['entry']['id']: i for i in entries}
        if any(i not in eligible for i in ids):
            raise ValueError('A selected run belongs to another machine or pipeline')
        found = set()
        for id_ in ids:
            item = eligible[id_]; entry, run = item['entry'], item['run']
            if entry['scheme'] not in schemes:
                raise ValueError('A selected run does not match the selected schemes')
            source = answers_file(run['folder'])
            if source is None or (source.stat().st_mtime_ns, source.stat().st_size) != item['source_stamp']:
                raise ValueError('A source report changed. Reload the machine folders.')
            with source.open(encoding='utf-8-sig', newline='') as handle:
                records = {r['question']: r for r in csv.DictReader(handle)}
            run_models = set(r['model'] for r in run['questions'].values()) & set(models)
            if not run_models:
                raise ValueError('A selected run does not match the selected models')
            for model in sorted(run_models):
                if any(q['text'] not in run['questions'] or run['questions'][q['text']]['model'] != model for q in qs):
                    raise ValueError(f'{entry["name"]}: incomplete selected-question coverage for {model}')
                found.add((model, entry['scheme']))
                for q in qs:
                    rec = run['questions'][q['text']]; stored = records[q['text']]
                    previous = by_chain_turn.get((q['chain'], (q['turn'] or 0)-1))
                    previous_record = run['questions'].get(previous['text']) if previous else None
                    gap = rec['position']-previous_record['position']-1 if previous_record else None
                    row = dict(machine_index=machine['index'], machine=machine['name'], model=model,
                               pipeline=pipeline, original_question=q['serial'], question=q['text'],
                               topic=q['topic'], question_type=q['question_type'], chain=q['chain'], turn=q['turn'],
                               run=entry['name'], run_id=id_, scheme=entry['scheme'], seed=entry['seed'],
                               position=rec['position'], gap=gap, status=rec['status'],
                               series_id=id_+':'+model, series=f'{model} | {entry["scheme"]} | {entry["name"]}',
                               emissions_g=rec['emissions_g'], energy_wh=rec['energy_wh'], latency_s=rec['latency_s'])
                    row.update(input_tokens=metric(stored, 'input_tokens'), output_tokens=metric(stored, 'output_tokens'))
                    for key in ('reconstructed_input_tokens','reconstructed_output_tokens'):
                        row[key]=metric(stored,key)
                    row['token_count_method']=stored.get('token_count_method','Ollama-reported if available')
                    row['measurement_basis'] = run['metadata'].get('Measurement basis', 'codecarbon_machine')
                    for target in ('load', 'eval', 'prompt_eval'):
                        row['ollama_' + target + '_duration_s'] = metric(stored, 'ollama_' + target + '_duration_ns', 1e-9)
                    for part in ('cpu', 'gpu', 'ram', 'ane'):
                        row[part+'_energy_wh'] = metric(stored, part+'_energy_kwh', 1000)
                    for phase in ('retrieval','generation'):
                        for target, field, scale in (('emissions_g','emissions_kg',1000),('energy_wh','energy_kwh',1000),('latency_s','latency_s',1)):
                            row[phase+'_'+target] = metric(stored, phase+'_'+field, scale)
                    raw.append(row)
            metadata.append(dict(machine=machine['name'], **run['metadata']))
        expected = {(m,s) for m in models for s in schemes}
        if found != expected:
            missing = ', '.join(f'{m} / {s}' for m,s in sorted(expected-found))
            raise ValueError(f'{machine["name"]}: no complete selected run for {missing}')
        chosen.append(dict(machine=machine['index'], name=machine['name'], questions=[q['serial'] for q in qs],
                           models=sorted(models), schemes=sorted(schemes), runs=ids))
    if payload.get('compare', False) and len(chosen)>1:
        first = chosen[0]
        for other in chosen[1:]:
            for key, description in [('questions','questions'), ('models','models'), ('schemes','shuffling schemes')]:
                if other[key] != first[key]:
                    raise ValueError(f'Cross-machine comparison requires the same selected {description}. Adjust selections or turn off comparison.')
    if len({r['measurement_basis'] for r in raw}) > 1:
        raise ValueError('Select one measurement method: CodeCarbon machine energy and idle-subtracted Apple SoC energy cannot be combined.')
    means = average_rows(raw)
    displayed = means if mode == 'schemes' else raw
    if any(r['status']!='ok' for r in raw):
        warnings.append('Recorded failed/unknown attempts are included and labelled; averages include their recorded costs.')
    if any(r['emissions_g'] is None for r in raw):
        warnings.append('Missing measurements stay unavailable, including any mean that depends on them.')
    if mode=='schemes' and any(c['schemes'] and len(c['schemes'])==1 for c in chosen):
        warnings.append('Only one scheme is selected for a machine; its mean does not measure variation across schemes.')
    if len({m.get('country_iso_code') for m in metadata})>1:
        warnings.append('Recorded carbon locations differ; compare energy alongside emissions.')
    stats = []
    grouped = defaultdict(list)
    for row in displayed:
        grouped[(row['machine_index'],row['series_id'])].append(row)
    for group in grouped.values():
        for key in METRICS:
            vals = [r[key] for r in group if r[key] is not None]
            if not vals: continue
            stats.append(dict(machine=group[0]['machine'], model=group[0]['model'], series=group[0]['series'],
                              metric=METRICS[key], questions=len(group), valid=len(vals),
                              total=sum(vals) if len(vals)==len(group) else None,
                              mean=statistics.mean(vals), sample_variance=statistics.variance(vals) if len(vals)>1 else None,
                              sample_stddev=statistics.stdev(vals) if len(vals)>1 else None,
                              minimum=min(vals), maximum=max(vals)))
    return dict(averaging=mode, compare=bool(payload.get('compare')), pipeline=pipeline,
                machines=chosen, rows=displayed, raw=raw, averages=means,
                followups=[r for r in displayed if r['chain']], metadata=metadata, statistics=stats,
                metrics=METRICS, warnings=warnings)
