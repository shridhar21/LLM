"""Compare completed batches by exact question text; generate a local PDF and CSV. check04"""
import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from generate_report import ANSWER_FILES, ROOT, answers_file, fmt, number


def read_json(path):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value
    except (ValueError, OSError) as exc:
        raise ValueError(f'{path}: invalid metadata: {exc}') from exc


def metric(row, key, scale=1):
    value = number(row.get(key))
    return value * scale if value is not None and value >= 0 else None


def load_run(folder):
    folder = Path(folder).resolve()
    source = answers_file(folder)
    if source is None:
        raise ValueError(f'{folder.name}: no per-question report found')
    with source.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows or 'question' not in fields:
        raise ValueError(f'{folder.name}: missing question rows/text column')
    rag = 'total_emissions_kg' in fields and 'retrieval_latency_s' in fields
    normal = 'emissions_kg' in fields and 'latency_s' in fields
    if rag == normal:
        raise ValueError(f'{folder.name}: cannot unambiguously identify normal/RAG pipeline')
    order = read_json(folder / 'question_order.json')
    measurement = read_json(folder / 'measurement_metadata.json')
    indexed = {}
    for position, row in enumerate(rows, 1):
        question = row['question']
        if question is None or not question.strip():
            raise ValueError(f'{folder.name}: blank question at row {position}')
        if question in indexed:
            raise ValueError(f'{folder.name}: duplicate exact question text at row {position}')
        stored_position = row.get('execution_position')
        if stored_position and number(stored_position) != position:
            raise ValueError(f'{folder.name}: execution positions do not match CSV sequence')
        indexed[question] = {
            'position': position, 'status': row.get('status', 'unknown'),
            'model': row.get('model_name') or 'Not recorded',
            'emissions_g': metric(row, 'total_emissions_kg' if rag else 'emissions_kg', 1000),
            'energy_wh': metric(row, 'energy_kwh', 1000),
            'latency_s': metric(row, 'total_latency_s' if rag else 'latency_s'),
        }
    planned = order.get('questions')
    if isinstance(planned, list) and len(planned) != len(rows):
        raise ValueError(f'{folder.name}: {len(rows)} result rows versus {len(planned)} planned questions; batch may be incomplete')
    summary = {}
    for name in ('summary.csv', 'exp1_llm_only_summary_per_query.csv', 'exp2_rag_summary_per_query.csv'):
        path = folder / name
        if path.exists():
            with path.open(encoding='utf-8-sig', newline='') as handle:
                summary = next(csv.DictReader(handle), {})
            break
    declared = number(summary.get('num_queries'))
    if declared is not None and declared != len(rows):
        raise ValueError(f'{folder.name}: summary question count disagrees with answer rows')
    def distinct(key):
        return ', '.join(sorted({str(row[key]) for row in rows if row.get(key)})) or 'Not recorded'
    metadata = {
        'Folder': folder.name, 'Model(s)': distinct('model_name'), 'Pipeline': 'RAG' if rag else 'Normal LLM',
        'Ordering': str(order.get('description') or order.get('mode') or distinct('ordering_mode')),
        'Seed': str(order.get('seed')) if order.get('seed') is not None else distinct('shuffle_seed'),
        'Questions': str(len(rows)), 'Runtime (s)': summary.get('total_runtime_s') or 'Not recorded',
        'Completion evidence': 'Summary present' if summary else 'No summary; completion unverified',
    }
    sessions = measurement.get('sessions', [])
    for key in ('codecarbon_version', 'os', 'effective_scope', 'measure_power_secs', 'pue', 'country_iso_code', 'hardware', 'backends', 'hardware_evidence'):
        metadata[key] = '; '.join(sorted({json.dumps(s[key], sort_keys=True) for s in sessions if key in s})) or 'Not recorded'
    return {'folder': folder, 'questions': indexed, 'metadata': metadata, 'summary': summary}


def validate(runs):
    if len(runs) < 2:
        raise ValueError('Select at least two distinct batch folders')
    if len({str(r['folder']) for r in runs}) != len(runs):
        raise ValueError('The same folder was selected more than once')
    if len({r['metadata']['Pipeline'] for r in runs}) != 1:
        raise ValueError('Normal LLM and RAG runs cannot be mixed')
    expected = set(runs[0]['questions'])
    for run in runs[1:]:
        actual = set(run['questions'])
        if actual != expected:
            raise ValueError(f"{run['folder'].name}: exact question texts differ ({len(expected-actual)} missing, {len(actual-expected)} extra/changed). Case, punctuation and whitespace must match.")


def delta(value, baseline):
    return value - baseline if value is not None and baseline is not None else None


def percentage(value, baseline):
    difference = delta(value, baseline)
    return difference / baseline * 100 if difference is not None and baseline != 0 else None


def total(run, key):
    values = [v[key] for v in run['questions'].values()]
    valid = [v for v in values if v is not None]
    return (sum(valid) if valid else None), len(valid), len(values)


def aligned_rows(runs, reference):
    records = []
    for qid, (question, baseline) in enumerate(runs[reference]['questions'].items(), 1):
        emissions = [run['questions'][question]['emissions_g'] for run in runs]
        available_emissions = [value for value in emissions if value is not None]
        mean_emissions = statistics.mean(available_emissions) if available_emissions else None
        for i, run in enumerate(runs):
            current = run['questions'][question]
            records.append({'question_key': f'Q{qid}', 'question': question, 'run': f'R{i+1}',
                            'folder': run['folder'].name, 'reference_run': f'R{reference+1}', **current,
                            'mean_emissions_g': mean_emissions,
                            'mean_emissions_valid_runs': len(available_emissions),
                            'mean_emissions_total_runs': len(runs),
                            'emissions_delta_g': delta(current['emissions_g'], baseline['emissions_g']),
                            'emissions_change_percent': percentage(current['emissions_g'], baseline['emissions_g'])})
    return records


def observations(runs, reference):
    notes = ['These are observed differences, not a causal estimate of the effect of shuffling. Output length, background activity and measurement variation may contribute.']
    models = {r['metadata']['Model(s)'] for r in runs}
    notes.append('Models differ or are unknown: model and ordering effects cannot be separated.' if len(models) > 1 or 'Not recorded' in models else 'Recorded model names match; this does not establish identical model versions or generation settings.')
    notes.append('Generation settings, token counts and RAG index identity are not consistently recorded by these reports; equality of those conditions cannot be verified.')
    complete = all(total(r, 'emissions_g')[1] == len(r['questions']) and all(v['status'] == 'ok' for v in r['questions'].values()) for r in runs)
    if complete:
        values = [total(r, 'emissions_g')[0] for r in runs]
        low, high = min(values), max(values)
        winners = ', '.join(f'R{i+1}' for i,v in enumerate(values) if v == low)
        notes.append(f'Lowest observed emissions: {winners}, {fmt(low)} g CO2e. Spread between highest and lowest: {fmt(high-low)} g ({fmt(percentage(high,low))}% relative to the lowest).')
    else:
        notes.append('No overall efficiency winner is assigned: emissions coverage or successful-request coverage is incomplete. Failed-request costs remain included in observed batch sums.')
    base = runs[reference]['questions']
    for i, run in enumerate(runs):
        if i == reference:
            continue
        pairs = [(q,v,base[q]) for q,v in run['questions'].items() if v['status']=='ok' and base[q]['status']=='ok' and v['emissions_g'] is not None and base[q]['emissions_g'] is not None]
        changes = [v['emissions_g']-b['emissions_g'] for _,v,b in pairs]
        same_sequence = list(run['questions']) == list(base)
        notes.append(f'R{i+1} versus R{reference+1}: '+('identical execution sequence. ' if same_sequence else 'different execution sequence. ')+f'{len(pairs)} valid successful question pairs; {sum(v<0 for v in changes)} lower, {sum(v>0 for v in changes)} higher, {sum(v==0 for v in changes)} unchanged emissions.')
        if pairs:
            q,v,b = max(pairs,key=lambda pair: abs(pair[1]['emissions_g']-pair[2]['emissions_g']))
            key = list(base).index(q)+1
            notes.append(f'Largest absolute change: Q{key}, {fmt(v["emissions_g"]-b["emissions_g"])} g; execution position {b["position"]} to {v["position"]}. Net change over these matched successful pairs: {fmt(sum(changes))} g.')
            for metric_name in ('energy_wh','latency_s'):
                available = [(v[metric_name],b[metric_name]) for _,v,b in pairs if v[metric_name] is not None and b[metric_name] is not None]
                if available:
                    notes.append(f'R{i+1}: {metric_name} change summed over {len(available)} matched successful pairs: {fmt(sum(a-b for a,b in available))}.')
    different = [k for k in runs[0]['metadata'] if k not in ('Folder','Questions') and len({r['metadata'][k] for r in runs})>1]
    notes.append('Recorded metadata differences: '+(', '.join(different) or 'none')+'. Unknown or unrecorded values do not establish equivalence.')
    notes.append('Historical zero defaults cannot be distinguished from measured zeros. Missing measurements are excluded and partial sums labelled. Percentage changes with a zero reference are N/A.')
    return notes


def write_pdf(runs, reference, records, output):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.graphics.shapes import Drawing, Rect, Line, Circle, String
    from report_charts import breakdown
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 12
    story = []
    def p(text, style='BodyText'):
        story.extend([Paragraph(escape(str(text)),styles[style]),Spacer(1,6)])
    def page(title):
        if story:
            story.append(PageBreak())
        p(title,'Title')
    def table(rows, widths):
        wrapped = [[Paragraph(escape(str(c)),styles['BodyText']) for c in row] for row in rows]
        t = Table(wrapped,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9ef')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.white]),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,10)])
    def total_label(run,key):
        value,n,expected=total(run,key)
        return f'{fmt(value)}'+(' (partial)' if n<expected else '')+f' [{n}/{expected}]'
    page('Execution-order comparison')
    p(f"Pipeline: {runs[0]['metadata']['Pipeline']} | {len(runs)} runs | {len(runs[0]['questions'])} identical question texts | Reference: R{reference+1}")
    table([['Run','Folder / model']]+[[f'R{i+1}',f"{r['folder'].name}\n{r['metadata']['Model(s)']}"] for i,r in enumerate(runs)],[40,455])
    p('Main observations','Heading2')
    for note in observations(runs,reference):
        p(note)
    page('Run cost and coverage')
    p('Observed sums include failed requests. Coverage is valid readings / all questions. Partial sums cannot be treated as complete batch costs.')
    table([['Run','Emissions (g)','Energy (Wh)','Request time (s)','Failures / unknown']]+[[f'R{i+1}',total_label(r,'emissions_g'),total_label(r,'energy_wh'),total_label(r,'latency_s'),sum(v['status']!='ok' for v in r['questions'].values())] for i,r in enumerate(runs)],[35,115,115,115,115])
    for start in range(0,len(runs),6):
        page(f'Batch costs: R{start+1}-R{min(start+6,len(runs))}')
        for key,label in [('emissions_g','Emissions (g CO2e)'),('energy_wh','Energy (Wh)')]:
            story.append(breakdown(label,[(f'R{i+1}',[v[key] for v in runs[i]['questions'].values()]) for i in range(start,min(start+6,len(runs)))]))
    for start in range(0,len(runs),3):
        page('Metadata differences')
        subset=runs[start:start+3]
        p('Different recorded values are marked *. Not recorded means unavailable, not equal.')
        data=[['Setting']+[f'R{start+j+1}' for j in range(len(subset))]]
        for key in runs[0]['metadata']:
            varies=len({r['metadata'][key] for r in runs})>1
            data.append([key+(' *' if varies else '')]+[r['metadata'][key] for r in subset])
        table(data,[105]+[390/len(subset)]*len(subset))
    questions=list(runs[reference]['questions'])
    # Heatmap panels bound label density for large comparisons.
    for run_start in range(0,len(runs),6):
        subset=runs[run_start:run_start+6]
        for qstart in range(0,len(questions),22):
            page('Question emissions change (%)')
            p(f'Relative to R{reference+1}. Blue = lower, red = higher; grey = missing, failed, or zero reference. Q labels map to exact question text in the comparison table. Color saturates at +/-100%; cell numbers retain the actual change.')
            qs=questions[qstart:qstart+22]
            d=Drawing(495,42+23*len(qs))
            w=435/len(subset)
            for j in range(len(subset)):
                d.add(String(65+j*w,d.height-15,f'R{run_start+j+1}',fontSize=9))
            for k,q in enumerate(qs):
                y=d.height-40-23*k
                d.add(String(0,y+6,f'Q{qstart+k+1}',fontSize=9))
                b=runs[reference]['questions'][q]
                for j,r in enumerate(subset):
                    v=r['questions'][q]
                    change=percentage(v['emissions_g'],b['emissions_g']) if v['status']==b['status']=='ok' else None
                    amount=min(abs(change)/100,1) if change is not None else 0
                    fill=colors.HexColor('#eeeeee') if change is None else colors.Color(1,1-.65*amount,1-.65*amount) if change>0 else colors.Color(1-.65*amount,1-.35*amount,1)
                    d.add(Rect(55+j*w,y,w-2,21,fillColor=fill,strokeColor=None))
                    d.add(String(61+j*w,y+6,fmt(change),fontSize=8))
            story.append(d)
    for i,run in enumerate(runs):
        if i==reference:
            continue
        page(f'Largest changes: R{i+1} versus R{reference+1}')
        p('Absolute emissions changes for up to 12 matched successful questions. Left = decrease; right = increase. Labels show g CO2e.')
        pairs=[]
        for j,q in enumerate(questions):
            v,b=run['questions'][q],runs[reference]['questions'][q]
            diff=delta(v['emissions_g'],b['emissions_g'])
            if diff is not None and v['status']==b['status']=='ok':
                pairs.append((j+1,diff))
        pairs=sorted(pairs,key=lambda x:abs(x[1]),reverse=True)[:12]
        d=Drawing(495,55+28*len(pairs))
        limit=max((abs(v) for _,v in pairs),default=0) or 1
        d.add(Line(255,20,255,d.height-5,strokeColor=colors.grey))
        if not pairs:
            p('No valid successful measurement pairs.')
        for j,(qid,value) in enumerate(pairs):
            y=d.height-30-j*28
            width=150*abs(value)/limit
            d.add(String(0,y+5,f'Q{qid}',fontSize=9))
            d.add(Rect(255-width if value<0 else 255,y,width,18,fillColor=colors.HexColor('#346ab0' if value<0 else '#bc473b'),strokeColor=None))
            d.add(String(420,y+5,fmt(value),fontSize=9))
        story.append(d)
    for start in range(0,len(runs),5):
        page('Cumulative emissions through execution')
        p('X: number of questions executed. Y: cumulative g CO2e. Question identities at a given position can differ. Curves stop at the first missing measurement rather than implying zero cost. Failed-request measurements are included.')
        palette=['#168577','#346ab0','#bc473b','#8c55a3','#c28420']
        curves=[]
        for i in range(start,min(start+5,len(runs))):
            values=[0.]
            for row in runs[i]['questions'].values():
                if row['emissions_g'] is None:
                    break
                values.append(values[-1]+row['emissions_g'])
            curves.append((i,values))
        maximum=max((max(v) for _,v in curves),default=0) or 1
        d=Drawing(495,400)
        n=len(questions)
        for tick in range(5):
            y=55+tick*220/4
            d.add(Line(65,y,485,y,strokeColor=colors.lightgrey))
            d.add(String(0,y,fmt(maximum*tick/4),fontSize=8))
        for tick in sorted({round(n*i/4) for i in range(5)}):
            d.add(String(65+420*tick/n,38,str(tick),fontSize=8))
        for j,(i,values) in enumerate(curves):
            color=colors.HexColor(palette[j])
            d.add(String(65,380-j*12,f'R{i+1}: {len(values)-1}/{n} questions accumulated',fontSize=8,fillColor=color))
            for k in range(1,len(values)):
                d.add(Line(65+420*(k-1)/n,55+220*values[k-1]/maximum,65+420*k/n,55+220*values[k]/maximum,strokeColor=color,strokeWidth=1.5))
            d.add(Circle(65+420*(len(values)-1)/n,55+220*values[-1]/maximum,2,fillColor=color,strokeColor=None))
        story.append(d)
    page('Aligned question comparison')
    p('Q identifiers follow the reference run. Text matching is exact. Mean emissions are calculated per question across all selected runs, using recorded values only; missing readings are excluded and the contributor count is shown. Recorded costs for failed attempts are included. Percentage changes are unavailable for a zero reference.')
    for qi,q in enumerate(questions,1):
        p(f'Q{qi}: {q}','Heading3')
        data=[['Run','Position','Status','g CO2e','Mean g (n/runs)','Wh','Seconds','Delta g','Change %']]
        for record in records[qi*len(runs)-len(runs):qi*len(runs)]:
            mean_count=f"{record['mean_emissions_valid_runs']}/{record['mean_emissions_total_runs']}"
            mean_value=f"{fmt(record['mean_emissions_g'])} ({mean_count})"
            data.append([record['run'],record['position'],record['status'],fmt(record['emissions_g']),mean_value,
                         fmt(record['energy_wh']),fmt(record['latency_s']),fmt(record['emissions_delta_g']),
                         fmt(record['emissions_change_percent'])])
        table(data,[30,48,48,55,80,45,55,62,62])
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8)
        canvas.drawString(50,22,'Local comparison | exact question-text alignment')
        canvas.drawRightString(545,22,f'Page {doc.page}')
    SimpleDocTemplate(str(output),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)


def compare(folders, reference=0, output_root=None):
    runs=[load_run(f) for f in folders]
    validate(runs)
    if not 0<=reference<len(runs):
        raise ValueError('Reference run is outside the selected list')
    records=aligned_rows(runs,reference)
    root=Path(output_root) if output_root else ROOT.parent/'comparison_reports'
    directory=root/datetime.now().strftime('comparison_%Y-%m-%d_%H-%M-%S_%f')
    directory.mkdir(parents=True,exist_ok=False)
    write_pdf(runs,reference,records,directory/'comparison.pdf')
    with (directory/'aligned_comparison.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders',nargs='*',help='Batch folders (or names within emissions_reports)')
    parser.add_argument('--reference',type=int,help='Reference position in selected runs, starting at 1')
    args=parser.parse_args()
    try:
        if args.folders:
            folders=[Path(f) if Path(f).is_dir() else ROOT/f for f in args.folders]
        else:
            available=[p for p in sorted(ROOT.iterdir()) if p.is_dir() and p.name.startswith(('exp1_llm_only_','exp2_rag_')) and answers_file(p)] if ROOT.exists() else []
            if len(available)<2:
                raise ValueError('At least two batch folders are needed')
            print('Available batch reports:')
            for i,folder in enumerate(available,1):
                print(f'{i}. {folder.name}')
            raw=input('Choose folder numbers separated by commas (or q to quit): ').strip()
            if raw.lower()=='q':
                return
            indexes=[int(s.strip()) for s in raw.split(',')]
            if any(i<1 or i>len(available) for i in indexes):
                raise ValueError('Folder number outside displayed list')
            folders=[available[i-1] for i in indexes]
        runs=[load_run(f) for f in folders]
        validate(runs)
        default=next((i for i,r in enumerate(runs) if r['metadata']['Ordering'] in ('original','Keep original order')),0)
        for i,r in enumerate(runs,1):
            print(f'R{i}: {r["folder"].name} ({r["metadata"]["Model(s)"]})')
        ref=args.reference
        if ref is None:
            response=input(f'Reference run number [default {default+1}]: ').strip()
            ref=int(response) if response else default+1
        print('Saved comparison to:',compare(folders,ref-1))
    except (OSError,ValueError,ImportError) as exc:
        print('Comparison not generated:',exc)
        raise SystemExit(1)


if __name__=='__main__':
    try:
        main()
    except (KeyboardInterrupt,EOFError):
        print('\nCancelled.')
