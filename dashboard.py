"""Local three-machine report explorer. Launch: python dashboard.py"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import threading
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from flask import Flask, abort, jsonify, request, send_file

import compare_reports as comparison
import generate_report as individual
import plot_run_tables as plots
from question_order import load_bank, MODES

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.json.sort_keys = False
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
CATALOGUES = {}
ARTIFACTS = {}
RENDER_LOCK = threading.RLock()
OUTPUT = ROOT / 'dashboard_outputs'


def scalar(value):
    return individual.number(value)


def identify(path):
    return hashlib.sha256(str(path).encode()).hexdigest()[:20]


def discover(roots):
    if len(roots) != 3:
        raise ValueError('Choose three machine folders')
    resolved = [Path(r['path']).expanduser().resolve() for r in roots]
    if len(set(resolved)) != 3:
        raise ValueError('Choose three different machine folders')
    catalogue = {}
    machines = []
    for index, (spec, root) in enumerate(zip(roots, resolved)):
        if not root.is_dir():
            raise ValueError(f'Machine {index+1}: folder does not exist')
        machine = {'index': index, 'name': str(spec.get('name') or f'Machine {index+1}')[:50],
                   'path': str(root), 'runs': [], 'excluded': []}
        # Immediate children are run folders; never recurse into backup/export trees.
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or individual.answers_file(folder) is None:
                continue
            try:
                completion=comparison.read_json(folder/'run_status.json')
                if completion and completion.get('status') != 'completed':
                    raise ValueError('Run is not completed: '+str(completion.get('status','unknown')))
                run = comparison.load_run(folder)
                if run['metadata']['Completion evidence'] == 'No summary; completion unverified':
                    raise ValueError('No summary; completion unverified')
                entry = dict(id=identify(folder.resolve()), name=folder.name,
                             machine=index, machine_name=machine['name'],
                             model=run['metadata']['Model(s)'], pipeline=run['metadata']['Pipeline'],
                             scheme=run['ordering'].get('mode') or 'unknown',
                             seed=run['ordering'].get('seed'), questions=len(run['questions']))
                source=individual.answers_file(folder)
                catalogue[entry['id']] = {'entry': entry, 'run': run,
                                           'source_stamp':(source.stat().st_mtime_ns,source.stat().st_size)}
                machine['runs'].append(entry)
            except (ValueError, OSError, KeyError) as exc:
                machine['excluded'].append({'name': folder.name, 'reason': str(exc)})
        machines.append(machine)
    return catalogue, machines


def selected(payload):
    catalogue = CATALOGUES.get(payload.get('dataset'))
    if catalogue is None:
        raise ValueError('Load the three machine folders first')
    ids = payload.get('ids', [])
    if not ids or len(ids) != len(set(ids)) or any(i not in catalogue for i in ids):
        raise ValueError('Select at least one valid distinct run')
    items = [catalogue[i] for i in ids]
    for item in items:
        source=individual.answers_file(item['run']['folder'])
        if source is None or (source.stat().st_mtime_ns,source.stat().st_size)!=item['source_stamp']:
            raise ValueError('A source report changed. Reload the machine folders before continuing.')
    runs = [i['run'] for i in items]
    if len(runs) > 1:
        comparison.validate(runs)
    # Exact text alignment with workbook order, not row position or folder name.
    bank = load_bank()
    wanted = set(runs[0]['questions'])
    bank = bank[bank['Question'].isin(wanted)].copy()
    if len(bank) != len(wanted) or set(bank['Question']) != wanted or bank['Question'].duplicated().any():
        raise ValueError('Selected question texts are missing or ambiguous in questionbank.xlsx')
    bank['original_serial'] = bank.index + 1
    return items, bank


def raw_rows(run):
    with individual.answers_file(run['folder']).open(encoding='utf-8-sig', newline='') as handle:
        return {row['question']: row for row in csv.DictReader(handle)}


def prepared(items, bank):
    """Adapter for the existing custom graph generator, for any selected folders."""
    qs = bank['Question'].tolist()
    matrix = np.array([[item['run']['questions'][q]['emissions_g'] for item in items] for q in qs], dtype=float)
    positions = np.array([[item['run']['questions'][q]['position'] for item in items] for q in qs])
    raw = [raw_rows(i['run']) for i in items]
    components = np.array([[[scalar(rows[q].get(f'{part}_energy_kwh')) for part in ('cpu','gpu','ram')]
                            for rows in raw] for q in qs], dtype=float)
    latency = np.array([[i['run']['questions'][q]['latency_s'] for i in items] for q in qs], dtype=float)
    palette_bank = load_bank()
    topics = list(dict.fromkeys(palette_bank['topic_sheet']))
    data = dict(labels=[f'{r.question_type} - {r.topic_sheet}' for r in bank.itertuples()],
                serials=bank['original_serial'].to_numpy(), topics=bank['topic_sheet'].to_numpy(),
                types=bank['question_type'].to_numpy(), position=positions, emissions_g=matrix,
                components=components, energy=components.sum(axis=2), latency=latency,
                topic_palette=dict(zip(topics, plots.TOPIC_COLORS)))
    schemes=[mode[0] for mode in MODES]
    data['run_colors']=[plots.plt.get_cmap('tab20')((schemes.index(i['entry']['scheme'])
                         if i['entry']['scheme'] in schemes else 19) % 20) for i in items]
    details = []
    for k, item in enumerate(items):
        chains = []
        for chain in dict.fromkeys(bank['conversation_chain']):
            if not chain:
                continue
            subset = bank[bank['conversation_chain'] == chain].sort_values('conversation_turn')
            indexes = [qs.index(q) for q in subset['Question']]
            turns = subset['conversation_turn'].tolist()
            # Do not fabricate missing earlier turns or assume historical chain validity.
            if turns != list(range(1, max(turns)+1)):
                raise ValueError('A selected follow-up chain is missing earlier turns')
            pos = positions[indexes,k]
            if (np.diff(pos) <= 0).any():
                raise ValueError(f'{item["entry"]["name"]}: follow-up turn order is broken')
            chains.append(dict(chain=chain, topic=subset.iloc[0]['topic_sheet'],
                               first_question=int(subset.iloc[0]['original_serial']),
                               original_positions=subset['original_serial'].tolist(),
                               positions=pos.tolist(), gaps=(np.diff(pos)-1).tolist(),
                               turn_emissions_g=matrix[indexes,k].tolist()))
        details.append(chains)
    return data, details


def tables(items, bank):
    rows, metadata, summary = [], [], []
    for item in items:
        entry, run = item['entry'], item['run']
        raw = raw_rows(run)
        for _, q in bank.iterrows():
            record = run['questions'][q['Question']]
            row = dict(machine=entry['machine_name'], run=entry['name'],
                       scheme=entry['scheme'], seed=entry['seed'], original_question=int(q['original_serial']),
                       topic=q['topic_sheet'], question_type=q['question_type'], question=q['Question'],
                       **record)
            for part in ('cpu','gpu','ram'):
                row[f'{part}_energy_kwh'] = scalar(raw[q['Question']].get(f'{part}_energy_kwh'))
            for phase in ('retrieval','generation'):
                for metric in ('emissions_kg','energy_kwh','latency_s'):
                    row[f'{phase}_{metric}'] = scalar(raw[q['Question']].get(f'{phase}_{metric}'))
            rows.append(row)
        def complete(key):
            values = [r[key] for r in run['questions'].values()]
            return sum(values) if all(v is not None for v in values) else None
        summary.append(dict(machine=entry['machine_name'],run=entry['name'],model=entry['model'],
                            scheme=entry['scheme'],seed=entry['seed'],questions=len(run['questions']),
                            successful=sum(r['status']=='ok' for r in run['questions'].values()),
                            emissions_g=complete('emissions_g'),energy_wh=complete('energy_wh'),
                            latency_s=complete('latency_s')))
        metadata.append(dict(machine=entry['machine_name'],**run['metadata']))
    followups = []
    for chain in dict.fromkeys(bank['conversation_chain']):
        if not chain:
            continue
        subset = bank[bank['conversation_chain']==chain].sort_values('conversation_turn')
        previous = {}
        for _, q in subset.iterrows():
            row = dict(original_question=int(q['original_serial']),topic=q['topic_sheet'],turn=int(q['conversation_turn']),chain=chain)
            for index,item in enumerate(items):
                local_index=sum(i['entry']['machine']==item['entry']['machine'] for i in items[:index+1])
                label = f'{item["entry"]["machine_name"]} / R{local_index}'
                rec = item['run']['questions'][q['Question']]
                row[label+' position'] = rec['position']
                row[label+' gap'] = rec['position']-previous[index]-1 if index in previous else None
                previous[index] = rec['position']
            for index,item in enumerate(items):
                local_index=sum(i['entry']['machine']==item['entry']['machine'] for i in items[:index+1])
                label=f'{item["entry"]["machine_name"]} / R{local_index}'
                row[label+' emissions_g']=item['run']['questions'][q['Question']]['emissions_g']
            followups.append(row)
    averages=[]
    for _, q in bank.iterrows():
        vals=[i['run']['questions'][q['Question']]['emissions_g'] for i in items]
        averages.append(dict(original_question=int(q['original_serial']),question=q['Question'],
                             mean_emissions_g=sum(vals)/len(vals) if all(v is not None for v in vals) else None,
                             valid=sum(v is not None for v in vals),expected=len(vals)))
    return {'overview':summary,'questions':rows,'metadata':metadata,'followups':followups,'averages':averages}


def custom_files(items, bank, folder, page_filter=None, maximum=None):
    data, chains = prepared(items, bank)
    if not np.isfinite(data['emissions_g']).all():
        raise ValueError('Custom graphs require available emissions for every selected question; missing values are not zero')
    labels = [f'R{i+1}' for i in range(len(items))]
    result = plots.generate(ROOT/'questionbank.xlsx',folder,prepared=(data,chains),run_labels=labels,
                            page_filter=page_filter,shared_maximum=maximum)
    (folder/'run_key.csv').write_text(csv_text([dict(label=label,machine=i['entry']['machine_name'],
                                     folder=i['entry']['name'],scheme=i['entry']['scheme'],seed=i['entry']['seed'])
                                     for label,i in zip(labels,items)]),encoding='utf-8')
    return result


def csv_text(rows):
    buffer=io.StringIO(newline='')
    if rows:
        writer=csv.DictWriter(buffer,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    return buffer.getvalue()


@app.before_request
def local_only():
    if request.host.split(':')[0] not in ('127.0.0.1','localhost'):
        abort(403)
    origin=request.headers.get('Origin')
    if origin and urlsplit(origin).netloc != request.host:
        abort(403)


@app.errorhandler(ValueError)
@app.errorhandler(OSError)
def invalid(exc):
    return jsonify(error=str(exc)),400


@app.get('/')
def home():
    return send_file(ROOT/'dashboard_ui'/'index.html')


@app.post('/api/browse')
def browse():
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk();root.withdraw()
    try:
        root.attributes('-topmost',True)
        path=filedialog.askdirectory(title='Choose a machine folder containing emission runs',parent=root)
    finally:
        root.destroy()
    return jsonify(path=path)


@app.post('/api/load')
def load():
    catalogue,machines=discover(request.get_json()['machines'])
    token=uuid.uuid4().hex
    CATALOGUES[token]=catalogue
    return jsonify(dataset=token,machines=machines)


@app.post('/api/analyse')
def analyse():
    items,bank=selected(request.get_json())
    result=tables(items,bank)
    warnings=[]
    if len({i['entry']['model'] for i in items})>1:
        warnings.append('Different models selected; hardware and ordering effects cannot be isolated.')
    if len({i['run']['metadata']['country_iso_code'] for i in items})>1:
        warnings.append('Carbon locations differ. Compare energy as well as emissions.')
    if any(i['run']['metadata']['measure_power_secs']=='Not recorded' for i in items):
        warnings.append('Some measurement settings were not recorded.')
    if any(r['status']!='ok' for i in items for r in i['run']['questions'].values()):
        warnings.append('Failed/unknown questions are included and explicitly labelled in the tables.')
    result.update(warnings=warnings,question_count=len(bank),runs=[i['entry'] for i in items])
    return jsonify(result)


def artifact(path):
    token=uuid.uuid4().hex
    ARTIFACTS[token]=path.resolve()
    return '/artifact/'+token


@app.get('/artifact/<token>')
def download(token):
    path=ARTIFACTS.get(token)
    if path is None or not path.is_file():abort(404)
    return send_file(path,as_attachment=request.args.get('download')=='1')


def new_folder():
    path=OUTPUT/uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def individual_file(run, folder):
    staged=folder/'run';staged.mkdir()
    for source in run['folder'].iterdir():
        if source.is_file() and source.suffix in ('.csv','.json'):
            shutil.copy2(source,staged/source.name)
    return individual.generate(staged)


def make_graphs(payload):
    items,bank=selected(payload)
    kind=payload.get('kind','execution');page=max(1,int(payload.get('page',1)))
    if page>(len(bank)+31)//32:raise ValueError('Question page is outside the selected question set')
    all_values=[r['emissions_g'] for i in items for r in i['run']['questions'].values() if r['emissions_g'] is not None]
    maximum=max(all_values,default=0)*1.08 or 1
    panels=[]
    with RENDER_LOCK:
        for machine in range(3):
            group=[i for i in items if i['entry']['machine']==machine]
            if not group:
                panels.append({'machine':machine,'missing':'No matching run selected'});continue
            folder=new_folder()
            if kind.startswith('report_'):
                if kind=='report_individual':
                    chosen=payload.get('run_ids',{}).get(str(machine))
                    group=[next((i for i in group if i['entry']['id']==chosen),group[0])]
                    pdf=individual_file(group[0]['run'],folder)
                elif kind=='report_comparison':
                    runs=[i['run'] for i in group]
                    try:
                        comparison.validate(runs);reference=comparison.original_reference(runs)
                    except ValueError as exc:
                        panels.append(dict(machine=machine,missing=str(exc)));continue
                    pdf=folder/'comparison.pdf'
                    comparison.write_pdf(runs,reference,comparison.aligned_rows(runs,reference),pdf)
                else:raise ValueError('Unknown report view')
                panels.append(dict(machine=machine,url=artifact(pdf),pdf=True,
                                   label=group[0]['entry']['machine_name'],
                                   key=[dict(label=f'R{k+1}',**i['entry']) for k,i in enumerate(group)]))
                continue
            if kind=='execution':
                chosen=payload.get('run_ids',{}).get(str(machine))
                group=[next((i for i in group if i['entry']['id']==chosen),group[0])]
                filename='01_emissions_R1.png'
            elif kind=='range':filename=f'10_question_emissions_{page:02}.png'
            elif kind=='average':filename='11_average_question_emissions.png'
            elif kind=='phases':
                if group[0]['entry']['pipeline']!='RAG':raise ValueError('Phase graphs require RAG runs')
                import matplotlib.pyplot as plt
                from matplotlib.colors import Normalize,LinearSegmentedColormap
                start=(page-1)*32;questions=bank['Question'].tolist()[start:start+32]
                values=[comparison.phase_value(i['run'],q,p) for i in items for q in bank['Question'] for p in ('retrieval','generation')]
                bound=max((v for v in values if v is not None),default=0) or 1
                fig,axes=plt.subplots(1,2,figsize=(12,10))
                fig.subplots_adjust(left=.1,right=.93,top=.90,bottom=.15,wspace=.35)
                for ax,phase in zip(axes,('retrieval','generation')):
                    matrix=np.array([[comparison.phase_value(i['run'],q,phase) for i in group] for q in questions],dtype=float)
                    cmap=LinearSegmentedColormap.from_list('recorded_phase_scale',[(.95,.97,1),(.10,.34,.47)])
                    cmap.set_bad('#e2e8f0')
                    im=ax.imshow(matrix,aspect='auto',cmap=cmap,norm=Normalize(0,bound))
                    ax.set_xticks(range(len(group)),[f'R{k+1}' for k in range(len(group))])
                    ax.set_yticks(range(len(questions)),[f'Q{n}' for n in bank['original_serial'].tolist()[start:start+32]])
                    ax.set_title(phase.title());fig.colorbar(im,ax=ax,label='g CO2eq')
                filename='phases.png';fig.savefig(folder/filename,dpi=160);plt.close(fig)
                panels.append(dict(machine=machine,url=artifact(folder/filename),label=group[0]['entry']['machine_name'],
                                   key=[dict(label=f'R{k+1}',**i['entry']) for k,i in enumerate(group)]));continue
            else:raise ValueError('Unknown graph type')
            custom_files(group,bank,folder/'render',page_filter={filename[:-4]},maximum=maximum)
            panels.append(dict(machine=machine,url=artifact(folder/'render'/filename),label=group[0]['entry']['machine_name'],
                               key=[dict(label=f'R{k+1}',**i['entry']) for k,i in enumerate(group)]))
    return panels


@app.post('/api/graph')
def graph():
    return jsonify(panels=make_graphs(request.get_json()))


@app.post('/api/export')
def export():
    payload=request.get_json();items,bank=selected(payload);kind=payload.get('kind')
    folder=new_folder()
    with RENDER_LOCK:
        if kind in ('csv','xlsx'):
            data=tables(items,bank)
            if kind=='csv':
                name=payload.get('table','questions')
                if name not in data:raise ValueError('Unknown table')
                rows=data[name]
                question=payload.get('question','all')
                if question!='all':rows=[r for r in rows if r.get('original_question')==int(question)]
                query=str(payload.get('search','')).lower()
                if query:rows=[r for r in rows if any(query in str(v).lower() for v in r.values())]
                path=folder/(name+'.csv');path.write_text(csv_text(rows),encoding='utf-8-sig')
            else:
                from openpyxl import Workbook
                wb=Workbook();wb.remove(wb.active)
                for name,rows in data.items():
                    ws=wb.create_sheet(name)
                    if rows:
                        ws.append(list(rows[0]))
                        for row in rows:
                            ws.append(list(row.values()))
                        for cells in ws:
                            for cell in cells:
                                if isinstance(cell.value,str):cell.data_type='s'
                path=folder/'comparison_tables.xlsx';wb.save(path)
        elif kind=='comparison':
            runs=[i['run'] for i in items];comparison.validate(runs)
            reference=comparison.original_reference(runs)
            records=comparison.aligned_rows(runs,reference)
            path=folder/'comparison.pdf';comparison.write_pdf(runs,reference,records,path)
            (folder/'aligned_comparison.csv').write_text(csv_text(records),encoding='utf-8-sig')
        elif kind=='custom':
            from pypdf import PdfReader,PdfWriter
            writer=PdfWriter()
            for machine in range(3):
                group=[i for i in items if i['entry']['machine']==machine]
                if not group:continue
                # A compact key makes the R labels and machine ownership explicit in the PDF.
                from reportlab.pdfgen import canvas
                from reportlab.lib.utils import simpleSplit
                key=folder/f'machine_{machine+1}_key.pdf'
                canvas_key=canvas.Canvas(str(key),pagesize=(1100,700))
                canvas_key.setFont('Helvetica-Bold',18)
                canvas_key.drawString(45,655,group[0]['entry']['machine_name']+' - selected runs')
                y=615
                for k,item in enumerate(group):
                    canvas_key.setFont('Helvetica',10)
                    text=f'R{k+1}: {item["entry"]["name"]} | {item["entry"]["model"]} | {item["entry"]["scheme"]} | seed: {item["entry"]["seed"]}'
                    for line in simpleSplit(text,'Helvetica',10,1000):
                        if y<40:canvas_key.showPage();canvas_key.setFont('Helvetica',10);y=650
                        canvas_key.drawString(45,y,line);y-=16
                    y-=10
                canvas_key.save()
                for key_page in PdfReader(key).pages:writer.add_page(key_page)
                pdf,_=custom_files(group,bank,folder/f'machine_{machine+1}')
                for page in PdfReader(pdf).pages:writer.add_page(page)
            path=folder/'custom_graphs.pdf'
            with path.open('wb') as handle:writer.write(handle)
        elif kind=='individual':
            entry=next((i for i in items if i['entry']['id']==payload.get('run')),None)
            if entry is None:raise ValueError('Choose a run')
            path=individual_file(entry['run'],folder)
        elif kind=='side-by-side':
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import ImageReader
            panels=make_graphs({**payload,'kind':payload.get('graph_kind','execution')})
            if any(p.get('pdf') for p in panels):
                raise ValueError('Download full PDFs from the report panels; three-machine image export is for graph views')
            path=folder/'three_machines.pdf'
            pdf=canvas.Canvas(str(path),pagesize=(3*1584,780))
            for panel in panels:
                x=panel['machine']*1584
                pdf.setFont('Helvetica-Bold',20)
                pdf.drawString(x+30,745,panel.get('label',f'Machine {panel["machine"]+1}'))
                if 'url' in panel:
                    image=ARTIFACTS[panel['url'].split('/')[-1]]
                    pdf.drawImage(ImageReader(str(image)),x+15,20,width=1554,height=700,preserveAspectRatio=True,anchor='c')
                else:
                    pdf.setFont('Helvetica',16);pdf.drawString(x+30,700,panel['missing'])
            pdf.save()
        else:raise ValueError('Unknown export type')
    response=dict(url=artifact(path),name=path.name)
    if kind=='comparison':response['aligned_url']=artifact(folder/'aligned_comparison.csv')
    return jsonify(response)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8501)
    args=parser.parse_args()
    print(f'Open http://127.0.0.1:{args.port} in your browser. Ctrl+C stops the dashboard.')
    app.run(host='127.0.0.1',port=args.port,debug=False,threaded=True,use_reloader=False)


if __name__=='__main__':main()
