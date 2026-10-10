"""Graceful interruption, partial-run evidence, and cancellable local requests."""
import csv
import functools
import json
import math
import signal
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_current = ContextVar('run_cancellation', default=None)
_cleanup_depth = ContextVar('cleanup_depth', default=0)


@contextmanager
def protect_cleanup(defer_interrupt=False):
    """A repeated Ctrl+C must not interrupt tracker cleanup or file publication."""
    previous = None
    outermost = _cleanup_depth.get() == 0
    depth_token = _cleanup_depth.set(_cleanup_depth.get() + 1)
    interrupted = False
    def handler(*_):
        nonlocal interrupted
        interrupted = True
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
        _cleanup_depth.reset(depth_token)
    if interrupted and defer_interrupt and outermost:
        raise KeyboardInterrupt


def safe_finalization(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with protect_cleanup(defer_interrupt=True):
            return function(*args, **kwargs)
    return wrapped


def configure_run(folder, planned, model, rag=False, indexing=False):
    state = _current.get()
    if state is not None:
        state.update(folder=Path(folder), planned=planned, model=model,
                     rag=rag, indexing=indexing)


def begin_question(identity, question_id, question):
    state = _current.get()
    if state is not None:
        state['question'] = dict(identity, question_id=question_id, question=question)
        state['phases'] = {}
        state.pop('retrieval_detail', None)


def track_phase(tracker, phase):
    state = _current.get()
    if state is not None:
        state.setdefault('phases', {})[phase] = {
            'tracker': tracker, 'started': time.time(), 'stopped': False,
        }
        if state.get('rag') or state.get('indexing'):
            from rag_reporting import register_phase
            register_phase(state, tracker, phase)


def phase_stopped(tracker, emissions):
    state = _current.get()
    if state:
        for name, phase in state.get('phases', {}).items():
            if phase['tracker'] is tracker:
                phase.update(stopped=True, emissions=emissions,
                             latency=time.time() - phase['started'])
                if state.get('rag') or state.get('indexing'):
                    from rag_reporting import stopped_phase
                    stopped_phase(state, tracker, name, emissions, phase['latency'])


def question_saved():
    state = _current.get()
    if state is not None:
        state.pop('question', None)
        state['phases'] = {}


def _rows(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _write_status(state, cancelled):
    folder = state.get('folder')
    if folder is None:
        return
    folder.mkdir(parents=True, exist_ok=True)
    rows = _rows(folder / 'answers.csv')
    counts = {key: sum(row.get('status') == key for row in rows)
              for key in ('ok', 'error', 'cancelled')}
    status = {'schema_version': 1, 'status': 'cancelled' if cancelled else 'completed',
              'pipeline': 'rag' if state.get('rag') else 'normal',
              'planned_queries': state['planned'], 'attempted_queries': len(rows),
              'completed_queries': counts['ok'] + counts['error'],
              'successful_queries': counts['ok'], 'failed_queries': counts['error'],
              'cancelled_queries': counts['cancelled'],
              'unstarted_queries': max(0, state['planned'] - len(rows)),
              'actual_execution': [dict(question=row['question'], status=row['status'],
                                        position=i) for i, row in enumerate(rows, 1)],
              'total_runtime_s': time.time() - state['started']}
    if state.get('indexing'):
        status['kind'] = 'indexing'
        status['index_published'] = state.get('index_published', False)
    temporary = folder / 'run_status.json.tmp'
    temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
    temporary.replace(folder / 'run_status.json')


def _save_cancelled(state):
    from measurement import stop_tracker, tracker_run_id, finite
    folder = state.get('folder')
    if folder is None:
        return
    folder.mkdir(parents=True, exist_ok=True)
    phases = state.get('phases', {})
    for phase in phases.values():
        if not phase['stopped']:
            stop_tracker(phase['tracker'], folder)
    raw = _rows(folder / 'emissions.csv') or _rows(folder / 'temp_emissions.csv')
    def metrics(name):
        phase = phases.get(name)
        if phase is None:
            return None
        row = next((r for r in reversed(raw)
                    if r.get('run_id') == tracker_run_id(phase['tracker'])), {})
        return {'emissions': phase.get('emissions', float('nan')),
                'latency': phase.get('latency', time.time() - phase['started']),
                **{key: finite(row.get(column)) for key, column in
                   [('energy', 'energy_consumed'), ('cpu', 'cpu_energy'),
                    ('gpu', 'gpu_energy'), ('ram', 'ram_energy')]}}
    if state.get('question') and phases:
        path = folder / 'answers.csv'
        rows = _rows(path)
        question = state['question']
        # A signal during CSV publication can arrive after its row was persisted.
        already_saved = rows and rows[-1].get('execution_position') == str(question['execution_position'])
        if not already_saved:
            result = dict(question, answer='', model_name=state['model'],
                          status='cancelled', error='Interrupted by Ctrl+C; measurements are partial')
            if state['rag']:
                parts = {name: metrics(name) for name in ('retrieval', 'generation')}
                for name, values in parts.items():
                    for target, key in [('latency_s', 'latency'), ('emissions_kg', 'emissions'), ('energy_kwh', 'energy')]:
                        result[f'{name}_{target}'] = values[key] if values else float('nan')
                def observed(key):
                    values = [p[key] for p in parts.values() if p is not None]
                    return sum(values) if values else float('nan')
                result.update(total_latency_s=observed('latency'), total_emissions_kg=observed('emissions'),
                              total_emissions_g=observed('emissions') * 1000, energy_kwh=observed('energy'),
                              chunks=[], retrieved_k=0)
                for key in ('cpu', 'gpu', 'ram'):
                    result[f'{key}_energy_kwh'] = observed(key)
            else:
                values = metrics('generation') or dict.fromkeys(('emissions', 'latency', 'energy', 'cpu', 'gpu', 'ram'), float('nan'))
                result.update(latency_s=values['latency'], emissions_kg=values['emissions'],
                              emissions_g=values['emissions'] * 1000, energy_kwh=values['energy'])
                for key in ('cpu', 'gpu', 'ram'):
                    result[f'{key}_energy_kwh'] = values[key]
            # Reuse the existing schema so completed rows are never reinterpreted.
            fields = list(rows[0]) if rows else list(result)
            with path.open('a', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fields, extrasaction='ignore')
                if not rows:
                    writer.writeheader()
                writer.writerow(result)
    _write_status(state, True)
    if not state.get('indexing'):
        rows = _rows(folder / 'answers.csv')
        emissions = 'total_emissions_kg' if state['rag'] else 'emissions_kg'
        def observed(column):
            values = [finite(row.get(column)) for row in rows]
            valid = [v for v in values if math.isfinite(v)]
            return sum(valid) if valid else ''
        status = json.loads((folder / 'run_status.json').read_text())
        summary = {key: value for key, value in status.items() if key not in ('actual_execution', 'schema_version')}
        summary.update(model_name=state['model'], num_queries=len(rows),
                       total_emissions_kg='', total_energy_kwh='',
                       observed_emissions_kg=observed(emissions), observed_energy_kwh=observed('energy_kwh'))
        with (folder / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, list(summary))
            writer.writeheader()
            writer.writerow(summary)


def cancellable_run(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        state = {'started': time.time(), 'phases': {}}
        token = _current.set(state)
        try:
            result = function(*args, **kwargs)
            with protect_cleanup():
                _write_status(state, False)
            return result
        except KeyboardInterrupt:
            print('\nStopping run; saving partial measurements. Please wait...')
            with protect_cleanup():
                _save_cancelled(state)
            print('Run cancelled. Completed results and available partial costs were preserved.')
            print('Ollama connection closure was requested; server-side generation may stop asynchronously.')
        finally:
            try:
                if state.get('folder') and (state.get('rag') or state.get('indexing')):
                    from rag_reporting import finalize_details
                    with protect_cleanup():
                        status_path = state['folder'] / 'run_status.json'
                        status = json.loads(status_path.read_text()).get('status') if status_path.exists() else 'error'
                        try:
                            finalize_details(state['folder'], status)
                        except Exception as exc:
                            print(f'WARNING: could not finalize modular RAG report: {exc}')
            finally:
                _current.reset(token)
    return wrapped


def publish_index(folder, index, chunks, embeddings, metadata, sources):
    """Stage all five index files; rollback a failed publication as one unit."""
    import faiss
    import numpy as np
    folder = Path(folder)
    names = ('index.faiss', 'chunks.json', 'embeddings.npy', 'corpus_meta.csv', 'chunk_sources.csv')
    with tempfile.TemporaryDirectory(prefix='index-staging-', dir=folder.parent) as temp:
        stage = Path(temp)
        faiss.write_index(index, str(stage / names[0]))
        (stage / names[1]).write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding='utf-8')
        np.save(stage / names[2], embeddings)
        metadata.to_csv(stage / names[3], index=False)
        sources.to_csv(stage / names[4], index=False)
        backup = stage / 'previous'
        backup.mkdir()
        for name in names:
            if (folder / name).exists():
                shutil.copy2(folder / name, backup / name)
        replaced = []
        with protect_cleanup(defer_interrupt=True):
            try:
                for name in names:
                    (stage / name).replace(folder / name)
                    replaced.append(name)
            except BaseException:
                for name in reversed(replaced):
                    if (backup / name).exists():
                        (backup / name).replace(folder / name)
                    else:
                        (folder / name).unlink()
                raise
            state = _current.get()
            if state is not None:
                state['index_published'] = True


def query_local_model(prompt, model):
    """Buffer streamed Ollama output; main thread remains responsive to Ctrl+C."""
    import requests
    finished = threading.Event()
    cancelled = threading.Event()
    resources = {}
    result = {}
    def worker():
        try:
            with requests.Session() as session:
                resources['session'] = session
                with session.post('http://127.0.0.1:11434/api/generate',
                                  json={'model': model, 'prompt': prompt, 'stream': True, 'keep_alive': '15m'},
                                  timeout=9000, stream=True) as response:
                    resources['response'] = response
                    response.raise_for_status()
                    pieces = []
                    last = {}
                    for line in response.iter_lines():
                        if cancelled.is_set():
                            return
                        if line:
                            last = json.loads(line)
                            if last.get('error'):
                                raise RuntimeError(last['error'])
                            pieces.append(last.get('response', last.get('message', {}).get('content', '')))
                    if not last.get('done'):
                        raise RuntimeError('Ollama response ended before completion; partial output is not a successful answer')
                    result['answer'] = ''.join(pieces).strip() if pieces else str(last)
        except Exception as exc:
            result['error'] = exc
        finally:
            finished.set()
    threading.Thread(target=worker, daemon=True).start()
    try:
        while not finished.wait(.1):
            pass
    except KeyboardInterrupt:
        cancelled.set()
        # Close asynchronously: a socket close itself can block behind a read.
        def close():
            for name in ('response', 'session'):
                resource = resources.get(name)
                if resource is not None:
                    resource.close()
        threading.Thread(target=close, daemon=True).start()
        raise
    if 'error' in result:
        raise result['error']
    return result['answer']
