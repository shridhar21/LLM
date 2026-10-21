"""Graceful interruption, partial-run evidence, and cancellable local requests."""
import csv
import functools
import json
import math
import os
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
    if not indexing:
        from ollama_metrics import configure_session
        configure_session(folder)
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
                'measurement_basis': row.get('measurement_basis', getattr(phase['tracker'], '_measurement_basis', 'codecarbon_machine')),
                'measurement_backend': row.get('measurement_backend', getattr(phase['tracker'], '_measurement_backend', 'codecarbon')),
                'attribution_status': row.get('attribution_status', ''),
                **{key: finite(row.get(column)) for key, column in
                   [('energy', 'energy_consumed'), ('cpu', 'cpu_energy'),
                    ('gpu', 'gpu_energy'), ('ram', 'ram_energy'), ('ane', 'ane_energy')]}}
    if state.get('question') and phases:
        path = folder / 'answers.csv'
        rows = _rows(path)
        question = state['question']
        # A signal during CSV publication can arrive after its row was persisted.
        already_saved = rows and rows[-1].get('execution_position') == str(question['execution_position'])
        if not already_saved:
            result = dict(question, answer='', model_name=state['model'], input_tokens=None, output_tokens=None,
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
                for key in ('cpu', 'gpu', 'ram', 'ane'):
                    result[f'{key}_energy_kwh'] = observed(key)
                last = parts['generation'] or parts['retrieval'] or {}
                for key in ('measurement_basis', 'measurement_backend', 'attribution_status'):
                    result[key] = last.get(key, '')
            else:
                values = metrics('generation') or dict.fromkeys(('emissions', 'latency', 'energy', 'cpu', 'gpu', 'ram', 'ane'), float('nan'))
                result.update(latency_s=values['latency'], emissions_kg=values['emissions'],
                              emissions_g=values['emissions'] * 1000, energy_kwh=values['energy'])
                for key in ('cpu', 'gpu', 'ram', 'ane'):
                    result[f'{key}_energy_kwh'] = values[key]
                for key in ('measurement_basis', 'measurement_backend', 'attribution_status'):
                    result[key] = values.get(key, '')
            from ollama_metrics import append_csv, persist_optional_metrics
            generation = phases.get('generation')
            if generation:
                result.update(persist_optional_metrics('', generation['tracker'], folder, result))
            # Expand by column name, retaining completed rows and all new fields.
            append_csv(path, {k: str(v) if isinstance(v, float) and not math.isfinite(v) else v
                              for k, v in result.items()})
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
        propagate_cancel = kwargs.pop('_propagate_cancel', False)
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
            if propagate_cancel:
                raise
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


class ModelAnswer(str):
    """A normal answer string carrying the usage reported by this Ollama request."""
    def __new__(cls, text, input_tokens=None, output_tokens=None, *, metrics=None):
        instance = super().__new__(cls, text)
        metrics = metrics if isinstance(metrics, dict) else {}
        instance.ollama_metrics = dict(metrics)
        input_tokens = metrics.get('prompt_eval_count', input_tokens)
        output_tokens = metrics.get('eval_count', output_tokens)
        instance.input_tokens = input_tokens if type(input_tokens) is int and input_tokens >= 0 else None
        instance.output_tokens = output_tokens if type(output_tokens) is int and output_tokens >= 0 else None
        for name in ('prompt_eval_count', 'prompt_eval_cached_count', 'eval_count', 'total_duration',
                     'load_duration', 'prompt_eval_duration', 'eval_duration'):
            value = metrics.get(name)
            setattr(instance, name, value if type(value) is int and value >= 0 else None)
        instance.done_reason = metrics.get('done_reason')
        return instance


def ollama_base_url():
    """One local endpoint for selection, context checks, warm-up and inference."""
    from urllib.parse import urlsplit
    value = (os.getenv('OLLAMA_URL') or os.getenv('OLLAMA_HOST') or 'http://127.0.0.1:11434').strip().rstrip('/')
    if not value.startswith(('http://', 'https://')):
        value = 'http://' + value
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('localhost', '127.0.0.1', '::1', '0.0.0.0', '::') or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        raise ValueError('Ollama must use a local HTTP(S) address, e.g. http://127.0.0.1:11435')
    host = '127.0.0.1' if parsed.hostname in ('0.0.0.0', '::') else parsed.hostname
    host = '[' + host + ']' if ':' in host else host
    return f'{parsed.scheme}://{host}:{parsed.port or 11434}'


def query_local_model(prompt, model, messages=None, options=None):
    """Shared cancellable request with detailed, optional per-run metrics."""
    from ollama_metrics import query_local_model as request
    return request(prompt, model, messages=messages, options=options)
