"""Cancellation regression tests without loading models or contacting Ollama."""
import ast
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import time
import uuid
import signal
import threading
import _thread
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd

import cancellation
import measurement
from rag_reporting import start_details
from question_order import identity, save_order

ROOT = Path(__file__).resolve().parent


class Tracker:
    def __init__(self, folder, filename, name):
        self.run_id = uuid.uuid4().hex
        self.folder, self.filename, self._project_name = folder, filename, name
        self.stops = 0

    def start(self):
        pass

    def stop(self):
        self.stops += 1
        path = self.folder / self.filename
        with path.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, ['run_id', 'energy_consumed', 'cpu_energy', 'gpu_energy', 'ram_energy'])
            if path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(dict(run_id=self.run_id, energy_consumed=.003,
                                 cpu_energy=.001, gpu_energy=0, ram_energy=.002))
        return .001


def load_functions(filename, query, trackers):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    names = ('process_queries', 'latest_row_for_run')
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(pd=pd, Path=Path, datetime=datetime, time=time, uuid=uuid, os=os,
                     identity=identity, save_order=save_order, query_llm=query,
                     **{name: getattr(measurement, name) for name in
                        ('finite', 'reading', 'complete_sum', 'coverage', 'stop_tracker', 'tracker_run_id')},
                     **{name: getattr(cancellation, name) for name in
                        ('cancellable_run', 'configure_run', 'begin_question', 'track_phase', 'question_saved', 'protect_cleanup')})
    def make(name, folder, output_file='emissions.csv'):
        tracker = Tracker(folder, output_file, name)
        trackers.append(tracker)
        return tracker
    namespace['make_tracker'] = make
    namespace['start_details'] = start_details
    namespace['build_augmented_prompt'] = lambda query, contexts: query
    def retrieve_context(assets, embedder, query):
        embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return ['context']
    namespace['retrieve_context'] = retrieve_context
    exec(compile(ast.Module(body=funcs, type_ignores=[]), filename, 'exec'), namespace)
    return namespace['process_queries']


class CancellationTests(unittest.TestCase):
    def execute(self, rag=False, interrupt=None, warmup=False, manual=False):
        trackers = []
        def query(prompt, model):
            if (warmup and prompt == 'Warmup') or prompt == interrupt:
                raise KeyboardInterrupt
            return 'answer:' + prompt
        function = load_functions('run.py' if rag else 'normal_llm.py', query, trackers)
        questions = pd.DataFrame([{'Question ID': i, 'Question': q} for i, q in enumerate(('A', 'B', 'C'), 1)])
        class Embedder:
            def encode(self, queries, **kwargs):
                if interrupt == 'retrieval' and queries == ['B']:
                    raise KeyboardInterrupt
                return np.zeros((1, 2))
        index = SimpleNamespace(search=lambda emb, k: (np.array([[0.]]), np.array([[0]])))
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, contextlib.chdir(temp), contextlib.redirect_stdout(io.StringIO()):
            if rag:
                function(questions, not manual, index, ['context'], Embedder(), 'fixture')
            else:
                function(questions, not manual, 'fixture')
            folder = next(Path('emissions_reports').iterdir())
            status = json.loads((folder / 'run_status.json').read_text())
            rows = cancellation._rows(folder / 'answers.csv')
            summary = cancellation._rows(folder / 'summary.csv')
            from generate_report import generate
            from pypdf import PdfReader
            pdf_text = '\n'.join(page.extract_text() for page in PdfReader(generate(folder)).pages)
            if interrupt or warmup:
                from compare_reports import load_run
                with self.assertRaisesRegex(ValueError, 'cancelled/incomplete'):
                    load_run(folder)
            return status, rows, summary, trackers, pdf_text

    def test_normal_completion_unchanged(self):
        status, rows, summary, trackers, text = self.execute()
        self.assertEqual(status['status'], 'completed')
        self.assertEqual([r['answer'] for r in rows], ['answer:A', 'answer:B', 'answer:C'])
        self.assertAlmostEqual(float(summary[0]['total_emissions_kg']), .003)
        self.assertAlmostEqual(float(summary[0]['total_energy_kwh']), .009)
        self.assertTrue(all(t.stops == 1 for t in trackers))
        self.assertNotIn('CANCELLED / INCOMPLETE', text)

    def test_normal_cancel_preserves_completed_and_partial(self):
        status, rows, summary, trackers, text = self.execute(interrupt='B')
        self.assertEqual([r['status'] for r in rows], ['ok', 'cancelled'])
        self.assertEqual(status['unstarted_queries'], 1)
        self.assertEqual(status['completed_queries'], 1)
        self.assertEqual(float(rows[1]['energy_kwh']), .003)
        self.assertEqual(summary[0]['total_energy_kwh'], '')
        self.assertAlmostEqual(float(summary[0]['observed_energy_kwh']), .006)
        self.assertTrue(all(t.stops == 1 for t in trackers))
        self.assertIn('CANCELLED / INCOMPLETE', text)
        self.assertIn('exclude cancelled attempts', text)

    def test_rag_completion_unchanged(self):
        status, rows, summary, trackers, _ = self.execute(rag=True)
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(len(trackers), 6)
        self.assertTrue(all(t.stops == 1 for t in trackers))
        self.assertAlmostEqual(float(summary[0]['total_energy_kwh']), .018)
        self.assertEqual(float(rows[0]['cpu_energy_kwh']), .002)

    def test_rag_cancel_generation_preserves_both_phases(self):
        status, rows, _, trackers, _ = self.execute(rag=True, interrupt='B')
        self.assertEqual(status['cancelled_queries'], 1)
        self.assertEqual(float(rows[1]['retrieval_energy_kwh']), .003)
        self.assertEqual(float(rows[1]['generation_energy_kwh']), .003)
        self.assertEqual(float(rows[1]['energy_kwh']), .006)
        self.assertTrue(all(t.stops == 1 for t in trackers))

    def test_rag_cancel_retrieval_does_not_start_generation(self):
        status, rows, _, trackers, _ = self.execute(rag=True, interrupt='retrieval')
        self.assertEqual(len(trackers), 3)
        self.assertEqual(rows[1]['generation_energy_kwh'], 'nan')
        self.assertEqual(float(rows[1]['energy_kwh']), .003)
        self.assertTrue(all(t.stops == 1 for t in trackers))

    def test_cancel_warmup_generates_empty_partial_report(self):
        status, rows, _, trackers, text = self.execute(warmup=True)
        self.assertEqual(rows, [])
        self.assertEqual(trackers, [])
        self.assertEqual(status['unstarted_queries'], 3)
        self.assertIn('CANCELLED / INCOMPLETE', text)

    def test_manual_cancel(self):
        status, rows, _, trackers, _ = self.execute(manual=True, interrupt='B')
        self.assertEqual(status['status'], 'cancelled')
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(t.stops == 1 for t in trackers))

    def test_interrupt_during_tracker_stop_finalizes_exactly_once(self):
        original = Tracker.stop
        def stop(tracker):
            result = original(tracker)
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return result
        with patch.object(Tracker, 'stop', stop):
            status, rows, _, trackers, _ = self.execute()
        self.assertEqual(status['status'], 'cancelled')
        self.assertEqual(len(trackers), 1)
        self.assertEqual(trackers[0].stops, 1)
        self.assertEqual(float(rows[0]['emissions_kg']), .001)

    def test_repeated_interrupt_during_cancellation_cleanup_is_suppressed(self):
        original = Tracker.stop
        def stop(tracker):
            result = original(tracker)
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return result
        with patch.object(Tracker, 'stop', stop):
            status, rows, _, trackers, _ = self.execute(interrupt='A')
        self.assertEqual(status['status'], 'cancelled')
        self.assertEqual(trackers[0].stops, 1)
        self.assertEqual(rows[0]['status'], 'cancelled')

    def test_interrupt_after_csv_save_does_not_duplicate_answer(self):
        original = cancellation.question_saved
        def saved():
            original()
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        with patch.object(cancellation, 'question_saved', saved):
            status, rows, _, trackers, _ = self.execute()
        self.assertEqual(status['status'], 'cancelled')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'ok')
        self.assertEqual(status['unstarted_queries'], 2)

    def test_index_success_and_deferred_interrupt_are_consistent(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory(dir=ROOT) as temp:
                folder = Path(temp) / 'index'
                folder.mkdir()
                fake = SimpleNamespace(write_index=lambda index, path: Path(path).write_bytes(b'new'))
                original = Path.replace
                def replace(path, target):
                    if interrupted and path.name == 'index.faiss':
                        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                    return original(path, target)
                with patch.dict('sys.modules', {'faiss': fake}), patch.object(Path, 'replace', replace):
                    if interrupted:
                        with self.assertRaises(KeyboardInterrupt):
                            cancellation.publish_index(folder, None, ['new'], np.zeros((1, 2)), pd.DataFrame({'a':[1]}), pd.DataFrame({'b':[2]}))
                    else:
                        cancellation.publish_index(folder, None, ['new'], np.zeros((1, 2)), pd.DataFrame({'a':[1]}), pd.DataFrame({'b':[2]}))
                self.assertEqual((folder / 'index.faiss').read_bytes(), b'new')
                self.assertEqual(json.loads((folder / 'chunks.json').read_text()), ['new'])
                np.testing.assert_array_equal(np.load(folder / 'embeddings.npy'), np.zeros((1, 2)))
                self.assertEqual(pd.read_csv(folder / 'corpus_meta.csv')['a'].tolist(), [1])
                self.assertEqual(pd.read_csv(folder / 'chunk_sources.csv')['b'].tolist(), [2])

    def test_index_publication_preserves_old_index_on_interrupt(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp) / 'index'
            folder.mkdir()
            old = folder / 'index.faiss'
            old.write_bytes(b'old')
            def interrupt(*args):
                raise KeyboardInterrupt
            with patch.dict('sys.modules', {'faiss': SimpleNamespace(write_index=interrupt)}):
                with self.assertRaises(KeyboardInterrupt):
                    cancellation.publish_index(folder, None, [], np.zeros((0, 2)), pd.DataFrame(), pd.DataFrame())
            self.assertEqual(old.read_bytes(), b'old')

    def test_cancelled_indexing_finalizes_tracker_and_records_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp)
            tracker = Tracker(folder, 'emissions.csv', 'indexing')
            @cancellation.cancellable_run
            def indexing():
                cancellation.configure_run(folder, 0, 'embedder', indexing=True)
                cancellation.track_phase(tracker, 'indexing')
                tracker.start()
                raise KeyboardInterrupt
            with contextlib.redirect_stdout(io.StringIO()):
                indexing()
            status = json.loads((folder / 'run_status.json').read_text())
            self.assertEqual(status['status'], 'cancelled')
            self.assertEqual(status['kind'], 'indexing')
            self.assertFalse(status['index_published'])
            self.assertEqual(tracker.stops, 1)
            self.assertEqual(len(cancellation._rows(folder / 'emissions.csv')), 1)

    def test_index_publication_rolls_back_failed_commit(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp) / 'index'
            folder.mkdir()
            names = ('index.faiss', 'chunks.json', 'embeddings.npy', 'corpus_meta.csv', 'chunk_sources.csv')
            for name in names:
                (folder / name).write_bytes(b'old-' + name.encode())
            original = Path.replace
            def replace(path, target):
                if path.name == 'embeddings.npy' and path.parent.name.startswith('index-staging'):
                    raise OSError('simulated commit failure')
                return original(path, target)
            fake = SimpleNamespace(write_index=lambda index, path: Path(path).write_bytes(b'new'))
            with patch.dict('sys.modules', {'faiss': fake}), patch.object(Path, 'replace', replace):
                with self.assertRaises(OSError):
                    cancellation.publish_index(folder, None, ['new'], np.zeros((1, 2)), pd.DataFrame(), pd.DataFrame())
            for name in names:
                self.assertEqual((folder / name).read_bytes(), b'old-' + name.encode())

    def test_streamed_response_preserves_answer_and_errors(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_lines(self):
                return iter([b'{"response":" Hello"}', b'{"response":" world "}', b'{"response":"","done":true}'])
        class Session:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def post(self, *args, **kwargs): return Response()
        with patch.dict('sys.modules', {'requests': SimpleNamespace(Session=Session)}):
            self.assertEqual(cancellation.query_local_model('q', 'model'), 'Hello world')
        with patch.object(Response, 'iter_lines', return_value=iter([b'{"error":"bad model"}'])):
            with patch.dict('sys.modules', {'requests': SimpleNamespace(Session=Session)}):
                with self.assertRaisesRegex(RuntimeError, 'bad model'):
                    cancellation.query_local_model('q', 'model')

    def test_local_http_request_can_be_interrupted_without_waiting_for_answer(self):
        import requests
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.end_headers()
                try:
                    for _ in range(30):
                        self.wfile.write((json.dumps({'response': 'x' * 600, 'done': False}) + '\n').encode())
                        self.wfile.flush()
                        time.sleep(.05)
                except OSError:
                    pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        original = requests.Session.post
        def local(session, url, **kwargs):
            return original(session, f'http://127.0.0.1:{server.server_port}/api/generate', **kwargs)
        timer = threading.Timer(.2, _thread.interrupt_main)
        try:
            with patch.object(requests.Session, 'post', local):
                timer.start()
                started = time.monotonic()
                with self.assertRaises(KeyboardInterrupt):
                    cancellation.query_local_model('q', 'fixture')
                self.assertLess(time.monotonic() - started, 1.2)
        finally:
            timer.cancel()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == '__main__':
    unittest.main()
