import ast
import contextlib
import io
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch
import cancellation
from rag_reporting import start_details
import time
import uuid

import pandas as pd
import requests

from measurement import reading, complete_sum, coverage
from model_selection import select_model


ROOT = Path(__file__).resolve().parent


def script_function(filename, name, namespace):
    """Test pipeline functions without loading the RAG embedding model or index."""
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    for helper in ('cancellable_run', 'configure_run', 'begin_question', 'track_phase',
                   'question_saved', 'query_local_model', 'protect_cleanup'):
        namespace.setdefault(helper, getattr(cancellation, helper))
    namespace.setdefault('start_details', start_details)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


class ModelSelectionTests(unittest.TestCase):
    def test_exact_names_and_invalid_choices(self):
        response = Mock()
        response.json.return_value = {'models': [{'name': 'llama3.2:3b'}, {'name': 'qwen3:4b'}]}
        with patch('model_selection.requests.get', return_value=response) as get, \
                patch('builtins.input', side_effect=['abc', '0', '3', '2']), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(select_model(), 'qwen3:4b')
        get.assert_called_once_with('http://127.0.0.1:11434/api/tags', timeout=10)
        response.raise_for_status.assert_called_once()
        self.assertIn('1. llama3.2:3b', output.getvalue())
        self.assertIn('2. qwen3:4b', output.getvalue())
        self.assertEqual(output.getvalue().count('Invalid selection'), 3)

    def test_unavailable_server(self):
        for error in (requests.ConnectionError('offline'), requests.Timeout('timeout')):
            with self.subTest(error=type(error).__name__), \
                    patch('model_selection.requests.get', side_effect=error), \
                    patch('builtins.input') as prompt:
                with self.assertRaisesRegex(ValueError, 'Check that Ollama is running'):
                    select_model()
                prompt.assert_not_called()

    def test_empty_invalid_and_http_error_lists(self):
        for data in ({'models': []}, {}, [], {'models': [None]}, {'models': [{'name': ''}]}):
            response = Mock()
            response.json.return_value = data
            with self.subTest(data=data), patch('model_selection.requests.get', return_value=response), \
                    patch('builtins.input') as prompt:
                with self.assertRaises(ValueError):
                    select_model()
                prompt.assert_not_called()
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError('404')
        with patch('model_selection.requests.get', return_value=response):
            with self.assertRaisesRegex(ValueError, 'Cannot list local Ollama models'):
                select_model()
        response = Mock()
        response.json.side_effect = ValueError('invalid JSON')
        with patch('model_selection.requests.get', return_value=response):
            with self.assertRaises(ValueError):
                select_model()

    def test_menus_pass_selected_model_for_every_run_option(self):
        for filename in ('normal_llm.py', 'run.py'):
            for choice in ('1', '2', '3', '4'):
                selected_questions = pd.DataFrame([{'Question ID': 1, 'Question': 'Test?'}])
                process = Mock()
                selector = Mock(return_value='qwen3:4b')
                namespace = {'pd': pd, 'select_model': selector,
                             'select_questions': Mock(return_value=selected_questions),
                             'process_queries': process, 'load_rag_index': Mock(return_value=('index', ['chunk'])),
                             'SentenceTransformer': Mock(return_value='embedder')}
                main = script_function(filename, 'main', namespace)
                inputs = [choice] + (['Custom?'] if choice == '4' else []) + ['5']
                with self.subTest(filename=filename, choice=choice), \
                        patch('builtins.input', side_effect=inputs), contextlib.redirect_stdout(io.StringIO()):
                    main()
                selector.assert_called_once()
                process.assert_called_once()
                self.assertEqual(process.call_args.kwargs['model_name'], 'qwen3:4b')
                self.assertEqual(process.call_args.kwargs['batch_mode'], choice in ('2', '3'))
                if choice != '4':
                    self.assertIs(process.call_args.args[0], selected_questions)
                    namespace['select_questions'].assert_called_once_with(choice)
                if filename == 'run.py':
                    namespace['SentenceTransformer'].assert_called_once_with('all-MiniLM-L6-v2')

    def test_selection_failure_returns_to_menu_without_starting_run(self):
        for filename in ('normal_llm.py', 'run.py'):
            namespace = {'select_model': Mock(side_effect=ValueError('No models installed')),
                         'select_questions': Mock(), 'process_queries': Mock(),
                         'load_rag_index': Mock(return_value=('index', [])), 'SentenceTransformer': Mock()}
            main = script_function(filename, 'main', namespace)
            with patch('builtins.input', side_effect=['2', '5']), contextlib.redirect_stdout(io.StringIO()):
                main()
            namespace['process_queries'].assert_not_called()
            namespace['select_questions'].assert_not_called()

    def test_selected_model_reaches_ollama_payload(self):
        for filename in ('normal_llm.py', 'run.py'):
            response = MagicMock()
            response.__enter__.return_value = response
            response.iter_lines.return_value = iter([b'{"response":" Answer ","done":true}'])
            session = MagicMock()
            session.__enter__.return_value = session
            session.post.return_value = response
            post = session.post
            query = script_function(filename, 'query_llm', {})
            with patch('requests.Session', return_value=session):
                self.assertEqual(query('Question?', 'qwen3:4b'), 'Answer')
            self.assertEqual(post.call_args.kwargs['json']['model'], 'qwen3:4b')
            with self.assertRaises(TypeError):
                query('Question?')

    def test_selected_model_used_for_warmup_queries_and_saved_results(self):
        questions = pd.DataFrame([{'Question ID': 1, 'Question': 'A?'}, {'Question ID': 2, 'Question': 'B?'}])
        telemetry = {'energy_consumed': .006, 'cpu_energy': .001, 'gpu_energy': .002, 'ram_energy': .003}
        for filename in ('normal_llm.py', 'run.py'):
            query = Mock(return_value='Answer')
            namespace = {'pd': pd, 'Path': Path, 'os': os, 'time': time, 'datetime': datetime, 'uuid': uuid,
                         'query_llm': query, 'make_tracker': Mock(return_value=Mock()),
                         'stop_tracker': Mock(return_value=.004), 'tracker_run_id': Mock(return_value='id'),
                         'latest_row_for_run': Mock(return_value=telemetry), 'reading': reading,
                         'complete_sum': complete_sum, 'coverage': coverage, 'save_order': Mock(),
                         'identity': lambda row, position: {'execution_position': position},
                         'retrieve_context': lambda assets, embedder, question: ['Context'],
                         'build_augmented_prompt': lambda question, chunks: question}
            process = script_function(filename, 'process_queries', namespace)
            original_directory = Path.cwd()
            with tempfile.TemporaryDirectory(dir=ROOT) as temp:
                try:
                    os.chdir(temp)
                    kwargs = {'model_name': 'qwen3:4b'}
                    if filename == 'run.py':
                        embedder = Mock()
                        embedder.encode.return_value.astype.return_value = 'vector'
                        index = Mock()
                        index.settings = {}
                        index.generation = None
                        index.search.return_value = ([[.5]], [[0]])
                        kwargs.update(index=index, chunks=['Context'], embedder=embedder)
                    with contextlib.redirect_stdout(io.StringIO()):
                        process(questions, batch_mode=True, **kwargs)
                    self.assertEqual(query.call_count, 3)
                    self.assertEqual(query.call_args_list[0].args[0], 'Warmup')
                    self.assertTrue(all(call.kwargs['model'] == 'qwen3:4b' for call in query.call_args_list))
                    folder = next(Path('emissions_reports').iterdir())
                    answers = pd.read_csv(folder / 'answers.csv')
                    summary = pd.read_csv(folder / 'summary.csv')
                    self.assertEqual(answers['model_name'].tolist(), ['qwen3:4b'] * 2)
                    self.assertEqual(summary.iloc[0]['model_name'], 'qwen3:4b' + ('+RAG' if filename == 'run.py' else ''))
                    multiplier = 2 if filename == 'run.py' else 1
                    self.assertAlmostEqual(answers.iloc[0]['energy_kwh'], .006 * multiplier)
                    self.assertAlmostEqual(summary.iloc[0]['total_energy_kwh'], .012 * multiplier)
                finally:
                    os.chdir(original_directory)


if __name__ == '__main__':
    unittest.main()
