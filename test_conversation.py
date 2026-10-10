"""Follow-up tests: no real model inference, downloads, or workbook edits."""
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pandas as pd

from conversation import annotate_chains, ConversationMemory, MissingHistoryError, validate_execution
from question_order import arrange, MODES, select_questions, load_bank, range_options, save_order
from test_cancellation import load_functions
from cancellation import query_local_model

ROOT = Path(__file__).resolve().parent


def chain_frame():
    return annotate_chains(pd.DataFrame([
        {'Question ID': 'A:1', 'Question': 'Write a sorting function.', 'topic_sheet': 'A', 'question_number': 1, 'question_type': 'Follow-Up 1'},
        {'Question ID': 'A:2', 'Question': 'Improve that function.', 'topic_sheet': 'A', 'question_number': 2, 'question_type': 'Follow-Up 2'},
        {'Question ID': 'A:3', 'Question': 'What is its complexity?', 'topic_sheet': 'A', 'question_number': 3, 'question_type': 'Follow-Up 3'},
        {'Question ID': 'A:4', 'Question': 'Independent question.', 'topic_sheet': 'A', 'question_number': 4, 'question_type': 'Direct'},
        {'Question ID': 'B:1', 'Question': 'Explain solar power.', 'topic_sheet': 'B', 'question_number': 1, 'question_type': 'Follow-Up 1'},
        {'Question ID': 'B:2', 'Question': 'What about its cost?', 'topic_sheet': 'B', 'question_number': 2, 'question_type': 'Follow-Up 2'},
    ]))


def show_response(limit=32768):
    response = Mock()
    response.json.return_value = {'model_info': {'general.architecture': 'llama', 'llama.context_length': limit}}
    return response


class ConversationTests(unittest.TestCase):
    def test_real_bank_has_eight_five_turn_chains(self):
        frame = load_bank()
        chains = frame[frame['conversation_chain'] != '']
        self.assertEqual(chains['conversation_chain'].nunique(), 8)
        for _, group in chains.groupby('conversation_chain'):
            self.assertEqual(group['conversation_turn'].tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(group['question_number'].tolist(), [13, 14, 15, 16, 17])

    def test_every_shuffle_keeps_each_chain_ordered(self):
        frame = chain_frame()
        for mode, *_ in MODES:
            for seed in (1, 42, 97):
                output = arrange(frame, mode, seed)
                for chain in set(frame['conversation_chain']) - {''}:
                    positions = output.index[output['conversation_chain'] == chain].tolist()
                    if mode not in ('within_topics', 'topics_within_topics', 'global'):
                        self.assertEqual(positions, list(range(positions[0], positions[0] + len(positions))))
                    self.assertEqual(output.loc[positions, 'conversation_turn'].tolist(), list(range(1, len(positions)+1)))
                self.assertEqual(set(output['Question ID']), set(frame['Question ID']))
                validate_execution(output)

    def test_free_shuffles_allow_interleaving_and_are_reproducible(self):
        frame = chain_frame()
        for mode in ('within_topics', 'topics_within_topics', 'global'):
            interleaved = False
            for seed in range(20):
                output = arrange(frame, mode, seed)
                repeated = arrange(frame, mode, seed)
                self.assertEqual(output['Question ID'].tolist(), repeated['Question ID'].tolist())
                positions = output.index[output['conversation_chain'] == 'A::A:1'].tolist()
                interleaved |= positions != list(range(positions[0], positions[0]+3))
                self.assertEqual(output.loc[positions, 'conversation_turn'].tolist(), [1, 2, 3])
                self.assertIn('allow_interleaving', output.attrs['ordering']['conversation_order_policy'])
            self.assertTrue(interleaved, mode)

    def test_question_type_shuffle_exempts_all_followup_chains(self):
        frame = chain_frame()
        extra = frame.iloc[:3].drop(columns=['conversation_chain', 'conversation_turn', 'conversation_prerequisites']).copy()
        extra['Question ID'] = ['A:5', 'A:6', 'A:7']
        extra['question_number'] = [5, 6, 7]
        frame = annotate_chains(pd.concat([frame.iloc[:4], extra, frame.iloc[4:]], ignore_index=True))
        for mode in ('questions', 'question_types_questions', 'topics_questions', 'all_levels'):
            for seed in range(10):
                output = arrange(frame, mode, seed)
                followups = output[(output['topic_sheet'] == 'A') & (output['conversation_turn'] > 0)]
                self.assertEqual(followups['Question ID'].tolist(), ['A:1', 'A:2', 'A:3', 'A:5', 'A:6', 'A:7'])

    def test_range_menu_distinguishes_interleaving_from_group_shuffle(self):
        frame = chain_frame()
        self.assertEqual([spec[0] for spec in range_options(frame.iloc[:4])],
                         ['original', 'question_types', 'within_topics'])
        both_chains = pd.concat([frame.iloc[:3], frame.iloc[4:]])
        modes = [spec[0] for spec in range_options(both_chains)]
        self.assertEqual(modes, ['original', 'topics', 'global'])
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = arrange(frame, 'global', 42)
            save_order(output, temp)
            saved = json.loads((Path(temp) / 'question_order.json').read_text())
            self.assertIn('allow_interleaving', saved['conversation_order_policy'])
            self.assertIn('other questions may run between', saved['description'])
            self.assertEqual([q['Question ID'] for q in saved['questions']], output['Question ID'].tolist())

    def test_single_followup_requires_explicit_prerequisite_choice(self):
        with patch('builtins.input', side_effect=['2', '15', '1']), patch('builtins.print'):
            selected = select_questions('1')
        self.assertEqual(selected['question_number'].tolist(), [13, 14, 15])
        self.assertEqual(len(selected.attrs['ordering']['selection']['prerequisites_added']), 2)
        with patch('builtins.input', side_effect=['2', '15', '2']), patch('builtins.print'):
            with self.assertRaisesRegex(ValueError, 'Selection cancelled'):
                select_questions('1')

    def test_single_chain_has_no_effective_shuffle(self):
        frame = chain_frame().iloc[:3]
        self.assertEqual([spec[0] for spec in range_options(frame)], ['original'])

    def test_invalid_chain_labels_and_missing_prerequisites_fail(self):
        frame = chain_frame()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            with self.assertRaisesRegex(ValueError, 'requires earlier turns'):
                ConversationMemory(arrange(frame.iloc[1:3]), temp, 'model')
        raw = frame.iloc[1:3].drop(columns=['conversation_chain', 'conversation_turn', 'conversation_prerequisites'])
        with self.assertRaisesRegex(ValueError, 'missing a contiguous'):
            annotate_chains(raw)
        raw = frame.iloc[:1].copy()
        raw['question_type'] = 'Follow up questions'
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            annotate_chains(raw)

    def test_history_contains_actual_answers_and_is_isolated_between_chains(self):
        frame = arrange(chain_frame())
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch('requests.post', return_value=show_response()):
            memory = ConversationMemory(frame, temp, 'model')
            query = Mock(side_effect=lambda prompt, **kwargs: 'Answer to ' + prompt)
            for _, row in frame.iterrows():
                memory.generate(query, row, row['Question'], 'model')
            first, second, third, independent, other, last = query.call_args_list
            self.assertEqual(len(first.kwargs['messages']), 1)
            self.assertEqual([m['role'] for m in second.kwargs['messages']], ['user', 'assistant', 'user'])
            self.assertEqual(second.kwargs['messages'][1]['content'], 'Answer to Write a sorting function.')
            self.assertEqual(len(third.kwargs['messages']), 5)
            self.assertNotIn('messages', independent.kwargs)
            self.assertEqual(len(other.kwargs['messages']), 1)
            self.assertNotIn('sorting', str(last.kwargs['messages']))
            saved = json.loads((Path(temp) / 'conversation_context.json').read_text())
            self.assertEqual(len(saved['requests']), 5)
            self.assertFalse(any(r['truncated'] for r in saved['requests']))
            reset = ConversationMemory(frame, temp, 'model')
            with self.assertRaises(MissingHistoryError):
                reset.generate(query, frame.iloc[1], frame.iloc[1]['Question'], 'model')

    def test_failed_prior_and_overflow_never_call_model_with_invented_history(self):
        frame = arrange(chain_frame())
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch('requests.post', return_value=show_response(8192)):
            memory = ConversationMemory(frame, temp, 'model')
            failing = Mock(side_effect=RuntimeError('model failed'))
            with self.assertRaises(RuntimeError):
                memory.generate(failing, frame.iloc[0], 'Prompt', 'model')
            good = Mock(return_value='Answer')
            with self.assertRaises(MissingHistoryError):
                memory.generate(good, frame.iloc[1], 'Prompt', 'model')
            good.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'context budget'):
                memory.generate(good, frame.iloc[0], 'x' * 10000, 'model')
            good.assert_not_called()
            self.assertEqual(memory.options['num_ctx'], 8192)

    def test_rag_retrieval_uses_opening_subject_and_prior_answer(self):
        frame = arrange(chain_frame())
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch('requests.post', return_value=show_response()):
            memory = ConversationMemory(frame, temp, 'model')
            memory.generate(Mock(return_value='def sort_items(items): ...'), frame.iloc[0], 'Evidence plus opening question', 'model')
            text = memory.retrieval_query(frame.iloc[1], frame.iloc[1]['Question'])
            self.assertTrue(text.startswith('Improve that function.'))
            self.assertIn('Write a sorting function.', text)
            self.assertIn('def sort_items', text)
            query = Mock(return_value='Improved function')
            memory.generate(query, frame.iloc[1], '[SOURCE 1]\nCurrent evidence\nImprove that function.', 'model')
            messages = query.call_args.kwargs['messages']
            self.assertIn('Current evidence', messages[-1]['content'])
            self.assertEqual(messages[0]['content'], 'Write a sorting function.')

    def test_chat_stream_payload_and_output_limit(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = iter([b'{"message":{"content":"Hello "}}',
                                                 b'{"message":{"content":"world"},"done":true,"done_reason":"stop"}'])
        session = MagicMock()
        session.__enter__.return_value = session
        session.post.return_value = response
        messages = [{'role': 'user', 'content': 'question'}]
        with patch('requests.Session', return_value=session):
            self.assertEqual(query_local_model('question', 'model', messages, {'num_ctx': 8192}), 'Hello world')
        self.assertTrue(session.post.call_args.args[0].endswith('/api/chat'))
        self.assertEqual(session.post.call_args.kwargs['json']['messages'], messages)
        response.iter_lines.return_value = iter([b'{"message":{"content":"partial"},"done":true,"done_reason":"length"}'])
        with patch('requests.Session', return_value=session):
            with self.assertRaisesRegex(RuntimeError, 'incomplete answer'):
                query_local_model('q', 'model', messages)

    def test_pipeline_integration_normal_and_rag_keep_csv_and_phase_counts(self):
        # Two chains and an independent question execute between each other's turns.
        frame = arrange(chain_frame()).iloc[[0, 4, 3, 1, 5, 2]].copy()
        frame['execution_position'] = range(1, len(frame)+1)
        for rag in (False, True):
            trackers, calls = [], []
            def query(prompt, model, **kwargs):
                calls.append((prompt, kwargs))
                return 'Answer to ' + prompt
            runner = load_functions('run.py' if rag else 'normal_llm.py', query, trackers)
            with tempfile.TemporaryDirectory(dir=ROOT) as temp, contextlib.chdir(temp), \
                    patch('requests.post', return_value=show_response()), patch('builtins.print'):
                if rag:
                    embedder = Mock()
                    embedder.encode.return_value = np.ones((1, 2))
                    runner(frame, True, object(), embedder, 'model')
                else:
                    runner(frame, True, 'model')
                output = next(Path('emissions_reports').iterdir())
                rows = pd.read_csv(output / 'answers.csv')
                self.assertEqual(rows['status'].tolist(), ['ok'] * 6)
                self.assertEqual(len(trackers), 12 if rag else 6)
                self.assertEqual([len(options['messages']) for _, options in calls if 'messages' in options], [1, 1, 3, 3, 5])
                for (_, row), (prompt, options) in zip(frame.iterrows(), calls[-6:]):
                    previous = frame[frame['Question ID'].isin(row['conversation_prerequisites'])]
                    expected = []
                    for _, turn in previous.iterrows():
                        expected.extend([{'role': 'user', 'content': turn['Question']},
                                         {'role': 'assistant', 'content': 'Answer to ' + turn['Question']}])
                    if row['conversation_chain']:
                        self.assertEqual(options['messages'], expected + [{'role': 'user', 'content': prompt}])
                    else:
                        self.assertNotIn('messages', options)
                if rag:
                    supplied = [args.args[0][0] for args in embedder.encode.call_args_list]
                    self.assertIn('Answer to Write a sorting function.', supplied[3])
                    self.assertNotIn('solar power', supplied[3])
                    self.assertIn('Answer to Explain solar power.', supplied[4])
                    self.assertNotIn('sorting function', supplied[4])
                self.assertTrue(all(t.stops == 1 for t in trackers))
                self.assertTrue((output / 'conversation_context.json').exists())

    def test_interleaved_failure_blocks_only_its_own_chain(self):
        frame = arrange(chain_frame()).iloc[[0, 4, 3, 1, 5, 2]].copy()
        frame['execution_position'] = range(1, len(frame)+1)
        for rag in (False, True):
            trackers, calls = [], []
            def query(prompt, model, **kwargs):
                calls.append(prompt)
                if prompt == 'Write a sorting function.':
                    raise RuntimeError('Failed first answer')
                return 'Answer to ' + prompt
            runner = load_functions('run.py' if rag else 'normal_llm.py', query, trackers)
            with tempfile.TemporaryDirectory(dir=ROOT) as temp, contextlib.chdir(temp), \
                    patch('requests.post', return_value=show_response()), patch('builtins.print'):
                if rag:
                    embedder = Mock()
                    embedder.encode.return_value = np.ones((1, 2))
                    runner(frame, True, object(), embedder, 'model')
                else:
                    runner(frame, True, 'model')
                folder = next(Path('emissions_reports').iterdir())
                rows = pd.read_csv(folder / 'answers.csv')
                self.assertEqual(rows['status'].tolist(), ['error', 'ok', 'ok', 'error', 'ok', 'error'])
                self.assertNotIn('Improve that function.', calls)
                self.assertNotIn('What is its complexity?', calls)
                records = json.loads((folder / 'conversation_context.json').read_text())['requests']
                successful = [r for r in records if r['status'] == 'ok']
                self.assertEqual(successful[-1]['preceding_question_ids'], ['B:1'])
                self.assertTrue(all(t.stops == 1 for t in trackers))

    def test_cancelled_chain_preserves_history_and_partial_measurements(self):
        frame = arrange(chain_frame().iloc[:3])
        for rag in (False, True):
            trackers = []
            def query(prompt, model, **kwargs):
                if len(kwargs.get('messages', [])) == 3:
                    raise KeyboardInterrupt
                return 'First successful answer'
            runner = load_functions('run.py' if rag else 'normal_llm.py', query, trackers)
            with tempfile.TemporaryDirectory(dir=ROOT) as temp, contextlib.chdir(temp), \
                    patch('requests.post', return_value=show_response()), patch('builtins.print'):
                if rag:
                    embedder = Mock()
                    embedder.encode.return_value = np.ones((1, 2))
                    runner(frame, True, object(), embedder, 'model')
                else:
                    runner(frame, True, 'model')
                folder = next(Path('emissions_reports').iterdir())
                rows = pd.read_csv(folder / 'answers.csv')
                self.assertEqual(rows['status'].tolist(), ['ok', 'cancelled'])
                records = json.loads((folder / 'conversation_context.json').read_text())['requests']
                self.assertEqual([r['status'] for r in records], ['ok', 'cancelled'])
                self.assertEqual(records[1]['messages'][1]['content'], 'First successful answer')
                self.assertTrue(all(t.stops == 1 for t in trackers))

    def test_unavailable_model_information_prevents_followup_requests(self):
        import requests
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch('requests.post', side_effect=requests.ConnectionError('offline')):
            with self.assertRaisesRegex(ValueError, 'Cannot verify'):
                ConversationMemory(arrange(chain_frame()), temp, 'model')

    def test_encoder_truncation_is_recorded_without_truncating_generation_history(self):
        frame = arrange(chain_frame())
        class Tokenizer:
            def encode(self, text, **kwargs):
                chars = [ord(c) for c in text]
                return chars[:kwargs['max_length']] if kwargs.get('truncation') else chars
            def decode(self, ids, **kwargs):
                return ''.join(chr(i) for i in ids)
        from types import SimpleNamespace
        encoder = SimpleNamespace(max_seq_length=60, tokenizer=Tokenizer())
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch('requests.post', return_value=show_response()):
            memory = ConversationMemory(frame, temp, 'model')
            memory.generate(Mock(return_value='A' * 500), frame.iloc[0], 'Opening question', 'model')
            encoded = memory.retrieval_query(frame.iloc[1], frame.iloc[1]['Question'], encoder)
            self.assertEqual(len(encoded), 60)
            query = Mock(return_value='Answer')
            memory.generate(query, frame.iloc[1], 'Current evidence and question', 'model')
            self.assertEqual(query.call_args.kwargs['messages'][1]['content'], 'A' * 500)
            saved = json.loads((Path(temp) / 'conversation_context.json').read_text())['requests'][-1]
            self.assertTrue(saved['retrieval_context']['encoder_truncated'])
            self.assertFalse(saved['truncated'])


if __name__ == '__main__':
    unittest.main()
