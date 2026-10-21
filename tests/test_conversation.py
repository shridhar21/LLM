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
from tests.test_cancellation import load_functions
from cancellation import query_local_model, ModelAnswer

ROOT = Path(__file__).resolve().parent.parent


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
    def test_historical_reconstruction_uses_saved_history_and_keeps_reported_counts(self):
        import csv
        from reconstruct_tokens import reconstruct_rows, render_prompt, publish, read_csv
        rows=[dict(question_id='A:1',question='Opening question',answer='Prior answer',model_name='llama3.2:3b',
                   question_type='Follow-Up 1',execution_position='1',status='ok',input_tokens='101',output_tokens='7'),
              dict(question_id='A:2',question='Continue?',answer='Follow-up answer',model_name='llama3.2:3b',
                   question_type='Follow-Up 2',execution_position='5',status='ok',input_tokens='',output_tokens=''),
              dict(question_id='A:3',question='Blocked?',answer='',model_name='llama3.2:3b',
                   question_type='Follow-Up 3',execution_position='7',status='error',input_tokens='',output_tokens='')]
        messages=[{'role':'user','content':'Opening question'},{'role':'assistant','content':'Prior answer'},
                  {'role':'user','content':'Continue?'}]
        context={'model':'llama3.2:3b','requests':[
            dict(question_id='A:1',answer='Prior answer',execution_position=1,status='ok',messages=messages[:1]),
            dict(question_id='A:2',answer='Follow-up answer',execution_position=5,status='ok',messages=messages)]}
        calls=[]
        def counter(model,text,prompt):
            calls.append((model,text,prompt));return len(text)
        result=reconstruct_rows(rows,context,counter)
        self.assertIn('Prior answer',calls[3][1])
        self.assertTrue(calls[3][2])
        self.assertGreater(result[1]['reconstructed_input_tokens'],len('Continue?'))
        self.assertEqual(result[2]['reconstructed_output_tokens'],'')
        self.assertEqual(result[0]['input_tokens'],'101')
        self.assertEqual(result[0]['output_tokens'],'7')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path=Path(temp)/'answers.csv'
            with path.open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            original,fields,old=read_csv(path)
            publish(path,original,fields,old,result,{'test':True})
            _,_,updated=read_csv(path)
            self.assertEqual([{k:r[k] for k in fields} for r in updated],old)
            self.assertEqual(path.with_name(path.name+'.before_token_reconstruction.bak').read_bytes(),original)
            with self.assertRaises(FileExistsError):publish(path,original,fields,old,result,{})
        context['requests'][1]['messages']=[]
        partial=reconstruct_rows(rows,context,counter)
        self.assertEqual(partial[1]['reconstructed_input_tokens'],'')
        self.assertGreater(partial[1]['reconstructed_output_tokens'],0)
        for model in ('llama3.2:3b','gemma3:4b','qwen3:4b'):
            text=render_prompt(model,messages)
            self.assertIn('Prior answer',text)
        with self.assertRaises(ValueError):render_prompt('unknown',messages)

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
            self.assertIn('allowing other questions between turns', saved['description'])
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

    def test_token_counts_come_from_final_response_for_generate_and_chat(self):
        response=MagicMock();response.__enter__.return_value=response
        session=MagicMock();session.__enter__.return_value=session;session.post.return_value=response
        for messages in (None,[{'role':'user','content':'question'}]):
            response.iter_lines.return_value=iter([b'{"response":"answer","message":{"content":"answer"},"done":false}',
                b'{"done":true,"prompt_eval_count":123,"eval_count":0}'])
            with patch('requests.Session',return_value=session):
                answer=query_local_model('question','fixture',messages)
            self.assertIsInstance(answer,str);self.assertEqual(answer,'answer')
            self.assertEqual(answer.input_tokens,123);self.assertEqual(answer.output_tokens,0)
        response.iter_lines.return_value=iter([b'{"response":"next","done":true}'])
        with patch('requests.Session',return_value=session):answer=query_local_model('question','fixture')
        self.assertIsNone(answer.input_tokens);self.assertIsNone(answer.output_tokens)
        self.assertIsNone(ModelAnswer('x',True,-1).input_tokens)
        self.assertIsNone(ModelAnswer('x',1,1.5).output_tokens)

    def test_shared_local_endpoint_and_extra_ollama_counters(self):
        from cancellation import ollama_base_url
        with patch.dict('os.environ', {'OLLAMA_URL':'127.0.0.1:11435', 'OLLAMA_HOST':'127.0.0.1:11434'}):
            self.assertEqual(ollama_base_url(), 'http://127.0.0.1:11435')
        with patch.dict('os.environ', {'OLLAMA_URL':'http://example.com:11434'}):
            with self.assertRaises(ValueError): ollama_base_url()
        answer = ModelAnswer('answer', metrics=dict(prompt_eval_count=21, eval_count=7, total_duration=99,
                             prompt_eval_cached_count=0, eval_duration=40, load_duration=10, done_reason='stop'))
        self.assertEqual((answer.input_tokens, answer.output_tokens, answer.total_duration), (21,7,99))
        self.assertEqual(answer.prompt_eval_cached_count, 0)
        self.assertEqual(answer.done_reason, 'stop')

    def test_token_counts_reach_normal_rag_and_followup_csvs(self):
        frame=arrange(chain_frame()).iloc[:3].copy()
        for rag in (False,True):
            trackers=[]
            runner=load_functions('run.py' if rag else 'normal_llm.py',lambda *a,**kw:ModelAnswer('answer',321,42),trackers)
            with tempfile.TemporaryDirectory(dir=ROOT) as temp,contextlib.chdir(temp),patch('requests.post',return_value=show_response()),patch('builtins.print'):
                if rag:
                    embedder=Mock();embedder.encode.return_value=np.ones((1,2))
                    runner(frame,True,object(),embedder,'model')
                else:runner(frame,True,'model')
                folder=next(Path('emissions_reports').iterdir())
                rows=pd.read_csv(folder/'answers.csv')
                self.assertEqual(rows.input_tokens.tolist(),[321]*3)
                self.assertEqual(rows.output_tokens.tolist(),[42]*3)
                summary = pd.read_csv(folder/'summary.csv')
                self.assertEqual(summary.iloc[0]['total_input_tokens'], 963)
                self.assertEqual(summary.iloc[0]['total_output_tokens'], 126)
                self.assertIn('inference_call_wall_s', rows.columns)
                self.assertIn('request_id', rows.columns)
                history=json.loads((folder/'conversation_context.json').read_text())
                self.assertTrue(all(r['input_tokens']==321 for r in history['requests']))
                if rag:
                    details=json.loads((folder/'rag_details.json').read_text())
                    self.assertTrue(all(r['output_tokens']==42 for r in details['queries']))

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


class RequestMetricsTests(unittest.TestCase):
    def setUp(self):
        import ollama_metrics as om
        self.om = om
        self.token = om._LAST.set(None)

    def tearDown(self):
        self.om._LAST.reset(self.token)

    def test_normalized_counts_rates_and_missing_cache(self):
        row = self.om.normalize(dict(prompt_eval_count=20, eval_count=10, eval_duration=2000000000,
                                     prompt_eval_duration=1000000000, prompt_eval_cached_count=5))
        self.assertEqual(row['input_tokens'], 20)
        self.assertEqual(row['uncached_prompt_tokens'], 15)
        self.assertEqual(row['prompt_cache_hit_ratio'], .25)
        self.assertEqual(row['output_tokens_per_s'], 5)
        self.assertIsNone(self.om.normalize({'eval_count':10})['prompt_cache_hit_ratio'])
        bad = self.om.normalize(dict(prompt_eval_count=2, prompt_eval_cached_count=3))
        self.assertEqual(bad['cache_count_consistency'], 'cached_exceeds_prompt')
        self.assertIsNone(bad['uncached_prompt_tokens'])

    def test_sampled_memory_summary_is_not_energy_and_missing_stays_missing(self):
        sampler = object.__new__(self.om.MemorySampler)
        sampler.interval, sampler.capture_uss, sampler.errors = .25, False, set()
        sampler.rows = [dict(monotonic_ns=t*1000000000, sample_overhead_s=.001, system_swap_used_bytes=0,
                             python=dict(rss_bytes=rss, uss_bytes=None),
                             ollama=dict(rss_bytes=None, uss_bytes=None),
                             combined=dict(rss_bytes=None, uss_bytes=None)) for t,rss in ((0,100),(1,300),(2,200))]
        row = sampler.summary()
        self.assertEqual(row['python_rss_bytes_peak'],300)
        self.assertEqual(row['python_rss_bytes_mean'],225)
        self.assertEqual(row['python_rss_bytes_delta'],100)
        self.assertIsNone(row['ollama_rss_bytes_peak'])
        self.assertIsNone(row['python_uss_bytes_mean'])
        self.assertFalse(row['memory_sampler_measures_ram_energy'])

    def test_stream_records_first_chunk_and_keeps_authoritative_usage(self):
        import requests
        events = [dict(message={'thinking':'thinking'}), dict(message={'content':'answer'}),
                  dict(done=True, prompt_eval_count=12, eval_count=7, eval_duration=1000000000)]
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [json.dumps(e).encode() for e in events]
        session = MagicMock()
        session.__enter__.return_value = session
        session.post.return_value = response
        with patch.object(requests, 'Session', return_value=session):
            answer = self.om.query_local_model('q','model',messages=[dict(role='user',content='q')])
        self.assertEqual(answer, 'answer')
        self.assertEqual((answer.input_tokens, answer.output_tokens), (12,7))
        self.assertEqual(answer.eval_duration, 1000000000)
        self.assertEqual(answer.request_metrics['thinking_characters'],8)
        self.assertLessEqual(answer.request_metrics['client_first_generated_chunk_s'], answer.request_metrics['client_first_visible_chunk_s'])
        self.assertTrue(session.post.call_args.args[0].endswith('/api/chat'))

    def test_original_model_answer_constructor_and_local_guard_preserved(self):
        answer = ModelAnswer('answer', 321, 42)
        self.assertEqual((answer.input_tokens, answer.output_tokens), (321,42))
        with patch.dict('os.environ', {'OLLAMA_URL':'https://example.com'}):
            with self.assertRaises(ValueError): self.om.base_url()

    def test_csv_expansion_retains_column_alignment_and_rejects_corruption(self):
        import csv
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'answers.csv'
            self.om.append_csv(path, dict(question='a', energy_kwh=.123456789123456))
            self.om.append_csv(path, dict(output_tokens=42, question='b', energy_kwh=0))
            with path.open(newline='') as handle: rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]['energy_kwh']), .123456789123456)
            self.assertEqual(rows[0]['output_tokens'], '')
            self.assertEqual(rows[1]['question'], 'b')
            path.write_text('a,b\n1\n', encoding='utf-8')
            with self.assertRaises(ValueError): self.om.append_csv(path, {'a':2})

    def test_cross_platform_generation_energy_only_and_request_id_matching(self):
        import csv
        from types import SimpleNamespace
        for backend in ('codecarbon', 'apple_powermetrics'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                tracker = SimpleNamespace(run_id='generation', _measurement_backend=backend)
                row = dict(question_id='q', execution_position=1, model_name='model', status='ok',
                           generation_energy_kwh=.001, generation_emissions_kg=.0005,
                           energy_kwh=.010, total_emissions_kg=.005, input_tokens=20, output_tokens=10,
                           measurement_backend=backend, measurement_basis='idle_subtracted_apple_soc' if backend=='apple_powermetrics' else 'codecarbon_machine')
                if backend == 'codecarbon':
                    self.om.append_csv(folder/'emissions.csv', dict(run_id='retrieval', energy_consumed=.009))
                    self.om.append_csv(folder/'emissions.csv', dict(run_id='generation', energy_consumed=.001))
                else:
                    self.om.append_csv(folder/'powermetrics_requests.csv', dict(request_id='generation', energy_kwh=.001,
                        incremental_soc_energy_j=3600, raw_soc_energy_j=4000, carbon_intensity_g_per_kwh=500, emissions_g=.5))
                record = dict(request_id='generation', eval_count=10, prompt_eval_count=20, total_reported_tokens=30,
                              request_status='ok', generation_request_id='generation')
                self.om._LAST.set(record)
                added = self.om.persist_metrics('', tracker, folder, row)
                self.assertAlmostEqual(added['request_joules_per_output_token'], 360)
                self.assertNotIn('energy_kwh', added)
                self.assertNotIn('input_tokens', added)
                self.assertEqual(added['generation_energy_measurement_basis'], row['measurement_basis'])
                if backend == 'codecarbon': self.assertEqual(added['generation_incremental_soc_energy_j'], '')
                with (folder/'ollama_requests.csv').open(newline='') as handle: saved = next(csv.DictReader(handle))
                self.assertEqual(saved['request_id'], 'generation')
                self.om._LAST.set(dict(request_id='other', eval_count=999))
                unmatched = self.om.persist_metrics('', tracker, folder, row)
                self.assertEqual(unmatched['request_status'], 'metrics_unavailable')
                self.assertEqual(unmatched['request_joules_per_output_token'], '')

    def test_optional_metrics_failure_does_not_block_primary_answer(self):
        with patch.object(self.om, 'persist_metrics', side_effect=OSError('file locked')):
            with self.assertWarns(UserWarning):
                row = self.om.persist_optional_metrics('', object(), Path('.'), {})
            self.assertIn('file locked', row['request_metrics_error'])

    def test_measured_generate_keeps_tokens_and_records_failures(self):
        from types import SimpleNamespace
        tracker = SimpleNamespace(run_id='test')
        sampler = Mock()
        sampler.start.return_value = sampler
        sampler.summary.return_value = dict(memory_sample_count=2)
        sampler.rows = []
        with patch.object(self.om, 'MemorySampler', return_value=sampler), tempfile.TemporaryDirectory() as temp:
            answer = self.om.measured_generate(lambda:ModelAnswer('answer',321,42), tracker=tracker, directory=temp)
            self.assertEqual(answer.output_tokens, 42)
            self.assertEqual(self.om._LAST.get()['eval_count'], 42)
            self.assertEqual(self.om._LAST.get()['memory_sample_count'], 2)
            self.assertGreaterEqual(self.om.last_inference_wall_s(), 0)
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                self.om.measured_generate(Mock(side_effect=RuntimeError('failed')), tracker=tracker, directory=temp)
            self.assertEqual(self.om._LAST.get()['request_status'], 'error')
            self.assertIsNone(self.om._LAST.get()['eval_count'])
            with self.assertRaises(KeyboardInterrupt):
                self.om.measured_generate(Mock(side_effect=KeyboardInterrupt), tracker=tracker, directory=temp)
            self.assertTrue((Path(temp)/'ollama_attempts.jsonl').is_file())


if __name__ == '__main__':
    unittest.main()
