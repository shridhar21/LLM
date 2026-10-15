import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from question_order import MODES, arrange, grouped, load_bank, select_questions, range_options, save_order, identity


class OrderingTests(unittest.TestCase):
    def test_repeated_same_mode_does_not_reuse_existing_folder(self):
        from question_order import batch_folder_name
        frame=arrange(self.bank,'original',42)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent) as temp:
            import contextlib
            with contextlib.chdir(temp):
                first=batch_folder_name('exp1_llm_only','fixed','model:3b',frame)
                (Path('emissions_reports')/first).mkdir(parents=True)
                second=batch_folder_name('exp1_llm_only','fixed','model:3b',frame)
                self.assertNotEqual(first,second)
                self.assertTrue(second.endswith('-run2'))

    def test_multiple_orders_keep_selection_and_use_independent_seeds(self):
        with patch('builtins.input', side_effect=['2', '0,2', '11,1,11']), \
                patch('builtins.print'), patch('question_order.secrets.randbits', side_effect=[10,20,30]):
            runs = select_questions('2', allow_multiple=True)
        self.assertEqual([r.attrs['ordering']['mode'] for r in runs], ['global','original','global'])
        self.assertEqual([r.attrs['ordering']['seed'] for r in runs], [10,None,30])
        for r, mode, seed in zip(runs, ['global','original','global'], [10,20,30]):
            self.assertEqual(r['Question ID'].tolist(), arrange(self.bank,mode,seed)['Question ID'].tolist())
        runs[0].iloc[0, runs[0].columns.get_loc('Question')] = 'changed'
        self.assertNotIn('changed',runs[2]['Question'].tolist())

    def test_range_multiple_uses_only_available_options_and_retains_selection(self):
        with patch('builtins.input',side_effect=['1','1','6','2','11','1,2']),patch('builtins.print'):
            runs=select_questions('3',allow_multiple=True)
        expected=self.bank.iloc[:6]
        for run in runs:
            self.assertEqual(set(run['Question ID']),set(expected['Question ID']))
        self.assertEqual([r.attrs['ordering']['mode'] for r in runs],['original','questions'])

    def test_opt_in_single_run_keeps_dataframe_return(self):
        with patch('builtins.input',side_effect=['1','4']),patch('builtins.print'):
            run=select_questions('2',allow_multiple=True)
        self.assertIsInstance(run,pd.DataFrame)
        self.assertEqual(run.attrs['ordering']['mode'],'question_types_questions')

    @classmethod
    def setUpClass(cls):
        cls.bank = load_bank()

    def test_workbook_and_all_modes(self):
        self.assertEqual(len(self.bank), 160)
        self.assertEqual(self.bank['topic_sheet'].nunique(), 8)
        self.assertTrue(self.bank['Question ID'].is_unique)
        for mode, _, shuffle_topics, shuffle_types, shuffle_questions, freely in MODES:
            a = arrange(self.bank, mode, 42)
            b = arrange(self.bank, mode, 42)
            self.assertEqual(a['Question ID'].tolist(), b['Question ID'].tolist())
            self.assertEqual(set(a['Question ID']), set(self.bank['Question ID']))
            self.assertEqual(a['execution_position'].tolist(), list(range(1,161)))
            if mode != 'global':
                # Topics must remain contiguous.
                runs = [v for i,v in enumerate(a['topic_sheet']) if i == 0 or v != a.iloc[i-1]['topic_sheet']]
                self.assertEqual(len(runs), 8)
                if not shuffle_topics:
                    self.assertEqual(runs, list(dict.fromkeys(self.bank['topic_sheet'])))
                if not shuffle_questions and not freely:
                    for topic in runs:
                        original = self.bank[self.bank['topic_sheet'] == topic].reset_index(drop=True)
                        output = a[a['topic_sheet'] == topic].reset_index(drop=True)
                        original_groups = {tuple(original.loc[g, 'Question ID']) for g in grouped(original)}
                        output_groups = {tuple(output.loc[g, 'Question ID']) for g in grouped(output)}
                        self.assertEqual(original_groups, output_groups)

    def test_entire_has_no_topic_prompt(self):
        with patch('builtins.input', side_effect=['1']) as prompt, patch('builtins.print'):
            result = select_questions('2')
        self.assertEqual(len(result), 160)
        self.assertEqual(prompt.call_count, 1)

    def test_range_selects_topic_and_serials_before_order(self):
        with patch('builtins.input', side_effect=['2','2','6','1']) as prompt, patch('builtins.print'):
            result = select_questions('3')
        self.assertEqual(set(result['topic_sheet']), {'Coding'})
        self.assertEqual(result['question_number'].tolist(), [2,3,4,5,6])
        self.assertEqual(prompt.call_count, 4)

    def test_single_question_range_omits_shuffle(self):
        with patch('builtins.input', side_effect=['1','1','1','1']), patch('builtins.print') as output:
            result = select_questions('3')
        self.assertEqual(len(result), 1)
        self.assertFalse(any('2. Shuffle' in str(call) for call in output.call_args_list))

    def test_free_range_all_modes_preserve_selection_and_order(self):
        free_choice = str(self.bank['topic_sheet'].nunique()+1)
        selected = self.bank.iloc[:40]
        self.assertEqual([m[0] for m in range_options(selected)], [m[0] for m in MODES])
        for number, spec in enumerate(MODES, 1):
            with patch('builtins.input', side_effect=[free_choice,'1','40',str(number)]), \
                    patch('builtins.print'), patch('question_order.secrets.randbits', return_value=42):
                result = select_questions('3')
            self.assertEqual(result['Question ID'].tolist(), arrange(selected,spec[0],42)['Question ID'].tolist())
            self.assertEqual(set(result['workbook_position']),set(range(1,41)))
            self.assertEqual(result.attrs['ordering']['selection']['kind'],'free_range')

    def test_free_range_boundary_validation_and_saved_identity(self):
        free_choice = str(self.bank['topic_sheet'].nunique()+1)
        with patch('builtins.input', side_effect=[free_choice,'0','20','19','21','1']), patch('builtins.print'):
            result = select_questions('3')
        self.assertEqual(result['Question ID'].tolist(),self.bank.iloc[19:21]['Question ID'].tolist())
        self.assertEqual(result['topic_sheet'].nunique(),2)
        self.assertEqual(identity(result.iloc[0],1)['workbook_position'],20)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent) as temp:
            save_order(result,Path(temp))
            recorded=json.loads((Path(temp)/'question_order.json').read_text())
        self.assertEqual(recorded['selection'],{'kind':'free_range','start':20,'end':21,'numbering':'workbook_position'})
        self.assertEqual([q['workbook_position'] for q in recorded['questions']],[20,21])

    def test_free_range_within_one_sheet_uses_single_topic_labels(self):
        free_choice = str(self.bank['topic_sheet'].nunique()+1)
        with patch('builtins.input',side_effect=[free_choice,'2','6','1']),patch('builtins.print') as output:
            result=select_questions('3')
        lines=[str(call.args[0]) for call in output.call_args_list if call.args]
        menu=lines[lines.index('\nChoose question execution order:')+1:]
        self.assertFalse(any('topic sheet' in line.lower() for line in menu))
        self.assertEqual(result['workbook_position'].tolist(),[2,3,4,5,6])

    def test_multi_topic_range_hides_equivalent_operations(self):
        frame=pd.DataFrame({'topic_sheet':['A','B'],'question_type':['Direct','Direct']})
        self.assertEqual([spec[0] for spec in range_options(frame)],['original','topics'])
        frame=pd.DataFrame({'topic_sheet':['A','A','B','B'],'question_type':['Direct']*4})
        self.assertEqual([spec[0] for spec in range_options(frame)],
                         ['original','questions','topics','topics_questions','global'])


if __name__ == '__main__':
    unittest.main()
