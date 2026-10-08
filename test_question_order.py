import unittest
from unittest.mock import patch
import pandas as pd
from question_order import MODES, arrange, grouped, load_bank, select_questions


class OrderingTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
