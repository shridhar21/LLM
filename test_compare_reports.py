import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compare_reports import load_run, validate, compare, aligned_rows, observations, percentage, execution_sequences, shuffle_scheme, SHUFFLE_SCHEMES, original_reference


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.a = self.fixture('a', ['Question A?', 'Question B?', 'Question C?'], 'model-a', [0, .002, .004])
        self.b = self.fixture('b', ['Question C?', 'Question A?', 'Question B?'], 'model-b', [.005, .001, .001])
        self.workbook_patch = patch('compare_reports.workbook_question_sequence', return_value=['Question A?','Question B?','Question C?'])
        self.workbook = self.workbook_patch.start()
        self.addCleanup(self.workbook_patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, name, questions, model, values, rag=False):
        directory=self.root/name
        directory.mkdir()
        fields=['question','model_name','status','topic_sheet','question_number','execution_position','energy_kwh']
        fields += ['total_emissions_kg','total_latency_s','retrieval_latency_s'] if rag else ['emissions_kg','latency_s']
        with (directory/'answers.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            for i,(q,v) in enumerate(zip(questions,values),1):
                row={'question':q,'model_name':model,'status':'ok','topic_sheet':name,'question_number':100+i,'execution_position':i,'energy_kwh':v,
                     'total_emissions_kg' if rag else 'emissions_kg':v,'total_latency_s' if rag else 'latency_s':i}
                if rag:
                    row['retrieval_latency_s']=.01
                writer.writerow(row)
        (directory/'question_order.json').write_text(json.dumps({'mode':'original' if name=='a' else 'global','seed':42,'questions':[{}]*len(questions)}))
        return directory

    def test_text_only_alignment_and_different_models(self):
        runs=[load_run(self.a),load_run(self.b)]
        validate(runs)
        records=aligned_rows(runs,0)
        self.assertEqual(records[1]['question'],'Question A?')
        self.assertEqual(records[1]['position'],2)
        self.assertEqual(records[1]['mean_emissions_g'],0.5)
        self.assertEqual(records[1]['mean_emissions_valid_runs'],2)
        self.assertEqual(records[1]['mean_emissions_total_runs'],2)
        self.assertIsNone(records[1]['emissions_change_percent'])
        self.assertEqual(records[3]['emissions_change_percent'],-50)
        self.assertTrue(any('Models differ' in s for s in observations(runs,0)))

    def test_execution_sequence_movements_and_nonfirst_reference(self):
        runs=[load_run(self.a),load_run(self.b)]
        sequences=execution_sequences(runs,0)
        self.assertEqual([(c['question_key'],c['shift'],c['movement']) for c in sequences[1]['cells']],
                         [('Q3',-2,'Earlier by 2'),('Q1',1,'Later by 1'),('Q2',1,'Later by 1')])
        self.assertEqual((sequences[1]['moved'],sequences[1]['unchanged'],sequences[1]['largest_shift']),(3,0,2))
        self.assertEqual(sequences[0]['sequence'],'Identical sequence')
        reversed_reference=execution_sequences(runs,1)
        self.assertEqual([c['question_key'] for c in reversed_reference[0]['cells']],['Q2','Q3','Q1'])
        self.assertEqual([c['shift'] for c in reversed_reference[0]['cells']],[-1,-1,2])
        self.assertEqual(reversed_reference[1]['moved'],0)

    def test_recorded_schemes_and_missing_metadata(self):
        for mode in SHUFFLE_SCHEMES:
            scheme=shuffle_scheme({'ordering':{'mode':mode,'seed':'0'}})
            self.assertNotEqual(scheme['topics'],'Not recorded')
            self.assertEqual(scheme['seed'],'0')
        self.assertIn('mix',shuffle_scheme({'ordering':{'mode':'global'}})['groups'])
        self.assertEqual(shuffle_scheme({'ordering':{'mode':'original'}})['seed'],'Not applicable')
        (self.b/'question_order.json').unlink()
        missing=load_run(self.b)
        self.assertEqual(shuffle_scheme(missing)['description'],'Scheme not recorded')
        self.assertEqual(shuffle_scheme(missing)['topics'],'Not recorded')
        unknown=shuffle_scheme({'ordering':{'mode':'future_mode','description':'Recorded custom rule','seed':'8'}})
        self.assertEqual(unknown['description'],'Recorded custom rule')
        self.assertEqual(unknown['question_types'],'Not recorded')

    def test_sequence_unchanged_despite_shuffled_scheme(self):
        c=self.fixture('c',['Question A?','Question B?','Question C?'],'model-a',[.001,.002,.003])
        sequences=execution_sequences([load_run(self.a),load_run(c)],0)
        self.assertEqual(sequences[1]['sequence'],'Identical sequence')
        self.assertEqual(sequences[1]['moved'],0)
        self.assertIn('shuffle',sequences[1]['scheme']['description'])

    def test_csv_scheme_fallback_and_long_question_key(self):
        from pypdf import PdfReader
        question='Long question: '+('Explain the experiment in detail. '*300)
        folders=[self.fixture('long_a',[question,'Short question'],'model-a',[.001,.002]),
                 self.fixture('long_b',['Short question',question],'model-a',[.002,.001])]
        (folders[1]/'question_order.json').unlink()
        path=folders[1]/'answers.csv'
        with path.open(newline='',encoding='utf-8') as handle:
            reader=csv.DictReader(handle)
            fields=reader.fieldnames+['ordering_mode','shuffle_seed']
            rows=list(reader)
        with path.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            writer.writerows([{**row,'ordering_mode':'global','shuffle_seed':'0'} for row in rows])
        scheme=shuffle_scheme(load_run(folders[1]))
        self.assertEqual(scheme['seed'],'0')
        self.assertIn('mix',scheme['groups'])
        self.workbook.return_value=[question,'Short question']
        output=compare(list(reversed(folders)),output_root=self.root/'long_output')
        text=' '.join(p.extract_text() for p in PdfReader(output/'comparison.pdf').pages)
        self.assertIn('Question label key',text)
        self.assertIn('Execution sequence comparison',text)

    def test_changed_whitespace_is_rejected(self):
        c=self.fixture('c',['Question A? ','Question B?','Question C?'],'model-a',[1,2,3])
        with self.assertRaisesRegex(ValueError,'exact question texts differ'):
            validate([load_run(self.a),load_run(c)])

    def test_duplicate_text_and_mixed_pipelines_rejected(self):
        c=self.fixture('c',['same','same'],'m',[1,2])
        with self.assertRaisesRegex(ValueError,'duplicate exact'):
            load_run(c)
        d=self.fixture('d',['Question A?','Question B?','Question C?'],'m',[1,2,3],rag=True)
        with self.assertRaisesRegex(ValueError,'cannot be mixed'):
            validate([load_run(self.a),load_run(d)])

    def test_missing_data_and_zero_reference(self):
        c=self.fixture('c',['Question A?','Question B?','Question C?'],'m',[None,0,.001])
        runs=[load_run(self.a),load_run(c)]
        validate(runs)
        self.assertTrue(any('No overall efficiency winner' in s for s in observations(runs,0)))
        records=aligned_rows(runs,0)
        self.assertEqual(records[0]['mean_emissions_g'],0)
        self.assertEqual(records[0]['mean_emissions_valid_runs'],1)
        self.assertEqual(records[0]['mean_emissions_total_runs'],2)
        self.assertIsNone(percentage(1,0))
        self.assertIsNone(percentage(None,1))

    def test_pdf_and_csv(self):
        from pypdf import PdfReader
        out=compare([self.a,self.b],output_root=self.root/'outputs')
        text=' '.join(p.extract_text() for p in PdfReader(out/'comparison.pdf').pages)
        for expected in ('Metadata differences','Aligned question comparison','Question A?','Cumulative emissions','model-b','Mean g',
                         'Recorded shuffling schemes','Question label key',
                         'Execution sequence comparison','Earlier by 2','Later by 1','Seed: 42'):
            self.assertIn(expected,text)
        self.assertNotIn('Which levels can move?',text)
        self.assertNotIn('Execution movement summary',text)
        self.assertGreater(text.index('Question label key'),text.index('Aligned question comparison'))
        self.assertIn('Question label key',PdfReader(out/'comparison.pdf').pages[-1].extract_text())
        with (out/'aligned_comparison.csv').open(encoding='utf-8',newline='') as handle:
            records=list(csv.DictReader(handle))
        self.assertEqual(len(records),6)
        self.assertEqual(records[1]['position'],'2')
        self.assertEqual(records[1]['mean_emissions_g'],'0.5')
        self.assertEqual(records[1]['mean_emissions_valid_runs'],'2')

    def test_invalid_selection_creates_no_output(self):
        with self.assertRaises(ValueError):
            compare([self.a,self.a],output_root=self.root/'outputs')
        self.assertFalse((self.root/'outputs').exists())

    def test_original_reference_is_automatic_and_verified_in_pdf_csv(self):
        from pypdf import PdfReader
        runs=[load_run(self.b),load_run(self.a)]
        self.assertEqual(original_reference(runs),1)
        output=compare([self.b,self.a],output_root=self.root/'auto_output')
        text=' '.join(page.extract_text() for page in PdfReader(output/'comparison.pdf').pages)
        self.assertIn('Reference: R2',text)
        self.assertIn('matches the selected questions in questionbank.xlsx',text)
        with (output/'aligned_comparison.csv').open(newline='',encoding='utf-8') as handle:
            rows=list(csv.DictReader(handle))
        self.assertTrue(all(row['reference_run']=='R2' for row in rows))
        self.assertEqual(rows[0]['question_key'],'Q1')
        self.assertEqual(rows[0]['question'],'Question A?')

    def test_original_reference_checks_actual_sequence_and_subset(self):
        self.workbook.return_value=['Unselected before','Question A?','Unselected between','Question B?','Question C?']
        runs=[load_run(self.b),load_run(self.a)]
        runs[0]['ordering']['mode']='original'
        self.assertEqual(original_reference(runs),1)
        c=self.fixture('c',['Question A?','Question B?','Question C?'],'model-a',[.001]*3)
        self.assertEqual(original_reference([load_run(c),load_run(self.a)]),1)
        (self.a/'question_order.json').unlink()
        self.assertEqual(original_reference([load_run(self.b),load_run(self.a)]),1)

    def test_missing_original_or_changed_workbook_creates_no_output(self):
        c=self.fixture('c',['Question B?','Question C?','Question A?'],'model-a',[.001]*3)
        with self.assertRaisesRegex(ValueError,'Include a run'):
            compare([self.b,c],output_root=self.root/'no_original')
        self.assertFalse((self.root/'no_original').exists())
        self.workbook.return_value=['Question A?','Question B?']
        with self.assertRaisesRegex(ValueError,'missing from questionbank'):
            compare([self.a,self.b],output_root=self.root/'missing_text')
        self.assertFalse((self.root/'missing_text').exists())
        self.workbook.return_value=['Question A?','Question A?','Question B?','Question C?']
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            original_reference([load_run(self.a),load_run(self.b)])

    def test_reference_matches_real_workbook_across_topic_sheets(self):
        from question_order import load_bank
        bank=load_bank()
        self.workbook.return_value=bank['Question'].tolist()
        selected=bank.iloc[[18,19,20,21]]['Question'].tolist()
        a=self.fixture('actual_original',selected,'model-a',[.001]*4)
        b=self.fixture('actual_shuffled',list(reversed(selected)),'model-a',[.001]*4)
        self.assertEqual(original_reference([load_run(b),load_run(a)]),1)

    def test_incomplete_manifest_and_reference_validation(self):
        (self.a/'question_order.json').write_text(json.dumps({'questions':[{}]*4}))
        with self.assertRaisesRegex(ValueError,'incomplete'):
            load_run(self.a)
        with self.assertRaises(ValueError):
            compare([self.b],output_root=self.root/'outputs')

    def test_multi_run_missing_data_pdf(self):
        from pypdf import PdfReader
        folders=[self.a,self.b]
        for i in range(5):
            folders.append(self.fixture(f'other{i}', ['Question B?','Question C?','Question A?'], 'model-a', [None,.002,.003]))
        out=compare(folders,output_root=self.root/'outputs')
        text=' '.join(p.extract_text() for p in PdfReader(out/'comparison.pdf').pages)
        self.assertIn('partial',text)
        self.assertIn('R7',text)
        self.assertIn('No overall efficiency winner',text)


if __name__=='__main__':
    unittest.main()
