import csv
import json
import tempfile
import unittest
from pathlib import Path

from compare_reports import load_run, validate, compare, aligned_rows, observations, percentage


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.a = self.fixture('a', ['Question A?', 'Question B?', 'Question C?'], 'model-a', [0, .002, .004])
        self.b = self.fixture('b', ['Question C?', 'Question A?', 'Question B?'], 'model-b', [.005, .001, .001])

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
        self.assertIsNone(records[1]['emissions_change_percent'])
        self.assertEqual(records[3]['emissions_change_percent'],-50)
        self.assertTrue(any('Models differ' in s for s in observations(runs,0)))

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
        self.assertIsNone(percentage(1,0))
        self.assertIsNone(percentage(None,1))

    def test_pdf_and_csv(self):
        from pypdf import PdfReader
        out=compare([self.a,self.b],output_root=self.root/'outputs')
        text=' '.join(p.extract_text() for p in PdfReader(out/'comparison.pdf').pages)
        for expected in ('Metadata differences','Aligned question comparison','Question A?','Cumulative emissions','model-b'):
            self.assertIn(expected,text)
        with (out/'aligned_comparison.csv').open(encoding='utf-8',newline='') as handle:
            records=list(csv.DictReader(handle))
        self.assertEqual(len(records),6)
        self.assertEqual(records[1]['position'],'2')

    def test_invalid_selection_creates_no_output(self):
        with self.assertRaises(ValueError):
            compare([self.a,self.a],output_root=self.root/'outputs')
        self.assertFalse((self.root/'outputs').exists())

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
        out=compare(folders,reference=1,output_root=self.root/'outputs')
        text=' '.join(p.extract_text() for p in PdfReader(out/'comparison.pdf').pages)
        self.assertIn('partial',text)
        self.assertIn('R7',text)
        self.assertIn('No overall efficiency winner',text)


if __name__=='__main__':
    unittest.main()
