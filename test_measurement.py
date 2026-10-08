import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from measurement import complete_sum, coverage, finite, reading, stop_tracker
from generate_report import generate


class MeasurementTests(unittest.TestCase):
    def test_telemetry_requires_current_run(self):
        import ast
        import pandas as pd
        for filename in ('normal_llm.py', 'run.py'):
            tree = ast.parse(Path(filename).read_text(encoding='utf-8'))
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'latest_row_for_run')
            namespace = {'pd': pd, 'Path': Path}
            exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), namespace)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'emissions.csv'
                pd.DataFrame([{'run_id': 'old', 'energy_consumed': 99}, {'run_id': 'current', 'energy_consumed': 0}]).to_csv(path, index=False)
                read = namespace['latest_row_for_run']
                self.assertIsNone(read(path, set(), 'missing'))
                self.assertIsNone(read(path, {'current'}, 'current'))
                self.assertEqual(read(path, set(), 'current')['energy_consumed'], 0)

    def test_missing_and_true_zero(self):
        self.assertTrue(math.isnan(reading(None, 'energy')))
        self.assertTrue(math.isnan(finite(float('inf'))))
        self.assertEqual(finite(0), 0)
        self.assertEqual(finite(1e-15), 1e-15)
        self.assertTrue(math.isnan(complete_sum([1, None])))
        self.assertTrue(math.isnan(complete_sum([])))
        self.assertEqual(coverage([0, None, 2]), {'valid': 2, 'expected': 3, 'complete': False, 'observed_sum': 2})

    def test_metadata_and_pdf(self):
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            tracker = SimpleNamespace(stop=lambda: None, _tracking_mode='machine', _run_id='sample',
                                      _conf={}, _hardware=[])
            self.assertTrue(math.isnan(stop_tracker(tracker, folder)))
            metadata = json.loads((folder / 'measurement_metadata.json').read_text())
            self.assertFalse(metadata['sessions'][0]['emissions_available'])
            with (folder / 'answers.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['status', 'model_name', 'latency_s', 'energy_kwh', 'emissions_kg'])
                writer.writeheader()
                writer.writerows([{'status': 'ok', 'model_name': 'fixture', 'latency_s': 1, 'energy_kwh': .001, 'emissions_kg': .001},
                                  {'status': 'error', 'model_name': 'fixture', 'latency_s': 2}])
            text = '\n'.join(p.extract_text() for p in PdfReader(generate(folder)).pages)
            self.assertIn('(partial)', text)
            self.assertIn('Measurement scope and reliability', text)
            self.assertIn('machine', text)
            (folder / 'measurement_metadata.json').unlink()
            text = '\n'.join(p.extract_text() for p in PdfReader(generate(folder)).pages)
            self.assertIn('UNVERIFIED', text)


if __name__ == '__main__':
    unittest.main()
