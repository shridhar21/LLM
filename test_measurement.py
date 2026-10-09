import csv
import json
import math
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from measurement import complete_sum, coverage, finite, reading, stop_tracker, load_codecarbon_config, make_codecarbon_tracker, tracker_run_id
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
                current_id = uuid.uuid4()
                tracker = SimpleNamespace(run_id=current_id, _run_id='wrong-legacy-id')
                pd.DataFrame([{'run_id': 'old', 'energy_consumed': 99},
                              {'run_id': str(current_id), 'energy_consumed': .003, 'cpu_energy': .001,
                               'gpu_energy': 0, 'ram_energy': .002},
                              {'run_id': 'unrelated-latest', 'energy_consumed': 88}]).to_csv(path, index=False)
                read = namespace['latest_row_for_run']
                self.assertIsNone(read(path, set(), 'missing'))
                self.assertIsNone(read(path, set(), tracker_run_id(SimpleNamespace())))
                self.assertIsNone(read(path, {str(current_id)}, tracker_run_id(tracker)))
                row = read(path, set(), tracker_run_id(tracker))
                self.assertEqual(row['energy_consumed'], .003)
                self.assertEqual(row['cpu_energy'], .001)
                self.assertEqual(row['gpu_energy'], 0)
                self.assertEqual(row['ram_energy'], .002)

    def test_tracker_id_version_compatibility(self):
        self.assertEqual(tracker_run_id(SimpleNamespace(run_id='current', _run_id='older')), 'current')
        self.assertEqual(tracker_run_id(SimpleNamespace(_run_id='older')), 'older')
        self.assertEqual(tracker_run_id(SimpleNamespace(run_id=None, _run_id='older')), 'older')
        self.assertIsNone(tracker_run_id(SimpleNamespace()))
        self.assertIsNone(tracker_run_id(SimpleNamespace(run_id='', _run_id='unknown')))

    def test_missing_and_true_zero(self):
        self.assertTrue(math.isnan(reading(None, 'energy')))
        self.assertTrue(math.isnan(finite(float('inf'))))
        self.assertEqual(finite(0), 0)
        self.assertEqual(finite(1e-15), 1e-15)
        self.assertTrue(math.isnan(complete_sum([1, None])))
        self.assertTrue(math.isnan(complete_sum([])))
        self.assertEqual(coverage([0, None, 2]), {'valid': 2, 'expected': 3, 'complete': False, 'observed_sum': 2})

    def test_codecarbon_config_profiles(self):
        inference = load_codecarbon_config('inference')
        indexing = load_codecarbon_config('rag_indexing')
        self.assertEqual(inference['tracking_mode'], 'machine')
        self.assertFalse(inference['save_to_api'])
        self.assertEqual(inference['measure_power_secs'], 1)
        self.assertEqual(indexing['measure_power_secs'], 1)
        self.assertNotIn('tracking_mode', indexing)
        self.assertNotIn('save_to_api', indexing)

    def test_tracker_uses_named_config_profile(self):
        captured = {}

        class FakeTracker:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            with patch.dict('sys.modules', {'codecarbon': SimpleNamespace(OfflineEmissionsTracker=FakeTracker)}):
                tracker = make_codecarbon_tracker('inference', 'test-run', Path(temp) / 'reports')
        self.assertEqual(captured['project_name'], 'test-run')
        self.assertEqual(captured['tracking_mode'], 'machine')
        self.assertFalse(captured['save_to_api'])
        self.assertEqual(tracker._app_config_profile, 'inference')
        self.assertEqual(tracker._app_config_settings, load_codecarbon_config('inference'))

    def test_metadata_and_pdf(self):
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            tracker = SimpleNamespace(stop=lambda: None, _tracking_mode='machine', run_id='sample',
                                      _conf={}, _hardware=[], _app_config_profile='inference',
                                      _app_config_settings={'tracking_mode': 'machine'},
                                      _app_config_path='codecarbon_config.json')
            self.assertTrue(math.isnan(stop_tracker(tracker, folder)))
            metadata = json.loads((folder / 'measurement_metadata.json').read_text())
            self.assertFalse(metadata['sessions'][0]['emissions_available'])
            self.assertEqual(metadata['sessions'][0]['run_id'], 'sample')
            self.assertEqual(metadata['sessions'][0]['config_profile'], 'inference')
            self.assertEqual(metadata['sessions'][0]['configured_settings'], {'tracking_mode': 'machine'})
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
