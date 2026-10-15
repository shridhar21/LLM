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
    def test_codecarbon_append_retains_blank_columns_and_header_order(self):
        from codecarbon.output_methods.file import FileOutput
        from measurement import _AlignedFileOutput, _CSV_330_COLUMNS
        values=dict.fromkeys(_CSV_330_COLUMNS, '')
        values.update(run_id='one', cpu_energy=1.234567890123456e-12,
                      cloud_region=None, longitude=None, latitude=None)
        with tempfile.TemporaryDirectory() as temp:
            handler=_AlignedFileOutput(FileOutput('emissions.csv',temp))
            handler.out(SimpleNamespace(values=values),None)
            values={**values,'run_id':'two','ram_power':None}
            handler.out(SimpleNamespace(values=dict(reversed(list(values.items())))),None)
            with (Path(temp)/'emissions.csv').open(newline='') as f:
                raw=list(csv.reader(f))
            self.assertEqual([len(row) for row in raw],[38,38,38])
            second=dict(zip(raw[0],raw[2]))
            self.assertEqual(second['run_id'],'two')
            self.assertEqual(second['longitude'],'')
            self.assertEqual(second['ram_power'],'')
            self.assertEqual(float(second['cpu_energy']),values['cpu_energy'])

    def test_historical_csv_repair_is_lossless_and_backed_up(self):
        from measurement import _CSV_330_COLUMNS, repair_emissions_csv
        row=dict.fromkeys(_CSV_330_COLUMNS,'')
        row.update(run_id='fixture',codecarbon_version='3.3.0',os='Windows-11',python_version='3.14.3',
                   cpu_count='12',gpu_count='0',ram_total_size='15.69796371459961',tracking_mode='machine',
                   on_cloud='N',pue='1.0',wue='0.0',cpu_utilization_percent='5.1',gpu_utilization_percent='0',
                   ram_utilization_percent='94.6',ram_used_gb='14.847126007080078',
                   emissions='0.00030089052413053417',cpu_energy='1.234567890123456789e-12')
        full=[row[k] for k in _CSV_330_COLUMNS]
        short=[v for k,v in zip(_CSV_330_COLUMNS,full) if k not in ('cloud_region','longitude','latitude')]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'emissions.csv'
            with path.open('w',newline='') as f: csv.writer(f).writerows([_CSV_330_COLUMNS,full,short])
            before=path.read_bytes()
            self.assertEqual(repair_emissions_csv(path),1)
            self.assertEqual(path.with_name(path.name+'.before_alignment_fix.bak').read_bytes(),before)
            with path.open(newline='') as f: repaired=list(csv.reader(f))
            self.assertEqual(repaired,[_CSV_330_COLUMNS,full,full])
            self.assertEqual(repair_emissions_csv(path),0)
            with path.open('w',newline='') as f: csv.writer(f).writerows([_CSV_330_COLUMNS,short[:-1]])
            broken=path.read_bytes()
            with self.assertRaises(ValueError):repair_emissions_csv(path)
            self.assertEqual(path.read_bytes(),broken)

    def test_factory_installs_only_csv_alignment_adapter(self):
        from codecarbon.output_methods.file import FileOutput
        from measurement import _AlignedFileOutput
        with tempfile.TemporaryDirectory() as temp:
            csv_output=FileOutput('emissions.csv',temp)
            other=SimpleNamespace()
            fake=SimpleNamespace(_output_handlers=[csv_output,other])
            with patch('codecarbon.OfflineEmissionsTracker',return_value=fake):
                tracker=make_codecarbon_tracker('inference','fixture',temp)
            self.assertIsInstance(tracker._output_handlers[0],_AlignedFileOutput)
            self.assertIs(tracker._output_handlers[0].delegate,csv_output)
            self.assertIs(tracker._output_handlers[1],other)

    def test_telemetry_requires_current_run(self):
        import ast
        import pandas as pd
        for filename in ('normal_llm.py', 'run.py'):
            tree = ast.parse((Path(__file__).resolve().parent.parent / filename).read_text(encoding='utf-8'))
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
        self.assertEqual(indexing['tracking_mode'], 'machine')
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
