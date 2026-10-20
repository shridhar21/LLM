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


class ApplePowerTests(unittest.TestCase):
    def monitor(self, *, seconds=2, samples=2, baseline=2):
        import threading
        from powermetrics_ollama import OllamaPowerMonitor
        monitor = object.__new__(OllamaPowerMonitor)
        monitor._lock = threading.Lock()
        monitor.power_samples = [dict(end_wall_ns=i * 1000000000, elapsed_ns=1000000000,
                                     cpu_w=2., gpu_w=3., ane_w=1., soc_w=6.) for i in range(1, samples + 1)]
        monitor.baseline = dict(cpu_w=1., gpu_w=1., ane_w=0., soc_w=baseline)
        monitor.sample_interval_ms = 250
        monitor.carbon_intensity = 500.
        monitor.ollama_pid_source = 'test fixture'
        monitor._process_delta = lambda *args: {'python_ollama_cpu_share': .99}
        monitor._append_request = lambda result: None
        window = SimpleNamespace(start_wall_ns=0, end_wall_ns=seconds * 1000000000, request_id='q', phase='generation')
        return monitor, window

    def test_non_mac_never_loads_mac_config_or_backend(self):
        from measurement import use_apple_powermetrics
        for system in ('Windows', 'Linux'):
            with patch('measurement.platform.system', return_value=system), patch.dict('os.environ', {'LLM_MAC_POWER_BACKEND': 'invalid'}), patch('pathlib.Path.read_text', side_effect=AssertionError('Mac config accessed')):
                self.assertFalse(use_apple_powermetrics())

    def test_apple_selector_and_explicit_legacy_override(self):
        from measurement import use_apple_powermetrics
        with patch('measurement.platform.system', return_value='Darwin'), patch('measurement.platform.machine', return_value='arm64'), patch.dict('os.environ', {'LLM_MAC_POWER_BACKEND': 'auto'}):
            self.assertTrue(use_apple_powermetrics())
            with patch('mac_power.PowermetricsTracker', return_value='apple') as factory:
                self.assertEqual(make_codecarbon_tracker('inference', 'test', 'ignored'), 'apple')
                factory.assert_called_once()
            with patch.dict('os.environ', {'LLM_MAC_POWER_BACKEND': 'codecarbon'}):
                self.assertFalse(use_apple_powermetrics())
        with patch('measurement.platform.system', return_value='Darwin'), patch('measurement.platform.machine', return_value='x86_64'), patch.dict('os.environ', {'LLM_MAC_POWER_BACKEND': 'auto'}):
            self.assertFalse(use_apple_powermetrics())

    def test_plist_parser_missing_ane_and_process_values(self):
        from powermetrics_ollama import _power_sample
        sample = _power_sample({'processor': {'cpu_power': 2000, 'gpu_power': 3000, 'combined_power': 6000},
                                'tasks': [{'cpu_power': 999999}]}, 10)
        self.assertEqual(sample['cpu_w'], 2)
        self.assertIsNone(sample['ane_w'])
        self.assertIsNone(_power_sample({'tasks': [{'cpu_power': 999999}]}, 10))

    def test_stream_parser_accepts_null_and_xml_boundaries(self):
        import io, plistlib, threading
        monitor, _ = self.monitor()
        monitor.power_samples = []
        record = {'elapsed_ns': 250000000, 'processor': {'cpu_power': 2000, 'gpu_power': 3000, 'combined_power': 6000}}
        raw = plistlib.dumps(record)
        monitor._proc = SimpleNamespace(stdout=io.BytesIO(raw + b'\0' + raw))
        monitor._stop, monitor._raw_handle = threading.Event(), io.BytesIO()
        monitor._power_jsonl, monitor._stderr_handle = io.StringIO(), io.BytesIO()
        monitor.session_tag = 'test'
        monitor._read_power()
        self.assertEqual(len(monitor.power_samples), 2)
        self.assertTrue(all(s['ane_w'] is None for s in monitor.power_samples))

    def test_energy_integration_baseline_and_carbon_units(self):
        monitor, window = self.monitor()
        with patch('powermetrics_ollama.time.sleep'):
            result = monitor.finish_window(window)
        self.assertEqual(result['raw_soc_energy_j'], 12)
        self.assertEqual(result['incremental_soc_energy_j_signed'], 8)
        self.assertAlmostEqual(result['energy_kwh'], 8 / 3600000)
        self.assertAlmostEqual(result['emissions_g'], 8 / 3600000 * 500)
        self.assertEqual(result['coverage_ratio'], 1)

    def test_partial_negative_and_zero_energy_are_distinct(self):
        for samples, baseline, expected in [(1, 2, None), (2, 10, None), (2, 6, 0)]:
            monitor, window = self.monitor(samples=samples, baseline=baseline)
            with patch('powermetrics_ollama.time.sleep'):
                result = monitor.finish_window(window)
            self.assertEqual(result['energy_kwh'], expected)
            if baseline == 10:
                self.assertEqual(result['incremental_soc_energy_j_signed'], -8)
        monitor, window = self.monitor()
        monitor.carbon_intensity = None
        with patch('powermetrics_ollama.time.sleep'):
            result = monitor.finish_window(window)
        self.assertGreater(result['energy_kwh'], 0)
        self.assertIsNone(result['emissions_g'])

    def test_short_phase_is_integrated_and_overlap_not_double_counted(self):
        monitor, _ = self.monitor(samples=1)
        raw, signed, duration, covered, _ = monitor._integrate(980000000, 1000000000)
        self.assertAlmostEqual(raw['soc'], .12)
        self.assertAlmostEqual(covered, .02)
        monitor.power_samples.append(dict(monitor.power_samples[0]))
        raw, _, _, covered, _ = monitor._integrate(0, 1000000000)
        self.assertEqual(raw['soc'], 6)
        self.assertEqual(covered, 1)

    def test_missing_ane_stays_unavailable(self):
        monitor, window = self.monitor()
        monitor.baseline['ane_w'] = None
        for sample in monitor.power_samples:
            sample['ane_w'] = None
        with patch('powermetrics_ollama.time.sleep'):
            result = monitor.finish_window(window)
        self.assertIsNone(result['raw_ane_energy_j'])
        self.assertIsNone(result['incremental_ane_energy_j_signed'])
        self.assertIsNotNone(result['energy_kwh'])

    def test_mac_adapter_evidence_schema_and_rag_retrieval(self):
        from mac_power import PowermetricsTracker
        monitor, window = self.monitor()
        with patch('powermetrics_ollama.time.sleep'):
            result = monitor.finish_window(window)
        fake = SimpleNamespace(start=lambda: None, calibrate_idle=lambda *a: None, stop=lambda: None)
        class Window:
            def __enter__(self): return self
            def __exit__(self, *args): self.result = result
        fake.request = lambda *a, **k: Window()
        with tempfile.TemporaryDirectory() as temp, patch('mac_power.OllamaPowerMonitor', return_value=fake) as factory:
            tracker = PowermetricsTracker('inference', 'RAG_ret_q1_test', temp)
            tracker.start()
            self.assertFalse(factory.call_args.kwargs['require_ollama'])
            self.assertAlmostEqual(stop_tracker(tracker, temp), result['emissions_g'] / 1000)
            with (Path(temp)/'emissions.csv').open(newline='') as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row['measurement_basis'], 'idle_subtracted_apple_soc')
            self.assertEqual(row['ram_energy'], 'nan')
            self.assertGreater(float(row['ane_energy']), 0)
            meta = json.loads((Path(temp)/'measurement_metadata.json').read_text())
            self.assertEqual(meta['sessions'][0]['measurement_backend'], 'apple_powermetrics')
            with (Path(temp)/'answers.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, ['model_name','status','latency_s','energy_kwh','emissions_kg','ane_energy_kwh','ram_energy_kwh','measurement_basis'])
                writer.writeheader()
                writer.writerow(dict(model_name='test', status='ok', latency_s=2, energy_kwh=row['energy_consumed'],
                                     emissions_kg=row['emissions'], ane_energy_kwh=row['ane_energy'],
                                     ram_energy_kwh='nan', measurement_basis=row['measurement_basis']))
            from pypdf import PdfReader
            text = '\n'.join(p.extract_text() for p in PdfReader(generate(Path(temp))).pages)
            self.assertIn('ANE energy', text)
            self.assertIn('RAM energy is N/A', text)

    def test_start_failure_closes_monitor_and_does_not_fallback(self):
        from mac_power import PowermetricsTracker
        from unittest.mock import Mock
        fake = Mock()
        fake.start.side_effect = PermissionError('sudo unavailable')
        with tempfile.TemporaryDirectory() as temp, patch('mac_power.OllamaPowerMonitor', return_value=fake):
            tracker = PowermetricsTracker('inference', 'LLM_only_q1', temp)
            with self.assertRaises(PermissionError): tracker.start()
            fake.stop.assert_called_once()
            self.assertFalse(tracker.started)

    def test_comparisons_reject_different_energy_definitions(self):
        from compare_reports import validate
        runs = [dict(folder=Path('a'), metadata={'Pipeline':'Normal LLM', 'Measurement basis':'codecarbon_machine'}, questions={'q':{}}),
                dict(folder=Path('b'), metadata={'Pipeline':'Normal LLM', 'Measurement basis':'idle_subtracted_apple_soc'}, questions={'q':{}})]
        with self.assertRaisesRegex(ValueError, 'different energy definitions'): validate(runs)

    def test_request_csv_preserves_header_order_and_rejects_drift(self):
        from powermetrics_ollama import OllamaPowerMonitor
        with tempfile.TemporaryDirectory() as temp:
            monitor = object.__new__(OllamaPowerMonitor)
            monitor._request_csv = Path(temp) / 'requests.csv'
            monitor._append_request({'a': 1, 'b': 2})
            monitor._append_request({'b': 4, 'a': 3})
            with monitor._request_csv.open(newline='') as handle:
                self.assertEqual(list(csv.DictReader(handle)), [{'a':'1','b':'2'}, {'a':'3','b':'4'}])
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                monitor._append_request({'a': 5})

    def test_cleanup_even_when_process_termination_fails(self):
        import io
        import threading
        from unittest.mock import Mock
        from powermetrics_ollama import OllamaPowerMonitor
        monitor = object.__new__(OllamaPowerMonitor)
        monitor._stop = threading.Event()
        monitor._proc = Mock()
        monitor._proc.poll.return_value = None
        monitor._proc.send_signal.side_effect = OSError('signal failed')
        monitor._proc.kill.side_effect = OSError('kill failed')
        monitor._proc.stdout = io.BytesIO()
        streams = [io.BytesIO() for _ in range(4)]
        for name, stream in zip(('_power_jsonl','_process_jsonl','_raw_handle','_stderr_handle'), streams):
            setattr(monitor, name, stream)
        with self.assertRaises(OSError): monitor.stop()
        self.assertTrue(monitor._proc.stdout.closed)
        self.assertTrue(all(s.closed for s in streams))

    def test_ollama_pid_matches_configured_listening_port(self):
        from powermetrics_ollama import _discover_ollama_roots
        from unittest.mock import Mock
        proc = Mock(pid=123, info={'pid':123, 'name':'ollama', 'cmdline':['ollama','serve'], 'ppid':1})
        proc.net_connections.return_value = [SimpleNamespace(status='LISTEN', laddr=SimpleNamespace(port=11435))]
        with patch.dict('os.environ', {'OLLAMA_URL':'http://127.0.0.1:11435', 'OLLAMA_SERVER_PID':'123'}), patch('powermetrics_ollama.psutil.process_iter', return_value=[proc]):
            self.assertEqual(_discover_ollama_roots()[0], [123])
            proc.net_connections.return_value[0].laddr.port = 11434
            self.assertEqual(_discover_ollama_roots()[0], [])


if __name__ == '__main__':
    unittest.main()
