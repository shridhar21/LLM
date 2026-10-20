"""Apple-only adapter. The cross-platform factory imports this module lazily."""
import csv
import json
import math
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from powermetrics_ollama import OllamaPowerMonitor

CONFIG = Path(__file__).with_name('mac_power_config.json')
BASIS = 'idle_subtracted_apple_soc'


def load_settings():
    settings = json.loads(CONFIG.read_text(encoding='utf-8'))
    overrides = {'sample_interval_ms': ('POWER_SAMPLE_INTERVAL_MS', int),
                 'process_interval_ms': ('PROCESS_SAMPLE_INTERVAL_MS', int),
                 'idle_baseline_seconds': ('POWER_IDLE_BASELINE_SECONDS', float),
                 'idle_settle_seconds': ('POWER_IDLE_SETTLE_SECONDS', float),
                 'carbon_intensity_g_per_kwh': ('POWER_CARBON_INTENSITY_G_PER_KWH', float)}
    for key, (name, cast) in overrides.items():
        if os.getenv(name):
            settings[key] = cast(os.environ[name])
    for key in overrides:
        value = settings.get(key)
        if value is None and key == 'carbon_intensity_g_per_kwh':
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid Mac power setting: {key}')
    if settings['idle_baseline_seconds'] <= 0:
        raise ValueError('Idle baseline must be positive')
    return settings


class PowermetricsTracker:
    """CodeCarbon-compatible methods; explicitly different measurement scope."""
    def __init__(self, profile, project_name, output_dir, output_file='emissions.csv'):
        import psutil
        self.output_dir, self.output_file = Path(output_dir), output_file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self._run_id = str(uuid.uuid4())
        self._project_name, self._app_config_profile = project_name, profile
        self.settings = self._app_config_settings = load_settings()
        self._app_config_path = str(CONFIG)
        self._measurement_backend, self._measurement_basis = 'apple_powermetrics', BASIS
        self._tracking_mode = BASIS
        self._measure_power_secs = self.settings['sample_interval_ms'] / 1000
        self._pue = 1.0
        self._conf = dict(cpu_model=platform.processor() or 'Apple Silicon',
                          cpu_count=psutil.cpu_count(), gpu_model='Apple integrated GPU rail', gpu_count=1,
                          ram_total_size=psutil.virtual_memory().total / 1024 ** 3)
        self._hardware = []
        self.monitor = self.window = self.result = None
        self.started, self.prepare_time_s = False, 0.0
        self.require_ollama = profile == 'inference' and not project_name.startswith('RAG_ret_')
        path = self.output_dir / output_file
        if path.exists() and path.stat().st_size:
            with path.open(encoding='utf-8-sig', newline='') as handle:
                rows = csv.DictReader(handle)
                if 'measurement_basis' not in (rows.fieldnames or []) or any(r['measurement_basis'] != BASIS for r in rows):
                    raise ValueError('This folder contains another energy-measurement method. Use a new run folder; historical files were preserved.')

    def start(self):
        if self.started:
            raise RuntimeError('Tracker already started')
        t0 = time.perf_counter()
        try:
            self.monitor = OllamaPowerMonitor(self.output_dir,
                sample_interval_ms=self.settings['sample_interval_ms'],
                process_interval_ms=self.settings['process_interval_ms'],
                carbon_intensity_g_per_kwh=self.settings['carbon_intensity_g_per_kwh'],
                require_ollama=self.require_ollama, session_tag=self.run_id)
            self.monitor.start()
            self.monitor.calibrate_idle(self.settings['idle_baseline_seconds'], self.settings['idle_settle_seconds'])
            self.window = self.monitor.request(self.run_id, phase=self._app_config_profile)
            self.window.__enter__()
            self.started = True
            from cancellation import _current
            state = _current.get()
            for phase in (state or {}).get('phases', {}).values():
                if phase['tracker'] is self:
                    phase['started'] = time.time()
        except BaseException:
            if self.monitor is not None:
                self.monitor.stop()
            raise
        finally:
            self.prepare_time_s = time.perf_counter() - t0
        return self

    @staticmethod
    def nonnegative(value, scale=1):
        return value / scale if value is not None and math.isfinite(value) and value >= 0 else float('nan')

    def _row(self):
        r = self.result
        duration = r['duration_s']
        row = dict(timestamp=datetime.now(timezone.utc).isoformat(), project_name=self._project_name,
                   run_id=self.run_id, experiment_id=self.run_id, duration=duration,
                   emissions=self.nonnegative(r['emissions_g'], 1000),
                   energy_consumed=self.nonnegative(r['energy_kwh']), ram_power=float('nan'), ram_energy=float('nan'),
                   measurement_backend=self._measurement_backend, measurement_basis=BASIS,
                   carbon_intensity_g_per_kwh=self.settings['carbon_intensity_g_per_kwh'],
                   tracking_mode=BASIS, os=platform.platform(), python_version=platform.python_version(),
                   **self._conf, attribution_status=r['attribution_status'], powermetrics_coverage_ratio=r['coverage_ratio'])
        for part in ('cpu', 'gpu', 'ane'):
            row[part + '_power'] = self.nonnegative(r['raw_' + part + '_energy_j'], duration) if duration > 0 else float('nan')
            row[part + '_energy'] = self.nonnegative(r['incremental_' + part + '_energy_j_signed'], 3600000)
            row[part + '_energy_j_signed'] = r['incremental_' + part + '_energy_j_signed']
        row.update(raw_soc_energy_j=r['raw_soc_energy_j'], incremental_soc_energy_j_signed=r['incremental_soc_energy_j_signed'],
                   incremental_soc_energy_j=r['incremental_soc_energy_j'],
                   python_cpu_delta_s=r.get('python_cpu_delta_s'), ollama_cpu_delta_s=r.get('ollama_cpu_delta_s'),
                   python_ollama_cpu_share=r.get('python_ollama_cpu_share'), background_busy_cpu_s=r.get('background_busy_cpu_s'),
                   python_rss_bytes_end=r.get('python_rss_bytes_end'), ollama_rss_bytes_end=r.get('ollama_rss_bytes_end'),
                   ollama_pids=json.dumps(r.get('ollama_pids')), measurement_prepare_time_s=self.prepare_time_s)
        return row

    def stop(self):
        if not self.started:
            return float('nan')
        try:
            self.window.__exit__(None, None, None)
            self.result = self.window.result
            row = self._row()
            path = self.output_dir / self.output_file
            exists = path.exists() and path.stat().st_size > 0
            fields = list(row)
            if exists:
                with path.open(encoding='utf-8-sig', newline='') as handle:
                    fields = next(csv.reader(handle))
                if set(fields) != set(row):
                    raise ValueError('Incompatible measurement CSV columns; start a new run folder')
            with path.open('a', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fields)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            return row['emissions']
        finally:
            self.started = False
            self.monitor.stop()
