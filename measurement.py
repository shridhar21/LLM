"""Missing-safe measurements and runtime evidence for local batch reports."""
import csv
import io
import json
import math
import os
import platform
import re
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from cancellation import safe_finalization

CODECARBON_CONFIG_PATH = Path(__file__).resolve().with_name('codecarbon_config.json')

# CodeCarbon 3.3.0's first CSV row includes these fields; its append writer
# incorrectly drops None-valued cloud_region/longitude/latitude afterwards.
_CSV_330_COLUMNS = ('timestamp project_name run_id experiment_id duration emissions emissions_rate '
                    'cpu_power gpu_power ram_power cpu_energy gpu_energy ram_energy energy_consumed '
                    'water_consumed country_name country_iso_code region cloud_provider cloud_region '
                    'os python_version codecarbon_version cpu_count cpu_model gpu_count gpu_model '
                    'longitude latitude ram_total_size tracking_mode cpu_utilization_percent '
                    'gpu_utilization_percent ram_utilization_percent ram_used_gb on_cloud pue wue').split()


def aligned_emissions_rows(path):
    """Validate rows and restore only the confirmed 3.3.0 missing-field layout."""
    original = Path(path).read_bytes()
    rows = list(csv.reader(io.StringIO(original.decode('utf-8-sig'), newline='')))
    if not rows:
        return original, rows, 0
    header = rows[0]
    repaired = 0
    for index, row in enumerate(rows[1:], 1):
        if len(row) == len(header):
            continue
        if header != _CSV_330_COLUMNS or len(row) != 35:
            raise ValueError(f'{path}: unrecognised CSV layout on record {index}; not repaired')
        candidate = row[:19] + [''] + row[19:26] + ['', ''] + row[26:]
        record = dict(zip(header, candidate))
        try:
            valid = (record['codecarbon_version'] == '3.3.0' and
                     record['os'].startswith('Windows-') and
                     re.fullmatch(r'\d+\.\d+\.\d+', record['python_version']) and
                     int(record['cpu_count']) > 0 and int(record['gpu_count']) >= 0 and
                     float(record['ram_total_size']) > 0 and
                     record['tracking_mode'] in ('machine', 'process') and
                     record['on_cloud'] in ('N', 'Y') and float(record['pue']) >= 1 and
                     float(record['wue']) >= 0 and
                     all(0 <= float(record[k]) <= 100 for k in
                         ('cpu_utilization_percent', 'gpu_utilization_percent', 'ram_utilization_percent')))
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError(f'{path}: record {index} does not match the known alignment bug')
        rows[index] = candidate
        repaired += 1
    return original, rows, repaired


def repair_emissions_csv(path):
    """Insert missing empty fields without changing any recorded value; keep a backup."""
    path = Path(path)
    original, rows, count = aligned_emissions_rows(path)
    if not count:
        return 0
    backup = path.with_name(path.name + '.before_alignment_fix.bak')
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError(f'{backup}: existing backup differs; refusing to replace it')
    else:
        with backup.open('xb') as handle:
            handle.write(original)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, prefix='.csv-alignment-', delete=False) as handle:
            temporary = Path(handle.name)
            csv.writer(handle).writerows(rows)
        if path.read_bytes() != original:
            raise ValueError(f'{path}: changed during repair; refusing to overwrite an active file')
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return count


class _AlignedFileOutput:
    """Keep CodeCarbon's other output behaviours; preserve every CSV field on append."""
    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def out(self, total, delta):
        path = Path(self.delegate.save_file_path)
        if self.delegate.on_csv_write != 'append' or not path.exists() or not path.stat().st_size:
            return self.delegate.out(total, delta)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            header = next(csv.reader(handle))
        values = dict(total.values)
        if len(header) != len(set(header)) or set(header) != set(values):
            return self.delegate.out(total, delta)
        # DictWriter follows the file's header order, retaining blank/None cells.
        with path.open('a', encoding='utf-8', newline='') as handle:
            csv.DictWriter(handle, fieldnames=header).writerow(values)


def load_codecarbon_config(profile):
    """Load one named CodeCarbon configuration from the project config file."""
    try:
        config = json.loads(CODECARBON_CONFIG_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read CodeCarbon configuration {CODECARBON_CONFIG_PATH}: {exc}') from exc
    if not isinstance(config, dict) or not isinstance(config.get(profile), dict):
        raise ValueError(f'CodeCarbon configuration profile {profile!r} is missing or invalid')
    return dict(config[profile])


def make_codecarbon_tracker(profile, project_name, output_dir, output_file='emissions.csv'):
    """Create an offline tracker using the selected project configuration profile."""
    from codecarbon import OfflineEmissionsTracker

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / output_file).exists():
        repair_emissions_csv(output_dir / output_file)
    settings = load_codecarbon_config(profile)
    tracker = OfflineEmissionsTracker(
        project_name=project_name,
        output_dir=str(output_dir),
        output_file=output_file,
        **settings,
    )
    tracker._app_config_profile = profile
    tracker._app_config_settings = settings
    tracker._app_config_path = str(CODECARBON_CONFIG_PATH)
    if isinstance(getattr(tracker, '_output_handlers', None), list):
        from codecarbon.output_methods.file import FileOutput
        tracker._output_handlers = [_AlignedFileOutput(handler) if isinstance(handler, FileOutput)
                                    else handler for handler in tracker._output_handlers]
    return tracker


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def reading(row, key):
    return finite(row.get(key)) if row else float('nan')


def complete_sum(values):
    values = [finite(v) for v in values]
    return sum(values) if values and all(math.isfinite(v) for v in values) else float('nan')


def coverage(values):
    values = [finite(v) for v in values]
    valid = [v for v in values if math.isfinite(v)]
    return {'valid': len(valid), 'expected': len(values), 'complete': len(valid) == len(values) and bool(values),
            'observed_sum': sum(valid) if valid else None}


def tracker_run_id(tracker):
    """Get the current CodeCarbon ID, with compatibility for older tracker versions."""
    for attribute in ('run_id', '_run_id'):
        value = getattr(tracker, attribute, None)
        if value is not None:
            value = str(value).strip()
            if value and value.lower() not in ('unknown', 'none', 'nan'):
                return value
    return None


@safe_finalization
def stop_tracker(tracker, directory):
    """Preserve failure as missing and save evidence from the actual tracker instance."""
    error = None
    try:
        result = finite(tracker.stop())
    except Exception as exc:
        result = float('nan')
        error = f'{type(exc).__name__}: {exc}'
    from cancellation import phase_stopped
    phase_stopped(tracker, result)
    path = Path(directory) / 'measurement_metadata.json'
    try:
        metadata = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'schema_version': 1, 'sessions': []}
        try:
            installed = version('codecarbon')
        except PackageNotFoundError:
            installed = 'unknown'
        conf = getattr(tracker, '_conf', {})
        conf = conf if isinstance(conf, dict) else {}
        resources = getattr(tracker, '_resource_tracker', None)
        backends = {}
        for part in ('cpu', 'gpu', 'ram'):
            backend = getattr(resources, part + '_tracker', None) or getattr(tracker, '_' + part + '_tracker', None)
            backends[part] = str(backend) if backend else 'unknown / not exposed by this version'
        hardware_evidence = []
        for device in getattr(tracker, '_hardware', []):
            hardware_evidence.append({'class': type(device).__name__,
                                      'mode': str(getattr(device, '_mode', 'unknown'))})
        session = {
            'recorded_at': datetime.now(timezone.utc).isoformat(),
            'project_name': str(getattr(tracker, '_project_name', 'unknown')),
            'run_id': tracker_run_id(tracker) or 'unknown',
            'config_file': str(getattr(tracker, '_app_config_path', CODECARBON_CONFIG_PATH)),
            'config_profile': str(getattr(tracker, '_app_config_profile', 'unknown')),
            'configured_settings': getattr(tracker, '_app_config_settings', {}),
            'codecarbon_version': installed, 'os': platform.platform(),
            'requested_scope': 'machine',
            'effective_scope': str(getattr(tracker, '_tracking_mode', conf.get('tracking_mode', 'unknown'))),
            'measure_power_secs': getattr(tracker, '_measure_power_secs', 1),
            'pue': getattr(tracker, '_pue', 1.0), 'country_iso_code': 'IND',
            'backends': backends,
            'hardware_evidence': hardware_evidence,
            'hardware': {key: conf.get(key, 'unknown') for key in ('cpu_model', 'gpu_model', 'gpu_count', 'ram_total_size')},
            'stop_error': error, 'emissions_available': math.isfinite(result),
        }
        metadata['sessions'].append(session)
        metadata['scope_note'] = ('Machine scope includes supported CPU/GPU/RAM activity from local Ollama and other processes; '
                                  'it does not isolate model energy or measure wall-socket whole-system power. '
                                  'Unknown backends are unverified; RAM is an energy estimate, not memory usage. '
                                  'A local endpoint does not prove all accelerator hardware is supported.')
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
        temporary.replace(path)
    except Exception as exc:
        print(f'WARNING: could not record measurement evidence: {exc}')
    return result
