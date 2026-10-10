"""Missing-safe measurements and runtime evidence for local batch reports."""
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from cancellation import safe_finalization

CODECARBON_CONFIG_PATH = Path(__file__).resolve().with_name('codecarbon_config.json')


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
