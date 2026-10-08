"""Missing-safe measurements and runtime evidence for local batch reports."""
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


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


def stop_tracker(tracker, directory):
    """Preserve failure as missing and save evidence from the actual tracker instance."""
    error = None
    try:
        result = finite(tracker.stop())
    except Exception as exc:
        result = float('nan')
        error = f'{type(exc).__name__}: {exc}'
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
            'run_id': str(getattr(tracker, '_run_id', 'unknown')),
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
