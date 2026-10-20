"""Apple-Silicon power measurement with explicit Ollama/Python attribution evidence.

Rail joules come from powermetrics and are machine/SoC measurements.  Process
CPU time and RSS are attribution evidence; they are not converted into exact
process joules.  Use a dedicated Ollama server and an idle baseline.
"""
from __future__ import annotations

import csv
import json
import math
import os
import plistlib
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import psutil
except ImportError as exc:
    raise RuntimeError("powermetrics_ollama.py requires psutil: pip install psutil") from exc


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)


def _find_number(record: dict, names: tuple[str, ...]) -> Optional[float]:
    wanted = {name.casefold() for name in names}
    for node in _walk(record):
        for key, value in node.items():
            if str(key).casefold() in wanted:
                found = _finite(value)
                if found is not None:
                    return found
    return None


def _timestamp_ns(record: dict, fallback: int) -> int:
    for node in _walk(record):
        for key, value in node.items():
            if str(key).casefold() != "timestamp":
                continue
            if hasattr(value, "timestamp"):
                try:
                    return int(value.timestamp() * 1_000_000_000)
                except (OverflowError, OSError, ValueError):
                    pass
            number = _finite(value)
            if number is not None:
                if number > 1e17:
                    return int(number)
                if number > 1e14:
                    return int(number * 1_000)
                if number > 1e11:
                    return int(number * 1_000_000)
                if number > 1e8:
                    return int(number * 1_000_000_000)
    return fallback


def _power_sample(record: dict, received_wall_ns: int) -> Optional[dict]:
    elapsed_ns = _find_number(record, ("elapsed_ns", "elapsed nanoseconds"))
    # Process/coalition proxy values must never masquerade as aggregate rail watts.
    aggregate = next((record[k] for k in ('processor', 'power') if isinstance(record.get(k), dict)), record)
    rails = {str(k).casefold(): _finite(v) for k, v in aggregate.items() if not isinstance(v, (dict, list))}
    cpu_mw = rails.get('cpu_power', rails.get('cpu power'))
    gpu_mw = rails.get('gpu_power', rails.get('gpu power'))
    ane_mw = rails.get('ane_power', rails.get('ane power'))
    combined_mw = rails.get('combined_power', rails.get('combined power', rails.get('package_power')))
    if all(value is None for value in (cpu_mw, gpu_mw, ane_mw, combined_mw)):
        return None
    end_ns = received_wall_ns  # receipt time is on the same wall-clock used by request boundaries
    if elapsed_ns is None or elapsed_ns <= 0:
        elapsed_ns = None
    return {
        "end_wall_ns": end_ns,
        "received_wall_ns": received_wall_ns,
        "elapsed_ns": int(elapsed_ns) if elapsed_ns is not None else None,
        "cpu_w": cpu_mw / 1000.0 if cpu_mw is not None else None,
        "gpu_w": gpu_mw / 1000.0 if gpu_mw is not None else None,
        "ane_w": ane_mw / 1000.0 if ane_mw is not None else None,
        "soc_w": combined_mw / 1000.0 if combined_mw is not None else None,
    }


def _busy_cpu_seconds(times) -> float:
    fields = ("user", "nice", "system", "irq", "softirq", "steal")
    return sum(float(getattr(times, field, 0.0)) for field in fields)


def _process_total(proc) -> Optional[float]:
    try:
        value = proc.cpu_times()
        return float(value.user + value.system)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def _descendants(root_pid: Optional[int]) -> list:
    if root_pid is None:
        return []
    try:
        root = psutil.Process(root_pid)
        return [root] + root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return []


def _discover_ollama_roots() -> tuple[list[int], str]:
    from urllib.parse import urlsplit
    from cancellation import ollama_base_url
    port = urlsplit(ollama_base_url()).port
    configured = os.getenv("OLLAMA_SERVER_PID", "").strip()
    if configured:
        try:
            target = int(configured)
        except ValueError:
            raise RuntimeError("OLLAMA_SERVER_PID must be an integer PID")
        if target <= 0:
            raise RuntimeError('OLLAMA_SERVER_PID must be a positive PID')
    else:
        target = None
    roots = []
    for proc in psutil.process_iter(("pid", "name", "cmdline", "ppid")):
        try:
            name = (proc.info.get("name") or "").casefold()
            command = " ".join(proc.info.get("cmdline") or []).casefold()
            if (target is None or proc.pid == target) and ('ollama' in name and 'serve' in command):
                if any(c.status == psutil.CONN_LISTEN and c.laddr.port == port
                       for c in proc.net_connections(kind='inet')):
                    roots.append((proc.info["pid"], proc.info.get("ppid") or 0))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    root_ids = {pid for pid, _ in roots}
    top = sorted(pid for pid, parent in roots if parent not in root_ids)
    return top, f'local listening port {port}' + (' and explicit PID' if target else '')


@dataclass
class RequestWindow:
    monitor: "OllamaPowerMonitor"
    request_id: str
    phase: str
    start_wall_ns: int = 0
    end_wall_ns: int = 0
    result: Optional[dict] = None

    def __enter__(self):
        self.start_wall_ns = time.time_ns()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end_wall_ns = time.time_ns()
        self.result = self.monitor.finish_window(self)
        return False


class OllamaPowerMonitor:
    """Continuously sample SoC rails and collect process-attribution evidence."""

    def __init__(self, output_dir, sample_interval_ms=250, process_interval_ms=100,
                 carbon_intensity_g_per_kwh=None, require_ollama=True, session_tag=None):
        if sys.platform != "darwin":
            raise RuntimeError("powermetrics measurement requires macOS")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_interval_ms = int(sample_interval_ms)
        self.process_interval_ms = int(process_interval_ms)
        if self.sample_interval_ms < 100:
            raise ValueError("Use a powermetrics interval of at least 100 ms")
        if self.process_interval_ms < 10:
            raise ValueError('Use a process interval of at least 10 ms')
        supplied = carbon_intensity_g_per_kwh
        if supplied is None:
            supplied = os.getenv("POWER_CARBON_INTENSITY_G_PER_KWH", "")
        self.carbon_intensity = _finite(supplied)
        if supplied not in (None, '') and (self.carbon_intensity is None or self.carbon_intensity < 0):
            raise ValueError('Carbon intensity must be a finite, non-negative g CO2eq/kWh value')
        self.require_ollama = bool(require_ollama)
        self.session_tag = str(session_tag or f"session-{time.time_ns()}")
        self.power_samples: list[dict] = []
        self.process_samples: list[dict] = []
        self.baseline = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        self._proc = None
        self._stderr_handle = None
        self._raw_handle = None
        self._power_jsonl = None
        self._process_jsonl = None
        self._request_csv = self.output_dir / "powermetrics_requests.csv"
        self.ollama_roots, self.ollama_pid_source = _discover_ollama_roots() if self.require_ollama else ([], 'not required for local retrieval/indexing')
        if self.require_ollama and len(self.ollama_roots) != 1:
            raise RuntimeError(
                "Expected exactly one dedicated Ollama server. Set OLLAMA_SERVER_PID to its PID; "
                f"discovered {self.ollama_roots or 'none'}."
            )

    def _command(self):
        executable = "/usr/bin/powermetrics"
        if not Path(executable).exists():
            raise RuntimeError("/usr/bin/powermetrics was not found")
        command = [executable, "--samplers", "tasks,cpu_power,gpu_power",
                   "--show-process-energy", "--show-process-gpu", "--show-process-coalition",
                   "--format", "plist", "--sample-rate", str(self.sample_interval_ms),
                   "--sample-count", "-1", "--buffer-size", "0"]
        return command if os.geteuid() == 0 else ["sudo", "-n", *command]

    def start(self):
        if self._proc is not None:
            return self
        try:
            self._stderr_handle = (self.output_dir / f"powermetrics_{self.session_tag}.stderr.log").open("wb")
            self._raw_handle = (self.output_dir / f"powermetrics_{self.session_tag}.pliststream").open("wb")
            self._power_jsonl = (self.output_dir / "powermetrics_samples.jsonl").open("a", encoding="utf-8")
            self._process_jsonl = (self.output_dir / "process_attribution_samples.jsonl").open("a", encoding="utf-8")
            self._proc = subprocess.Popen(self._command(), stdout=subprocess.PIPE,
                                          stderr=self._stderr_handle, bufsize=0)
        except BaseException:
            self._close_files()
            raise
        self._threads = [
            threading.Thread(target=self._read_power, name="powermetrics-reader", daemon=True),
            threading.Thread(target=self._read_processes, name="process-attribution", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        try:
            deadline = time.monotonic() + max(5.0, self.sample_interval_ms / 1000 * 5)
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    raise RuntimeError('powermetrics exited; check its stderr log and sudo -n permissions. No CPU-load fallback was used.')
                with self._lock:
                    if self.power_samples:
                        return self
                time.sleep(0.05)
            raise RuntimeError('No usable CPU/GPU/SoC samples. Inspect the raw stream for this macOS version.')
        except BaseException:
            self.stop()
            raise

    def _read_power(self):
        buffer = bytearray()
        stream = self._proc.stdout
        while not self._stop.is_set():
            block = stream.read(65536)
            if not block:
                break
            self._raw_handle.write(block)
            self._raw_handle.flush()
            buffer.extend(block)
            while b'</plist>' in buffer:
                end = buffer.index(b'</plist>') + len(b'</plist>')
                chunk, buffer = bytes(buffer[:end]).strip(b'\x00 \r\n'), buffer[end:]
                if not chunk.strip():
                    continue
                try:
                    record = plistlib.loads(bytes(chunk))
                    sample = _power_sample(record, time.time_ns())
                except Exception as exc:
                    self._stderr_handle.write(f"\nPLIST parse error: {exc}\n".encode())
                    self._stderr_handle.flush()
                    continue
                if sample is None:
                    continue
                sample["session_tag"] = self.session_tag
                if sample["elapsed_ns"] is None:
                    sample["elapsed_ns"] = self.sample_interval_ms * 1_000_000
                if sample["soc_w"] is None:
                    available = [sample[key] for key in ("cpu_w", "gpu_w", "ane_w") if sample[key] is not None]
                    sample["soc_w"] = sum(available) if len(available) == 3 else None
                if any(sample[key] is None or sample[key] < 0 for key in ('cpu_w', 'gpu_w', 'soc_w')):
                    continue
                with self._lock:
                    self.power_samples.append(sample)
                self._power_jsonl.write(json.dumps(sample, separators=(",", ":")) + "\n")
                self._power_jsonl.flush()

    def _target_snapshot(self):
        python_procs = _descendants(os.getpid())
        excluded = set()
        if self._proc is not None:
            excluded = {proc.pid for proc in _descendants(self._proc.pid)}
        python_procs = [proc for proc in python_procs if proc.pid not in excluded]
        ollama_procs = []
        seen = set()
        for root in self.ollama_roots:
            for proc in _descendants(root):
                if proc.pid not in seen:
                    seen.add(proc.pid)
                    ollama_procs.append(proc)
        python_procs = [proc for proc in python_procs if proc.pid not in seen]
        def totals(processes):
            cpu = 0.0
            rss = 0
            pids = []
            for proc in processes:
                value = _process_total(proc)
                if value is None:
                    continue
                cpu += value
                pids.append(proc.pid)
                try:
                    rss += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            return cpu, rss, sorted(pids)
        python_cpu, python_rss, python_pids = totals(python_procs)
        ollama_cpu, ollama_rss, ollama_pids = totals(ollama_procs)
        return {
            "wall_ns": time.time_ns(),
            "session_tag": self.session_tag,
            "system_busy_cpu_s": _busy_cpu_seconds(psutil.cpu_times()),
            "python_cpu_s": python_cpu,
            "python_rss_bytes": python_rss,
            "python_pids": python_pids,
            "ollama_cpu_s": ollama_cpu,
            "ollama_rss_bytes": ollama_rss,
            "ollama_pids": ollama_pids,
        }

    def _read_processes(self):
        while not self._stop.is_set():
            sample = self._target_snapshot()
            with self._lock:
                self.process_samples.append(sample)
            self._process_jsonl.write(json.dumps(sample, separators=(",", ":")) + "\n")
            self._process_jsonl.flush()
            self._stop.wait(self.process_interval_ms / 1000.0)

    def calibrate_idle(self, seconds=10.0, settle_seconds=2.0):
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(settle_seconds) or settle_seconds < 0:
            raise ValueError('Idle baseline must be positive; settling time must be non-negative')
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        start = time.time_ns()
        time.sleep(float(seconds))
        end = time.time_ns()
        with self._lock:
            selected = [s for s in self.power_samples if start <= s["end_wall_ns"] <= end]
        minimum = max(3, int(seconds * 1000 / self.sample_interval_ms * 0.5))
        if len(selected) < minimum:
            raise RuntimeError(f"Only {len(selected)} idle samples were captured; need at least {minimum}")
        self.baseline = {}
        for key in ('cpu_w', 'gpu_w', 'ane_w', 'soc_w'):
            values = [s[key] for s in selected if s[key] is not None]
            self.baseline[key] = statistics.median(values) if values else None
        self.baseline.update({"samples": len(selected), "start_wall_ns": start, "end_wall_ns": end,
                              "kind": "model-resident idle"})
        (self.output_dir / f"powermetrics_baseline_{self.session_tag}.json").write_text(
            json.dumps(self.baseline, indent=2), encoding="utf-8")
        return dict(self.baseline)

    def request(self, request_id, phase="generation"):
        if self.baseline is None:
            raise RuntimeError("Call calibrate_idle() after workload warm-up and before measured requests")
        return RequestWindow(self, str(request_id), str(phase))

    def _integrate(self, start_ns, end_ns):
        duration_s = (end_ns - start_ns) / 1e9
        with self._lock:
            samples = list(self.power_samples)
        joules = {key: 0.0 for key in ("cpu", "gpu", "ane", "soc")}
        covered = []
        used = 0
        invalid = set()
        previous_right = start_ns
        for sample in sorted(samples, key=lambda s: s['end_wall_ns']):
            sample_end = sample["end_wall_ns"]
            sample_start = sample_end - int(sample["elapsed_ns"])
            left, right = max(start_ns, sample_start, previous_right), min(end_ns, sample_end)
            if right <= left:
                continue
            seconds = (right - left) / 1e9
            for rail in joules:
                power = sample[f"{rail}_w"]
                if power is not None:
                    joules[rail] += power * seconds
                else:
                    invalid.add(rail)
            covered.append((left, right))
            previous_right = right
            used += 1
        covered.sort()
        merged = []
        for left, right in covered:
            if not merged or left > merged[-1][1]:
                merged.append([left, right])
            else:
                merged[-1][1] = max(merged[-1][1], right)
        coverage_s = sum(right - left for left, right in merged) / 1e9
        signed = {}
        for rail in joules:
            baseline = self.baseline.get(f'{rail}_w')
            if rail in invalid or baseline is None or not used:
                joules[rail], signed[rail] = None, None
            else:
                signed[rail] = joules[rail] - baseline * duration_s
        return joules, signed, duration_s, coverage_s, used

    def _process_delta(self, start_ns, end_ns):
        missing = dict.fromkeys(('system_busy_cpu_delta_s', 'python_cpu_delta_s', 'ollama_cpu_delta_s',
                                'python_ollama_cpu_share', 'background_busy_cpu_s', 'python_rss_bytes_end',
                                'ollama_rss_bytes_end', 'ollama_pids', 'python_pids'))
        missing['process_attribution_available'] = False
        with self._lock:
            samples = list(self.process_samples)
        before = [s for s in samples if s["wall_ns"] <= start_ns]
        after = [s for s in samples if s["wall_ns"] >= end_ns]
        first = before[-1] if before else (samples[0] if samples else None)
        last = after[0] if after else (samples[-1] if samples else None)
        if first is None or last is None or last["wall_ns"] <= first["wall_ns"]:
            return missing
        result = {"process_attribution_available": True}
        for key in ("system_busy_cpu_s", "python_cpu_s", "ollama_cpu_s"):
            result[key.replace("_s", "_delta_s")] = max(0.0, last[key] - first[key])
        busy = result["system_busy_cpu_delta_s"]
        workload = result["python_cpu_delta_s"] + result["ollama_cpu_delta_s"]
        result["python_ollama_cpu_share"] = min(1.0, workload / busy) if busy > 0 else None
        result["background_busy_cpu_s"] = max(0.0, busy - workload)
        result["ollama_pids"] = last["ollama_pids"]
        result["python_pids"] = last["python_pids"]
        result["ollama_rss_bytes_end"] = last["ollama_rss_bytes"]
        result["python_rss_bytes_end"] = last["python_rss_bytes"]
        return result

    def finish_window(self, window):
        time.sleep(self.sample_interval_ms / 1000.0 * 1.25)
        raw, signed, duration_s, coverage_s, sample_count = self._integrate(
            window.start_wall_ns, window.end_wall_ns)
        process = self._process_delta(window.start_wall_ns, window.end_wall_ns)
        coverage_ratio = min(1.0, coverage_s / duration_s) if duration_s > 0 else 0.0
        incremental_soc_j = signed['soc'] if coverage_ratio >= .95 and signed['soc'] is not None and signed['soc'] >= 0 else None
        energy_kwh = incremental_soc_j / 3_600_000.0 if incremental_soc_j is not None else None
        emissions_g = (energy_kwh * self.carbon_intensity
                       if self.carbon_intensity is not None and energy_kwh is not None else None)
        cpu_share = process.get("python_ollama_cpu_share")
        status = "CPU/process attribution supported; exact GPU process joules unavailable"
        if coverage_ratio < 0.95:
            status = "insufficient powermetrics coverage"
        elif signed['soc'] is None or signed['soc'] < 0:
            status = 'incremental energy unavailable: missing or below idle baseline'
        elif cpu_share is None:
            status = "process attribution unavailable"
        elif cpu_share < 0.90:
            status = "background CPU contamination detected; exact GPU process joules unavailable"
        result = {
            "request_id": window.request_id,
            "phase": window.phase,
            "start_wall_ns": window.start_wall_ns,
            "end_wall_ns": window.end_wall_ns,
            "duration_s": duration_s,
            "sample_count": sample_count,
            "coverage_ratio": coverage_ratio,
            "raw_cpu_energy_j": raw["cpu"],
            "raw_gpu_energy_j": raw["gpu"],
            "raw_ane_energy_j": raw["ane"],
            "raw_soc_energy_j": raw["soc"],
            "incremental_cpu_energy_j_signed": signed["cpu"],
            "incremental_gpu_energy_j_signed": signed["gpu"],
            "incremental_ane_energy_j_signed": signed["ane"],
            "incremental_soc_energy_j_signed": signed["soc"],
            "incremental_soc_energy_j": incremental_soc_j,
            "energy_kwh": energy_kwh,
            "carbon_intensity_g_per_kwh": self.carbon_intensity,
            "emissions_g": emissions_g,
            "measurement_scope": "idle-subtracted Apple SoC rails; not direct per-process joules",
            "timing_method": "power-sample receipt timestamps; request-boundary overlap; not hardware timestamp synchronization",
            "attribution_status": status,
            "ollama_pid_source": self.ollama_pid_source,
            **process,
        }
        self._append_request(result)
        return result

    def _append_request(self, result):
        exists = self._request_csv.exists() and self._request_csv.stat().st_size > 0
        fields = list(result)
        if exists:
            with self._request_csv.open(encoding='utf-8-sig', newline='') as handle:
                fields = next(csv.reader(handle))
            if set(fields) != set(result):
                raise ValueError('Incompatible powermetrics request CSV; use a new run folder')
        with self._request_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            if not exists:
                writer.writeheader()
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in result.items()})

    def stop(self):
        try:
            self._stop_and_save()
        finally:
            if self._proc is not None and self._proc.stdout is not None:
                self._proc.stdout.close()
            self._close_files()

    def _stop_and_save(self):
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.send_signal(signal.SIGTERM)
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
                self._proc.wait(timeout=5)
        for thread in self._threads:
            thread.join(timeout=2)
        metadata = {
            "schema_version": 1,
            "measurement_scope": "SoC rail energy with model-resident idle subtraction",
            "process_scope_note": "Python/Ollama CPU time supports attribution but does not turn shared rail power into exact process joules.",
            "powermetrics_command": self._command(),
            "sample_interval_ms": self.sample_interval_ms,
            "process_interval_ms": self.process_interval_ms,
            "ollama_root_pids": self.ollama_roots,
            "ollama_pid_source": self.ollama_pid_source,
            "python_pid": os.getpid(),
            "require_ollama": self.require_ollama,
            "session_tag": self.session_tag,
            "baseline": self.baseline,
            "carbon_intensity_g_per_kwh": self.carbon_intensity,
        }
        (self.output_dir / f"powermetrics_metadata_{self.session_tag}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")

    def _close_files(self):
        for name in ("_power_jsonl", "_process_jsonl", "_raw_handle", "_stderr_handle"):
            handle = getattr(self, name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, name, None)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False
