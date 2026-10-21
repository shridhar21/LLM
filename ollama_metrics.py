"""Ollama request metrics and sampled process memory.

No process-exclusive GPU or DRAM energy is claimed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import warnings
from contextvars import ContextVar

_STATE = ContextVar("ollama_metrics_state", default=None)
_LAST = ContextVar("ollama_metrics_last", default=None)
_LOCK = threading.RLock()
_VERSION = None

COUNTS = (
    "prompt_eval_count",
    "prompt_eval_cached_count",
    "eval_count",
)
DURATIONS = (
    "load_duration",
    "total_duration",
    "prompt_eval_duration",
    "eval_duration",
)


def base_url():
    from cancellation import ollama_base_url
    return ollama_base_url()


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def ratio(a, b):
    a, b = number(a), number(b)
    return a / b if a is not None and b is not None and b > 0 else None


def configure_session(folder):
    """Fetch version outside the per-question measurement window."""
    global _VERSION
    import requests
    _LAST.set(None)

    try:
        response = requests.get(base_url() + "/api/version", timeout=3)
        response.raise_for_status()
        _VERSION = response.json().get("version")
    except Exception:
        _VERSION = None


def cell(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def append_csv(path, row):
    """Expand an existing CSV header safely when adding metric columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: cell(v) for k, v in row.items()}

    with _LOCK:
        existing, fields = [], []
        if path.exists() and path.stat().st_size:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                existing = list(reader)
            if len(set(fields)) != len(fields) or any(None in item or any(v is None for v in item.values()) for item in existing):
                raise ValueError(f"Malformed CSV: {path}")

        extra = [k for k in row if k not in fields]
        if fields and not extra:
            with path.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fields).writerow(row)
            return

        fields += extra
        fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fields)
                writer.writeheader()
                writer.writerows(existing)
                writer.writerow(row)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def normalize(final):
    final = final or {}
    out = {}
    for key in COUNTS + DURATIONS:
        value = final.get(key)
        out[key] = value if type(value) is int and value >= 0 else None

    for key in DURATIONS:
        out[key + "_ns"] = out[key]
        out[key + "_s"] = (
            out[key] / 1e9 if out[key] is not None else None
        )

    p = out["prompt_eval_count"]
    c = out["prompt_eval_cached_count"]
    e = out["eval_count"]
    cache_valid = p is not None and c is not None and c <= p
    uncached = p - c if cache_valid else None

    out.update(
        input_tokens=p,
        input_tokens_cached=c,
        output_tokens=e,
        uncached_prompt_tokens=uncached,
        cached_prompt_count_available=c is not None,
        cache_count_consistency=(
            "unavailable" if p is None or c is None
            else "ok" if cache_valid else "cached_exceeds_prompt"
        ),
        prompt_cache_hit_ratio=ratio(c, p) if cache_valid else None,
        output_tokens_per_s=ratio(e, out["eval_duration_s"]),
        uncached_prompt_tokens_per_s=ratio(
            uncached, out["prompt_eval_duration_s"]
        ),
        effective_prompt_tokens_per_s=ratio(
            p, out["prompt_eval_duration_s"]
        ),
        total_reported_tokens=(
            p + e if p is not None and e is not None else None
        ),
        done=final.get("done") is True,
        done_reason=final.get("done_reason"),
        response_truncated=final.get("done_reason") == "length",
    )
    return out


class MemorySampler:
    """RSS/optional USS for Python and one explicitly selected Ollama tree."""

    def __init__(self, request_id):
        import psutil

        self.psutil = psutil
        self.request_id = request_id
        self.interval = max(
            0.05, float(os.getenv("METRICS_MEMORY_INTERVAL_MS", "250")) / 1000
        )
        self.capture_uss = os.getenv("METRICS_CAPTURE_USS", "0") == "1"
        self.rows = []
        self.errors = set()
        self.event = threading.Event()
        self.thread = None
        self.root = None

        try:
            from urllib.parse import urlsplit
            port = urlsplit(base_url()).port
            selected = os.getenv('OLLAMA_SERVER_PID')
            matches = []
            for candidate in psutil.process_iter(('pid', 'name', 'cmdline')):
                try:
                    if selected and candidate.pid != int(selected):
                        continue
                    if 'ollama' not in (candidate.info.get('name') or '').lower():
                        continue
                    if any(c.status == psutil.CONN_LISTEN and c.laddr.port == port
                           for c in candidate.net_connections(kind='inet')):
                        matches.append(candidate.pid)
                except psutil.Error:
                    continue
            if len(matches) != 1:
                raise ValueError('Cannot identify one Ollama server at the configured port; set OLLAMA_SERVER_PID if needed')
            pid = matches[0]
            process = psutil.Process(pid)
            if "ollama" not in process.name().lower():
                raise ValueError("Selected PID is not an Ollama process")
            self.root = (pid, process.create_time())
        except (KeyError, ValueError, psutil.Error) as exc:
            self.errors.add("Ollama root unavailable: " + str(exc))

    def tree(self, process):
        try:
            processes = [process] + process.children(recursive=True)
            return list({p.pid: p for p in processes}.values())
        except self.psutil.Error as exc:
            self.errors.add("Process tree: " + str(exc))
            return []

    def group(self, processes):
        rss, uss, details = 0, 0, []
        private_complete = self.capture_uss
        complete = True

        for process in processes:
            try:
                info = process.memory_info()
                private = None
                if self.capture_uss:
                    try:
                        method = (
                            getattr(process, "memory_footprint", None)
                            or process.memory_full_info
                        )
                        private = getattr(method(), "uss", None)
                    except (self.psutil.Error, AttributeError):
                        pass
                rss += info.rss
                if private is None:
                    private_complete = False
                else:
                    uss += private

                details.append({
                    "pid": process.pid,
                    "create_time": process.create_time(),
                    "name": process.name(),
                    "rss_bytes": info.rss,
                    "uss_bytes": private,
                })
            except self.psutil.Error:
                complete = False
                self.errors.add("Process disappeared or access denied")

        return {
            "rss_bytes": rss if complete and details else None,
            "uss_bytes": uss if complete and private_complete and details else None,
            "processes": details,
        }

    def sample(self):
        psutil = self.psutil
        start = time.monotonic_ns()
        ollama = []

        if self.root:
            try:
                root = psutil.Process(self.root[0])
                if root.create_time() != self.root[1]:
                    raise ValueError("Ollama PID was reused")
                ollama = self.tree(root)
            except (psutil.Error, ValueError):
                self.errors.add("Ollama root missing or PID reused")

        excluded = {p.pid for p in ollama}
        python = []
        for process in self.tree(psutil.Process(os.getpid())):
            try:
                name = process.name().lower()
                if (
                    process.pid not in excluded
                    and name not in ("sudo", "powermetrics")
                    and "ollama" not in name
                ):
                    python.append(process)
            except psutil.Error:
                self.errors.add("Unable to inspect child process")

        py, ol = self.group(python), self.group(ollama)
        vm, swap = psutil.virtual_memory(), psutil.swap_memory()
        combined = {}
        for key in ("rss_bytes", "uss_bytes"):
            a, b = py[key], ol[key]
            combined[key] = (
                a + b if a is not None and b is not None else None
            )

        self.rows.append({
            "request_id": self.request_id,
            "monotonic_ns": start,
            "wall_time_ns": time.time_ns(),
            "python": py,
            "ollama": ol,
            "combined": combined,
            "system_available_bytes": vm.available,
            "system_swap_used_bytes": swap.used,
            "sample_overhead_s": (time.monotonic_ns() - start) / 1e9,
        })

    def loop(self):
        while not self.event.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self.errors.add("Sampling failure: " + repr(exc))

    def start(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.event.set()
        if self.thread:
            self.thread.join()
        self.sample()

    def summary(self):
        out = {
            "memory_sample_count": len(self.rows),
            "memory_sampling_interval_ms": self.interval * 1000,
            "memory_uss_requested": self.capture_uss,
            "memory_attribution_status": (
                "warning" if self.errors else "pid_trees_sampled"
            ),
            "memory_warnings": sorted(self.errors),
            "memory_sample_overhead_s": sum(
                r["sample_overhead_s"] for r in self.rows
            ),
            "memory_sampler_measures_ram_energy": False,
            "memory_scope": (
                "Process-tree RSS sums; shared mappings not deduplicated; "
                "not a GPU allocation or DRAM-energy measurement"
            ),
        }

        for group in ("python", "ollama", "combined"):
            for metric in ("rss_bytes", "uss_bytes"):
                values = [
                    (r[group][metric], r["monotonic_ns"])
                    for r in self.rows
                ]
                valid = [v for v, _ in values if v is not None]
                first = values[0][0] if values else None
                last = values[-1][0] if values else None
                peak = max(valid) if valid else None
                integral, covered = 0.0, 0.0

                for (a, ta), (b, tb) in zip(values, values[1:]):
                    if a is not None and b is not None:
                        dt = max(0, tb - ta) / 1e9
                        integral += (a + b) * 0.5 * dt
                        covered += dt

                quantities = {
                    "start": first,
                    "end": last,
                    "peak": peak,
                    "delta": (
                        last - first
                        if first is not None and last is not None else None
                    ),
                    "peak_above_start": (
                        peak - first
                        if peak is not None and first is not None else None
                    ),
                    "mean": integral / covered if covered else None,
                }
                for suffix, value in quantities.items():
                    out[f"{group}_{metric}_{suffix}"] = value
                    mib = metric.replace("_bytes", "_mib")
                    out[f"{group}_{mib}_{suffix}"] = (
                        value / 2**20 if value is not None else None
                    )
                out[f"{group}_{metric}_valid_samples"] = len(valid)
                out[f"{group}_{metric}_covered_s"] = covered

        if self.rows:
            out["system_swap_used_bytes_start"] = self.rows[0]["system_swap_used_bytes"]
            out["system_swap_used_bytes_end"] = self.rows[-1]["system_swap_used_bytes"]
        return out


def query_local_model(prompt, model, messages=None, options=None):
    """Stream text, retaining the final Ollama response and cancellation."""
    import requests
    from cancellation import ModelAnswer

    record = _STATE.get()
    if record is None:
        record = {}

    endpoint = "/api/generate" if messages is None else "/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "15m"),
    }
    payload["prompt" if messages is None else "messages"] = (
        prompt if messages is None else messages
    )
    if options is not None:
        payload["options"] = options

    think = os.getenv("OLLAMA_THINK")
    if think is not None:
        if think.lower() in ("true", "false"):
            payload["think"] = think.lower() == "true"
        elif think in ("low", "medium", "high"):
            payload["think"] = think
        else:
            raise ValueError("Invalid OLLAMA_THINK setting")

    record.update(
        ollama_url=base_url(),
        ollama_endpoint=endpoint,
        ollama_version=_VERSION,
        model_name=model,
        request_options_json=json.dumps(options or {}, sort_keys=True),
        request_think_setting=payload.get("think", "server_default"),
        raw_final_response=None,
        thinking_characters=0,
        client_first_generated_chunk_s=None,
        client_first_visible_chunk_s=None,
    )

    finished, cancelled = threading.Event(), threading.Event()
    resources, result = {}, {}

    def worker():
        start = time.monotonic_ns()
        record["client_http_start_wall_ns"] = time.time_ns()
        try:
            with requests.Session() as session:
                resources["session"] = session
                with session.post(
                    base_url() + endpoint,
                    json=payload,
                    stream=True,
                    timeout=(10, 9000),
                ) as response:
                    resources["response"] = response
                    response.raise_for_status()
                    pieces, last = [], {}

                    for line in response.iter_lines():
                        if cancelled.is_set():
                            return
                        if not line:
                            continue
                        event = json.loads(line)
                        last = event
                        if event.get("error"):
                            raise RuntimeError(event["error"])

                        message = event.get("message") or {}
                        content = event.get(
                            "response", message.get("content", "")
                        ) or ""
                        thinking = event.get(
                            "thinking", message.get("thinking", "")
                        ) or ""
                        elapsed = (time.monotonic_ns() - start) / 1e9

                        if (content or thinking) and record["client_first_generated_chunk_s"] is None:
                            record["client_first_generated_chunk_s"] = elapsed
                        if content and record["client_first_visible_chunk_s"] is None:
                            record["client_first_visible_chunk_s"] = elapsed

                        pieces.append(content)
                        record["thinking_characters"] += len(thinking)
                        if event.get("done") is True:
                            record["raw_final_response"] = event

                    if last.get("done") is not True:
                        raise RuntimeError('Ollama response ended before completion; partial output is not a successful answer')

                    if messages is not None and last.get("done_reason") == "length":
                        raise RuntimeError(
                            "Follow-up response reached its output/context limit; incomplete answer was not added to history."
                        )

                    answer = ModelAnswer("".join(pieces).strip(), metrics=last)
                    answer.request_metrics = record
                    result["answer"] = answer
        except Exception as exc:
            result["error"] = exc
        finally:
            record["client_http_wall_s"] = (
                time.monotonic_ns() - start
            ) / 1e9
            record["client_http_end_wall_ns"] = time.time_ns()
            finished.set()

    threading.Thread(target=worker, daemon=True).start()

    try:
        while not finished.wait(0.1):
            pass
    except KeyboardInterrupt:
        cancelled.set()
        record["client_cancel_requested"] = True
        record["server_cancellation_confirmed"] = False

        def close():
            for key in ("response", "session"):
                try:
                    if resources.get(key) is not None:
                        resources[key].close()
                except Exception:
                    pass

        threading.Thread(target=close, daemon=True).start()
        finished.wait(0.5)
        raise

    if "error" in result:
        raise result["error"]
    return result["answer"]


def measured_generate(generate, *args, tracker, directory, **kwargs):
    from measurement import tracker_run_id
    from cancellation import protect_cleanup

    rid = tracker_run_id(tracker)
    if not rid:
        raise RuntimeError("Missing tracker ID; cannot link metrics safely")

    record = {
        "request_id": str(rid),
        "generation_request_id": str(rid),
        "request_status": "running",
        "measurement_boundary": "conversation.generate",
        "thinking_tokens": None,
    }
    token = _STATE.set(record)
    _LAST.set(record)
    sampler = None

    try:
        sampler = MemorySampler(str(rid)).start()
    except Exception as exc:
        record["memory_attribution_status"] = "unavailable"
        record["memory_warnings"] = [repr(exc)]

    start = time.monotonic_ns()
    record["inference_start_wall_ns"] = time.time_ns()

    try:
        answer = generate(*args, **kwargs)
        if record.get('raw_final_response') is None:
            final = dict(getattr(answer, 'ollama_metrics', {}) or {})
            for key, attr in (('prompt_eval_count', 'input_tokens'), ('eval_count', 'output_tokens')):
                value = getattr(answer, attr, None)
                if key not in final and type(value) is int:
                    final[key] = value
            record['raw_final_response'] = final or None
        record["request_status"] = "ok"
        return answer
    except BaseException as exc:
        record["request_status"] = (
            "cancelled" if isinstance(exc, KeyboardInterrupt) else "error"
        )
        record["request_error"] = repr(exc)
        raise
    finally:
        record["inference_call_wall_s"] = (
            time.monotonic_ns() - start
        ) / 1e9
        record["inference_end_wall_ns"] = time.time_ns()
        _STATE.reset(token)

        with protect_cleanup():
            if sampler:
                try:
                    sampler.stop()
                    record.update(sampler.summary())
                    # Publish after tracker.stop() for ordinary completed requests.
                    record["_memory_samples"] = sampler.rows
                except Exception as exc:
                    record["memory_attribution_status"] = "unavailable"
                    record["memory_warnings"] = [repr(exc)]

            record.update(normalize(record.get("raw_final_response")))

            # Cancellation may bypass the ordinary answer-row publication.
            if record["request_status"] == "cancelled":
                try:
                    samples = record.pop("_memory_samples", [])
                    for sample in samples:
                        append_jsonl(Path(directory) / "request_memory_samples.jsonl", sample)
                    append_jsonl(Path(directory) / "ollama_attempts.jsonl", record)
                except Exception as exc:
                    warnings.warn(f'Additional cancellation diagnostics could not be saved: {exc}')


def last_inference_wall_s():
    return (_LAST.get() or {}).get("inference_call_wall_s", float("nan"))


def persist_metrics(answer, tracker, directory, question_row=None):
    """Called after tracker.stop(); preserves overall/RAG energy columns."""
    from measurement import tracker_run_id

    rid = str(tracker_run_id(tracker))
    source = getattr(answer, "request_metrics", None) or _LAST.get() or {}
    if source.get("request_id") != rid:
        source = {
            "request_id": rid,
            "generation_request_id": rid,
            "request_status": "metrics_unavailable",
        }

    record = dict(source)
    memory_samples = record.pop("_memory_samples", [])
    source.pop("_memory_samples", None)
    directory = Path(directory)

    power_path = directory / "powermetrics_requests.csv"
    matches = []
    if power_path.exists():
        with power_path.open(encoding="utf-8-sig", newline="") as handle:
            matches = [
                r for r in csv.DictReader(handle)
                if (r.get("request_id") or r.get("run_id")) == rid
            ]

    power = matches[0] if len(matches) == 1 else {}
    record["power_match_status"] = (
        "matched_by_id" if len(matches) == 1
        else "missing" if not matches else "duplicate_ids"
    )

    q = question_row or {}
    basis = q.get('measurement_basis') or getattr(tracker, '_measurement_basis', 'codecarbon_machine')
    backend = q.get('measurement_backend') or getattr(tracker, '_measurement_backend', 'codecarbon')
    # A RAG ratio uses generation only, never the retrieval + generation total.
    emissions_field = 'generation_emissions_kg' if 'generation_energy_kwh' in q else 'emissions_kg'
    if not matches and backend == 'codecarbon':
        # Read the exact tracker ID; do not attach the most recent other phase.
        for name in ('emissions.csv', 'temp_emissions.csv'):
            path = directory / name
            if path.is_file():
                with path.open(encoding='utf-8-sig', newline='') as handle:
                    candidates = [r for r in csv.DictReader(handle) if r.get('run_id') == rid]
                if len(candidates) == 1:
                    raw = candidates[0]
                    power = dict(energy_kwh=raw.get('energy_consumed'))
                    kg = number(q.get(emissions_field))
                    power['emissions_g'] = kg * 1000 if kg is not None else None
                    record['power_match_status'] = 'codecarbon_matched_by_id'
                    break
    record.update(measurement_basis=basis, measurement_backend=backend,
                  generation_energy_measurement_basis=basis)
    for key in ("question_id", "execution_position", "topic_sheet", "model_name", "status"):
        if key in q:
            record[key] = str(q[key])

    energy_j = number(power.get("incremental_soc_energy_j"))
    energy_kwh = number(power.get("energy_kwh"))
    if energy_j is None and energy_kwh is not None:
        energy_j = energy_kwh * 3_600_000
    if energy_kwh is None and energy_j is not None:
        energy_kwh = energy_j / 3_600_000

    ci = number(power.get("carbon_intensity_g_per_kwh"))
    record.update(
        generation_energy_j=energy_j,
        generation_incremental_soc_energy_j=(energy_j if backend == 'apple_powermetrics' else None),
        generation_energy_kwh_measured=energy_kwh,
        generation_raw_soc_energy_j=number(power.get("raw_soc_energy_j")),
        generation_incremental_soc_energy_j_signed=number(
            power.get("incremental_soc_energy_j_signed")
        ),
        power_coverage_ratio=number(power.get("coverage_ratio")),
        power_request_duration_s=number(power.get("duration_s")),
        power_attribution_status=power.get("attribution_status"),
        carbon_intensity_g_per_kwh=ci,
        generation_reported_emissions_g=number(power.get("emissions_g")),
        generation_recalculated_emissions_g=(
            energy_kwh * ci
            if energy_kwh is not None and ci is not None else None
        ),
        carbon_factor_unit_warning=ci is not None and 0 < ci < 1,
        request_joules_per_output_token=ratio(
            energy_j, record.get("eval_count")
        ),
        request_joules_per_prompt_token=ratio(
            energy_j, record.get("prompt_eval_count")
        ),
        request_joules_per_total_reported_token=ratio(
            energy_j, record.get("total_reported_tokens")
        ),
        token_energy_scope=(
            "Whole generation-request energy divided by tokens; "
            "NOT separately measured prefill/decode energy"
        ),
    )

    if record["carbon_factor_unit_warning"]:
        warnings.warn(
            "Carbon intensity below 1 g/kWh: verify g/kWh versus kg/kWh. "
            "No automatic correction was applied."
        )

    for sample in memory_samples:
        append_jsonl(directory / "request_memory_samples.jsonl", sample)
    append_jsonl(directory / "ollama_requests.jsonl", record)

    scalar = {
        k: v for k, v in record.items()
        if k != "raw_final_response"
    }
    append_csv(directory / "ollama_requests.csv", scalar)

    protected = {
        "status", "error", "model_name", "question_id",
        "execution_position", "topic_sheet", "input_tokens", "output_tokens",
        "measurement_basis", "measurement_backend",
    }
    return {
        k: cell(v) for k, v in scalar.items() if k not in protected
    }


def persist_optional_metrics(answer, tracker, directory, question_row):
    """Optional diagnostics must not prevent primary answer publication."""
    try:
        return persist_metrics(answer, tracker, directory, question_row)
    except Exception as exc:
        warnings.warn(f'Additional request metrics could not be saved: {exc}')
        return {'request_metrics_error': str(exc)}


def print_brief(row):
    print(
        "Ollama tokens — input:",
        row.get("input_tokens", ""),
        "| cached:", row.get("input_tokens_cached", ""),
        "| output:", row.get("output_tokens", ""),
    )
    print(
        "Ollama seconds — load:",
        row.get("load_duration_s", ""),
        "| prefill:", row.get("prompt_eval_duration_s", ""),
        "| decode:", row.get("eval_duration_s", ""),
        "| total:", row.get("total_duration_s", ""),
    )
    print(
        "Sampled peak RSS MiB — Python:",
        row.get("python_rss_mib_peak", ""),
        "| Ollama:", row.get("ollama_rss_mib_peak", ""),
        "| combined:", row.get("combined_rss_mib_peak", ""),
    )


def report(folder):
    folder = Path(folder)
    with (folder / "ollama_requests.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        all_rows = list(csv.DictReader(handle))

    ids = None
    answers = folder / "answers.csv"
    if answers.exists():
        with answers.open(encoding="utf-8-sig", newline="") as handle:
            current = list(csv.DictReader(handle))
        ids = {
            r.get("generation_request_id") or r.get("request_id")
            for r in current
        }
        ids.discard(None)
        ids.discard("")
        if current and not ids:
            raise SystemExit(
                "answers.csv has no request IDs. Run a NEW request "
                "after installing the update."
            )

    unique = {}
    duplicates = 0
    for row in all_rows:
        rid = row.get("request_id")
        if not rid or (ids is not None and rid not in ids):
            continue
        if rid in unique:
            duplicates += 1
        unique[rid] = row

    rows = list(unique.values())
    fields = [
        "prompt_eval_count", "prompt_eval_cached_count", "eval_count",
        "load_duration_s", "total_duration_s",
        "prompt_eval_duration_s", "eval_duration_s",
        "generation_energy_j",
        "generation_reported_emissions_g",
    ]
    summary = {
        "selected_requests": len(rows),
        "duplicate_rows_last_selected": duplicates,
        "scope": "Generation only; excludes RAG retrieval/indexing",
    }
    for field in fields:
        values = [number(r.get(field)) for r in rows]
        valid = [v for v in values if v is not None]
        summary["observed_sum_" + field] = sum(valid) if valid else None
        summary[field + "_valid_count"] = len(valid)

    # Overwrite only this derived summary, never original evidence.
    target = folder / "metrics_summary.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(summary))
        writer.writeheader()
        writer.writerow({k: cell(v) for k, v in summary.items()})

    lines = [
        "# Request metrics", "",
        "Scope: generation requests selected from current answers by ID.",
        "Totals are observed partial sums; check available-record counts.",
        f"Selected requests: {len(rows)}. Duplicate rows: {duplicates}.", "",
        "| ID | Input | Cached | Output | Load s | Prefill s | Decode s | Energy J |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [row.get("request_id", "")] + [
            row.get(key, "") for key in (
                "prompt_eval_count", "prompt_eval_cached_count", "eval_count",
                "load_duration_s", "prompt_eval_duration_s",
                "eval_duration_s", "generation_energy_j",
            )
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")

    lines += [
        "", "## Interpretation", "",
        "- Blank cached counts mean unavailable, not zero.",
        "- RSS sums can count shared mappings multiple times.",
        "- Peaks are sampled and may miss short-lived allocations.",
        "- Memory footprint is not DRAM energy or GPU allocation.",
        "- Request joules/token is not isolated decode/prefill energy.",
        "- Client, server and power-window durations have different boundaries.",
        "- Historical metrics cannot be recovered from answer text alone.",
        "",
        "Detailed memory, coverage and attribution fields are in "
        "ollama_requests.csv and answers.csv.",
    ]
    (folder / "metrics_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Written:", target, "and", folder / "metrics_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    report(parser.parse_args().report)
