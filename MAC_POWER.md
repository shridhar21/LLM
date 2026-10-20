# Platform-aware power measurement

Windows and Linux continue using the existing CodeCarbon profiles in
`codecarbon_config.json`. Native Apple Silicon macOS (`Darwin`, `arm64`) uses
the custom powermetrics adapter by default. Intel Macs and Python under Rosetta
use CodeCarbon in auto mode. `LLM_MAC_POWER_BACKEND=powermetrics` explicitly
requests the Apple adapter; `codecarbon` explicitly selects the legacy method.
These controls are ignored on Windows/Linux. Mac initialization errors stop
the run with a setup message; there is no silent switch to CPU-load estimates.

The default auto choice is in `mac_power_config.json`. Only a Mac loads it.
Sampling defaults are 250 ms power, 100 ms process activity, 5 s idle baseline
and 1 s settling. Override with `POWER_SAMPLE_INTERVAL_MS`,
`PROCESS_SAMPLE_INTERVAL_MS`, `POWER_IDLE_BASELINE_SECONDS` and
`POWER_IDLE_SETTLE_SECONDS`. Baseline acquisition occurs before each measured
phase and adds real runtime. It is excluded from the recorded query latency;
tracker-stop overhead may remain in wall-clock latency. Ollama's own timings
are saved separately in nanoseconds. Do not compare those with seconds directly.

## Mac setup

Use a native Python installation with psutil (`python -m pip install psutil`).
The monitor needs permission to run `/usr/bin/powermetrics` without an
interactive password prompt. Test in the Mac terminal:

```bash
sudo -n /usr/bin/powermetrics --samplers tasks,cpu_power,gpu_power --show-process-energy --show-process-gpu --show-process-coalition --format plist --sample-rate 250 --sample-count 1
```

If permission is denied, an administrator can configure narrowly scoped
passwordless access to `/usr/bin/powermetrics` using `sudo visudo`. The application
does not change sudoers or run the full Python project as root. Supported
powermetrics fields and flags must be verified on the target macOS version.

For inference, the monitor finds the Ollama server listening on the configured
local port. Set `OLLAMA_SERVER_PID` when needed; it must match that server.
The optional `python start_dedicated_ollama.py` starts a server on port 11435
on any OS and prints settings for the second experiment terminal. Keep its
terminal open. Model selection, warm-up, generation and follow-up capacity checks
all use `OLLAMA_URL`, then `OLLAMA_HOST`, then `http://127.0.0.1:11434`.
Only local addresses are accepted. Retrieval and indexing do not require an
Ollama server to be running for their measurement phase.

To calculate Mac emissions, provide the study's documented grid factor:

```bash
export POWER_CARBON_INTENSITY_G_PER_KWH='YOUR_NUMERIC_VALUE'
```

Otherwise energy is recorded but emissions remain unavailable. No country
factor is silently copied from CodeCarbon. The config file also accepts this
factor as a number. Existing historical runs are not modified.

## Meaning of the results

Mac primary energy is the SoC rail energy in the request window minus a
model-resident idle baseline. CPU/GPU/ANE rail fields are retained when exposed.
RAM **energy** remains unavailable; process RAM **usage** (RSS) is recorded as
evidence. The reported SoC total must not be interpreted as CPU + GPU + RAM.
Component values can differ from the SoC total and ANE may be unavailable.
Neither platform measures exact Ollama-only or wall-socket energy.

Raw `.pliststream` output, stderr logs, parsed power/process JSONL samples,
baseline JSON, per-phase request CSV and measurement metadata are saved inside
each new run folder. Coverage below 95% and negative incremental total energy
are unavailable in primary energy fields; signed raw diagnostics remain saved.
Background CPU share below 90% and missing process attribution are flagged.
Those flags are evidence for review, not a guarantee that GPU energy belongs
only to Ollama. Time alignment uses sample receipt times and boundary overlap,
not hardware clock synchronization. The parser retains missing ANE as missing.

Reports display the measurement backend/scope. RAG's modular workbook adds an
Apple evidence sheet. Comparisons, dashboard selections and table exports reject
mixing `idle_subtracted_apple_soc` with historical `codecarbon_machine` totals.
The old datasets and dissertation comparisons keep their original CodeCarbon
meaning. Use new run folders for the new backend. The existing normal/RAG menus,
question selection and follow-up-safe shuffling remain in place.
