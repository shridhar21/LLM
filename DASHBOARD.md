# Local machine-comparison dashboard

From the project folder:

```powershell
python -m pip install -r requirements-dashboard.txt
python dashboard.py
```

Open **http://127.0.0.1:8501**. Stop with **Ctrl+C**. After code updates, restart
the command and refresh the browser. Use `--port 8502` if necessary.
Dependencies are installed once; charts and scripts are subsequently served
locally. No Ollama server or internet connection is needed to inspect reports.

## Selection flow

Choose one to three parent folders, each containing one machine's emission-run
folders. Leave unused paths empty. Select Normal LLM or RAG, then independently
configure each machine:

1. Questions: all available, original serials (`4, 13-17, 74`), topic sheets,
   or question types across sheets. Within sheets, choose all, follow-ups only,
   or particular question types. Preview the original serials and question text.
2. Models: one, several, or all models with complete selected-question coverage.
3. Shuffling schemes: one, several, or all available schemes.
4. Runs: select the recorded runs, including which repeats to include.

Question identity is exact text; serial numbers come from the current
questionbank. Different full run question sets are acceptable when every chosen
run covers the selected questions. Missing coverage is reported explicitly.
Follow-up selection is analysis of saved results; it does not rerun inference.
Follow-up gaps use the immediately preceding chain turn in the full source run,
even if that preceding turn is outside the analysis selection.

## Averaging

- **No averaging:** every run remains separate.
- **Average across selected shuffling schemes:** one mean for each question,
  model and machine. Never pool questions, models or machines into this mean.

If a scheme has repeated runs, average that question's repeated values within
the scheme first, then average the scheme means with equal scheme weights.
Example: scheme A has 0 and 4 g, scheme B has 10 g; the result is 6 g.
All selected schemes (including original order if selected) contribute.
The tables record scheme count, run count, source folders and valid measurement
counts. A missing contributing measurement makes that metric's mean unavailable;
a recorded zero remains zero. Failed/unknown attempts are labelled and their
recorded costs remain included, with a warning.

For direct cross-machine comparison, selected questions, models and schemes
must match. One averaging mode applies to every participating machine.
Turn off comparison to independently explore different machine selections.
RAG and Normal LLM cannot be mixed in an analysis.

## Interactive graphs

Graphs stack vertically by default; side-by-side layout is optional. Select one
or several graph types for the same data, and choose emissions, energy,
component energy, latency or available RAG phase measurements.

- Execution-order lines: one graph per run with every selected original serial
  labelled at its actual execution position; topic-coloured segments and points.
- Original-order lines, per-question means, points and bars.
- Per-question dot/min-max range comparisons (emissions use 0.025 g ticks).
- Question/run heatmaps and RAG retrieval/generation heatmaps.
- Histograms of successful question costs.
- Total selected-question cost bars, hardware energy stacks, RAG phase stacks,
  and topic/question-type totals.

Models always have separate graph groups. Averaging disables execution-order
graphs because there is no shared execution sequence across shuffled runs.
Means use original questionbank order. Other summaries explicitly sum the
selected questions; in averaging mode, these are sums of per-question means.

Drag to zoom, wheel to zoom, pan or reset using the toolbar, hide traces using
the legend, or enter axis limits below a graph. Matching graphs can synchronize
their axis ranges. Hover shows question identity and recorded context. The
camera button downloads the current graph as PNG. Chart interactions use the
locally served [Plotly library](https://plotly.com/javascript/configuration-options/).

## Tables and exports

Tables include displayed values, raw selected measurements, per-question means,
separate follow-up chain tables, settings/metadata and descriptive statistics.
Statistics describe variation across the selected questions; they are not
confidence intervals for a causal shuffle effect.

CSV exports contain the full selected table (table search is display-only).
Excel includes all selected-data tables. Files are written to unique directories
under `dashboard_outputs`; original report files are read-only.

Existing full-run PDF generators remain under **Downloads & full-run reports**,
available in no-averaging mode. They use full original runs, not the interactive
question filter. Full comparison requires an actual original-order reference
and identical complete question sets. Custom full-run graphs require one model
per machine. Existing standalone report/inference commands remain available.

This dashboard binds only to localhost. Public hosting would require separate
folder access and authentication arrangements.
