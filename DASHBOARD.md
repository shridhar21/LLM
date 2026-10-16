# Local machine-comparison dashboard

From the project folder:

```powershell
python -m pip install -r requirements-dashboard.txt
python dashboard.py
```

Open **http://127.0.0.1:8501** in a browser. Stop the server with **Ctrl+C**.
Use `python dashboard.py --port 8502` if the default port is occupied.
The server binds only to this computer. No public hosting, account, internet,
Ollama server or inference is needed to explore existing reports.

Choose three different parent folders, one per PC/laptop. Each parent contains
its own emission-run folders. The parents may be copied onto this computer or
be accessible network folders. They need not use e1-e5 names.

1. Name each machine and choose or paste its parent folder path.
2. Load folders, filter model/pipeline/shuffle scheme, and tick completed runs.
3. Analyse. Questions must have identical exact text across selected runs.
   The local questionbank verifies original identities and labels. Cancelled,
   incomplete and invalid folders are listed under exclusions.
4. Select the graph and question page. Execution plots can match a shuffle
   scheme across the three machines or use individually chosen runs. Missing
   matches are shown as unavailable, never substituted with another scheme.
5. Click a graph for its full-resolution PNG or switch a machine to full-width
   view. Across-machine views use shared scales, units and topic colours.
6. Download tables or PDFs from Reports & downloads.

## Included views

- Execution-order emissions: separate landscape graphs, every question labelled
  by original serial in actual execution order, topic-coloured segments/points.
- Per-question dot/range comparisons: 32 questions per page, original order,
  0.025 g CO2eq tick spacing, three-decimal labels and fixed run colours.
- Average question emissions over the selected runs of each machine.
- Follow-up tables: original serial, each run's position, intervening questions
  outside that chain, and recorded per-turn emissions. First gap is unavailable.
- Overview, per-question energy/latency/emissions, averages, metadata, settings
  and per-question RAG phase measurements. Missing measurements stay missing.
- RAG retrieval/generation emission heatmaps, with a common absolute scale.
- Existing full comparison and individual-run PDFs, generated through the
  existing report engines. The full comparison requires a selected run whose
  actual order matches the questionbank subset; different models are flagged.
  Both report types can also be viewed in three side-by-side PDF panels.

Different seeds under the same shuffle scheme need not yield identical execution
sequences. Selected run names and seeds remain visible beside each graph and in
the PDF run keys. Recorded failed questions are labelled rather than silently
discarded; these are observed costs, not proven causal effects of shuffling.

## Files and exports

`dashboard_outputs` contains generated exports and cached preview files, in
separate uniquely named directories. Original reports are never written to.
Individual PDFs are generated from temporary report copies inside this directory.
Existing terminal pipelines and report commands remain available as before.

The original custom CLI still accepts its e1-e5 workbook. The dashboard adds a
direct-folder adapter, so a run_tables.xlsx export is not needed for the UI.
CSV downloads honour the current table search/question filters; Excel includes
all analysis tables. The full comparison export
also provides its existing aligned CSV. The three-machine-view PDF is deliberately
wide so each panel retains its labels when zoomed.

This is a local personal dashboard. Do not expose the development server to the
public internet. If you later want shared hosting, choose that separately so
folder access and authentication can be designed for that environment.
