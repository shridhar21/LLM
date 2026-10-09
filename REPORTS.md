# Emission reports

## Comparing batch reports

Run `python compare_reports.py`, choose at least two folder numbers separated by
commas, and choose a reference run. Alternatively:
`python compare_reports.py folder_name_1 folder_name_2 --reference 1`.
Names are resolved under `emissions_reports`; explicit folder paths also work.
Each comparison creates `comparison_reports/comparison_<timestamp>/comparison.pdf`
and `aligned_comparison.csv`. PDF support uses `requirements-report.txt`.

Matching uses exact question text only (including whitespace, case and punctuation).
Duplicate texts, differing question sets and mixed normal/RAG pipelines are rejected.
Different models are allowed and flagged as a confounding difference. The report
compares recorded metadata, coverage, observed costs, per-question changes and
execution-order cumulative emissions. Partial data never qualifies for an overall
efficiency ranking. Historical metadata omissions are explicit; missing configuration
cannot establish that shuffling was the only experimental difference.
The aligned question comparison also reports each question's mean emissions across
the selected folders, with the number of recorded readings shown; missing readings
are excluded. This is a per-question mean, not a mean of batch totals. The same mean
and contributor counts are included in `aligned_comparison.csv`.

## Question selection and shuffling

Both `python normal_llm.py` and `python run.py` use `questionbank.xlsx`.
Entire-set batches include all sheets without a topic-selection prompt and offer
11 execution-order options: eight hierarchical combinations (including original),
two free within-topic combinations, and global free shuffle.
Range batches first select a topic, display serial-number/question-type mapping,
and select an inclusive serial range. Their ordering menu shows only distinct,
effective options. Single-question runs also select a topic and serial number.

Hierarchical shuffles group by topic sheet and question type. Follow-Up 1-5 form
one group; their order changes only when questions within that group are shuffled.
Every shuffled batch uses a recorded random seed. To reproduce its order in Python,
use `question_order.arrange(selected_frame, mode, seed)` on the same selected data.
`question_order.json` stores the chosen mode, seed and planned sequence. Each
answer row stores topic, original serial, question type and execution position.
The optional PDF includes ordering information and an execution-to-question mapping.

## Measurement reliability

New inference runs explicitly request CodeCarbon machine scope so supported local
hardware activity includes the separate Ollama process. This includes background
activity and is not a wall-socket measurement or isolated model measurement.
`codecarbon_config.json` is the source of the tracker settings: normal LLM and RAG
inference use the `inference` profile; RAG indexing uses `rag_indexing`. The indexing
profile intentionally leaves `tracking_mode` and `save_to_api` to CodeCarbon's
installed-version defaults. Each inference session also records the config profile
and settings snapshot in its metadata; per-batch PDFs and comparison metadata tables
display the recorded profile and settings.
`measurement_metadata.json` records each session's effective scope, tracker version,
hardware/backend evidence, sampling interval, country, PUE and stop failures.
Unknown backends remain unverified. Actual coverage must be checked on the machine
running the experiment; the PDF generator does not infer it from its own machine.

Missing/non-finite inference measurements stay blank in CSV output. RAG totals
require both phases. Summary total costs are blank when incomplete; coverage and
observed-sum columns preserve available measurements. PDF partial sums are labelled.
Raw telemetry is matched to the current tracker run ID instead of reusing old rows.
Historical reports without metadata are explicitly labelled unverified in the PDF.

## Optional PDF report

Install PDF support with `python -m pip install -r requirements-report.txt`, then run
`python generate_report.py`. Type an exact batch folder name from the displayed list.
One `report.pdf` is generated in that folder; replacement requires confirmation.
No PDFs are generated automatically. Both current and historical batch filenames
are supported. Manual queries and indexing folders are excluded.

The PDF includes run metadata, measured cost totals, sample statistics, per-question
plots, distributions, hardware energy and RAG phase totals, and data-quality counts.
Charts use per-question bars (points for large batches), a median reference line,
labelled histogram bins, and horizontal hardware/phase comparisons. Red marks
failed requests, X marks missing readings, and amber marks partial component sums.
Statistics exclude failed/unknown requests and missing values; cost totals include
all available measurements. Existing zero defaults cannot be recovered as missing.

New runs use a single CSV per report purpose:

- `answers.csv`: query results, appended after each question; no duplicate live-results file or final rewrite.
- `summary.csv`: one batch or indexing summary, inside the existing pipeline-specific timestamped folder.
- `emissions.csv`: raw CodeCarbon measurements, retained for batch and indexing runs.

Manual queries still use `temp_manual_run`, replacing its answers on each run and removing temporary raw telemetry after completion.

Existing historical reports are preserved. The PDF generators also support older per-question report filenames.

Column names, precision, units, model identifiers and calculated totals remain compatible with existing consumers. Kilogram/gram columns and RAG totals are retained for compatibility; no measurement values are rounded or changed by this cleanup.
