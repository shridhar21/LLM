# Emission reports

## Stopping a run safely

Press Ctrl+C during normal LLM, RAG, or indexing execution. Completed answers
remain saved; the active tracker is finalized and an interrupted question is
recorded as `cancelled`, with available partial phase costs. No remaining questions
are started. `run_status.json` records planned, attempted, completed, successful,
failed, cancelled, and unstarted counts, plus the actual attempted sequence.
Cancelled inference runs receive a partial `summary.csv`: observed costs are
separate from blank full-batch totals. Cancellation during warmup records zero
attempts. Ctrl+C during native embedding/search may be delayed until control
returns to Python. Repeated Ctrl+C is suppressed while cancellation cleanup runs.

Ollama responses are streamed internally and buffered into the same complete
answer string; answers are not streamed to the terminal. On cancellation the
connection is closed asynchronously, without stopping the shared Ollama server.
Server-side termination is best-effort, not guaranteed instantaneous.

Generate the PDF as usual: incomplete runs are prominently labelled and cancelled
attempts are excluded from successful-request statistics. Comparison rejects
cancelled runs, even if their last question was the interrupted one. Historical
report formats remain supported.

Index updates are staged before publication. Ctrl+C during the short publication
step is deferred until all index files form a consistent update; publication
failures roll back the previous files. An interrupted build before publication
leaves the existing index unchanged. This protects interruption and normal write
failures, not power loss or concurrent index writers. There is no resume feature.

## Selecting a model for a run

In both `python normal_llm.py` and `python run.py`, choosing a run option
first displays the models installed in local Ollama. Enter a model's number,
then select questions or enter your custom prompt as before. Select a model
again for each new run. That exact model is used for batch warm-up and all
queries, and is recorded through the existing answer/summary/report fields.
There is no hardcoded generation-model fallback. Ollama must be running and
have at least one model installed; otherwise no query run starts. The RAG
embedding model remains `all-MiniLM-L6-v2`. Code calling `query_llm` or
`process_queries` directly must now supply the generation model explicitly.

## Comparing batch reports

RAG-to-RAG PDFs also include a table for every identical question, comparing
retrieval emissions and generation emissions across selected runs in g CO2e.
The final overview before the Question label key contains two aligned heatmaps:
questions down the rows, selected runs across columns, with separate retrieval
and generation panels. All panels/pages share one absolute emissions colour scale;
darker cells mean greater emissions. These values are not reference-run differences.
Large comparisons split into readable question/run panels. Missing phase readings
are N/A, true zeros remain zero, and recorded failed/unknown costs are marked *.
Historical missing phases are never inferred from totals. Normal LLM comparisons,
the comparison CSV schema, and existing cancelled/incomplete-run exclusion remain
unchanged. Regenerate a comparison through the existing selector to see the additions;
historical PDF files are not automatically rewritten.

Run `python compare_reports.py`, choose at least two folder numbers separated by
commas. The reference is chosen automatically from runs whose actual question
sequence matches `questionbank.xlsx` sheet/row order for the selected question set.
Include at least one such run. An explicitly recorded original-order run is
preferred; ties use the first matching folder selected. For range runs, only the
selected questions are checked. Missing/ambiguous workbook texts or the absence
of an original-order run prevent comparison. Alternatively:
`python compare_reports.py folder_name_1 folder_name_2`.
Names are resolved under `emissions_reports`; explicit folder paths also work.
Each comparison creates `comparison_reports/comparison_<timestamp>/comparison.pdf`
and `aligned_comparison.csv`. Install `requirements-report.txt` for PDF generation
and reading the workbook used to verify the reference.

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
The comparison PDF includes each run's recorded shuffling scheme and seed,
side-by-side execution sequences, and an exact question-label key as its last
section. Question labels follow the selected reference.
Arrows and earlier/later annotations show position changes relative to that run;
colours indicate movement rather than emissions performance. Sequence panels repeat
the reference alongside up to two other runs. Missing scheme metadata is labelled
explicitly rather than inferred from the execution order.

## Advanced local RAG retrieval

Install the additional retrieval packages with
`python -m pip install -r requirements-rag.txt`, then run `python rag_setter.py`
once before using `python run.py`. The new index cannot use legacy flat chunks.
No migration or real-corpus rebuild is performed merely by changing the code.

The new assets live in immutable generations under `indexes/rag_faiss/advanced/`.
`CURRENT.json` identifies the complete active generation. Each generation contains
`index.faiss`, `documents.jsonl`, `parents.jsonl`, `faiss_chunk_ids.json`,
`embeddings.npy`, `corpus_meta.csv`, and `index_config.json` (settings and checksums).
The original five flat-index files are preserved. Prior advanced generations are
also retained. Failed or cancelled builds do not switch the active generation.
Later additions/modifications reuse unchanged document embeddings; absent corpus
files are retained, matching the existing no-implicit-deletion behavior.

Heading-aware sections become parent passages (approximately 900 regex-estimated
tokens) and child passages (180-token target, 240-token oversized-paragraph splitting,
one-sentence overlap where it fits). These are not model-tokenizer counts. Child
records retain document, filename, heading, parent and child identities. Parents
are not embedded. Corpus extraction still uses the existing MarkItDown converter.

Normalized MiniLM vectors use FAISS `IndexFlatIP` for cosine similarity. LangChain
`Document` and `BM25Retriever` provide the sparse integration; BM25 is rebuilt
from saved children on load. Dense and positive-scoring sparse candidates (up to
15 each) are fused using RRF with constant 60. MMR selects up to four children
using .75 query relevance and .25 redundancy, not the numerical RRF scores.
Evidence passes if there is a selection and the best dense similarity is at least
.30, or the leading fused candidate has both dense and sparse support. This is
a heuristic, not proof of answer correctness. Unique selected parents supply at
most 12,000 characters of evidence text (excluding prompt instructions and labels).

Evidence prompts include `[SOURCE N]`, filename and heading, and request citations
and evidence-only answers. If evidence is rejected or no parent text is usable,
the original question is sent to the same selected Ollama model without context.
The existing `chunks` column stores the labelled parent context; `retrieved_k`
counts those supplied parent sources, not the number of selected child candidates.
Existing CSV column names and comparison calculations remain unchanged. Retrieval
stays inside the retrieval tracker; prompt construction stays inside generation.
Corpus indexing now uses two non-overlapping measured groups: document preparation,
then embedding/indexing/storage including publication and saved-asset validation.
The indexing summary sums both groups. In indexing summaries, `chunk_size=180` means an approximate
token target and `chunk_overlap=1` means one sentence, not legacy character counts.
Use `rag_setter.legacy_main()` only for explicit legacy-index reproduction; the
normal RAG menu requires the new advanced assets and never silently falls back to
the old L2 pipeline.

The RAG runner now receives the loaded assets directly; its unused separate
`chunks` argument was removed. Python callers use
`process_queries(queries_df, batch_mode, index, embedder, model_name)`.
This does not change the `chunks` column saved in answer reports.

### Modular measurement workbook and RAG PDF details

New RAG batches automatically save `modular_emissions.xlsx` inside their emission
folder when the run completes or is cancelled. Its sheets are **Group measurements**
(two measured rows per attempted question), **Step timings**, **Evidence**, and
**Configuration**. Group rows include exact question text, identity, execution
position, selected model, statuses, CodeCarbon session ID, raw tracker duration,
wall time, emissions in kg CO2e, and total/CPU/GPU/RAM energy in kWh. Numeric
precision is retained, real zeros stay zero, and missing readings stay blank.
Duration and wall time are different: wall time includes tracker lifecycle and
bookkeeping. Hardware coverage limitations still apply.

Indexing runs with actual corpus changes save a separate workbook containing the
two corpus groups plus configuration. Corpus costs are not copied into every
question's totals. Unchanged-corpus checks do not create measured group rows.

`rag_details.json` preserves decisions, timings, source identities and group records
incrementally, so completed/partial evidence survives cancellation and Excel export
failures. The workbook is written after measurements stop. If Excel holds the output
file open, close it and retry with `python rag_reporting.py <run-folder>`; this exports
the saved readings without rerunning models. Normal LLM runs do not use this feature.
Manual RAG queries retain JSON details but do not automatically export the batch workbook.

Generate RAG PDFs through the existing `python generate_report.py` selector. New
sections show evidence-backed/fallback/unrecorded counts and fallback reasons,
per-question retrieval/generation emissions stacks, embedding/FAISS/BM25/fusion-MMR
and gate/parent-expansion timing stacks, an evidence-selection table, successful
RAG-versus-fallback mean costs with valid counts, and the saved retrieval configuration
and index-generation ID. Timings do not imply separately measured step emissions.
Failed/cancelled costs remain visible, but mode-average costs use successful readings
only. Historical missing decisions are marked unrecorded, never inferred from chunks.
Normal LLM PDFs remain unchanged. PDFs are still generated only on request.

## Question selection and shuffling

Both `python normal_llm.py` and `python run.py` use `questionbank.xlsx`.
Entire-set batches include all sheets without a topic-selection prompt and offer
11 execution-order options: eight hierarchical combinations (including original),
two free within-topic combinations, and global free shuffle.
Range batches first select a topic, display serial-number/question-type mapping,
and select an inclusive serial range. Their ordering menu shows only distinct,
effective options. Single-question runs also select a topic and serial number.
The range topic menu also offers **Free range across topic sheets**. Its combined
mapping uses workbook-wide serials in sheet/row order and shows each question's
topic, original sheet serial, and question type. Choose inclusive start/end
workbook serials, then an execution order. Multi-topic ranges support all 11
ordering schemes, with equivalent or ineffective choices removed; ranges within
one topic use the single-topic menu. Results retain original question identities
and add `workbook_position`; `question_order.json` also records the selected bounds.

Hierarchical shuffles group by topic sheet and question type. Shuffling questions
within a question type leaves the follow-up question type in its original order.
Question-type shuffles can move that group while retaining its internal order.
Free shuffles within topics or across the workbook randomly choose among available
questions: only the next turn of each follow-up chain is available. Other questions
and chains may run between its turns, but each chain retains its original turn order
and its separate question/answer history. This is random selection among available
questions, not a claim of uniform sampling of all valid execution permutations.
Independent questions retain the previous rules.
Every shuffled batch uses a recorded random seed. To reproduce its order in Python,
use `question_order.arrange(selected_frame, mode, seed)` on the same selected data.
`question_order.json` stores the chosen mode, seed and planned sequence. Each
answer row stores topic, original serial, question type and execution position.
The optional PDF includes ordering information and an execution-to-question mapping.

## Follow-up conversational history

Both normal LLM and RAG runners recognize contiguous `Follow-Up 1`, `Follow-Up 2`,
etc. within a topic sheet. Follow-Up 1 opens the conversation; each subsequent turn
receives all earlier original questions and successful generated answers from that
chain. Other chains and independent questions receive none of that history. Every
run starts fresh. Ambiguous labels, gaps, or a chain without its opening turn are
rejected rather than inferred from question wording. The workbook is not modified.

Follow-up generation uses streamed `/api/chat`; independent questions and warm-up
continue using `/api/generate`. RAG adds the current turn's retrieved evidence to
the current user message; previous evidence passages are not copied into history.
Retrieval includes the current question, the opening subject and the latest two
question/answer pairs without an additional LLM rewriting call. The MiniLM encoder
has its own shorter input window. Where tokenizer information is available, its
retrieval-only truncation and actual supplied retrieval text are recorded. Full
generation history is never silently truncated or summarized.

Partial single/range selections ask whether to include and execute missing earlier
turns, explicitly recording their costs, or cancel. A chain with a failed prior
answer cannot proceed: dependent attempts are marked errors without an Ollama
generation request. Unrelated questions can still run. Cancellation preserves prior
successful turns and partial measurements through the existing cleanup flow.

`conversation_config.json` controls the requested context window (default 32,768),
answer reserve/output limit (2,048), and template margin (1,024). The context window
is capped at the selected model's advertised capacity, verified through `/api/show`
before measurements begin. A conservative UTF-8-byte input budget, with additional
per-message overhead allowance, rejects oversized prompts; it is not an actual
token count and can reject inputs that would fit under the model tokenizer. Adjust
these settings explicitly if needed. Outputs reported by Ollama as ending at a
length limit are treated as errors rather than inserted as complete prior answers.

`conversation_context.json` in each run folder records chain IDs, turns, supplied
messages, preceding question IDs, context settings, failures, and encoder truncation
when relevant. It contains the actual experiment prompts/answers, not shared memory
for future runs. `question_order.json` records the chain-order policy and added
prerequisites. Existing CSV columns and energy/emissions arithmetic are unchanged;
the measured requests themselves are longer and can therefore cost more. Historical
stateless follow-up results are not equivalent to new history-aware runs.

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
