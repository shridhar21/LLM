import os
import time
import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from markitdown import MarkItDown
from measurement import make_codecarbon_tracker, stop_tracker
from cancellation import cancellable_run, configure_run, track_phase, publish_index


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".csv",
    ".txt",
    ".md",
    ".html",
    ".htm",
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


# ============================================================
# FILE HASHING
# ============================================================

def get_file_hash(path: Path):
    """
    Calculate SHA-256 hash for a file.

    Used to determine whether an existing corpus document
    has actually changed.
    """

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# EXISTING METADATA
# ============================================================

def load_existing_metadata(meta_path: Path):

    if not meta_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(meta_path)

    except Exception as e:
        print(f"Warning: Could not read metadata: {e}")
        return pd.DataFrame()


# ============================================================
# DOCUMENT CHANGE DETECTION
# ============================================================

def find_document_changes(
    corpus_dir: Path,
    existing_meta: pd.DataFrame
):
    """
    Categorize corpus files into:

        new
        unchanged
        modified

    using SHA-256 hashes.
    """

    existing_hashes = {}

    if (
        not existing_meta.empty
        and "filename" in existing_meta.columns
        and "file_hash" in existing_meta.columns
    ):

        for _, row in existing_meta.iterrows():

            existing_hashes[str(row["filename"])] = str(
                row["file_hash"]
            )

    new_files = []
    unchanged_files = []
    modified_files = []

    for p in sorted(corpus_dir.iterdir()):

        if not p.is_file():
            continue

        extension = p.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:

            print(
                f"Skipping unsupported file: {p.name}"
            )

            continue

        current_hash = get_file_hash(p)

        filename = p.name

        if filename not in existing_hashes:

            new_files.append(
                (p, current_hash)
            )

        elif existing_hashes[filename] == current_hash:

            unchanged_files.append(p)

        else:

            modified_files.append(
                (p, current_hash)
            )

    return (
        new_files,
        unchanged_files,
        modified_files,
    )


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_document(
    path: Path,
    markdown_converter: MarkItDown
):
    """
    Convert a document to text using MarkItDown.

    Returns None if conversion fails or produces no text.
    """

    try:

        result = markdown_converter.convert(
            str(path)
        )

        content = (
            result.text_content or ""
        ).strip()

        if not content:

            print(
                f"  ⚠ No text extracted from {path.name}"
            )

            return None

        return content

    except Exception as e:

        print(
            f"  ✗ Failed to convert {path.name}: "
            f"{type(e).__name__}: {e}"
        )

        return None


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_documents(
    files,
    start_doc_id,
    markdown_converter
):
    """
    Extract text from new/modified documents.
    """

    texts = []
    metadata = []

    for path, file_hash in files:

        print(
            f"Processing: {path.name}"
        )

        content = extract_document(
            path,
            markdown_converter
        )

        if content is None:
            continue

        doc_id = (
            start_doc_id + len(texts)
        )

        texts.append(content)

        metadata.append({

            "doc_id": doc_id,

            "filename": path.name,

            "path": str(path),

            "file_type": path.suffix.lower(),

            "file_hash": file_hash,

            "num_chars": len(content),

        })

        print(
            f"  ✓ Extracted "
            f"{len(content):,} characters"
        )

    return (
        texts,
        pd.DataFrame(metadata)
    )


# ============================================================
# CHUNKING
# ============================================================

def chunk_texts(
    texts,
    doc_ids,
    start_chunk_id=0,
    chunk_size=800,
    overlap=200
):
    """
    Split documents into overlapping character chunks.
    """

    chunks = []
    sources = []

    for text, doc_id in zip(
        texts,
        doc_ids
    ):

        start = 0

        while start < len(text):

            end = min(
                len(text),
                start + chunk_size
            )

            chunk = (
                text[start:end]
                .strip()
            )

            if chunk:

                chunk_id = (
                    start_chunk_id
                    + len(chunks)
                )

                chunks.append(chunk)

                sources.append({

                    "doc_id": doc_id,

                    "chunk_id": chunk_id,

                    "start": start,

                    "end": end,

                    "num_chars": len(chunk),

                })

            if end == len(text):
                break

            start = end - overlap

    return (
        chunks,
        pd.DataFrame(sources)
    )


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def generate_embeddings(
    embedder,
    chunks
):
    """
    Generate embeddings only for the supplied chunks.
    """

    if not chunks:

        return np.empty(
            (0, 384),
            dtype="float32"
        )

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings.astype(
        "float32"
    )


# ============================================================
# FAISS INDEX CONSTRUCTION
# ============================================================

def build_faiss_index(embeddings):
    """
    Build a fresh IndexFlatL2 from already-computed embeddings.
    """

    if len(embeddings) == 0:

        raise ValueError(
            "Cannot build FAISS index: "
            "no embeddings available."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# CODECARBON
# ============================================================

def latest_emissions_row(
    emissions_csv: Path
):

    if not emissions_csv.exists():
        return None

    try:

        df = pd.read_csv(
            emissions_csv
        )

    except Exception:

        return None

    if df.empty:
        return None

    return df.iloc[-1].to_dict()


# ============================================================
# MAIN
# ============================================================

@cancellable_run
def legacy_main():

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    CORPUS_DIR = Path(
        "rag_corpus"
    )

    RAG_INDEX_DIR = Path(
        "indexes/rag_faiss"
    )

    OUTDIR = Path(
        f"emissions_reports/"
        f"rag_indexing_{timestamp}"
    )
    configure_run(OUTDIR, 0, EMBEDDING_MODEL, indexing=True)

    CORPUS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RAG_INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------------

    meta_path = (
        RAG_INDEX_DIR
        / "corpus_meta.csv"
    )

    sources_path = (
        RAG_INDEX_DIR
        / "chunk_sources.csv"
    )

    chunks_path = (
        RAG_INDEX_DIR
        / "chunks.json"
    )

    embeddings_path = (
        RAG_INDEX_DIR
        / "embeddings.npy"
    )

    index_path = (
        RAG_INDEX_DIR
        / "index.faiss"
    )

    emissions_csv = (
        OUTDIR
        / "emissions.csv"
    )

    summary_path = (
        OUTDIR
        / "summary.csv"
    )

    # --------------------------------------------------------
    # LOAD EXISTING METADATA
    # --------------------------------------------------------

    existing_meta = load_existing_metadata(
        meta_path
    )

    # --------------------------------------------------------
    # CHECK FOR CORPUS CHANGES
    #
    # This check happens on every script execution.
    # The expensive indexing pipeline does NOT.
    # --------------------------------------------------------

    print(
        "\nChecking corpus for changes..."
    )

    (
        new_files,
        unchanged_files,
        modified_files,
    ) = find_document_changes(
        CORPUS_DIR,
        existing_meta
    )

    print(
        f"  New files:       "
        f"{len(new_files)}"
    )

    print(
        f"  Unchanged files: "
        f"{len(unchanged_files)}"
    )

    print(
        f"  Modified files:  "
        f"{len(modified_files)}"
    )

    # --------------------------------------------------------
    # NO WORK REQUIRED
    # --------------------------------------------------------

    if not new_files and not modified_files:

        print(
            "\nNo new or modified documents."
        )

        print(
            "FAISS index is already up to date."
        )

        return

    # --------------------------------------------------------
    # LOAD EXISTING CHUNKS
    # --------------------------------------------------------

    if chunks_path.exists():

        with open(
            chunks_path,
            "r",
            encoding="utf-8"
        ) as f:

            existing_chunks = json.load(f)

    else:

        existing_chunks = []

    # --------------------------------------------------------
    # LOAD EXISTING SOURCES
    # --------------------------------------------------------

    if sources_path.exists():

        existing_sources = pd.read_csv(
            sources_path
        )

    else:

        existing_sources = pd.DataFrame()

    # --------------------------------------------------------
    # LOAD EXISTING EMBEDDINGS
    # --------------------------------------------------------

    if embeddings_path.exists():

        existing_embeddings = np.load(
            embeddings_path
        ).astype("float32")

    else:

        existing_embeddings = np.empty(
            (0, 384),
            dtype="float32"
        )

    # --------------------------------------------------------
    # LOAD EXISTING FAISS INDEX
    # --------------------------------------------------------

    if index_path.exists():

        existing_index = faiss.read_index(
            str(index_path)
        )

    else:

        existing_index = None

    # ========================================================
    # START CODECARBON ONLY NOW
    #
    # The cheap hash/change detection above is not included
    # in the indexing benchmark.
    # ========================================================

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nRouting CodeCarbon telemetry to "
        f"{OUTDIR}..."
    )

    tracker = make_codecarbon_tracker(
        'rag_indexing',
        project_name=(
            f"RAG_incremental_index_"
            f"{uuid.uuid4().hex[:8]}"
        ),
        output_dir=OUTDIR,
        output_file="emissions.csv",
    )

    start_time = time.time()

    track_phase(tracker, 'indexing')
    tracker.start()

    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    print(
        "\nLoading embedding model..."
    )

    embedder = SentenceTransformer(
        EMBEDDING_MODEL
    )

    markdown_converter = MarkItDown()

    # ========================================================
    # DETERMINE WHETHER THIS IS:
    #
    # 1. NEW FILE ADDITION ONLY
    # 2. MODIFICATION
    # ========================================================

    only_new_files = (
        len(new_files) > 0
        and len(modified_files) == 0
    )

    has_modifications = (
        len(modified_files) > 0
    )

    # ========================================================
    # CASE 1:
    #
    # ONLY NEW FILES
    #
    # Truly incremental:
    #
    # existing FAISS index
    #        +
    # new embeddings
    #
    # No reconstruction.
    # ========================================================

    if only_new_files:

        print(
            "\nMode: INCREMENTAL ADDITION"
        )

        print(
            "Only new documents detected."
        )

        # ----------------------------------------------------
        # Determine next document ID
        # ----------------------------------------------------

        if existing_meta.empty:

            next_doc_id = 0

        else:

            next_doc_id = (
                int(
                    existing_meta[
                        "doc_id"
                    ].max()
                )
                + 1
            )

        # ----------------------------------------------------
        # Extract documents
        # ----------------------------------------------------

        new_texts, new_meta = load_documents(
            new_files,
            next_doc_id,
            markdown_converter
        )

        if not new_texts:

            stop_tracker(tracker, OUTDIR)

            print(
                "\nNo documents could be extracted."
            )

            return

        # ----------------------------------------------------
        # Chunk
        # ----------------------------------------------------

        print(
            "\nChunking new documents..."
        )

        new_doc_ids = (
            new_meta[
                "doc_id"
            ].tolist()
        )

        start_chunk_id = len(
            existing_chunks
        )

        new_chunks, new_sources = chunk_texts(

            new_texts,

            new_doc_ids,

            start_chunk_id,

            CHUNK_SIZE,

            CHUNK_OVERLAP,
        )

        print(
            f"Created "
            f"{len(new_chunks)} new chunks."
        )

        # ----------------------------------------------------
        # Embed ONLY new chunks
        # ----------------------------------------------------

        print(
            "\nGenerating embeddings "
            "for new chunks..."
        )

        new_embeddings = generate_embeddings(
            embedder,
            new_chunks
        )

        # ----------------------------------------------------
        # Add to FAISS
        # ----------------------------------------------------

        print(
            "\nAdding new vectors to FAISS..."
        )

        if existing_index is not None:

            index = existing_index

            index.add(
                new_embeddings
            )

        else:

            index = build_faiss_index(
                new_embeddings
            )

        # ----------------------------------------------------
        # Combine stored data
        # ----------------------------------------------------

        all_chunks = (
            existing_chunks
            + new_chunks
        )

        all_embeddings = np.vstack(
            [
                existing_embeddings,
                new_embeddings
            ]
        )

        if existing_meta.empty:

            combined_meta = new_meta

        else:

            combined_meta = pd.concat(
                [
                    existing_meta,
                    new_meta
                ],
                ignore_index=True
            )

        if existing_sources.empty:

            combined_sources = (
                new_sources
            )

        else:

            combined_sources = pd.concat(
                [
                    existing_sources,
                    new_sources
                ],
                ignore_index=True
            )

    # ========================================================
    # CASE 2:
    #
    # MODIFIED DOCUMENT(S)
    #
    # We do NOT re-embed unchanged chunks.
    #
    # Existing embeddings belonging to unchanged documents
    # are reused.
    #
    # Only modified documents receive new embeddings.
    #
    # A fresh FAISS index is then constructed from:
    #
    #     unchanged embeddings
    #             +
    #     new embeddings
    #
    # ========================================================

    elif has_modifications:

        print(
            "\nMode: MODIFIED DOCUMENT"
        )

        print(
            "Reusing embeddings for "
            "unchanged documents."
        )

        print(
            "Only modified documents "
            "will be re-embedded."
        )

        modified_names = {
            path.name
            for path, _ in modified_files
        }

        # ----------------------------------------------------
        # Identify modified document IDs
        # ----------------------------------------------------

        modified_doc_ids = set()

        if not existing_meta.empty:

            modified_doc_ids = set(
                existing_meta[
                    existing_meta[
                        "filename"
                    ].isin(
                        modified_names
                    )
                ]["doc_id"]
            )

        # ----------------------------------------------------
        # Retain metadata for unchanged documents
        # ----------------------------------------------------

        if existing_meta.empty:

            retained_meta = (
                pd.DataFrame()
            )

        else:

            retained_meta = (
                existing_meta[
                    ~existing_meta[
                        "filename"
                    ].isin(
                        modified_names
                    )
                ].copy()
            )

        # ----------------------------------------------------
        # Retain chunks AND embeddings belonging to
        # unchanged documents.
        #
        # IMPORTANT:
        # FAISS vector order must remain synchronized with
        # chunks.json and chunk_sources.csv.
        # ----------------------------------------------------

        if (
            not existing_sources.empty
            and len(existing_embeddings) > 0
        ):

            keep_mask = (
                ~existing_sources[
                    "doc_id"
                ].isin(
                    modified_doc_ids
                )
            )

            retained_sources = (
                existing_sources[
                    keep_mask
                ].copy()
            )

            retained_chunks = [
                chunk
                for chunk, keep
                in zip(
                    existing_chunks,
                    keep_mask.tolist()
                )
                if keep
            ]

            retained_embeddings = (
                existing_embeddings[
                    keep_mask.to_numpy()
                ]
            )

        else:

            retained_sources = (
                pd.DataFrame()
            )

            retained_chunks = []

            retained_embeddings = (
                np.empty(
                    (0, 384),
                    dtype="float32"
                )
            )

        # ----------------------------------------------------
        # Determine document IDs for modified documents.
        #
        # We KEEP their existing doc_id so that references
        # remain stable.
        # ----------------------------------------------------

        existing_doc_ids = {}

        if not existing_meta.empty:

            existing_doc_ids = dict(
                zip(
                    existing_meta[
                        "filename"
                    ],
                    existing_meta[
                        "doc_id"
                    ]
                )
            )

        modified_doc_id_list = []

        for path, _ in modified_files:

            modified_doc_id_list.append(
                existing_doc_ids[
                    path.name
                ]
            )

        # ----------------------------------------------------
        # Extract modified documents
        # ----------------------------------------------------

        modified_texts, modified_meta = (
            load_documents(
                modified_files,
                0,
                markdown_converter
            )
        )

        # Replace the temporary doc IDs assigned above with
        # their ORIGINAL document IDs.
        #
        # This preserves document identity across updates.

        for i in range(
            len(modified_meta)
        ):

            original_doc_id = (
                modified_doc_id_list[i]
            )

            modified_meta.loc[
                i,
                "doc_id"
            ] = original_doc_id

        if not modified_texts:

            stop_tracker(tracker, OUTDIR)

            print(
                "\nModified documents "
                "could not be extracted."
            )

            return

        # ----------------------------------------------------
        # Chunk modified documents
        # ----------------------------------------------------

        print(
            "\nChunking modified documents..."
        )

        modified_doc_ids_final = (
            modified_meta[
                "doc_id"
            ].astype(int).tolist()
        )

        start_chunk_id = (
            len(retained_chunks)
        )

        modified_chunks, modified_sources = (
            chunk_texts(

                modified_texts,

                modified_doc_ids_final,

                start_chunk_id,

                CHUNK_SIZE,

                CHUNK_OVERLAP,
            )
        )

        print(
            f"Created "
            f"{len(modified_chunks)} "
            f"replacement chunks."
        )

        # ----------------------------------------------------
        # Embed ONLY modified chunks
        # ----------------------------------------------------

        print(
            "\nGenerating embeddings only "
            "for modified chunks..."
        )

        modified_embeddings = (
            generate_embeddings(
                embedder,
                modified_chunks
            )
        )

        # ----------------------------------------------------
        # Construct complete replacement corpus
        #
        # Existing embeddings are REUSED.
        # ----------------------------------------------------

        all_chunks = (
            retained_chunks
            + modified_chunks
        )

        all_embeddings = np.vstack(
            [
                retained_embeddings,
                modified_embeddings
            ]
        )

        combined_meta = pd.concat(
            [
                retained_meta,
                modified_meta
            ],
            ignore_index=True
        )

        combined_sources = pd.concat(
            [
                retained_sources,
                modified_sources
            ],
            ignore_index=True
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # FAISS itself is reconstructed, but the embeddings
        # of unchanged documents are NOT regenerated.
        # ----------------------------------------------------

        print(
            "\nReconstructing FAISS index "
            "using reused + new embeddings..."
        )

        index = build_faiss_index(
            all_embeddings
        )

    else:

        stop_tracker(tracker, OUTDIR)

        print(
            "\nUnexpected indexing state."
        )

        return

    # ========================================================
    # END CODECARBON TRACKING
    # ========================================================

    emissions_kg = (
        stop_tracker(tracker, OUTDIR)
        or 0.0
    )

    runtime_s = (
        time.time()
        - start_time
    )

    # ========================================================
    # READ CODECARBON TELEMETRY
    # ========================================================

    row = latest_emissions_row(
        emissions_csv
    )

    energy_kwh = (
        row.get(
            "energy_consumed",
            0.0
        )
        if row
        else 0.0
    )

    cpu_kwh = (
        row.get(
            "cpu_energy",
            0.0
        )
        if row
        else 0.0
    )

    gpu_kwh = (
        row.get(
            "gpu_energy",
            0.0
        )
        if row
        else 0.0
    )

    ram_kwh = (
        row.get(
            "ram_energy",
            0.0
        )
        if row
        else 0.0
    )

    # ========================================================
    # SAVE EVERYTHING
    # ========================================================

    print(
        "\nSaving updated index and metadata..."
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    publish_index(RAG_INDEX_DIR, index, all_chunks, all_embeddings,
                  combined_meta, combined_sources)

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_row = {

        "num_new_docs":
            len(new_files),

        "num_modified_docs":
            len(modified_files),

        "num_unchanged_docs":
            len(unchanged_files),

        "num_new_chunks":
            (
                len(new_chunks)
                if only_new_files
                else 0
            ),

        "num_replacement_chunks":
            (
                len(modified_chunks)
                if has_modifications
                else 0
            ),

        "total_docs_in_index":
            len(combined_meta),

        "total_chunks_in_index":
            len(all_chunks),

        "index_runtime_s":
            runtime_s,

        "index_emissions_kg":
            emissions_kg,

        "index_emissions_g":
            emissions_kg * 1000.0,

        "total_energy_kwh":
            energy_kwh,

        "cpu_energy_kwh":
            cpu_kwh,

        "gpu_energy_kwh":
            gpu_kwh,

        "ram_energy_kwh":
            ram_kwh,

        "embedding_model":
            EMBEDDING_MODEL,

        "chunk_size":
            CHUNK_SIZE,

        "chunk_overlap":
            CHUNK_OVERLAP,

        "index_type":
            "IndexFlatL2",

    }

    pd.DataFrame(
        [summary_row]
    ).to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n"
        + "=" * 55
    )

    print(
        "Incremental RAG Index Build Complete"
    )

    print(
        "=" * 55
    )

    print(
        f"New documents:          "
        f"{len(new_files)}"
    )

    print(
        f"Modified documents:     "
        f"{len(modified_files)}"
    )

    print(
        f"Unchanged documents:    "
        f"{len(unchanged_files)}"
    )

    print(
        f"Total documents:        "
        f"{len(combined_meta)}"
    )

    print(
        f"Total chunks:           "
        f"{len(all_chunks)}"
    )

    print(
        f"Total embeddings:       "
        f"{len(all_embeddings)}"
    )

    print(
        f"Runtime:                "
        f"{runtime_s:.2f} s"
    )

    print(
        f"Energy:                 "
        f"{energy_kwh:.6f} kWh"
    )

    print(
        f"  ├─ CPU:               "
        f"{cpu_kwh:.6f} kWh"
    )

    print(
        f"  ├─ GPU:               "
        f"{gpu_kwh:.6f} kWh"
    )

    print(
        f"  └─ RAM:               "
        f"{ram_kwh:.6f} kWh"
    )

    print(
        f"CO2eq:                  "
        f"{emissions_kg * 1000:.4f} g"
    )

    print(
        f"\nIndex:                  "
        f"{index_path}"
    )

    print(
        f"Embeddings:             "
        f"{embeddings_path}"
    )

    print(
        f"Metadata:               "
        f"{meta_path}"
    )

    print(
        f"Sources:                "
        f"{sources_path}"
    )

    print(
        f"Report:                 "
        f"{OUTDIR}"
    )

    print(
        "=" * 55
    )


@cancellable_run
def main():
    """Build the advanced index without replacing the legacy flat-chunk assets."""
    from advanced_rag import index_corpus
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    index_corpus(Path('rag_corpus'), Path('indexes/rag_faiss'),
                 Path(f'emissions_reports/rag_indexing_{timestamp}'), SUPPORTED_EXTENSIONS,
                 extract_document, MarkItDown, SentenceTransformer,
                 make_codecarbon_tracker, latest_emissions_row)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Advanced RAG indexing failed: {exc}')
