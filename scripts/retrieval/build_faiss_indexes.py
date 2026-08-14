"""
Project 2B — Build FAISS Indexes
===================================

Purpose
-------
Builds deterministic FAISS retrieval indexes from frozen
multimodal embedding artifacts exported by Project 2A.

This script converts embedding matrices into efficient
vector search structures used throughout Project 2B.

Current Implementation
----------------------
Builds:

1. Text Retrieval Index
2. Whisper Retrieval Index

using:

    faiss.IndexFlatIP

Since all embeddings are exported after L2 normalization:

    cosine_similarity(a, b)
        =
    inner_product(a, b)

therefore:

    IndexFlatIP

can be used directly for semantic retrieval.

Project Context
---------------
Project 2A produces:

    Vision Embeddings
    Text Embeddings
    Whisper Embeddings

Project 2B operationalizes these embeddings into:

    FAISS Retrieval
    Nearest-Neighbor Search
    Semantic Search
    Constraint-Aware Retrieval
    Grounded Multimodal Retrieval

This script is the SECOND stage of the 2B pipeline.

Pipeline Position
-----------------

2A/artifacts/final
            ↓
test_load_artifacts.py
            ↓
build_faiss_indexes.py
            ↓
retrieve_topk.py
retrieve_topk_vision.py
            ↓
retrieval pipeline

Inputs
------
2A/artifacts/final/

    text_embeddings.npy
    whisper_embeddings.npy

    text_index.json
    whisper_index.json

Expected Embedding Shapes
-------------------------

text_embeddings
    (70, 256)

whisper_embeddings
    (11, 256)

Validation Performed
--------------------

Normalization Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~
Verifies:

    ||z||₂ ≈ 1

for every embedding.

This guarantees:

    inner_product
        =
    cosine_similarity

Metadata Validation
~~~~~~~~~~~~~~~~~~~
Loads and preserves:

    text_index.json
    whisper_index.json

for later retrieval interpretation.

Outputs
-------
2B/indexes/

    text_index.faiss
    whisper_index.faiss

    text_metadata.json
    whisper_metadata.json

Output Description
------------------

text_index.faiss
~~~~~~~~~~~~~~~~
FAISS semantic retrieval index for
disaster report embeddings.

whisper_index.faiss
~~~~~~~~~~~~~~~~~~~
FAISS semantic retrieval index for
speech/transcript embeddings.

text_metadata.json
~~~~~~~~~~~~~~~~~~
Metadata mapping retrieval IDs back to:

    event
    bucket
    text evidence

whisper_metadata.json
~~~~~~~~~~~~~~~~~~~~~
Metadata mapping retrieval IDs back to:

    event
    bucket
    transcript evidence

Usage
-----

Default:

PYTHONPATH=2B python \
2B/scripts/retrieval/build_faiss_indexes.py

Custom artifact path:

PYTHONPATH=2B python \
2B/scripts/retrieval/build_faiss_indexes.py \
    --artifact_dir 2A/artifacts/final \
    --output_dir 2B/indexes

Expected Output
---------------

============================================================
2B — Build FAISS Indexes
============================================================

Loading embeddings...
------------------------------------------------------------
text_embeddings     : (70, 256)
whisper_embeddings  : (11, 256)

Verifying normalization...
------------------------------------------------------------
text_norm_mean      : 1.000000
whisper_norm_mean   : 1.000000

Building FAISS indexes...
------------------------------------------------------------
text_index.ntotal       : 70
whisper_index.ntotal    : 11

Saving indexes...
------------------------------------------------------------
Saved: 2B/indexes/text_index.faiss
Saved: 2B/indexes/whisper_index.faiss
Saved: 2B/indexes/text_metadata.json
Saved: 2B/indexes/whisper_metadata.json

============================================================
[OK] FAISS indexes built successfully.
============================================================

Files Written
-------------
text_index.faiss
whisper_index.faiss

text_metadata.json
whisper_metadata.json

Role in Project 2B
---------------------
This script creates the retrieval backend used by:

    retrieve_topk.py
    retrieve_operational.py
    retrieve_constrained.py
    demo_retrieval.py
    grounded_multimodal_summary.py
    evaluate_retrieval_metrics.py

Without these FAISS indexes,
semantic retrieval cannot be performed.

Author
------
Project 2B
Operational Multimodal Retrieval Systems
"""
Project 2B — Build FAISS Indexes
===================================

Purpose
-------
Builds deterministic FAISS vector indexes from frozen
2A multimodal embedding artifacts.

Current Scope
-------------
This initial implementation builds:

1. Text FAISS index
2. Whisper FAISS index

using:

    faiss.IndexFlatIP

Since all embeddings are already L2-normalized,
inner product == cosine similarity.

This script is foundational for:
- retrieval
- semantic auditing
- XE evaluation
- grounded synthesis

Inputs
------
2A/artifacts/final/

    text_embeddings.npy
    whisper_embeddings.npy

    text_index.json
    whisper_index.json

Outputs
-------
2B/indexes/

    text_index.faiss
    whisper_index.faiss

    text_metadata.json
    whisper_metadata.json

Usage
-----

From project root:

PYTHONPATH=2B python \
    2B/scripts/retrieval/build_faiss_indexes.py

Optional custom paths:

PYTHONPATH=2B python \
    2B/scripts/retrieval/build_faiss_indexes.py \
    --artifact_dir 2A/artifacts/final \
    --output_dir 2B/indexes

Expected Output
---------------

===================================================
2B — Build FAISS Indexes
===================================================

Loading embeddings...
---------------------------------------------------
text_embeddings     : (70, 256)
whisper_embeddings  : (11, 256)

Verifying normalization...
---------------------------------------------------
text_norm_mean      : 1.000000
whisper_norm_mean   : 1.000000

Building FAISS indexes...
---------------------------------------------------
text_index.ntotal       : 70
whisper_index.ntotal    : 11

Saving indexes...
---------------------------------------------------
Saved:
2B/indexes/text_index.faiss
2B/indexes/whisper_index.faiss

[OK] FAISS indexes built successfully.

Author
------
Project 2B

"""

import os
import json
import argparse

import numpy as np
import faiss


# =========================================================
# Utility Functions
# =========================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def compute_norm_mean(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    return float(norms.mean())


def verify_normalization(name, embeddings, tol=1e-3):
    """
    Verifies embeddings are approximately unit normalized.
    """

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=tol):
        raise ValueError(
            f"[ERROR] {name} embeddings are not normalized."
        )


def build_index(embeddings):
    """
    Builds FAISS IndexFlatIP index.

    Since embeddings are L2-normalized:
        inner product == cosine similarity
    """

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings.astype(np.float32))

    return index


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 60)
    print("2B — Build FAISS Indexes")
    print("=" * 60)

    artifact_dir = args.artifact_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # Paths
    # =====================================================

    text_emb_path = os.path.join(
        artifact_dir,
        "text_embeddings.npy"
    )

    whisper_emb_path = os.path.join(
        artifact_dir,
        "whisper_embeddings.npy"
    )

    text_index_json_path = os.path.join(
        artifact_dir,
        "text_index.json"
    )

    whisper_index_json_path = os.path.join(
        artifact_dir,
        "whisper_index.json"
    )

    # =====================================================
    # Load Embeddings
    # =====================================================

    print("\nLoading embeddings...")
    print("-" * 50)

    text_embeddings = np.load(text_emb_path)
    whisper_embeddings = np.load(whisper_emb_path)

    print(f"text_embeddings     : {text_embeddings.shape}")
    print(f"whisper_embeddings  : {whisper_embeddings.shape}")

    # =====================================================
    # Verify Normalization
    # =====================================================

    print("\nVerifying normalization...")
    print("-" * 50)

    verify_normalization(
        "text",
        text_embeddings
    )

    verify_normalization(
        "whisper",
        whisper_embeddings
    )

    text_norm_mean = compute_norm_mean(text_embeddings)
    whisper_norm_mean = compute_norm_mean(
        whisper_embeddings
    )

    print(f"text_norm_mean      : {text_norm_mean:.6f}")
    print(f"whisper_norm_mean   : {whisper_norm_mean:.6f}")

    # =====================================================
    # Load Metadata
    # =====================================================

    print("\nLoading metadata...")
    print("-" * 50)

    text_metadata = load_json(text_index_json_path)
    whisper_metadata = load_json(
        whisper_index_json_path
    )

    print(f"text_metadata rows      : {len(text_metadata)}")
    print(
        f"whisper_metadata rows   : "
        f"{len(whisper_metadata)}"
    )

    # =====================================================
    # Build FAISS Indexes
    # =====================================================

    print("\nBuilding FAISS indexes...")
    print("-" * 50)

    text_index = build_index(text_embeddings)

    whisper_index = build_index(
        whisper_embeddings
    )

    print(
        f"text_index.ntotal       : "
        f"{text_index.ntotal}"
    )

    print(
        f"whisper_index.ntotal    : "
        f"{whisper_index.ntotal}"
    )

    # =====================================================
    # Save Indexes
    # =====================================================

    print("\nSaving indexes...")
    print("-" * 50)

    text_index_output_path = os.path.join(
        output_dir,
        "text_index.faiss"
    )

    whisper_index_output_path = os.path.join(
        output_dir,
        "whisper_index.faiss"
    )

    faiss.write_index(
        text_index,
        text_index_output_path
    )

    faiss.write_index(
        whisper_index,
        whisper_index_output_path
    )

    # =====================================================
    # Save Metadata
    # =====================================================

    text_metadata_output_path = os.path.join(
        output_dir,
        "text_metadata.json"
    )

    whisper_metadata_output_path = os.path.join(
        output_dir,
        "whisper_metadata.json"
    )

    save_json(
        text_metadata,
        text_metadata_output_path
    )

    save_json(
        whisper_metadata,
        whisper_metadata_output_path
    )

    print(f"Saved: {text_index_output_path}")
    print(f"Saved: {whisper_index_output_path}")

    print(
        f"Saved: {text_metadata_output_path}"
    )

    print(
        f"Saved: {whisper_metadata_output_path}"
    )

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 60)
    print("[OK] FAISS indexes built successfully.")
    print("=" * 60)


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to 2A exported artifacts"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="2B/indexes",
        help="Directory to save FAISS indexes"
    )

    args = parser.parse_args()

    main(args)