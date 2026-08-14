"""
Project 2B — Operational Retrieval
=====================================

Purpose
-------
Implements the first operational retrieval policy layer
for Project 2B.

Unlike raw nearest-neighbor retrieval, this script applies:

    1. semantic similarity
    2. same-event preference
    3. same-bucket preference

to produce:

    • more stable retrievals
    • more interpretable retrievals
    • more auditable retrievals
    • more operationally realistic retrievals

This script represents the transition from:

    retrieval diagnostics

to:

    retrieval stabilization.

Project Context
---------------
Project 2A learns a shared multimodal embedding space:

    Vision Embeddings
    Text Embeddings
    Whisper Embeddings

Project 2B operationalizes those embeddings into:

    FAISS Retrieval
    Semantic Search
    Constraint-Aware Retrieval
    Grounded Multimodal Retrieval

Earlier diagnostics revealed:

    vision→vision same_bucket_ratio ≈ 0.9790

while:

    vision→text same_bucket_ratio ≈ 0.0153

indicating:

    strong visual semantics

but:

    weak unrestricted cross-modal retrieval.

This script introduces a metadata-aware retrieval
stabilization layer.

Pipeline Position
-----------------

2A/artifacts/final
            ↓
build_faiss_indexes.py
            ↓
retrieve_topk.py
            ↓
debug_cross_modal_stats.py
            ↓
retrieve_operational.py
            ↓
retrieve_constrained.py
            ↓
grounded_multimodal_summary.py

Current Scope
-------------

This script implements:

    deterministic reranking

using:

    metadata-aware bonuses

This script does NOT implement:

    learning-to-rank
    neural reranking
    LLM reranking
    cross-encoder reranking
    reinforcement learning

The goal is:

    operational stabilization

rather than:

    model retraining.

Operational Scoring
-------------------

Final ranking score:

    final_score =
        raw_similarity
        + same_event_bonus
        + same_bucket_bonus

where:

raw_similarity
~~~~~~~~~~~~~~
Cosine similarity produced by FAISS retrieval.

same_event_bonus
~~~~~~~~~~~~~~~~
Additional score applied when:

    candidate_event == query_event

same_bucket_bonus
~~~~~~~~~~~~~~~~~
Additional score applied when:

    candidate_bucket == query_bucket

Inputs
------

From 2A/artifacts/final/

    vision_embeddings.npy
    vision_index.json

From 2B/indexes/

    text_index.faiss
    text_metadata.json

Expected Artifact Sizes
-----------------------

vision_embeddings
    (694, 256)

text_embeddings
    (70, 256)

whisper_embeddings
    (11, 256)

Retrieval Flow
--------------

Vision Query
      ↓
Vision Embedding
      ↓
FAISS Retrieval
      ↓
Candidate Pool
      ↓
Metadata-Aware Reranking
      ↓
Final Top-K Retrieval

Outputs
-------

Console Retrieval Report

Displays:

    query metadata

        query_idx
        event
        bucket
        patch_id

and:

    reranked retrieval results

        raw_score
        same_event_bonus
        same_bucket_bonus
        final_score

        source_event
        source_bucket
        evidence text

Usage
-----

Default Retrieval

PYTHONPATH=2B python \
2B/scripts/retrieval/retrieve_operational.py \
    --query_idx 100 \
    --topk 5

Custom Bonus Configuration

PYTHONPATH=2B python \
2B/scripts/retrieval/retrieve_operational.py \
    --query_idx 100 \
    --topk 5 \
    --same_event_bonus 0.10 \
    --same_bucket_bonus 0.05

Larger Candidate Pool

PYTHONPATH=2B python \
2B/scripts/retrieval/retrieve_operational.py \
    --query_idx 100 \
    --candidate_pool_size 20

Example Output
--------------

============================================================
Query
============================================================

query_idx      : 100
event          : hurricane-florence
bucket         : flooding

============================================================
Top-5 Operational Retrievals
============================================================

[1]

raw_score          : 0.0301
same_event_bonus   : 0.1000
same_bucket_bonus  : 0.0500

final_score        : 0.1801

event              : hurricane-florence
bucket             : flooding

"...retrieved evidence..."

------------------------------------------------------------

Files Written
-------------
None.

This script performs reranking and prints results
to the terminal.

Key Findings
------------

Project 2B showed that:

    metadata-aware retrieval bonuses

can significantly improve retrieval stability
without retraining the embedding models.

This observation motivated:

    constrained retrieval

implemented later in:

    retrieve_constrained.py

Role in Project 2B
---------------------

This script serves as:

    the first operational retrieval layer

used to demonstrate:

    metadata-aware reranking
    retrieval stabilization
    operational retrieval policies
    deterministic retrieval control

before introducing:

    hard retrieval constraints
    retrieval evaluation
    grounded retrieval

Author
------
Project 2B
Multimodal Retrieval Systems Engineering
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


def load_faiss_index(path):
    return faiss.read_index(path)


def print_separator():
    print("-" * 60)


# =========================================================
# Operational Scoring
# =========================================================
# ---------------------------------------------------------
# Metadata-aware retrieval scoring function.
#
# Combines:
#
#     semantic similarity
#
# with:
#
#     event prior
#     bucket prior
#
# to improve retrieval stability.
# ---------------------------------------------------------
def compute_operational_score(
    raw_score,
    query_event,
    query_bucket,
    candidate_event,
    candidate_bucket,
    same_event_bonus,
    same_bucket_bonus
):
    """
    Computes operational retrieval score.
    """
	
	# Initialize reranking bonuses.
    event_bonus = 0.0
    bucket_bonus = 0.0

    if candidate_event == query_event:
        event_bonus = same_event_bonus

    if candidate_bucket == query_bucket:
        bucket_bonus = same_bucket_bonus

    final_score = (
        raw_score
        + event_bonus
        + bucket_bonus
    )

    return {
        "raw_score": raw_score,
        "event_bonus": event_bonus,
        "bucket_bonus": bucket_bonus,
        "final_score": final_score
    }


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 60)
    print("2B — Operational Retrieval")
    print("=" * 60)

    artifact_dir = args.artifact_dir
    index_dir = args.index_dir
	
	# ---------------------------------------------------------
	# Load vision embeddings exported from Project 2A.
	#
	# Each embedding serves as a retrieval query.
	# ---------------------------------------------------------
    # =====================================================
    # Load Vision Queries
    # =====================================================

    print("\nLoading vision artifacts...")
    print_separator()

    vision_embeddings = np.load(
        os.path.join(
            artifact_dir,
            "vision_embeddings.npy"
        )
    )

    vision_metadata = load_json(
        os.path.join(
            artifact_dir,
            "vision_index.json"
        )
    )

    print(
        f"vision_embeddings : "
        f"{vision_embeddings.shape}"
    )

    # ---------------------------------------------------------
	# Load FAISS retrieval backend and associated metadata.
	#
	# Retrieval candidates are drawn from this index before
	# operational reranking is applied.
	# ---------------------------------------------------------
	# =====================================================
    # Load Text Retrieval Backend
    # =====================================================

    print("\nLoading retrieval backend...")
    print_separator()

    text_index = load_faiss_index(
        os.path.join(
            index_dir,
            "text_index.faiss"
        )
    )

    text_metadata = load_json(
        os.path.join(
            index_dir,
            "text_metadata.json"
        )
    )

    print(
        f"text_index.ntotal : "
        f"{text_index.ntotal}"
    )

    # =====================================================
    # Validate Query
    # =====================================================

    query_idx = args.query_idx

    if query_idx < 0 or query_idx >= len(vision_embeddings):
        raise ValueError(
            f"Invalid query_idx: {query_idx}"
        )

    query_embedding = vision_embeddings[
        query_idx
    ]

    query_meta = vision_metadata[
        query_idx
    ]

    query_event = query_meta["event_id"]
    query_bucket = query_meta["damage_bucket"]

    # =====================================================
    # Query Info
    # =====================================================

    print("\n" + "=" * 60)
    print("Query")
    print("=" * 60)

    print(f"\nquery_idx      : {query_idx}")
    print(f"event          : {query_event}")
    print(f"bucket         : {query_bucket}")
    print(
        f"patch_id       : "
        f"{query_meta['patch_id']}"
    )

	# ---------------------------------------------------------
	# Stage 1:
	#
	# Retrieve an initial candidate pool using
	# semantic similarity alone.
	# ---------------------------------------------------------
    # =====================================================
    # Raw Retrieval
    # =====================================================

    query_embedding = np.expand_dims(
        query_embedding.astype(np.float32),
        axis=0
    )

    # Retrieve a larger candidate pool than final top-k.
	#
	# Reranking will determine the final ranking order.
    raw_scores, raw_indices = text_index.search(
        query_embedding,
        args.candidate_pool_size
    )

	# ---------------------------------------------------------
	# Stage 2:
	#
	# Apply metadata-aware bonuses to stabilize retrieval.
	#
	# This transforms:
	#
	#     raw retrieval
	#
	# into:
	#
	#     operational retrieval
	# ---------------------------------------------------------
    # =====================================================
    # Operational Reranking
    # =====================================================

    reranked_results = []

    for raw_score, idx in zip(
        raw_scores[0],
        raw_indices[0]
    ):

        meta = text_metadata[idx]

        candidate_event = meta["source_event"]
        candidate_bucket = meta["damage_bucket"]

        scoring = compute_operational_score(
            raw_score=float(raw_score),
            query_event=query_event,
            query_bucket=query_bucket,
            candidate_event=candidate_event,
            candidate_bucket=candidate_bucket,
            same_event_bonus=args.same_event_bonus,
            same_bucket_bonus=args.same_bucket_bonus
        )

        reranked_results.append({
            "idx": int(idx),
            "metadata": meta,
            **scoring
        })

	# Rank candidates using operational score.
    # =====================================================
    # Sort By Final Score
    # =====================================================

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x["final_score"],
        reverse=True
    )

	# ---------------------------------------------------------
	# Display reranked retrieval results together with
	# bonus contributions and final ranking scores.
	# ---------------------------------------------------------
    # =====================================================
    # Print Results
    # =====================================================

    print("\n" + "=" * 60)
    print(
        f"Top-{args.topk} Operational Retrievals"
    )
    print("=" * 60)

    for rank, result in enumerate(
        reranked_results[:args.topk],
        start=1
    ):

        meta = result["metadata"]

        print(f"\n[{rank}]")

        print(
            f"raw_score          : "
            f"{result['raw_score']:.4f}"
        )

        print(
            f"same_event_bonus   : "
            f"{result['event_bonus']:.4f}"
        )

        print(
            f"same_bucket_bonus  : "
            f"{result['bucket_bonus']:.4f}"
        )

        print(
            f"\nfinal_score        : "
            f"{result['final_score']:.4f}"
        )

        print(
            f"\nevent              : "
            f"{meta['source_event']}"
        )

        print(
            f"bucket             : "
            f"{meta['damage_bucket']}"
        )

        print("\n" + meta["text"])

        print_separator()

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 60)
    print("[OK] Operational retrieval complete.")
    print("=" * 60)


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query_idx",
        type=int,
        required=True,
        help="Vision query index"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Final top-k results"
    )

    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=10,
        help="Raw retrieval candidate pool size"
    )

    parser.add_argument(
        "--same_event_bonus",
        type=float,
        default=0.10,
        help="Same-event reranking bonus"
    )

    parser.add_argument(
        "--same_bucket_bonus",
        type=float,
        default=0.05,
        help="Same-bucket reranking bonus"
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to 2A artifacts"
    )

    parser.add_argument(
        "--index_dir",
        type=str,
        default="2B/indexes",
        help="Path to retrieval indexes"
    )

    args = parser.parse_args()

    main(args)