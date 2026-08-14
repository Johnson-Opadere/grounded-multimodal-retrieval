"""
Project 2B — Vision Retrieval Statistics Debugger
====================================================

Purpose
-------
Performs observability and diagnostic analysis of the
vision embedding manifold using vision-to-vision retrieval.

This script is one of the most important diagnostic tools
in Project 2B.

It quantifies whether the visual embedding space learned
meaningful disaster semantics before analyzing cross-modal
retrieval behavior.

Scientific Motivation
---------------------

This script helps answer:

    • How strong is same-event clustering?
    • How strong is same-bucket clustering?
    • Does cross-event semantic transfer emerge?
    • Are retrieval neighborhoods diverse?
    • Is the embedding manifold collapsing?
    • Are certain disaster categories dominating retrieval?

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

Before evaluating cross-modal retrieval, we must verify
that the visual embedding manifold itself is healthy.

Pipeline Position
-----------------

2A/artifacts/final
            ↓
test_load_artifacts.py
            ↓
retrieve_topk_vision.py
            ↓
debug_retrieval_stats.py
            ↓
debug_cross_modal_stats.py
            ↓
retrieve_operational.py

Inputs
------

From 2A/artifacts/final/

    vision_embeddings.npy
    vision_index.json

Expected Artifact Sizes
-----------------------

vision_embeddings
    (694, 256)

text_embeddings
    (70, 256)

whisper_embeddings
    (11, 256)

Current Scope
-------------

This script computes:

1. same_event_ratio
2. same_bucket_ratio
3. avg_unique_events@K
4. retrieval score statistics
5. bucket-level retrieval behavior
6. event diversity diagnostics

This script does NOT compute:

    Recall@K
    MRR
    XE Recall@K
    constraint evaluation
    grounded retrieval

Those are handled later in the pipeline.

Retrieval Method
----------------

FAISS Backend
~~~~~~~~~~~~~

Uses:

    faiss.IndexFlatIP

All embeddings are L2-normalized.

Therefore:

    cosine_similarity(a,b)
        ==
    inner_product(a,b)

which allows exact nearest-neighbor retrieval.

Metric Definitions
------------------

same_event_ratio
~~~~~~~~~~~~~~~~
Fraction of retrieved neighbors originating from
the same disaster event.

same_bucket_ratio
~~~~~~~~~~~~~~~~~
Fraction of retrieved neighbors belonging to the
same disaster category.

avg_unique_events@K
~~~~~~~~~~~~~~~~~~~
Average number of unique source events appearing
within the top-K neighborhood.

score_mean
~~~~~~~~~~
Mean similarity score across all retrieved neighbors.

score_std
~~~~~~~~~
Standard deviation of similarity scores.

Bucket-Level Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~
Reports:

    same_bucket_ratio
    unique_events_seen
    normalized_event_diversity

for each disaster category.

Usage
-----

Default:

PYTHONPATH=2B python \
2B/scripts/retrieval/debug_retrieval_stats.py \
    --topk 5

Evaluate subset of queries:

PYTHONPATH=2B python \
2B/scripts/retrieval/debug_retrieval_stats.py \
    --topk 5 \
    --num_queries 200

Expected Output
---------------

============================================================
Global Retrieval Statistics
============================================================

same_event_ratio        : 0.7726
same_bucket_ratio       : 0.9790
avg_unique_events@5     : 1.55

score_mean              : 0.9965
score_std               : 0.0204

============================================================
Bucket-Level Behavior
============================================================

Bucket: flooding
------------------------------------------------------------
same_bucket_ratio       : 0.9763
unique_events_seen      : 8

Bucket: wildfire
------------------------------------------------------------
same_bucket_ratio       : 0.9952
unique_events_seen      : 4

Files Written
-------------
None.

This script performs diagnostics only and prints
statistics to the terminal.

Key Findings
------------

Project 2B diagnostics revealed:

    same_event_ratio  ≈ 0.7726
    same_bucket_ratio ≈ 0.9790

These results indicate:

    • strong disaster-category organization
    • strong event clustering
    • healthy visual embedding geometry
    • successful cross-event semantic grouping

This became critical evidence that:

    the visual embedding manifold was healthy

and that:

    cross-modal alignment was the primary
    retrieval bottleneck.

Role in Project 2B
---------------------

This script serves as:

    the primary vision-manifold observability tool

used to validate:

    semantic clustering
    event structure
    retrieval diversity
    embedding stability

before introducing:

    cross-modal diagnostics
    retrieval constraints
    operational routing

Author
------
Project 2B
Multimodal Retrieval Systems Engineering
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import faiss


# =========================================================
# Utility Functions
# =========================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def print_separator():
    print("-" * 60)


# ---------------------------------------------------------
# Build exact FAISS retrieval backend for inspecting
# the visual embedding manifold.
#
# Used exclusively for observability and diagnostics.
# ---------------------------------------------------------
# =========================================================
# Build Vision Index
# =========================================================

def build_vision_index(embeddings):

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(
        embeddings.astype(np.float32)
    )

    return index


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 60)
    print("2B — Retrieval Statistics Debugger")
    print("=" * 60)

    artifact_dir = args.artifact_dir

    # =====================================================
    # Paths
    # =====================================================

    vision_emb_path = os.path.join(
        artifact_dir,
        "vision_embeddings.npy"
    )

    vision_index_path = os.path.join(
        artifact_dir,
        "vision_index.json"
    )

    # =====================================================
    # Load Data
    # =====================================================

    print("\nLoading vision artifacts...")
    print_separator()

    vision_embeddings = np.load(
        vision_emb_path
    )

    vision_metadata = load_json(
        vision_index_path
    )

    print(
        f"vision_embeddings : "
        f"{vision_embeddings.shape}"
    )

    print(
        f"vision_metadata   : "
        f"{len(vision_metadata)}"
    )

    # =====================================================
    # Build FAISS Index
    # =====================================================

    print("\nBuilding in-memory vision index...")
    print_separator()

    vision_index = build_vision_index(
        vision_embeddings
    )

    print(
        f"vision_index.ntotal : "
        f"{vision_index.ntotal}"
    )

    # =====================================================
    # Determine Query Range
    # =====================================================

    total_queries = len(vision_embeddings)

    if args.num_queries is None:
        num_queries = total_queries
    else:
        num_queries = min(
            args.num_queries,
            total_queries
        )

    print("\nEvaluation Setup")
    print_separator()

    print(f"topk                : {args.topk}")
    print(f"num_queries         : {num_queries}")

	# ---------------------------------------------------------
	# Aggregate retrieval statistics across all evaluated
	# queries.
	#
	# These metrics summarize global manifold behavior.
	# ---------------------------------------------------------
	# =====================================================
    # Global Statistics
    # =====================================================

    total_neighbors = 0

    same_event_count = 0
    same_bucket_count = 0

    unique_event_counts = []

    all_scores = []

	# ---------------------------------------------------------
	# Collect retrieval behavior separately for each
	# disaster category.
	#
	# Helps identify:
	#     - dominant buckets
	#     - weak buckets
	#     - diversity differences
	# ---------------------------------------------------------
    # =====================================================
    # Bucket-Level Statistics
    # =====================================================

    bucket_stats = defaultdict(
        lambda: {
            "same_bucket_count": 0,
            "neighbor_count": 0,
            "unique_events": []
        }
    )

    # =====================================================
    # Main Retrieval Loop
    # =====================================================

    print("\nRunning retrieval analysis...")
    print_separator()

    # ---------------------------------------------------------
	# Evaluate retrieval neighborhoods for every query.
	#
	# Each query retrieves:
	#
	#     top-K visual neighbors
	#
	# excluding the query itself.
	# ---------------------------------------------------------
	for query_idx in range(num_queries):

        query_embedding = vision_embeddings[
            query_idx
        ]

        query_meta = vision_metadata[
            query_idx
        ]

        query_event = query_meta["event_id"]

        query_bucket = query_meta[
            "damage_bucket"
        ]

        query_embedding = np.expand_dims(
            query_embedding.astype(np.float32),
            axis=0
        )

        # retrieve extra because top-1 is self
        scores, indices = vision_index.search(
            query_embedding,
            args.topk + 1
        )

        retrieved_events = set()

        neighbor_counter = 0

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            # skip self-match
            if idx == query_idx:
                continue

            neighbor_meta = vision_metadata[idx]

            neighbor_event = neighbor_meta[
                "event_id"
            ]

            neighbor_bucket = neighbor_meta[
                "damage_bucket"
            ]

            all_scores.append(float(score))

            total_neighbors += 1
            neighbor_counter += 1

            retrieved_events.add(
                neighbor_event
            )
			
			# Measures event-level clustering strength.
            # =============================================
            # Same Event
            # =============================================

            if neighbor_event == query_event:
                same_event_count += 1

            # Measures disaster-category clustering strength.
			# =============================================
            # Same Bucket
            # =============================================

            if neighbor_bucket == query_bucket:

                same_bucket_count += 1

                bucket_stats[query_bucket][
                    "same_bucket_count"
                ] += 1

            bucket_stats[query_bucket][
                "neighbor_count"
            ] += 1

            bucket_stats[query_bucket][
                "unique_events"
            ].append(neighbor_event)

            # =============================================
            # Stop At top-k
            # =============================================

            if neighbor_counter >= args.topk:
                break

        unique_event_counts.append(
            len(retrieved_events)
        )

    # ---------------------------------------------------------
	# Convert raw counts into interpretable retrieval
	# observability metrics.
	# ---------------------------------------------------------
	# =====================================================
    # Global Metrics
    # =====================================================

    same_event_ratio = (
        same_event_count / total_neighbors
    )

    same_bucket_ratio = (
        same_bucket_count / total_neighbors
    )

    avg_unique_events = np.mean(
        unique_event_counts
    )

    score_mean = np.mean(all_scores)
    score_std = np.std(all_scores)

    # =====================================================
    # Print Global Results
    # =====================================================

    print("\n" + "=" * 60)
    print("Global Retrieval Statistics")
    print("=" * 60)

    print(
        f"\nsame_event_ratio        : "
        f"{same_event_ratio:.4f}"
    )

    print(
        f"same_bucket_ratio       : "
        f"{same_bucket_ratio:.4f}"
    )

    print(
        f"avg_unique_events@{args.topk} : "
        f"{avg_unique_events:.2f}"
    )

    print(
        f"\nscore_mean              : "
        f"{score_mean:.4f}"
    )

    print(
        f"score_std               : "
        f"{score_std:.4f}"
    )

	# ---------------------------------------------------------
	# Display retrieval behavior separately for each
	# disaster category.
	#
	# Useful for detecting category-specific retrieval
	# strengths and weaknesses.
	# ---------------------------------------------------------
    # =====================================================
    # Bucket-Level Results
    # =====================================================

    print("\n" + "=" * 60)
    print("Bucket-Level Behavior")
    print("=" * 60)

    for bucket_name, stats in sorted(
        bucket_stats.items()
    ):

        bucket_same_ratio = (
            stats["same_bucket_count"]
            / stats["neighbor_count"]
        )

        unique_events = len(
            set(stats["unique_events"])
        )

        avg_unique_events_bucket = (
            unique_events / max(1, num_queries)
        )

        print(f"\nBucket: {bucket_name}")
        print_separator()

        print(
            f"same_bucket_ratio       : "
            f"{bucket_same_ratio:.4f}"
        )

        print(
            f"unique_events_seen      : "
            f"{unique_events}"
        )

        print(
            f"normalized_event_diversity : "
            f"{avg_unique_events_bucket:.4f}"
        )

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 60)
    print("[OK] Retrieval statistics complete.")
    print("=" * 60)


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k neighbors"
    )

    parser.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help="Number of queries to evaluate"
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to 2A artifacts"
    )

    args = parser.parse_args()

    main(args)