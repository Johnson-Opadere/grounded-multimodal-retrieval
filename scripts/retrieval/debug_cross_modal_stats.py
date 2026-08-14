"""
Project 2B — Cross-Modal Retrieval Statistics
================================================

Purpose
-------
Performs observability and diagnostic analysis of
cross-modal retrieval behavior.

Given:

    vision query embeddings

the system retrieves:

    text evidence

and optionally:

    whisper evidence

using semantic nearest-neighbor retrieval.

This script is one of the most important diagnostic
components in Project 2B.

Scientific Motivation
---------------------

This script helps answer:

    • How much semantic structure survives across modalities?
    • How strong is vision ↔ language alignment?
    • Does cross-event semantic transfer emerge?
    • Is retrieval semantically meaningful?
    • Are generic damage concepts dominating retrieval?
    • How severe is modality degradation?

This script complements:

    debug_retrieval_stats.py

which analyzed:

    vision → vision retrieval

while this script analyzes:

    vision → text retrieval

and optionally:

    vision → whisper retrieval

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

This script provides the critical bridge between:

    representation quality

and:

    retrieval quality

Pipeline Position
-----------------

2A/artifacts/final
            ↓
retrieve_topk.py
            ↓
debug_retrieval_stats.py
            ↓
debug_cross_modal_stats.py
            ↓
retrieve_operational.py
            ↓
retrieve_constrained.py

Inputs
------

From 2A/artifacts/final/

    vision_embeddings.npy
    vision_index.json

From 2B/indexes/

    text_index.faiss
    text_metadata.json

Optional:

    whisper_index.faiss
    whisper_metadata.json

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

Computes:

1. same_bucket_ratio
2. same_event_ratio
3. avg_unique_events@K
4. retrieval score statistics
5. bucket-level behavior
6. modality degradation analysis

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

which enables exact semantic retrieval.

Metric Definitions
------------------

same_bucket_ratio
~~~~~~~~~~~~~~~~~
Fraction of retrieved evidence belonging to the
same disaster category as the query.

same_event_ratio
~~~~~~~~~~~~~~~~
Fraction of retrieved evidence originating from
the same disaster event as the query.

avg_unique_events@K
~~~~~~~~~~~~~~~~~~~
Average number of unique source events appearing
within the retrieved neighborhood.

score_mean
~~~~~~~~~~
Average similarity score across all retrieved evidence.

score_std
~~~~~~~~~
Standard deviation of retrieval scores.

Bucket-Level Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~
Reports:

    same_bucket_ratio
    unique_events_seen

for each disaster category.

Usage
-----

Text Retrieval Analysis

PYTHONPATH=2B python \
2B/scripts/retrieval/debug_cross_modal_stats.py \
    --topk 5

Text + Whisper Analysis

PYTHONPATH=2B python \
2B/scripts/retrieval/debug_cross_modal_stats.py \
    --topk 5 \
    --use_whisper

Subset Evaluation

PYTHONPATH=2B python \
2B/scripts/retrieval/debug_cross_modal_stats.py \
    --topk 5 \
    --num_queries 100

Expected Output
---------------

============================================================
Text Retrieval Statistics
============================================================

same_bucket_ratio       : 0.0153
same_event_ratio        : 0.0625
avg_unique_events@5     : 3.68

score_mean              : 0.0373
score_std               : 0.0340

============================================================
Cross-Modal Diagnosis
============================================================

vision->vision same_bucket_ratio : 0.9790
vision->text same_bucket_ratio   : 0.0153

Interpretation:

Cross-modal alignment remains significantly weaker
than intra-vision semantic organization.

Files Written
-------------
None.

This script performs diagnostics only and prints
statistics to the terminal.

Key Findings
------------

Project 2B diagnostics revealed:

    vision->vision same_bucket_ratio ≈ 0.9790

while:

    vision->text same_bucket_ratio ≈ 0.0153

This established one of the central findings
of Project 2B:

    the visual embedding manifold was healthy,

but:

    unrestricted cross-modal retrieval
    remained unstable.

This observation directly motivated:

    metadata-aware retrieval constraints

which later became the primary operational
stabilization mechanism.

Role in Project 2B
---------------------

This script serves as:

    the primary cross-modal observability tool

used to diagnose:

    modality degradation
    semantic instability
    retrieval collapse
    cross-event transfer behavior

before introducing:

    constraint routing
    retrieval stabilization
    grounded retrieval

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


def load_faiss_index(path):
    return faiss.read_index(path)


def print_separator():
    print("-" * 60)


# =========================================================
# Evaluation Function
# =========================================================
# ---------------------------------------------------------
# Core cross-modal evaluation routine.
#
# Query:
#     vision embedding
#
# Retrieval:
#     text or whisper evidence
#
# Computes observability metrics describing
# cross-modal semantic behavior.
# ---------------------------------------------------------
def evaluate_cross_modal(
    vision_embeddings,
    vision_metadata,
    retrieval_index,
    retrieval_metadata,
    topk,
    modality_name,
    num_queries
):
    """
    Evaluates:
        vision -> language retrieval
    """

    total_neighbors = 0

    same_bucket_count = 0
    same_event_count = 0

    unique_event_counts = []

    all_scores = []

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

    for query_idx in range(num_queries):

        query_embedding = vision_embeddings[
            query_idx
        ]

        query_meta = vision_metadata[
            query_idx
        ]

        query_bucket = query_meta[
            "damage_bucket"
        ]

        query_event = query_meta[
            "event_id"
        ]

        query_embedding = np.expand_dims(
            query_embedding.astype(np.float32),
            axis=0
        )
		
		# Execute exact semantic nearest-neighbor retrieval.
        scores, indices = retrieval_index.search(
            query_embedding,
            topk
        )

        retrieved_events = set()

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            neighbor_meta = retrieval_metadata[
                idx
            ]

            neighbor_bucket = neighbor_meta[
                "damage_bucket"
            ]

            neighbor_event = neighbor_meta[
                "source_event"
            ]

            total_neighbors += 1

            all_scores.append(float(score))

            retrieved_events.add(
                neighbor_event
            )

			# Measures disaster-category semantic consistency.
            # =============================================
            # Same Bucket
            # =============================================

            if neighbor_bucket == query_bucket:

                same_bucket_count += 1

                bucket_stats[query_bucket][
                    "same_bucket_count"
                ] += 1

            # Measures event-level semantic consistency.
			# =============================================
            # Same Event
            # =============================================

            if neighbor_event == query_event:
                same_event_count += 1

            bucket_stats[query_bucket][
                "neighbor_count"
            ] += 1

            bucket_stats[query_bucket][
                "unique_events"
            ].append(neighbor_event)

        unique_event_counts.append(
            len(retrieved_events)
        )

	# ---------------------------------------------------------
	# Convert raw retrieval counts into interpretable
	# cross-modal diagnostics.
	# ---------------------------------------------------------    
	# =====================================================
    # Global Metrics
    # =====================================================

    same_bucket_ratio = (
        same_bucket_count / total_neighbors
    )

    same_event_ratio = (
        same_event_count / total_neighbors
    )

    avg_unique_events = np.mean(
        unique_event_counts
    )

    score_mean = np.mean(all_scores)
    score_std = np.std(all_scores)

    # =====================================================
    # Print Results
    # =====================================================

    print("\n" + "=" * 60)
    print(f"{modality_name} Retrieval Statistics")
    print("=" * 60)

    print(
        f"\nsame_bucket_ratio       : "
        f"{same_bucket_ratio:.4f}"
    )

    print(
        f"same_event_ratio        : "
        f"{same_event_ratio:.4f}"
    )

    print(
        f"avg_unique_events@{topk}     : "
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

    # =====================================================
    # Bucket-Level
    # =====================================================

    print("\n" + "=" * 60)
    print(f"{modality_name} Bucket-Level Behavior")
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

    return {
        "same_bucket_ratio": same_bucket_ratio,
        "same_event_ratio": same_event_ratio,
        "avg_unique_events": avg_unique_events,
        "score_mean": score_mean,
        "score_std": score_std
    }


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 60)
    print("2B — Cross-Modal Retrieval Statistics")
    print("=" * 60)

    artifact_dir = args.artifact_dir
    index_dir = args.index_dir

    # =====================================================
    # Load Vision Data
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

    print(
        f"vision_metadata   : "
        f"{len(vision_metadata)}"
    )

    # =====================================================
    # Determine Query Count
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
	# Primary evaluation:
	#
	# Vision
	#   →
	# Text
	#
	# Used to assess semantic transfer across modalities.
	# ---------------------------------------------------------    
	# =====================================================
    # TEXT RETRIEVAL
    # =====================================================

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

    text_results = evaluate_cross_modal(
        vision_embeddings=vision_embeddings,
        vision_metadata=vision_metadata,
        retrieval_index=text_index,
        retrieval_metadata=text_metadata,
        topk=args.topk,
        modality_name="Text",
        num_queries=num_queries
    )

	# ---------------------------------------------------------
	# Optional evaluation:
	#
	# Vision
	#   →
	# Whisper
	#
	# Provides an additional modality comparison.
	# ---------------------------------------------------------
    # =====================================================
    # WHISPER RETRIEVAL (OPTIONAL)
    # =====================================================

    whisper_results = None

    if args.use_whisper:

        whisper_index = load_faiss_index(
            os.path.join(
                index_dir,
                "whisper_index.faiss"
            )
        )

        whisper_metadata = load_json(
            os.path.join(
                index_dir,
                "whisper_metadata.json"
            )
        )

        whisper_results = evaluate_cross_modal(
            vision_embeddings=vision_embeddings,
            vision_metadata=vision_metadata,
            retrieval_index=whisper_index,
            retrieval_metadata=whisper_metadata,
            topk=args.topk,
            modality_name="Whisper",
            num_queries=num_queries
        )

	# ---------------------------------------------------------
	# Compare cross-modal retrieval quality against the
	# previously measured vision-manifold baseline.
	#
	# This highlights the gap between:
	#
	# vision → vision
	#
	# and:
	#
	# vision → language
	# ---------------------------------------------------------
    # =====================================================
    # Final Diagnosis
    # =====================================================

    print("\n" + "=" * 60)
    print("Cross-Modal Diagnosis")
    print("=" * 60)

    print(
        "\nvision->vision same_bucket_ratio : "
        "0.9790"
    )

    print(
        f"vision->text same_bucket_ratio   : "
        f"{text_results['same_bucket_ratio']:.4f}"
    )

    if whisper_results is not None:

        print(
            f"vision->whisper same_bucket_ratio: "
            f"{whisper_results['same_bucket_ratio']:.4f}"
        )

    print("\nInterpretation:")

    if text_results["same_bucket_ratio"] < 0.5:

        print(
            "Cross-modal alignment remains "
            "significantly weaker than "
            "intra-vision semantic organization."
        )

    else:

        print(
            "Cross-modal semantic transfer "
            "is emerging successfully."
        )

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 60)
    print("[OK] Cross-modal statistics complete.")
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
        help="Top-k retrieval count"
    )

    parser.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help="Number of queries to evaluate"
    )

    parser.add_argument(
        "--use_whisper",
        action="store_true",
        help="Enable whisper evaluation"
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
        help="Path to FAISS indexes"
    )

    args = parser.parse_args()

    main(args)