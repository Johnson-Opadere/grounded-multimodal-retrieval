"""
Project 2B — Retrieval Metrics Evaluation
============================================

Purpose
-------
Implements the final quantitative evaluation suite for
Project 2B.

This script evaluates retrieval quality across multiple
retrieval policies and quantifies the tradeoffs between:

    retrieval flexibility
    retrieval stability
    cross-event transfer
    semantic coherence

It serves as the final numerical evaluation stage of
the retrieval system.

Scientific Motivation
---------------------

Earlier diagnostics revealed:

Vision → Vision

    same_bucket_ratio ≈ 0.9790

while:

Vision → Text

    same_bucket_ratio ≈ 0.0153

This established that:

    visual representations were strong

but:

    unrestricted cross-modal retrieval
    remained unstable.

The key question became:

    Can metadata-aware retrieval constraints
    improve retrieval quality?

This script provides the quantitative answer.

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

This script evaluates the effectiveness of the final
retrieval architecture.

Pipeline Position
-----------------

2A/artifacts/final
            ↓
retrieve_topk.py
            ↓
debug_cross_modal_stats.py
            ↓
retrieve_operational.py
            ↓
retrieve_constrained.py
            ↓
evaluate_constraints.py
            ↓
evaluate_retrieval_metrics.py
            ↓
grounded_multimodal_summary.py

Retrieval Modes Evaluated
-------------------------

none
~~~~

Global unrestricted retrieval.

All candidates remain eligible.

Used as:

    baseline retrieval

same_bucket
~~~~~~~~~~~

Retrieval restricted to the same damage category.

Example:

    flooding
        →
    flooding

disaster_family
~~~~~~~~~~~~~~~

Retrieval restricted to related disaster families.

Example:

    hurricane-florence
        →
    hurricane-harvey

same_event
~~~~~~~~~~

Retrieval restricted to the same disaster event.

Example:

    hurricane-florence
        →
    hurricane-florence

Metrics Computed
----------------

Recall@1
~~~~~~~~

Measures whether the highest-ranked retrieval result
belongs to the correct damage bucket.

Recall@5
~~~~~~~~

Measures whether at least one relevant result appears
within the top-5 retrieved candidates.

MRR
~~~

Mean Reciprocal Rank.

Measures how highly relevant evidence is ranked.

XE Recall@5
~~~~~~~~~~~

Cross-Event Recall@5.

Measures retrieval of semantically relevant evidence
from different disaster events.

This is one of the most important metrics in 2B.

same_bucket_ratio
~~~~~~~~~~~~~~~~~

Fraction of retrieved evidence belonging to the same
damage category as the query.

same_event_ratio
~~~~~~~~~~~~~~~~

Fraction of retrieved evidence originating from the
same disaster event as the query.

avg_unique_events@K
~~~~~~~~~~~~~~~~~~~

Average number of unique source events appearing
within the retrieval neighborhood.

Measures retrieval diversity.

Relevance Definition
--------------------

Relevant Retrieval
~~~~~~~~~~~~~~~~~~

A retrieved candidate is considered relevant if:

    query_bucket
        ==
    candidate_bucket

Cross-Event Relevant Retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A retrieved candidate is considered cross-event relevant if:

    query_bucket
        ==
    candidate_bucket

and:

    query_event
        !=
    candidate_event

Inputs
------

From 2A/artifacts/final/

    vision_embeddings.npy
    vision_index.json

    text_embeddings.npy
    text_index.json

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
Constraint Mode
      ↓
Candidate Filtering
      ↓
Semantic Similarity Ranking
      ↓
Top-K Retrieval
      ↓
Metric Computation
      ↓
Final Evaluation

Outputs
-------

Console Evaluation Report

Displays:

    Recall@1
    Recall@5
    MRR
    XE Recall@5
    same_bucket_ratio
    same_event_ratio
    avg_unique_events@K

for every retrieval mode.

JSON Output

Default:

    2B/results/retrieval_metrics.json

contains:

    per-mode evaluation metrics

for:

    none
    same_bucket
    disaster_family
    same_event

Usage
-----

Default Evaluation

PYTHONPATH=2B python \
2B/scripts/retrieval/evaluate_retrieval_metrics.py

Custom Top-K

PYTHONPATH=2B python \
2B/scripts/retrieval/evaluate_retrieval_metrics.py \
    --topk 5

Custom Output Path

PYTHONPATH=2B python \
2B/scripts/retrieval/evaluate_retrieval_metrics.py \
    --save_path results/custom_metrics.json

Example Output
--------------

======================================================================
MODE: none
======================================================================

Recall@1                 : 0.0000
Recall@5                 : 0.0648
MRR                      : 0.0153
XE Recall@5              : 0.0173
same_bucket_ratio        : 0.0153
same_event_ratio         : 0.0625
avg_unique_events@K      : 3.6787

======================================================================
MODE: same_bucket
======================================================================

Recall@1                 : 1.0000
Recall@5                 : 1.0000
MRR                      : 1.0000
XE Recall@5              : 0.8516
same_bucket_ratio        : 1.0000
same_event_ratio         : 0.3752
avg_unique_events@K      : 1.4006

Files Written
-------------

Default:

    2B/results/retrieval_metrics.json

Role in Project 2B
---------------------

This script serves as:

    the final quantitative evaluation layer

used to measure:

    retrieval quality
    semantic stability
    cross-event transfer
    retrieval diversity
    operational tradeoffs

The resulting metrics form the primary quantitative
evidence presented in the README, slides, and
interview walkthroughs.

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


# =========================================================
# Disaster Families
# =========================================================
# ---------------------------------------------------------
# Disaster-family definitions used by the
# disaster_family retrieval mode.
#
# Enables controlled cross-event retrieval while
# preserving disaster-type consistency.
# ---------------------------------------------------------
DISASTER_FAMILIES = {

    "flooding_family": [
        "hurricane-harvey",
        "hurricane-florence",
        "hurricane-matthew",
        "hurricane-michael",
        "midwest-flooding"
    ],

    "wildfire_family": [
        "socal-fire",
        "santa-rosa-wildfire"
    ],

    "geological_family": [
        "mexico-earthquake",
        "palu-tsunami",
        "guatemala-volcano"
    ]
}


# =========================================================
# Utility Functions
# =========================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# ---------------------------------------------------------
# Computes cosine similarity between embeddings.
#
# Used for semantic ranking after metadata filtering.
# ---------------------------------------------------------
def cosine_similarity(a, b):

    numerator = np.dot(a, b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def separator():
    return "-" * 70


def get_disaster_family(event_id):

    for family_name, events in DISASTER_FAMILIES.items():

        if event_id in events:
            return family_name

    return "unknown_family"


# =========================================================
# Constraint Logic
# =========================================================
# ---------------------------------------------------------
# Constraint-routing function.
#
# Determines whether a retrieval candidate survives
# metadata-aware filtering.
# ---------------------------------------------------------
def candidate_passes_constraint(
    query_meta,
    candidate_meta,
    constraint_type
):

    query_event = query_meta["event_id"]
    query_bucket = query_meta["damage_bucket"]

    candidate_event = candidate_meta["source_event"]
    candidate_bucket = candidate_meta["damage_bucket"]

    # =====================================================
    # NONE
    # =====================================================

    if constraint_type == "none":
        return True

    # =====================================================
    # SAME EVENT
    # =====================================================

    elif constraint_type == "same_event":

        return candidate_event == query_event

    # =====================================================
    # SAME BUCKET
    # =====================================================

    elif constraint_type == "same_bucket":

        return candidate_bucket == query_bucket

    # =====================================================
    # DISASTER FAMILY
    # =====================================================

    elif constraint_type == "disaster_family":

        query_family = get_disaster_family(
            query_event
        )

        candidate_family = get_disaster_family(
            candidate_event
        )

        return query_family == candidate_family

    else:

        raise ValueError(
            f"Unknown constraint: {constraint_type}"
        )


# =========================================================
# Relevance Logic
# =========================================================
# ---------------------------------------------------------
# Relevance definition used by Recall@K and MRR.
#
# Relevant:
#     same damage bucket
# ---------------------------------------------------------
def is_relevant(
    query_meta,
    candidate_meta
):
    """
    Relevance definition:
    same damage bucket
    """

    return (
        query_meta["damage_bucket"]
        ==
        candidate_meta["damage_bucket"]
    )

# ---------------------------------------------------------
# Cross-event relevance definition.
#
# Relevant:
#     same bucket
#
# Different:
#     disaster event
# ---------------------------------------------------------
def is_cross_event_relevant(
    query_meta,
    candidate_meta
):
    """
    XE relevance:
    same bucket
    BUT different event
    """

    same_bucket = (
        query_meta["damage_bucket"]
        ==
        candidate_meta["damage_bucket"]
    )

    different_event = (
        query_meta["event_id"]
        !=
        candidate_meta["source_event"]
    )

    return (
        same_bucket
        and
        different_event
    )


# =========================================================
# Retrieval
# =========================================================
# ---------------------------------------------------------
# Executes constrained semantic retrieval.
#
# Stage 1:
#     metadata filtering
#
# Stage 2:
#     cosine similarity ranking
#
# Returns:
#     top-k retrieval candidates
# ---------------------------------------------------------
def retrieve_candidates(
    query_embedding,
    query_meta,
    text_embeddings,
    text_metadata,
    constraint_type,
    topk
):

    results = []

    for idx in range(len(text_embeddings)):

        meta = text_metadata[idx]

        passes = candidate_passes_constraint(
            query_meta=query_meta,
            candidate_meta=meta,
            constraint_type=constraint_type
        )

        if not passes:
            continue

        score = cosine_similarity(
            query_embedding,
            text_embeddings[idx]
        )

        results.append({
            "score": score,
            "metadata": meta
        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:topk]


# =========================================================
# Metrics
# =========================================================
# ---------------------------------------------------------
# Computes the complete evaluation suite for a
# single retrieval mode.
#
# Metrics:
#     Recall@1
#     Recall@5
#     MRR
#     XE Recall@5
#     same_bucket_ratio
#     same_event_ratio
#     avg_unique_events@K
# ---------------------------------------------------------
def compute_metrics(
    vision_embeddings,
    vision_metadata,
    text_embeddings,
    text_metadata,
    constraint_type,
    topk
):

    recall_at_1 = []
    recall_at_5 = []

    reciprocal_ranks = []

    xe_recall_at_5 = []

    same_bucket_hits = []
    same_event_hits = []

    unique_events = []

    # =====================================================
    # Loop Queries
    # =====================================================
	# ---------------------------------------------------------
	# Evaluate every vision query independently.
	#
	# Metrics are aggregated across the full corpus.
	# ---------------------------------------------------------
    for query_idx in range(
        len(vision_embeddings)
    ):

        query_embedding = vision_embeddings[
            query_idx
        ]

        query_meta = vision_metadata[
            query_idx
        ]

        results = retrieve_candidates(
            query_embedding=query_embedding,
            query_meta=query_meta,
            text_embeddings=text_embeddings,
            text_metadata=text_metadata,
            constraint_type=constraint_type,
            topk=topk
        )

        # =================================================
        # Empty Results
        # =================================================

        if len(results) == 0:

            recall_at_1.append(0)
            recall_at_5.append(0)

            reciprocal_ranks.append(0)

            xe_recall_at_5.append(0)

            same_bucket_hits.append(0)
            same_event_hits.append(0)

            unique_events.append(0)

            continue

		# Measures top-ranked retrieval correctness.
        # =================================================
        # Recall@1
        # =================================================

        top1_meta = results[0]["metadata"]

        r1 = int(
            is_relevant(
                query_meta,
                top1_meta
            )
        )

        recall_at_1.append(r1)

		# Measures whether any relevant evidence appears
		# within the top-k retrieval set.
        # =================================================
        # Recall@5
        # =================================================

        relevant_found = False

        for result in results:

            meta = result["metadata"]

            if is_relevant(
                query_meta,
                meta
            ):
                relevant_found = True
                break

        recall_at_5.append(
            int(relevant_found)
        )

		# Measures ranking quality of relevant evidence.
        # =================================================
        # MRR
        # =================================================

        rr = 0.0

        for rank, result in enumerate(
            results,
            start=1
        ):

            meta = result["metadata"]

            if is_relevant(
                query_meta,
                meta
            ):

                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

		# Measures cross-event semantic transfer capability.
        # =================================================
        # XE Recall@5
        # =================================================

        xe_found = False

        for result in results:

            meta = result["metadata"]

            if is_cross_event_relevant(
                query_meta,
                meta
            ):

                xe_found = True
                break

        xe_recall_at_5.append(
            int(xe_found)
        )

        # =================================================
        # same_bucket_ratio
        # =================================================

        bucket_matches = []

        for result in results:

            meta = result["metadata"]

            bucket_matches.append(
                int(
                    query_meta["damage_bucket"]
                    ==
                    meta["damage_bucket"]
                )
            )

        same_bucket_hits.append(
            np.mean(bucket_matches)
        )

        # =================================================
        # same_event_ratio
        # =================================================

        event_matches = []

        for result in results:

            meta = result["metadata"]

            event_matches.append(
                int(
                    query_meta["event_id"]
                    ==
                    meta["source_event"]
                )
            )

        same_event_hits.append(
            np.mean(event_matches)
        )

        # =================================================
        # avg_unique_events@K
        # =================================================

        retrieved_events = set()

        for result in results:

            meta = result["metadata"]

            retrieved_events.add(
                meta["source_event"]
            )

        unique_events.append(
            len(retrieved_events)
        )
		
	# ---------------------------------------------------------
	# Aggregate query-level metrics into corpus-level
	# evaluation statistics.
	# ---------------------------------------------------------
    # =====================================================
    # Final Aggregation
    # =====================================================

    metrics = {

        "Recall@1":
            float(np.mean(recall_at_1)),

        "Recall@5":
            float(np.mean(recall_at_5)),

        "MRR":
            float(np.mean(reciprocal_ranks)),

        "XE Recall@5":
            float(np.mean(xe_recall_at_5)),

        "same_bucket_ratio":
            float(np.mean(same_bucket_hits)),

        "same_event_ratio":
            float(np.mean(same_event_hits)),

        "avg_unique_events@K":
            float(np.mean(unique_events))
    }

    return metrics


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 70)
    print("2B — Retrieval Metrics Evaluation")
    print("=" * 70)

    artifact_dir = args.artifact_dir

    # =====================================================
    # Load Artifacts
    # =====================================================

    print("\nLoading artifacts...")
    print(separator())

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

    text_embeddings = np.load(
        os.path.join(
            artifact_dir,
            "text_embeddings.npy"
        )
    )

    text_metadata = load_json(
        os.path.join(
            artifact_dir,
            "text_index.json"
        )
    )

    print(
        f"vision_embeddings : "
        f"{vision_embeddings.shape}"
    )

    print(
        f"text_embeddings   : "
        f"{text_embeddings.shape}"
    )

	# ---------------------------------------------------------
	# Compare all retrieval policies using the same
	# evaluation framework.
	# ---------------------------------------------------------
    # =====================================================
    # Evaluate Modes
    # =====================================================

    retrieval_modes = [
        "none",
        "same_bucket",
        "disaster_family",
        "same_event"
    ]

    final_results = {}

    for mode in retrieval_modes:

        print("\n" + "=" * 70)
        print(f"MODE: {mode}")
        print("=" * 70)

        metrics = compute_metrics(
            vision_embeddings=vision_embeddings,
            vision_metadata=vision_metadata,
            text_embeddings=text_embeddings,
            text_metadata=text_metadata,
            constraint_type=mode,
            topk=args.topk
        )

        final_results[mode] = metrics

        for metric_name, value in metrics.items():

            print(
                f"{metric_name:<25}: "
                f"{value:.4f}"
            )

	# ---------------------------------------------------------
	# Persist evaluation results for:
	#
	#     README tables
	#     slide generation
	#     experiment tracking
	# ---------------------------------------------------------
    # =====================================================
    # Save Results
    # =====================================================

    if args.save_path is not None:

        os.makedirs(
            os.path.dirname(args.save_path),
            exist_ok=True
        )

        with open(args.save_path, "w") as f:

            json.dump(
                final_results,
                f,
                indent=4
            )

        print("\n" + "=" * 70)
        print(
            f"[OK] Metrics saved to:"
        )

        print(args.save_path)

	# ---------------------------------------------------------
	# Summarize the primary systems findings revealed
	# by the evaluation suite.
	# ---------------------------------------------------------
    # =====================================================
    # Final Observation
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL OBSERVATION")
    print("=" * 70)

    print(
        "\nThis evaluation compares unrestricted "
        "vs constrained operational retrieval."
    )

    print(
        "\nThe metrics quantify:"
    )

    print(
        "- retrieval quality"
    )

    print(
        "- cross-event transfer"
    )

    print(
        "- semantic stabilization"
    )

    print(
        "- grounding tradeoffs"
    )

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 70)
    print("[OK] Retrieval evaluation complete.")
    print("=" * 70)


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k retrieval"
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to 2A artifacts"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="2B/results/retrieval_metrics.json",
        help="Path to save evaluation metrics"
    )

    args = parser.parse_args()

    main(args)