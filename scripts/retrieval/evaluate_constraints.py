"""
Project 2B — Constraint Evaluation
=====================================

Purpose
-------
Evaluates and compares multiple metadata-aware retrieval
strategies introduced in Project 2B.

This script quantifies the impact of retrieval constraints
on semantic retrieval behavior.

Specifically, it compares:

    1. unrestricted retrieval
    2. same_bucket retrieval
    3. disaster_family retrieval
    4. same_event retrieval

using a common set of retrieval metrics.

Scientific Motivation
---------------------

Earlier diagnostics revealed:

Vision → Vision

    same_bucket_ratio ≈ 0.9790

while:

Vision → Text

    same_bucket_ratio ≈ 0.0153

This established that:

    unrestricted cross-modal retrieval

was substantially weaker than:

    visual semantic organization.

The key question became:

    Can metadata-aware retrieval constraints
    stabilize retrieval behavior?

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

This script evaluates the effectiveness of the retrieval
constraints introduced in:

    retrieve_constrained.py

Pipeline Position
-----------------

2A/artifacts/final
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

Retrieval Modes
---------------

none
~~~~

Unrestricted retrieval.

All candidates remain eligible.

Used as:

    baseline retrieval

same_bucket
~~~~~~~~~~~

Only candidates belonging to the same damage
category survive.

Example:

    flooding
        →
    flooding

same_event
~~~~~~~~~~

Only candidates originating from the same disaster
event survive.

Example:

    hurricane-florence
        →
    hurricane-florence

disaster_family
~~~~~~~~~~~~~~~

Allows retrieval among related disaster families.

Flooding Family
~~~~~~~~~~~~~~~

    hurricane-harvey
    hurricane-florence
    hurricane-matthew
    hurricane-michael
    midwest-flooding

Wildfire Family
~~~~~~~~~~~~~~~

    socal-fire
    santa-rosa-wildfire

Geological Family
~~~~~~~~~~~~~~~~~

    mexico-earthquake
    palu-tsunami
    guatemala-volcano

Metrics Computed
----------------

same_bucket_ratio
~~~~~~~~~~~~~~~~~

Fraction of retrieved evidence belonging to
the same damage category as the query.

same_event_ratio
~~~~~~~~~~~~~~~~

Fraction of retrieved evidence originating
from the same disaster event as the query.

avg_unique_events
~~~~~~~~~~~~~~~~~

Average number of unique events observed in
the retrieval neighborhood.

Measures retrieval diversity.

candidate_survival_rate
~~~~~~~~~~~~~~~~~~~~~~~

Average number of candidates surviving
metadata filtering.

Measures retrieval flexibility.

score_mean
~~~~~~~~~~

Mean semantic similarity score.

score_std
~~~~~~~~~

Standard deviation of similarity scores.

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
Semantic Similarity
      ↓
Top-K Retrieval
      ↓
Metric Aggregation

Outputs
-------

Console Evaluation Report

Displays:

    retrieval metrics

        same_bucket_ratio
        same_event_ratio
        avg_unique_events
        candidate_survival_rate

for every retrieval mode.

Also reports:

    score_mean
    score_std

for each mode.

Usage
-----

Full Evaluation

PYTHONPATH=2B python \
2B/scripts/retrieval/evaluate_constraints.py

Debug Evaluation

PYTHONPATH=2B python \
2B/scripts/retrieval/evaluate_constraints.py \
    --num_queries 100

Custom Top-K

PYTHONPATH=2B python \
2B/scripts/retrieval/evaluate_constraints.py \
    --topk 10

Example Output
--------------

============================================================
Constraint Evaluation Summary
============================================================

Mode                 same_bucket   same_event   avg_events
-----------------------------------------------------------
none                 0.0153        0.0625       3.68

same_bucket          1.0000        0.3752       1.40

disaster_family      0.0622        0.2767       2.93

same_event           0.3451        1.0000       1.00

============================================================
Score Statistics
============================================================

Mode                 score_mean    score_std
-----------------------------------------------------------
none                 0.0373        0.0340

same_bucket          ...

Files Written
-------------
None.

This script performs evaluation only and prints
results to the terminal.

Key Findings
------------

Project 2B demonstrated that:

    metadata-aware candidate generation

can dramatically improve retrieval coherence.

The strongest operational retrieval mode was:

    same_bucket

because it balanced:

    retrieval quality

and:

    retrieval flexibility.

This became the primary retrieval policy
used throughout the remainder of the project.

Role in Project 2B
---------------------

This script serves as:

    the retrieval-policy evaluation layer

used to compare:

    unrestricted retrieval

versus:

    constrained retrieval

and quantify:

    retrieval stabilization
    candidate diversity
    retrieval tradeoffs

before introducing:

    final retrieval metrics
    grounded multimodal retrieval

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
# Used after metadata filtering to rank retrieval
# candidates.
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


def get_disaster_family(event_id):

    for family_name, events in DISASTER_FAMILIES.items():

        if event_id in events:
            return family_name

    return "unknown_family"


# =========================================================
# Constraint Logic
# =========================================================
# ---------------------------------------------------------
# Constraint-gating function.
#
# Determines whether a retrieval candidate survives
# metadata filtering.
#
# This is the core mechanism behind constrained
# retrieval stabilization.
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
	
	# ---------------------------------------------------------
	# Disaster-family definitions used by the
	# disaster_family retrieval mode.
	#
	# Enables controlled cross-event retrieval while
	# preserving broad disaster-type consistency.
	# ---------------------------------------------------------
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
# Evaluate Single Constraint
# =========================================================
# ---------------------------------------------------------
# Evaluates a single retrieval mode across all
# selected query embeddings.
#
# Computes:
#     same_bucket_ratio
#     same_event_ratio
#     avg_unique_events
#     candidate_survival_rate
#     score statistics
# ---------------------------------------------------------
def evaluate_constraint_mode(
    mode_name,
    vision_embeddings,
    vision_metadata,
    text_embeddings,
    text_metadata,
    topk,
    num_queries
):

    total_neighbors = 0

    same_bucket_count = 0
    same_event_count = 0

    unique_event_counts = []

    all_scores = []

    candidate_survival_counts = []

    # =====================================================
    # Main Query Loop
    # =====================================================
	# ---------------------------------------------------------
	# Main evaluation loop.
	#
	# Each query is evaluated independently under the
	# selected retrieval constraint.
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

        candidate_results = []

		# ---------------------------------------------------------
		# Stage 1:
		#
		# Apply metadata constraints to generate the
		# retrieval candidate pool.
		# ---------------------------------------------------------
        # =================================================
        # Candidate Generation
        # =================================================

        for idx in range(len(text_embeddings)):

            candidate_meta = text_metadata[idx]
			
			# Apply selected retrieval constraint.
            passes = candidate_passes_constraint(
                query_meta=query_meta,
                candidate_meta=candidate_meta,
                constraint_type=mode_name
            )

            if not passes:
                continue
			
			# Stage 2:
			#
			# Rank surviving candidates using semantic similarity.
            score = cosine_similarity(
                query_embedding,
                text_embeddings[idx]
            )

            candidate_results.append({
                "score": score,
                "metadata": candidate_meta
            })

        candidate_survival_counts.append(
            len(candidate_results)
        )

        # =================================================
        # Sort Candidates
        # =================================================

        candidate_results = sorted(
            candidate_results,
            key=lambda x: x["score"],
            reverse=True
        )

        candidate_results = candidate_results[:topk]

        retrieved_events = set()
		
		# ---------------------------------------------------------
		# Compare all supported retrieval policies using
		# a common evaluation framework.
		# ---------------------------------------------------------
        # =================================================
        # Evaluate Results
        # =================================================

        for result in candidate_results:

            meta = result["metadata"]

            neighbor_event = meta["source_event"]
            neighbor_bucket = meta[
                "damage_bucket"
            ]

            total_neighbors += 1

            all_scores.append(result["score"])

            retrieved_events.add(
                neighbor_event
            )

            # =============================================
            # Same Bucket
            # =============================================

            if neighbor_bucket == query_bucket:
                same_bucket_count += 1

            # =============================================
            # Same Event
            # =============================================

            if neighbor_event == query_event:
                same_event_count += 1

        unique_event_counts.append(
            len(retrieved_events)
        )
	
	# ---------------------------------------------------------
	# Convert raw retrieval statistics into interpretable
	# evaluation metrics.
	# ---------------------------------------------------------
    # =====================================================
    # Aggregate Metrics
    # =====================================================

    if total_neighbors == 0:

        same_bucket_ratio = 0.0
        same_event_ratio = 0.0

    else:

        same_bucket_ratio = (
            same_bucket_count / total_neighbors
        )

        same_event_ratio = (
            same_event_count / total_neighbors
        )

    avg_unique_events = np.mean(
        unique_event_counts
    )

    avg_candidate_survival = np.mean(
        candidate_survival_counts
    )

    if len(all_scores) > 0:

        score_mean = np.mean(all_scores)
        score_std = np.std(all_scores)

    else:

        score_mean = 0.0
        score_std = 0.0

    return {
        "mode": mode_name,
        "same_bucket_ratio": same_bucket_ratio,
        "same_event_ratio": same_event_ratio,
        "avg_unique_events": avg_unique_events,
        "avg_candidate_survival":
            avg_candidate_survival,
        "score_mean": score_mean,
        "score_std": score_std
    }


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 60)
    print("2B — Constraint Evaluation")
    print("=" * 60)

    artifact_dir = args.artifact_dir

    # =====================================================
    # Load Data
    # =====================================================

    print("\nLoading artifacts...")

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

    # =====================================================
    # Query Count
    # =====================================================

    total_queries = len(vision_embeddings)

    if args.num_queries is None:

        num_queries = total_queries

    else:

        num_queries = min(
            args.num_queries,
            total_queries
        )

    print(
        f"\nnum_queries       : "
        f"{num_queries}"
    )

    print(
        f"topk              : "
        f"{args.topk}"
    )

    # =====================================================
    # Constraint Modes
    # =====================================================

    modes = [
        "none",
        "same_bucket",
        "disaster_family",
        "same_event"
    ]

    results = []

    # =====================================================
    # Evaluate Modes
    # =====================================================

    for mode_name in modes:

        print(
            f"\nEvaluating mode: {mode_name}"
        )

        result = evaluate_constraint_mode(
            mode_name=mode_name,
            vision_embeddings=vision_embeddings,
            vision_metadata=vision_metadata,
            text_embeddings=text_embeddings,
            text_metadata=text_metadata,
            topk=args.topk,
            num_queries=num_queries
        )

        results.append(result)
	
	# ---------------------------------------------------------
	# Display retrieval-policy comparison results.
	#
	# This table summarizes the tradeoffs between:
	#
	#     retrieval flexibility
	#
	# and:
	#
	#     retrieval stability
	# ---------------------------------------------------------
    # =====================================================
    # Print Summary Table
    # =====================================================

    print("\n" + "=" * 60)
    print("Constraint Evaluation Summary")
    print("=" * 60)

    header = (
        f"\n{'Mode':<20}"
        f"{'same_bucket':<15}"
        f"{'same_event':<15}"
        f"{'avg_events':<15}"
        f"{'cand_survival':<15}"
    )

    print(header)

    print("-" * 80)

    for result in results:

        row = (
            f"{result['mode']:<20}"
            f"{result['same_bucket_ratio']:<15.4f}"
            f"{result['same_event_ratio']:<15.4f}"
            f"{result['avg_unique_events']:<15.2f}"
            f"{result['avg_candidate_survival']:<15.2f}"
        )

        print(row)

    # =====================================================
    # Score Statistics
    # =====================================================

    print("\n" + "=" * 60)
    print("Score Statistics")
    print("=" * 60)

    header = (
        f"\n{'Mode':<20}"
        f"{'score_mean':<15}"
        f"{'score_std':<15}"
    )

    print(header)

    print("-" * 50)

    for result in results:

        row = (
            f"{result['mode']:<20}"
            f"{result['score_mean']:<15.4f}"
            f"{result['score_std']:<15.4f}"
        )

        print(row)

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 60)
    print("[OK] Constraint evaluation complete.")
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
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to 2A artifacts"
    )

    args = parser.parse_args()

    main(args)