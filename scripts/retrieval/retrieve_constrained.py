"""
Project 2B — Constrained Retrieval
=====================================

Purpose
-------
Implements metadata-constrained multimodal retrieval for
Project 2B.

This script represents the primary retrieval stabilization
mechanism introduced after diagnostic analysis revealed that:

    • vision representations were strong
    • unrestricted cross-modal retrieval was weak
    • global candidate generation was unstable

Instead of performing:

    global semantic retrieval

across the entire corpus,

this script first performs:

    metadata-aware candidate filtering

and only then performs:

    semantic ranking

within the surviving candidate set.

This approach dramatically improves retrieval stability.

Scientific Motivation
---------------------

Earlier diagnostics revealed:

Vision → Vision

    same_bucket_ratio ≈ 0.9790

while:

Vision → Text

    same_bucket_ratio ≈ 0.0153

This established one of the central findings
of Project 2B:

    semantic reranking only works when the
    candidate pool is already relevant.

Therefore:

    candidate generation becomes critical.

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

This script introduces the first hard retrieval constraints
used by the operational retrieval system.

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
grounded_multimodal_summary.py

Current Scope
-------------

This script supports:

    same_event filtering
    same_bucket filtering
    disaster_family filtering
    unrestricted retrieval

followed by:

    semantic similarity ranking

This script does NOT implement:

    neural rerankers
    learning-to-rank
    cross-encoders
    reinforcement learning
    LLM reranking

The objective is:

    deterministic retrieval stabilization

rather than:

    model retraining.

Constraint Types
----------------

1. none
~~~~~~~~

No filtering.

All text candidates remain eligible.

Used as:

    unrestricted baseline

2. same_event
~~~~~~~~~~~~~

Only candidates originating from the same disaster
event survive.

Example:

    hurricane-florence
        →
    hurricane-florence

3. same_bucket
~~~~~~~~~~~~~~

Only candidates belonging to the same damage
category survive.

Example:

    flooding
        →
    flooding

This became the primary operational constraint
used in Project 2B.

4. disaster_family
~~~~~~~~~~~~~~~~~~

Allows retrieval across related disaster families.

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

Retrieval Flow
--------------

Vision Query
      ↓
Metadata Constraint
      ↓
Candidate Filtering
      ↓
Semantic Similarity Ranking
      ↓
Top-K Retrieval

Inputs
------

From 2A/artifacts/final/

    vision_embeddings.npy
    vision_index.json

    text_embeddings.npy

From 2B/indexes/

    text_metadata.json

Expected Artifact Sizes
-----------------------

vision_embeddings
    (694, 256)

text_embeddings
    (70, 256)

whisper_embeddings
    (11, 256)

Similarity Computation
----------------------

Uses cosine similarity:

                  a · b
sim(a,b) = ------------------
           ||a|| × ||b||

Since embeddings are already L2-normalized:

    cosine similarity

provides direct semantic similarity measurement.

Outputs
-------

Console Retrieval Report

Displays:

    query metadata

        query_idx
        event
        bucket
        constraint

and:

    constrained retrieval results

        semantic_score
        source_event
        damage_bucket
        evidence text

Usage
-----

Unrestricted Retrieval

PYTHONPATH=2B python \
2B/scripts/retrieval/retrieve_constrained.py \
    --query_idx 100 \
    --constraint none

Same Bucket Retrieval

PYTHONPATH=2B python \
2B/scripts/retrieval/retrieve_constrained.py \
    --query_idx 100 \
    --constraint same_bucket

Disaster Family Retrieval

PYTHONPATH=2B python \
2B/scripts/retrieval/retrieve_constrained.py \
    --query_idx 100 \
    --constraint disaster_family

Same Event Retrieval

PYTHONPATH=2B python \
2B/scripts/retrieval/retrieve_constrained.py \
    --query_idx 100 \
    --constraint same_event

Example Output
--------------

============================================================
Query
============================================================

query_idx      : 100
event          : hurricane-florence
bucket         : flooding
constraint     : same_bucket

Generating constrained candidates...
------------------------------------------------------------
candidate_count : 10

============================================================
Top-5 Constrained Retrievals
============================================================

[1]

semantic_score : 0.0532

event          : hurricane-harvey
bucket         : flooding

"...retrieved evidence..."

------------------------------------------------------------

Files Written
-------------
None.

This script performs constrained retrieval and prints
results to the terminal.

Key Findings
------------

Project 2B demonstrated that:

    metadata-aware candidate generation

can dramatically improve retrieval quality even when:

    unrestricted cross-modal alignment remains weak.

This became the primary operational retrieval strategy
used throughout the remainder of the project.

Role in Project 2B
---------------------

This script serves as:

    the primary retrieval stabilization layer

used to demonstrate:

    metadata-aware retrieval routing
    constrained candidate generation
    semantic retrieval stabilization
    operational retrieval policies

before introducing:

    retrieval evaluation
    grounded retrieval
    multimodal evidence aggregation

Author
------
Project 2B
Multimodal Retrieval Systems Engineering
"""

import os
import json
import argparse

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


def print_separator():
    print("-" * 60)

# ---------------------------------------------------------
# Computes cosine similarity between two embeddings.
#
# Used after metadata filtering to rank surviving
# retrieval candidates.
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

# ---------------------------------------------------------
# Maps an event identifier to its disaster family.
#
# Used by disaster_family retrieval mode.
# ---------------------------------------------------------
def get_disaster_family(event_id):

    for family_name, events in DISASTER_FAMILIES.items():

        if event_id in events:
            return family_name

    return "unknown_family"


# =========================================================
# Candidate Filtering
# =========================================================
# ---------------------------------------------------------
# Metadata gating function.
#
# Determines whether a retrieval candidate survives
# constraint-based filtering before semantic ranking.
# ---------------------------------------------------------
def candidate_passes_constraint(
    query_meta,
    candidate_meta,
    constraint_type
):
    """
    Determines whether candidate survives metadata gating.
    """

    query_event = query_meta["event_id"]
    query_bucket = query_meta["damage_bucket"]

    candidate_event = candidate_meta["source_event"]
    candidate_bucket = candidate_meta["damage_bucket"]

    # =====================================================
    # SAME EVENT
    # =====================================================

    if constraint_type == "same_event":

        return candidate_event == query_event

    # =====================================================
    # SAME BUCKET
    # =====================================================

    elif constraint_type == "same_bucket":

        return candidate_bucket == query_bucket
	
	# ---------------------------------------------------------
	# Event groupings used by the disaster_family constraint.
	#
	# Enables controlled cross-event retrieval while preserving
	# broad disaster-type consistency.
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

    # =====================================================
    # NONE
    # =====================================================

    elif constraint_type == "none":

        return True

    else:

        raise ValueError(
            f"Unknown constraint: {constraint_type}"
        )


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 60)
    print("2B — Constrained Retrieval")
    print("=" * 60)

    artifact_dir = args.artifact_dir
    index_dir = args.index_dir

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

    # =====================================================
    # Load Text Metadata
    # =====================================================

    print("\nLoading text retrieval corpus...")
    print_separator()

    text_embeddings = np.load(
        os.path.join(
            artifact_dir,
            "text_embeddings.npy"
        )
    )

    text_metadata = load_json(
        os.path.join(
            index_dir,
            "text_metadata.json"
        )
    )

    print(
        f"text_embeddings : "
        f"{text_embeddings.shape}"
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
        f"constraint     : "
        f"{args.constraint}"
    )

	# ---------------------------------------------------------
	# Stage 1:
	#
	# Apply metadata constraints to reduce the candidate pool.
	#
	# This is the key stabilization mechanism of Project 2B.
	# ---------------------------------------------------------
    # =====================================================
    # Constrained Candidate Generation
    # =====================================================

    print("\nGenerating constrained candidates...")
    print_separator()

    candidate_results = []

    for idx in range(len(text_embeddings)):

        candidate_meta = text_metadata[idx]

        # =============================================
        # Metadata Gating
        # =============================================
		# Apply selected retrieval constraint.
        passes = candidate_passes_constraint(
            query_meta=query_meta,
            candidate_meta=candidate_meta,
            constraint_type=args.constraint
        )

        if not passes:
            continue
			
		# ---------------------------------------------------------
		# Stage 2:
		#
		# Rank surviving candidates using cosine similarity.
		#
		# Retrieval quality now depends on:
		#
		#     constrained candidate generation
		#         +
		#     semantic ranking
		# ---------------------------------------------------------
        # =============================================
        # Semantic Similarity
        # =============================================

        score = cosine_similarity(
            query_embedding,
            text_embeddings[idx]
        )

        candidate_results.append({
            "idx": idx,
            "score": score,
            "metadata": candidate_meta
        })

    # =====================================================
    # Sort By Semantic Similarity
    # =====================================================
	# Rank surviving candidates by semantic similarity.
    candidate_results = sorted(
        candidate_results,
        key=lambda x: x["score"],
        reverse=True
    )

    print(
        f"candidate_count : "
        f"{len(candidate_results)}"
    )

	# ---------------------------------------------------------
	# Display final constrained retrieval results.
	#
	# Useful for qualitative inspection of stabilization
	# behavior under different constraint modes.
	# ---------------------------------------------------------
    # =====================================================
    # Print Results
    # =====================================================

    print("\n" + "=" * 60)
    print(
        f"Top-{args.topk} Constrained Retrievals"
    )
    print("=" * 60)

    if len(candidate_results) == 0:

        print("\nNo candidates survived constraints.")

    else:

        for rank, result in enumerate(
            candidate_results[:args.topk],
            start=1
        ):

            meta = result["metadata"]

            print(f"\n[{rank}]")

            print(
                f"semantic_score : "
                f"{result['score']:.4f}"
            )

            print(
                f"\nevent          : "
                f"{meta['source_event']}"
            )

            print(
                f"bucket         : "
                f"{meta['damage_bucket']}"
            )

            print("\n" + meta["text"])

            print_separator()

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 60)
    print("[OK] Constrained retrieval complete.")
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
        help="Top-k results"
    )

    parser.add_argument(
        "--constraint",
        type=str,
        default="same_bucket",
        choices=[
            "none",
            "same_event",
            "same_bucket",
            "disaster_family"
        ],
        help="Metadata constraint type"
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