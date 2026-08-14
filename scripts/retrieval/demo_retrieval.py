"""
Project 2B — Interactive Retrieval Demo
==========================================

Purpose
-------
Provides an interactive demonstration of the retrieval
system developed in Project 2B.

This script compares multiple retrieval strategies
side-by-side for a single vision query and illustrates
how metadata-aware retrieval constraints affect
retrieval behavior.

For each query, the system compares:

    1. unrestricted retrieval
    2. same_bucket retrieval
    3. disaster_family retrieval
    4. same_event retrieval

allowing direct inspection of retrieval stabilization.

Scientific Motivation
---------------------

Project 2B diagnostics revealed:

Vision → Vision

    same_bucket_ratio ≈ 0.9790

while:

Vision → Text

    same_bucket_ratio ≈ 0.0153

This established one of the central findings of
Project 2B:

    unrestricted cross-modal retrieval
    was unstable.

The next question became:

    Can metadata-aware routing improve retrieval
    coherence without retraining the embedding model?

This demo provides a qualitative answer.

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

This script serves as the primary demonstration
interface for the retrieval system.

Pipeline Position
-----------------

2A/artifacts/final
            ↓
retrieve_constrained.py
            ↓
evaluate_constraints.py
            ↓
generate_retrieval_report.py
            ↓
demo_retrieval.py
            ↓
grounded_multimodal_summary.py

Retrieval Modes
---------------

none
~~~~

Global unrestricted retrieval.

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

disaster_family
~~~~~~~~~~~~~~~

Allows retrieval across related disaster events.

Example:

    hurricane-florence
        →
    hurricane-harvey

same_event
~~~~~~~~~~

Only candidates from the same disaster event
survive.

Example:

    hurricane-florence
        →
    hurricane-florence

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
Cosine Similarity Ranking
      ↓
Top-K Evidence
      ↓
Interactive Demonstration

Outputs
-------

Console Demonstration

Displays:

    query metadata

        query_idx
        event
        bucket
        patch_id
        disaster_family

and:

    retrieval results

        score
        source_event
        source_bucket
        evidence text

for every retrieval mode.

Interpretation Layer
--------------------

Each retrieval mode includes a human-readable
interpretation explaining:

    • retrieval behavior
    • retrieval flexibility
    • retrieval stability
    • operational implications

This makes the script useful for:

    interviews
    demos
    README walkthroughs
    qualitative auditing

Usage
-----

Flooding Example

PYTHONPATH=2B python \
2B/scripts/retrieval/demo_retrieval.py \
    --query_idx 100

Wildfire Example

PYTHONPATH=2B python \
2B/scripts/retrieval/demo_retrieval.py \
    --query_idx 600

Custom Top-K

PYTHONPATH=2B python \
2B/scripts/retrieval/demo_retrieval.py \
    --query_idx 100 \
    --topk 5

Example Output
--------------

======================================================================
QUERY
======================================================================

query_idx      : 100
event          : hurricane-florence
bucket         : flooding

======================================================================
RETRIEVAL MODE: none
======================================================================

Interpretation:

Global unrestricted semantic retrieval.
Most unstable retrieval mode.

[1]

score       : 0.0301
event       : palu-tsunami
bucket      : earthquake_collapse

Evidence:

"...retrieved evidence..."

======================================================================
RETRIEVAL MODE: same_bucket
======================================================================

Interpretation:

Semantic bucket gating stabilizes retrieval
semantics strongly.

[1]

score       : 0.0532
event       : hurricane-harvey
bucket      : flooding

Evidence:

"...retrieved evidence..."

======================================================================
FINAL OBSERVATION
======================================================================

This demo illustrates how metadata-aware
constraint routing stabilizes multimodal retrieval.

Files Written
-------------

None.

This script performs interactive retrieval
demonstration only.

Key Findings
------------

Project 2B demonstrated that:

    metadata-aware candidate generation

can substantially improve retrieval coherence
even when unrestricted cross-modal alignment
remains weak.

This became the primary operational retrieval
strategy used throughout the project.

Role in Project 2B
---------------------

This script serves as:

    the primary retrieval demonstration layer

used to:

    compare retrieval modes
    demonstrate stabilization
    support README examples
    support portfolio demonstrations
    support interview walkthroughs

before introducing:

    grounded multimodal retrieval

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


def separator():
    return "-" * 70


# ---------------------------------------------------------
# Computes cosine similarity between embeddings.
#
# Used to rank retrieval candidates after metadata
# filtering has been applied.
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
# the selected retrieval policy.
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
#     top-k retrieval results
# ---------------------------------------------------------
def run_retrieval(
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
# Interpretation
# =========================================================
# ---------------------------------------------------------
# Human-readable explanation for each retrieval mode.
#
# Used to make retrieval behavior easier to understand
# during demonstrations.
# ---------------------------------------------------------
def get_interpretation(mode_name):

    if mode_name == "none":

        return (
            "Global unrestricted semantic retrieval. "
            "Most unstable retrieval mode."
        )

    elif mode_name == "same_bucket":

        return (
            "Semantic bucket gating stabilizes "
            "retrieval semantics strongly."
        )

    elif mode_name == "disaster_family":

        return (
            "Disaster-family routing balances "
            "semantic flexibility and operational stability."
        )

    elif mode_name == "same_event":

        return (
            "Strongest operational grounding but "
            "most restrictive retrieval mode."
        )

    return ""


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 70)
    print("2B — Interactive Retrieval Demo")
    print("=" * 70)

    artifact_dir = args.artifact_dir

	# ---------------------------------------------------------
	# Load multimodal embeddings and metadata exported
	# from Project 2A.
	# ---------------------------------------------------------
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
	# Verify that the requested query index exists
	# within the vision embedding corpus.
	# ---------------------------------------------------------
    # =====================================================
    # Query Validation
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

	# ---------------------------------------------------------
	# Display query metadata for retrieval auditing
	# and demonstration purposes.
	# ---------------------------------------------------------
    # =====================================================
    # Query Display
    # =====================================================

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)

    print(f"\nquery_idx      : {query_idx}")
    print(f"event          : {query_event}")
    print(f"bucket         : {query_bucket}")

    print(
        f"patch_id       : "
        f"{query_meta['patch_id']}"
    )

    print(
        f"disaster_family: "
        f"{get_disaster_family(query_event)}"
    )

	# ---------------------------------------------------------
	# Compare all supported retrieval policies using
	# the same vision query.
	# ---------------------------------------------------------
    # =====================================================
    # Retrieval Modes
    # =====================================================

    modes = [
        "none",
        "same_bucket",
        "disaster_family",
        "same_event"
    ]

    for mode in modes:

        print("\n" + "=" * 70)
        print(f"RETRIEVAL MODE: {mode}")
        print("=" * 70)

        print(
            "\nInterpretation:"
        )

        print(
            get_interpretation(mode)
        )

        print("\n")

        results = run_retrieval(
            query_embedding=query_embedding,
            query_meta=query_meta,
            text_embeddings=text_embeddings,
            text_metadata=text_metadata,
            constraint_type=mode,
            topk=args.topk
        )

		# Handle cases where no candidates survive the
		# selected retrieval constraint.
        # =================================================
        # Empty Results
        # =================================================

        if len(results) == 0:

            print(
                "No candidates survived constraints."
            )

            continue

		# ---------------------------------------------------------
		# Display retrieval results together with
		# provenance information and supporting evidence.
		# ---------------------------------------------------------
        # =================================================
        # Print Results
        # =================================================

        for rank, result in enumerate(
            results,
            start=1
        ):

            meta = result["metadata"]

            print(separator())

            print(f"[{rank}]")

            print(
                f"score       : "
                f"{result['score']:.4f}"
            )

            print(
                f"event       : "
                f"{meta['source_event']}"
            )

            print(
                f"bucket      : "
                f"{meta['damage_bucket']}"
            )

            print("\nEvidence:\n")

            print(meta["text"])

            print("\n")

	# ---------------------------------------------------------
	# Summarize the central systems insight of Project 2B.
	#
	# Unrestricted retrieval was unstable while
	# metadata-aware routing improved retrieval coherence.
	# ---------------------------------------------------------
    # =====================================================
    # Final Diagnosis
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL OBSERVATION")
    print("=" * 70)

    print(
        "\nThis demo illustrates how metadata-aware "
        "constraint routing stabilizes multimodal "
        "retrieval under weak cross-modal alignment."
    )

    print(
        "\nKey systems insight:"
    )

    print(
        "Global unrestricted semantic retrieval "
        "was unstable, while constrained local "
        "retrieval produced more operationally "
        "coherent evidence."
    )

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 70)
    print("[OK] Demo retrieval complete.")
    print("=" * 70)


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
        default=3,
        help="Top-k retrieval count"
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to 2A artifacts"
    )

    args = parser.parse_args()

    main(args)