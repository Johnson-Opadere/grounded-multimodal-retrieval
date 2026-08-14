"""
Project 2B — Grounded Multimodal Summary
===========================================

Purpose
-------
Implements the final retrieval-and-grounding pipeline
for Project 2B.

This script combines:

    1. visual retrieval
    2. constrained text retrieval
    3. metadata-aware routing
    4. evidence aggregation
    5. grounded summary generation
    6. LLM-ready prompt construction

into a single operational workflow.

This is the final integration layer of Project 2B.

Scientific Motivation
---------------------

Project 2B diagnostics revealed:

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

Subsequent experiments showed:

    metadata-aware retrieval constraints

significantly improved retrieval coherence.

This script operationalizes that insight by combining:

    local semantic retrieval
        +
    metadata routing
        +
    evidence grounding

into a complete retrieval pipeline.

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

This script represents the final stage of the system.

Pipeline Position
-----------------

2A/artifacts/final
            ↓
retrieve_topk_vision.py
            ↓
retrieve_constrained.py
            ↓
evaluate_retrieval_metrics.py
            ↓
grounded_multimodal_summary.py

Operational Retrieval Pipeline
------------------------------

Query Disaster Patch
            ↓
Visual Neighbor Retrieval
            ↓
Constraint-Aware Text Retrieval
            ↓
Metadata Routing
            ↓
Evidence Aggregation
            ↓
Grounded Summary Prompt

Retrieval Components
--------------------

Visual Retrieval
~~~~~~~~~~~~~~~~

Retrieves:

    visually similar disaster patches

using:

    FAISS nearest-neighbor search

within the vision embedding manifold.

Text Retrieval
~~~~~~~~~~~~~~

Retrieves:

    semantically relevant textual evidence

using:

    metadata-aware constrained retrieval

Constraint Modes
----------------

none
~~~~

Global unrestricted retrieval.

same_bucket
~~~~~~~~~~~

Retrieval restricted to the same
damage category.

same_event
~~~~~~~~~~

Retrieval restricted to the same
disaster event.

disaster_family
~~~~~~~~~~~~~~~

Retrieval restricted to related
disaster families.

This is the default operational mode.

Inputs
------

From 2A/artifacts/final/

    vision_embeddings.npy
    vision_index.json

    text_embeddings.npy
    text_index.json

Image Repository

    data/images/hold/post_disaster/

Expected Artifact Sizes
-----------------------

vision_embeddings
    (694, 256)

text_embeddings
    (70, 256)

whisper_embeddings
    (11, 256)

Grounded Evidence Construction
------------------------------

Visual Evidence
~~~~~~~~~~~~~~~

For each query:

    retrieve visual neighbors

and report:

    similarity score
    event
    damage bucket
    patch identifier
    image path

Text Evidence
~~~~~~~~~~~~~

For each query:

    retrieve constrained textual evidence

and report:

    similarity score
    source event
    source bucket
    evidence text

Grounded Prompt
~~~~~~~~~~~~~~~

Aggregates:

    visual evidence
    textual evidence

into a structured prompt suitable for:

    LLM summarization
    analyst review
    retrieval auditing

Current Scope
-------------

This script performs:

    visual retrieval
    text retrieval
    metadata routing
    evidence aggregation
    prompt generation

This script does NOT perform:

    LLM API calls
    agent orchestration
    autonomous planning
    image generation
    answer synthesis

The objective is:

    grounded retrieval

rather than:

    autonomous generation.

Outputs
-------

Console Report

Displays:

    query metadata

        query_idx
        event
        bucket
        patch_id
        constraint

Visual Retrieval Section

Displays:

    retrieved visual neighbors

including:

    score
    event
    bucket
    patch_id
    image_path

Text Retrieval Section

Displays:

    retrieved evidence

including:

    score
    event
    bucket
    text evidence

Grounded Prompt Section

Displays:

    complete LLM-ready prompt

constructed from:

    visual evidence
    textual evidence

Usage
-----

Flooding Example

PYTHONPATH=2B python \
2B/scripts/retrieval/grounded_multimodal_summary.py \
    --query_idx 100

Wildfire Example

PYTHONPATH=2B python \
2B/scripts/retrieval/grounded_multimodal_summary.py \
    --query_idx 600

Same-Bucket Retrieval

PYTHONPATH=2B python \
2B/scripts/retrieval/grounded_multimodal_summary.py \
    --query_idx 100 \
    --constraint same_bucket

Disaster-Family Retrieval

PYTHONPATH=2B python \
2B/scripts/retrieval/grounded_multimodal_summary.py \
    --query_idx 100 \
    --constraint disaster_family

Custom Retrieval Sizes

PYTHONPATH=2B python \
2B/scripts/retrieval/grounded_multimodal_summary.py \
    --query_idx 100 \
    --visual_topk 5 \
    --text_topk 5

Example Output
--------------

======================================================================
QUERY
======================================================================

query_idx      : 100
event          : hurricane-florence
bucket         : flooding

======================================================================
RETRIEVED VISUAL NEIGHBORS
======================================================================

[1]

score       : 0.9984
event       : hurricane-harvey
bucket      : flooding

image_path  :
.../hurricane-harvey_00000296_post_disaster.png

======================================================================
RETRIEVED TEXT EVIDENCE
======================================================================

[1]

score       : 0.0532
event       : hurricane-harvey
bucket      : flooding

Evidence:

"...retrieved evidence..."

======================================================================
GROUNDED SUMMARY PROMPT
======================================================================

You are given grounded multimodal disaster retrieval
evidence...

Files Written
-------------

None.

This script performs retrieval and prompt generation only.

Key Findings
------------

Project 2B demonstrated that:

    metadata-aware retrieval routing

combined with:

    local semantic retrieval

produces substantially more coherent evidence than:

    unrestricted global retrieval.

This script operationalizes the final retrieval system.

Role in Project 2B
---------------------

This script serves as:

    the final integration layer

combining:

    visual retrieval
    constrained retrieval
    evidence aggregation
    retrieval grounding

into a single operational pipeline.

It represents the complete retrieval workflow
developed throughout Project 2B.

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
# Disaster Families
# =========================================================
# ---------------------------------------------------------
# Disaster-family definitions used by the
# disaster_family routing mode.
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
# Used for constrained text retrieval ranking.
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
# Metadata routing function.
#
# Determines whether a retrieval candidate survives
# the selected constraint strategy.
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
# Build Image Path
# =========================================================
# ---------------------------------------------------------
# Converts a patch identifier into the corresponding
# post-disaster image path.
#
# Example:
#
# hurricane-florence_00000465
#
# ->
#
# hurricane-florence_00000465_post_disaster.png
# ---------------------------------------------------------
def build_image_path(
    image_root,
    patch_id
):
    """
    Converts:
        hurricane-florence_00000465

    into:
        .../hurricane-florence_00000465_post_disaster.png
    """

    filename = (
        f"{patch_id}_post_disaster.png"
    )

    return os.path.join(
        image_root,
        filename
    )


# =========================================================
# Visual Retrieval
# =========================================================
# ---------------------------------------------------------
# Retrieves top-k nearest visual neighbors using
# FAISS similarity search.
#
# Used to provide visual grounding evidence.
# ---------------------------------------------------------
def retrieve_visual_neighbors(
    query_idx,
    vision_embeddings,
    vision_metadata,
    topk
):

	# Build exact FAISS retrieval backend for vision
	# nearest-neighbor search.
    index = faiss.IndexFlatIP(
        vision_embeddings.shape[1]
    )

    index.add(
        vision_embeddings.astype(np.float32)
    )

    query_embedding = np.expand_dims(
        vision_embeddings[query_idx].astype(np.float32),
        axis=0
    )

	# Retrieve nearest visual neighbors.
	#
	# The query itself will appear first and is removed.
    scores, indices = index.search(
        query_embedding,
        topk + 1
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == query_idx:
            continue

        results.append({
            "score": float(score),
            "metadata": vision_metadata[idx]
        })

    return results[:topk]


# =========================================================
# Text Retrieval
# =========================================================
# ---------------------------------------------------------
# Performs metadata-aware constrained text retrieval.
#
# Stage 1:
#     constraint filtering
#
# Stage 2:
#     cosine similarity ranking
#
# Returns:
#     top-k textual evidence
# ---------------------------------------------------------
def retrieve_text_evidence(
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
# Grounded Prompt Builder
# =========================================================
# ---------------------------------------------------------
# Constructs an LLM-ready grounded prompt from:
#
#     visual retrieval evidence
#     textual retrieval evidence
#
# This function centralizes prompt generation so
# downstream scripts (e.g. GPT-4o / Claude report
# generation) can reuse the exact same grounding
# logic without duplicating code.
#
# Returns
# -------
# str
#
#     Fully formatted grounded prompt suitable
#     for LLM consumption.
# ---------------------------------------------------------
def build_grounded_prompt(
    visual_results,
    text_results
):

    summary_lines = []

    summary_lines.append(
        "You are given grounded multimodal "
        "disaster retrieval evidence."
    )

    summary_lines.append("")

    summary_lines.append(
        "Summarize:"
    )

    summary_lines.append(
        "- likely disaster characteristics"
    )

    summary_lines.append(
        "- observed damage patterns"
    )

    summary_lines.append(
        "- operational implications"
    )

    summary_lines.append("")

    summary_lines.append(
        "Only use the retrieved evidence."
    )

    summary_lines.append(
        "Do not invent unsupported claims."
    )

    summary_lines.append("")
    summary_lines.append("")

    summary_lines.append(
        "VISUAL NEIGHBORS:"
    )

    for result in visual_results:

        meta = result["metadata"]

        summary_lines.append(
            f"- {meta['event_id']} | "
            f"{meta['damage_bucket']}"
        )

    summary_lines.append("")

    summary_lines.append(
        "TEXT EVIDENCE:"
    )

    for result in text_results:

        meta = result["metadata"]

        summary_lines.append(
            f"- {meta['text']}"
        )

    return "\n".join(summary_lines)

# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 70)
    print("2B — Grounded Multimodal Summary")
    print("=" * 70)

    artifact_dir = args.artifact_dir

	# ---------------------------------------------------------
	# Load embeddings and metadata exported from
	# Project 2A.
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
	# Verify query index exists within the vision corpus.
	# ---------------------------------------------------------
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

    query_image_path = build_image_path(
        args.image_root,
        query_meta["patch_id"]
    )

	# ---------------------------------------------------------
	# Display query metadata and image location for
	# retrieval auditing.
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
        f"constraint     : "
        f"{args.constraint}"
    )

    print(
        f"\nquery_image    : "
        f"{query_image_path}"
    )

	# ---------------------------------------------------------
	# Retrieve visually similar disaster patches.
	#
	# Provides visual grounding evidence.
	# ---------------------------------------------------------
    # =====================================================
    # Visual Retrieval
    # =====================================================

    print("\n" + "=" * 70)
    print("RETRIEVED VISUAL NEIGHBORS")
    print("=" * 70)

    visual_results = retrieve_visual_neighbors(
        query_idx=query_idx,
        vision_embeddings=vision_embeddings,
        vision_metadata=vision_metadata,
        topk=args.visual_topk
    )

    for rank, result in enumerate(
        visual_results,
        start=1
    ):

        meta = result["metadata"]

        image_path = build_image_path(
            args.image_root,
            meta["patch_id"]
        )

        print(separator())

        print(f"[{rank}]")

        print(
            f"score       : "
            f"{result['score']:.4f}"
        )

        print(
            f"event       : "
            f"{meta['event_id']}"
        )

        print(
            f"bucket      : "
            f"{meta['damage_bucket']}"
        )

        print(
            f"patch_id    : "
            f"{meta['patch_id']}"
        )

        print(
            f"\nimage_path  : "
            f"{image_path}"
        )

	# ---------------------------------------------------------
	# Retrieve textual evidence using metadata-aware
	# constrained retrieval.
	# ---------------------------------------------------------
    # =====================================================
    # Text Retrieval
    # =====================================================

    print("\n" + "=" * 70)
    print("RETRIEVED TEXT EVIDENCE")
    print("=" * 70)

    text_results = retrieve_text_evidence(
        query_embedding=query_embedding,
        query_meta=query_meta,
        text_embeddings=text_embeddings,
        text_metadata=text_metadata,
        constraint_type=args.constraint,
        topk=args.text_topk
    )

    for rank, result in enumerate(
        text_results,
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

	# ---------------------------------------------------------
	# Aggregate visual and textual evidence into an
	# LLM-ready grounded prompt.
	#
	# The prompt is intentionally constrained to
	# retrieved evidence only.
	# ---------------------------------------------------------
    # =====================================================
    # Grounded Summary Prompt
    # =====================================================

    print("\n" + "=" * 70)
    print("GROUNDED SUMMARY PROMPT")
    print("=" * 70)

    summary_prompt = build_grounded_prompt(
    visual_results=visual_results,
    text_results=text_results
)

    print("\n")
    print(summary_prompt)

	# ---------------------------------------------------------
	# Summarize the final systems insight of Project 2B.
	#
	# Metadata-aware retrieval routing improves retrieval
	# coherence without retraining embedding models.
	# ---------------------------------------------------------
    # =====================================================
    # Final Observation
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL OBSERVATION")
    print("=" * 70)

    print(
        "\nThis pipeline demonstrates grounded "
        "multimodal operational retrieval."
    )

    print(
        "\nThe system combines:"
    )

    print(
        "- visual semantic retrieval"
    )

    print(
        "- metadata-aware routing"
    )

    print(
        "- constrained text retrieval"
    )

    print(
        "- grounded evidence aggregation"
    )

    print(
        "- LLM-ready synthesis prompts"
    )

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 70)
    print("[OK] Grounded multimodal summary complete.")
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
        "--constraint",
        type=str,
        default="disaster_family",
        choices=[
            "none",
            "same_bucket",
            "same_event",
            "disaster_family"
        ],
        help="Constraint routing strategy"
    )

    parser.add_argument(
        "--visual_topk",
        type=int,
        default=3,
        help="Visual retrieval top-k"
    )

    parser.add_argument(
        "--text_topk",
        type=int,
        default=3,
        help="Text retrieval top-k"
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to 2A artifacts"
    )

    parser.add_argument(
        "--image_root",
        type=str,
        default=(
            "/mnt/ebs-data/cv_project2/"
            "data/images/hold/post_disaster"
        ),
        help="Path to post-disaster images"
    )

    args = parser.parse_args()

    main(args)