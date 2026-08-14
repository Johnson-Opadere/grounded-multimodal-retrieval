"""
Project 2B — Vision-to-Vision Retrieval
==========================================

Purpose
-------
Performs nearest-neighbor retrieval directly within the
vision embedding space produced by Project 2A.

Given:

    a vision query embedding

the system retrieves:

    top-k nearest visual neighbors

using dense vector similarity search.

This script is a critical diagnostic component of
Project 2B.

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

Before evaluating cross-modal retrieval quality,
it is important to determine whether the vision
embedding manifold itself is semantically coherent.

Scientific Motivation
----------------------

This script helps answer:

    Is the vision representation healthy?

If:

    vision → vision retrieval is strong

but:

    vision → text retrieval is weak

then:

    the vision encoder learned meaningful disaster
    semantics,

while:

    cross-modal alignment remains the primary bottleneck.

This distinction became one of the key findings
of Project 2B.

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
retrieval diagnostics

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

Retrieval Method
----------------

FAISS Backend
~~~~~~~~~~~~~

Uses:

    faiss.IndexFlatIP

All vision embeddings are L2-normalized.

Therefore:

    cosine_similarity(a,b)
        ==
    inner_product(a,b)

which allows exact semantic retrieval.

Query Flow
----------

Vision Query
      ↓
Vision Embedding
      ↓
FAISS Similarity Search
      ↓
Top-K Vision Neighbors

Outputs
-------

Console Retrieval Report

Displays:

    query metadata

        event
        bucket
        patch_id
        split

and:

    retrieved visual neighbors

        score
        event
        bucket
        patch_id

Usage
-----

Default:

PYTHONPATH=2B python \
2B/scripts/retrieval/retrieve_topk_vision.py \
    --query_idx 100 \
    --topk 5

Wildfire Example:

PYTHONPATH=2B python \
2B/scripts/retrieval/retrieve_topk_vision.py \
    --query_idx 600 \
    --topk 5

Example Output
--------------

============================================================
Query
============================================================

query_idx       : 100
event           : hurricane-florence
bucket          : flooding
patch_id        : hurricane-florence_00000465
split           : hold

============================================================
Top-5 Retrieved Vision Neighbors
============================================================

[1]

score       : 0.9984
event       : hurricane-harvey
bucket      : flooding
patch_id    : hurricane-harvey_00000296

------------------------------------------------------------

[2]

score       : 0.9984
event       : hurricane-harvey
bucket      : flooding
patch_id    : hurricane-harvey_00000190

------------------------------------------------------------

Files Written
-------------
None.

This script performs retrieval and prints results
to the terminal.

Key Findings
------------

Project 2B diagnostics showed:

    vision→vision same_bucket_ratio ≈ 0.979

indicating that the visual embedding manifold
learned highly coherent disaster semantics.

This result provided strong evidence that:

    vision representations were healthy

and that:

    cross-modal alignment remained the primary
    retrieval bottleneck.

Role in Project 2B
---------------------

This script serves as:

    the primary vision-manifold diagnostic

used to verify:

    semantic clustering
    disaster similarity structure
    cross-event visual coherence

before:

    cross-modal diagnostics
    retrieval stabilization
    constraint-aware routing

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


def print_separator():
    print("-" * 60)


def print_query_info(query_idx, meta):

    print("\n" + "=" * 60)
    print("Query")
    print("=" * 60)

    print(f"\nquery_idx       : {query_idx}")
    print(f"event           : {meta['event_id']}")
    print(f"bucket          : {meta['damage_bucket']}")
    print(f"patch_id        : {meta['patch_id']}")
    print(f"split           : {meta['split']}")


# ---------------------------------------------------------
# Pretty-print a retrieved visual neighbor.
#
# Displays:
#     rank
#     similarity score
#     event
#     bucket
#     patch identifier
# ---------------------------------------------------------
def print_neighbor(rank, score, meta):

    print(f"\n[{rank}]")

    print(f"score       : {score:.4f}")
    print(f"event       : {meta['event_id']}")
    print(f"bucket      : {meta['damage_bucket']}")
    print(f"patch_id    : {meta['patch_id']}")

    print_separator()


# =========================================================
# Build Vision Index
# =========================================================
# ---------------------------------------------------------
# Builds an in-memory FAISS retrieval index for
# vision embeddings.
#
# Retrieval uses:
#
#     IndexFlatIP
#
# which performs exact nearest-neighbor search
# using inner-product similarity.
# ---------------------------------------------------------
def build_vision_index(embeddings):
    """
    Builds in-memory FAISS index for vision embeddings.
    """

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

	# FAISS requires float32 vectors.
    index.add(
        embeddings.astype(np.float32)
    )

    return index


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 60)
    print("2B — Vision-to-Vision Retrieval")
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

	# ---------------------------------------------------------
	# Load frozen vision embeddings exported from 2A.
	#
	# Each row corresponds to a disaster patch and
	# serves as both:
	#
	#     retrieval query
	#
	# and:
	#
	#     retrieval candidate
	# ---------------------------------------------------------
    # =====================================================
    # Load Vision Embeddings
    # =====================================================

    print("\nLoading vision embeddings...")
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

	# ---------------------------------------------------------
	# Construct an exact nearest-neighbor retrieval
	# backend for vision-manifold inspection.
	#
	# This allows evaluation of:
	#
	# - semantic clustering
	# - event structure
	# - disaster similarity
	# ---------------------------------------------------------    
	# =====================================================
    # Build Vision Index
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
    # Validate Query Index
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

    print_query_info(
        query_idx,
        query_meta
    )

    # =====================================================
    # Run Retrieval
    # =====================================================

    print("\n" + "=" * 60)
    print(
        f"Top-{args.topk} Retrieved Vision Neighbors"
    )
    print("=" * 60)

    query_embedding = np.expand_dims(
        query_embedding.astype(np.float32),
        axis=0
    )

    
	# Retrieve top-k nearest visual neighbors.
	#
	# Note:
	# The closest result will usually be the query
	# itself, which is removed later.
    scores, indices = vision_index.search(
        query_embedding,
        args.topk + 1  # retrieve extra because top-1 will be self
    )

	# ---------------------------------------------------------
	# Display retrieved visual neighbors along with
	# similarity scores and metadata.
	#
	# Useful for qualitative inspection of the
	# learned visual embedding manifold.
	# ---------------------------------------------------------
    # =====================================================
    # Print Results
    # =====================================================

    rank = 1

    for score, idx in zip(scores[0], indices[0]):

        # skip self-match
        if idx == query_idx:
            continue

        meta = vision_metadata[idx]

        print_neighbor(
            rank=rank,
            score=float(score),
            meta=meta
        )

        rank += 1

        if rank > args.topk:
            break

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 60)
    print("[OK] Vision retrieval complete.")
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