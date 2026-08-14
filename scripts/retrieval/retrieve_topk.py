"""
Project 2B — Vision-to-Text Retrieval
========================================

Purpose
-------
Performs deterministic multimodal semantic retrieval using
frozen vision embeddings exported from Project 2A and
FAISS retrieval indexes built in Project 2B.

Given:

    a vision query embedding

the system retrieves:

    top-k nearest text evidence

and optionally:

    top-k nearest whisper evidence

using dense vector similarity search.

This is the FIRST operational retrieval script in Project 2B.

Project Context
---------------
Project 2A learns a shared multimodal embedding space:

    Vision Embeddings
    Text Embeddings
    Whisper Embeddings

Project 2B operationalizes those embeddings into:

    FAISS Retrieval
    Nearest-Neighbor Search
    Semantic Search
    Constraint-Aware Retrieval
    Grounded Multimodal Retrieval

This script demonstrates basic cross-modal retrieval before:

    diagnostics
    retrieval stabilization
    constraint routing
    grounded retrieval
    final evaluation

Pipeline Position
-----------------

2A/artifacts/final
            ↓
test_load_artifacts.py
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

Query Flow
----------

Vision Query
      ↓
Vision Embedding
      ↓
FAISS Similarity Search
      ↓
Top-K Text Retrieval
      ↓
Optional Whisper Retrieval

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

    retrieved evidence

        score
        source_event
        damage_bucket
        text

Usage
-----

Text Retrieval Only

PYTHONPATH=2B python \
2B/scripts/retrieval/retrieve_topk.py \
    --query_idx 421 \
    --topk 5

Text + Whisper Retrieval

PYTHONPATH=2B python \
2B/scripts/retrieval/retrieve_topk.py \
    --query_idx 421 \
    --topk 5 \
    --use_whisper

Example Output
--------------

============================================================
Query
============================================================

query_idx       : 421
event           : hurricane-harvey
bucket          : flooding
patch_id        : hurricane-harvey_00000123
split           : hold

============================================================
Top-5 Retrieved Text
============================================================

[1]

score   : 0.0372
event   : hurricane-michael
bucket  : flooding

floodwaters inundated homes and roads...

------------------------------------------------------------

[2]

score   : 0.0345
event   : midwest-flooding
bucket  : flooding

residential areas remained underwater...

------------------------------------------------------------

Files Written
-------------
None.

This script performs retrieval and prints results
to the terminal.

Role in Project 2B
---------------------

This script serves as:

    the first cross-modal retrieval engine

used to verify:

    semantic alignment
    retrieval plausibility
    embedding quality

before introducing:

    diagnostics
    observability
    retrieval constraints
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


def print_query_info(query_idx, query_meta):
    print("\n" + "=" * 60)
    print("Query")
    print("=" * 60)

    print(f"\nquery_idx       : {query_idx}")
    print(f"event           : {query_meta['event_id']}")
    print(f"bucket          : {query_meta['damage_bucket']}")
    print(f"patch_id        : {query_meta['patch_id']}")
    print(f"split           : {query_meta['split']}")


# ---------------------------------------------------------
# Pretty-print a retrieved evidence item.
#
# Displays:
#     rank
#     similarity score
#     event
#     bucket
#     evidence text
# ---------------------------------------------------------
def print_retrieval_block(
    rank,
    score,
    meta
):

    print(f"\n[{rank}]")

    print(f"score   : {score:.4f}")

    if "source_event" in meta:
        print(f"event   : {meta['source_event']}")

    if "damage_bucket" in meta:
        print(f"bucket  : {meta['damage_bucket']}")

    print("\n" + meta["text"])

    print_separator()


# =========================================================
# Retrieval Function
# =========================================================
# ---------------------------------------------------------
# Core retrieval routine.
#
# Performs exact nearest-neighbor retrieval using
# FAISS IndexFlatIP.
#
# Input:
#     query embedding
#
# Output:
#     top-k ranked retrieval results
#
# Each result contains:
#     rank
#     similarity score
#     metadata
# ---------------------------------------------------------
def retrieve_topk(
    query_embedding,
    index,
    metadata,
    topk
):
    """
    Runs FAISS top-k retrieval.
    """
	# FAISS expects float32 vectors.
    query_embedding = query_embedding.astype(np.float32)
    query_embedding = np.expand_dims(query_embedding, axis=0)

    # Execute exact semantic nearest-neighbor search.
	scores, indices = index.search(
        query_embedding,
        topk
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        meta = metadata[idx]

        results.append({
            "rank": rank,
            "score": float(score),
            "idx": int(idx),
            "metadata": meta
        })

    return results


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 60)
    print("2B — Retrieve Top-K")
    print("=" * 60)

    # =====================================================
    # Artifact Paths
    # =====================================================

    artifact_dir = args.artifact_dir
    index_dir = args.index_dir

    vision_emb_path = os.path.join(
        artifact_dir,
        "vision_embeddings.npy"
    )

    vision_index_path = os.path.join(
        artifact_dir,
        "vision_index.json"
    )

	# ---------------------------------------------------------
	# Vision embeddings serve as retrieval queries.
	#
	# Each query corresponds to a disaster patch exported
	# from Project 2A.
	# ---------------------------------------------------------
    # =====================================================
    # Load Vision Queries
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

    # =====================================================
    # Validate Query Index
    # =====================================================

    query_idx = args.query_idx

    if query_idx < 0 or query_idx >= len(vision_embeddings):
        raise ValueError(
            f"Invalid query_idx: {query_idx}"
        )

    query_embedding = vision_embeddings[query_idx]
    query_meta = vision_metadata[query_idx]

    print_query_info(
        query_idx,
        query_meta
    )

	# ---------------------------------------------------------
	# Primary cross-modal retrieval pathway:
	#
	# Vision
	#   →
	# Text
	#
	# Retrieves semantically related disaster reports.
	# ---------------------------------------------------------
    # =====================================================
    # TEXT RETRIEVAL
    # =====================================================

    print("\n" + "=" * 60)
    print(f"Top-{args.topk} Retrieved Text")
    print("=" * 60)

    text_index_path = os.path.join(
        index_dir,
        "text_index.faiss"
    )

    text_metadata_path = os.path.join(
        index_dir,
        "text_metadata.json"
    )

    text_index = load_faiss_index(
        text_index_path
    )

    text_metadata = load_json(
        text_metadata_path
    )

    text_results = retrieve_topk(
        query_embedding=query_embedding,
        index=text_index,
        metadata=text_metadata,
        topk=args.topk
    )

    for result in text_results:

        print_retrieval_block(
            rank=result["rank"],
            score=result["score"],
            meta=result["metadata"]
        )

	# ---------------------------------------------------------
	# Optional retrieval against speech/transcript evidence.
	#
	# Uses the same vision query embedding but searches
	# the whisper embedding index.
	# ---------------------------------------------------------
    # =====================================================
    # WHISPER RETRIEVAL (OPTIONAL)
    # =====================================================

    if args.use_whisper:

        print("\n" + "=" * 60)
        print(f"Top-{args.topk} Retrieved Whisper")
        print("=" * 60)

        whisper_index_path = os.path.join(
            index_dir,
            "whisper_index.faiss"
        )

        whisper_metadata_path = os.path.join(
            index_dir,
            "whisper_metadata.json"
        )

        whisper_index = load_faiss_index(
            whisper_index_path
        )

        whisper_metadata = load_json(
            whisper_metadata_path
        )

        whisper_results = retrieve_topk(
            query_embedding=query_embedding,
            index=whisper_index,
            metadata=whisper_metadata,
            topk=args.topk
        )

        for result in whisper_results:

            print_retrieval_block(
                rank=result["rank"],
                score=result["score"],
                meta=result["metadata"]
            )

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 60)
    print("[OK] Retrieval complete.")
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
        "--use_whisper",
        action="store_true",
        help="Enable whisper retrieval"
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="2A artifact directory"
    )

    parser.add_argument(
        "--index_dir",
        type=str,
        default="2B/indexes",
        help="FAISS index directory"
    )

    args = parser.parse_args()

    main(args)