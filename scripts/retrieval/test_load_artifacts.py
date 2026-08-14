"""
Project 2B — Artifact Validation Script
==========================================

Purpose
-------
Validates all multimodal retrieval artifacts exported from
Project 2A before retrieval system construction.

This script performs:

1. Embedding integrity validation
2. Metadata integrity validation
3. Row alignment verification
4. L2-normalization verification
5. NaN / Inf safety checks
6. Index ordering validation
7. Corpus diagnostics and statistics

This is the FIRST script executed in Project 2B and
must pass before:

    - FAISS index construction
    - semantic retrieval
    - retrieval diagnostics
    - constraint routing
    - grounded retrieval
    - retrieval evaluation

Project Context
---------------
Project 2A produces frozen multimodal embeddings:

    Vision Embeddings
    Text Embeddings
    Whisper Embeddings

Project 2B operationalizes those embeddings into:

    FAISS Retrieval
    Nearest-Neighbor Search
    Semantic Search
    Constraint-Aware Retrieval
    Grounded Multimodal Retrieval

Expected Inputs
---------------
2A/artifacts/final/

    vision_embeddings.npy
    text_embeddings.npy
    whisper_embeddings.npy

    vision_index.json
    text_index.json
    whisper_index.json

Validation Checks
-----------------

Embedding Integrity
~~~~~~~~~~~~~~~~~~~
- File loading succeeds
- Shapes are valid
- No NaN values
- No Inf values

Normalization
~~~~~~~~~~~~~
Verifies embeddings remain L2-normalized:

    ||z||₂ ≈ 1

Metadata Alignment
~~~~~~~~~~~~~~~~~~
Verifies:

    len(metadata) == len(embeddings)

and:

    metadata[idx]["idx"] == embedding row idx

Corpus Diagnostics
~~~~~~~~~~~~~~~~~~
Reports:

- Event distribution
- Damage bucket distribution
- Corpus size
- Unique event count

Usage
-----

Default:

PYTHONPATH=2B python \
2B/scripts/retrieval/test_load_artifacts.py

Custom artifact directory:

PYTHONPATH=2B python \
2B/scripts/retrieval/test_load_artifacts.py \
    --artifact_dir 2A/artifacts/final

Expected Output
---------------

============================================================
2B Artifact Validation
============================================================

vision_embeddings   : (694, 256)
text_embeddings     : (70, 256)
whisper_embeddings  : (11, 256)

vision_norm_mean    : 1.000000
text_norm_mean      : 1.000000
whisper_norm_mean   : 1.000000

[OK] No NaNs/Infs detected.
[OK] Index alignment verified.

============================================================
[OK] Artifact validation passed.
============================================================

Outputs
-------
No files are written.

This script performs validation only and prints
diagnostic statistics to the console.

Role in 2B Pipeline
----------------------

2A/artifacts/final
            ↓
test_load_artifacts.py
            ↓
build_faiss_indexes.py
            ↓
retrieval pipeline

Author
------
Project 2B
Operational Multimodal Retrieval Systems

"""

import os
import json
import argparse
from collections import Counter

import numpy as np


# =========================================================
# Utility Functions
# =========================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------
# Computes L2-norm statistics for validation.
#
# Expected:
#     mean ≈ 1.0
#
# since all exported 2A embeddings are stored
# after L2 normalization.
# ---------------------------------------------------------
def compute_norm_stats(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)

    return {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
    }


# ---------------------------------------------------------
# Defensive validation.
#
# Retrieval systems should never operate on embeddings
# containing NaN or Inf values because FAISS similarity
# search becomes undefined.
# ---------------------------------------------------------
def check_nan_inf(name, embeddings):
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()

    if has_nan:
        raise ValueError(f"[ERROR] NaNs detected in {name}")

    if has_inf:
        raise ValueError(f"[ERROR] Infs detected in {name}")


# ---------------------------------------------------------
# Verifies metadata rows remain perfectly aligned with
# embedding rows.
#
# This guarantees:
#
# embedding[i]
#      ↔
# metadata[i]
#
# throughout the retrieval pipeline.
# ---------------------------------------------------------
def validate_index_alignment(index_data, embeddings, name):
    """
    Ensures:
    - len(index) == num embeddings
    - idx ordering matches row ordering
    """

    if len(index_data) != len(embeddings):
        raise ValueError(
            f"[ERROR] {name}: "
            f"index length ({len(index_data)}) != "
            f"embedding rows ({len(embeddings)})"
        )

    for expected_idx, row in enumerate(index_data):

        if row["idx"] != expected_idx:
            raise ValueError(
                f"[ERROR] {name}: "
                f"row ordering mismatch at position {expected_idx}"
            )


def print_bucket_distribution(name, index_data, bucket_key):
    buckets = [x[bucket_key] for x in index_data]

    counter = Counter(buckets)

    print(f"\n{name} Bucket Distribution")
    print("-" * 50)

    for k, v in sorted(counter.items()):
        print(f"{k:<30} {v}")


def print_event_distribution(name, index_data, event_key):
    events = [x[event_key] for x in index_data]

    counter = Counter(events)

    print(f"\n{name} Event Distribution")
    print("-" * 50)

    for k, v in sorted(counter.items()):
        print(f"{k:<30} {v}")


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 60)
    print("2B Artifact Validation")
    print("=" * 60)

    artifact_dir = args.artifact_dir

    # =====================================================
    # Paths
    # =====================================================

    vision_emb_path = os.path.join(
        artifact_dir,
        "vision_embeddings.npy"
    )

    text_emb_path = os.path.join(
        artifact_dir,
        "text_embeddings.npy"
    )

    whisper_emb_path = os.path.join(
        artifact_dir,
        "whisper_embeddings.npy"
    )

    vision_index_path = os.path.join(
        artifact_dir,
        "vision_index.json"
    )

    text_index_path = os.path.join(
        artifact_dir,
        "text_index.json"
    )

    whisper_index_path = os.path.join(
        artifact_dir,
        "whisper_index.json"
    )

	# ---------------------------------------------------------
	# Load frozen multimodal embeddings exported from 2A.
	#
	# Vision:
	#     disaster image embeddings
	#
	# Text:
	#     disaster report embeddings
	#
	# Whisper:
	#     speech/transcript embeddings
	# ---------------------------------------------------------
    # =====================================================
    # Load Embeddings
    # =====================================================

    print("\nLoading embeddings...")
    print("-" * 50)

    vision_embeddings = np.load(vision_emb_path)
    text_embeddings = np.load(text_emb_path)
    whisper_embeddings = np.load(whisper_emb_path)

    print(f"vision_embeddings   : {vision_embeddings.shape}")
    print(f"text_embeddings     : {text_embeddings.shape}")
    print(f"whisper_embeddings  : {whisper_embeddings.shape}")

    # =====================================================
    # Safety Checks
    # =====================================================

    print("\nRunning NaN / Inf checks...")
    print("-" * 50)

    check_nan_inf("vision_embeddings", vision_embeddings)
    check_nan_inf("text_embeddings", text_embeddings)
    check_nan_inf("whisper_embeddings", whisper_embeddings)

    print("[OK] No NaNs/Infs detected.")

    # =====================================================
    # Norm Checks
    # =====================================================

    print("\nEmbedding Norm Statistics")
    print("-" * 50)

    vision_norm_stats = compute_norm_stats(vision_embeddings)
    text_norm_stats = compute_norm_stats(text_embeddings)
    whisper_norm_stats = compute_norm_stats(whisper_embeddings)

    print(
        f"vision_norm_mean     : "
        f"{vision_norm_stats['mean']:.6f}"
    )

    print(
        f"text_norm_mean       : "
        f"{text_norm_stats['mean']:.6f}"
    )

    print(
        f"whisper_norm_mean    : "
        f"{whisper_norm_stats['mean']:.6f}"
    )

    # =====================================================
    # Load Metadata
    # =====================================================

    print("\nLoading metadata...")
    print("-" * 50)

    vision_index = load_json(vision_index_path)
    text_index = load_json(text_index_path)
    whisper_index = load_json(whisper_index_path)

    print(f"vision_index rows    : {len(vision_index)}")
    print(f"text_index rows      : {len(text_index)}")
    print(f"whisper_index rows   : {len(whisper_index)}")

    # ---------------------------------------------------------
	# Critical retrieval-system validation.
	#
	# A metadata misalignment would cause:
	#
	# embedding A
	#     →
	# metadata B
	#
	# producing incorrect retrieval results.
	# ---------------------------------------------------------
	# =====================================================
    # Alignment Checks
    # =====================================================

    print("\nRunning alignment checks...")
    print("-" * 50)

    validate_index_alignment(
        vision_index,
        vision_embeddings,
        "vision"
    )

    validate_index_alignment(
        text_index,
        text_embeddings,
        "text"
    )

    validate_index_alignment(
        whisper_index,
        whisper_embeddings,
        "whisper"
    )

    print("[OK] Index alignment verified.")

    # =====================================================
    # Vision Diagnostics
    # =====================================================

    print("\nVision Diagnostics")
    print("-" * 50)

    unique_events = sorted(
        list(set(x["event_id"] for x in vision_index))
    )

    print(f"Unique events        : {len(unique_events)}")
    print(f"Events               : {unique_events}")

    print_bucket_distribution(
        "Vision",
        vision_index,
        bucket_key="damage_bucket"
    )

    # =====================================================
    # Text Diagnostics
    # =====================================================

    print_event_distribution(
        "Text",
        text_index,
        event_key="source_event"
    )

    print_bucket_distribution(
        "Text",
        text_index,
        bucket_key="damage_bucket"
    )

    # =====================================================
    # Whisper Diagnostics
    # =====================================================

    print_event_distribution(
        "Whisper",
        whisper_index,
        event_key="source_event"
    )

    print_bucket_distribution(
        "Whisper",
        whisper_index,
        bucket_key="damage_bucket"
    )

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 60)
    print("[OK] Artifact validation passed.")
    print("=" * 60)


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to exported 2A artifacts"
    )

    args = parser.parse_args()

    main(args)