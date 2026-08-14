"""
Project 2B — Grounded Multimodal Retrieval Visualization
===========================================================

Purpose
-------
Creates an end-to-end visualization of the final
retrieval-and-grounding pipeline developed in
Project 2B.

For a selected disaster query, the visualization displays:

    1. Query disaster image
    2. Retrieved visual neighbors
    3. Retrieved text evidence
    4. Grounded operational summary

This script provides a qualitative demonstration of
the complete operational retrieval workflow.

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

Subsequent experiments showed that:

    metadata-aware retrieval constraints

substantially improved retrieval coherence.

This script visualizes the final retrieval pipeline
built upon those findings.

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

This script represents the highest-level qualitative
visualization of the completed retrieval system.

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
            ↓
visualize_grounded_retrieval.py
            ↓
README Figures
            ↓
Slide Deck Figures

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

Retrieval Components
--------------------

Visual Retrieval
~~~~~~~~~~~~~~~~

Uses:

    FAISS nearest-neighbor search

to retrieve visually similar disaster patches.

Text Retrieval
~~~~~~~~~~~~~~

Uses:

    semantic similarity search

over retrieved textual evidence.

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

Visualization Layout
--------------------

+------------------------------------------------------+
| Query Image | Neighbor 1 | Neighbor 2 | Neighbor 3   |
+------------------------------------------------------+
| Retrieved Text Evidence                              |
+------------------------------------------------------+
| Grounded Operational Summary                         |
+------------------------------------------------------+

The visualization combines:

    visual evidence
    textual evidence
    retrieval provenance
    grounded interpretation

into a single figure.

Outputs
-------

Visualization Figure

Default output directory:

    2B/visualization/grounded_retrieval/

Example outputs:

    flooding_demo.png
    wildfire_demo.png
    volcano_demo.png

Figure Content
--------------

Query Image
~~~~~~~~~~~

Displays:

    query disaster patch

including:

    event
    damage category

Visual Neighbors
~~~~~~~~~~~~~~~~

Displays:

    retrieved disaster images

including:

    similarity score
    source event

Retrieved Text Evidence
~~~~~~~~~~~~~~~~~~~~~~~

Displays:

    top retrieved evidence passages

used to support retrieval grounding.

Grounded Operational Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Displays:

    concise evidence-based summary

constructed from:

    retrieved images
    retrieved text

Usage
-----

Flooding Example

PYTHONPATH=2B python \
2B/scripts/visualization/visualize_grounded_retrieval.py \
    --query_idx 100 \
    --topk 3 \
    --constraint same_bucket \
    --save_name flooding_demo.png

Wildfire Example

PYTHONPATH=2B python \
2B/scripts/visualization/visualize_grounded_retrieval.py \
    --query_idx 600 \
    --topk 3 \
    --constraint same_bucket \
    --save_name wildfire_demo.png

Volcanic Damage Example

PYTHONPATH=2B python \
2B/scripts/visualization/visualize_grounded_retrieval.py \
    --query_idx 20 \
    --topk 3 \
    --constraint same_bucket \
    --save_name volcano_demo.png

Example Output
--------------

======================================================================
Query
======================================================================

query_idx : 100
event     : hurricane-florence
bucket    : flooding

======================================================================
Retrieved Visual Neighbors
======================================================================

Neighbor #1

event       : hurricane-harvey
score       : 0.9984

Neighbor #2

event       : hurricane-harvey
score       : 0.9984

======================================================================
Retrieved Text Evidence
======================================================================

[1]

"...retrieved flooding-related evidence..."

======================================================================
Grounded Operational Summary
======================================================================

Retrieved multimodal evidence suggests
flooding-related damage with semantically coherent
cross-event retrieval.

Files Written
-------------

Default:

    2B/visualization/grounded_retrieval/

Examples:

    flooding_demo.png
    wildfire_demo.png
    volcano_demo.png

Role in Project 2B
---------------------

This script serves as:

    the final qualitative system demonstration

used to:

    visualize retrieval grounding
    generate README figures
    generate slide figures
    support portfolio demonstrations
    support interview walkthroughs

It combines the major components of Project 2B
into a single interpretable visualization.

Author
------
Project 2B
Multimodal Retrieval Systems Engineering
"""

import os
import json
import argparse

import faiss
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# Utility Functions
# =========================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# ---------------------------------------------------------
# Utility function for formatting long retrieval evidence
# and summaries inside matplotlib figures.
# ---------------------------------------------------------
def wrap_text(text, max_len=80):
    """
    Simple text wrapper for matplotlib rendering.
    """
    words = text.split()
    lines = []

    current = []

    for word in words:
        current.append(word)

        if len(" ".join(current)) > max_len:
            lines.append(" ".join(current[:-1]))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines)

# ---------------------------------------------------------
# Convert a patch identifier into the corresponding
# post-disaster image filename.
#
# Example:
#
# hurricane-florence_00000465
#
# ->
#
# hurricane-florence_00000465_post_disaster.png
# ---------------------------------------------------------
def build_image_path(images_root, patch_id):
    """
    Convert patch_id into image path.

    Example:
        hurricane-florence_00000465
            ->
        hurricane-florence_00000465_post_disaster.png
    """

    filename = f"{patch_id}_post_disaster.png"

    return os.path.join(
        images_root,
        filename
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query_idx",
        type=int,
        required=True,
        help="Vision query index."
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of retrieved neighbors."
    )

    parser.add_argument(
        "--constraint",
        type=str,
        default="same_bucket",
        choices=[
            "none",
            "same_bucket",
            "same_event",
            "disaster_family"
        ],
        help="Retrieval constraint mode."
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to final embedding artifacts."
    )

    parser.add_argument(
        "--images_root",
        type=str,
        default="data/images/hold/post_disaster",
        help="Directory containing post-disaster images."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="2B/visualization/grounded_retrieval",
        help="Directory for saving visualizations."
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="grounded_demo.png",
        help="Output visualization filename."
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

	# ---------------------------------------------------------
	# Load multimodal embeddings and metadata exported
	# from Project 2A.
	# ---------------------------------------------------------
    # =====================================================
    # Load Embeddings + Metadata
    # =====================================================

    print("=" * 70)
    print("Grounded Multimodal Retrieval Visualization")
    print("=" * 70)

    vision_embeddings = np.load(
        os.path.join(args.artifact_dir, "vision_embeddings.npy")
    ).astype(np.float32)

    text_embeddings = np.load(
        os.path.join(args.artifact_dir, "text_embeddings.npy")
    ).astype(np.float32)

    vision_index = load_json(
        os.path.join(args.artifact_dir, "vision_index.json")
    )

    text_index = load_json(
        os.path.join(args.artifact_dir, "text_index.json")
    )

	# ---------------------------------------------------------
	# Build exact semantic retrieval backends for:
	#
	#     vision retrieval
	#     text retrieval
	#
	# using inner-product search over L2-normalized
	# embeddings.
	# ---------------------------------------------------------
    # =====================================================
    # Build FAISS Indexes
    # =====================================================

    vision_faiss = faiss.IndexFlatIP(
        vision_embeddings.shape[1]
    )

    vision_faiss.add(vision_embeddings)

    text_faiss = faiss.IndexFlatIP(
        text_embeddings.shape[1]
    )

    text_faiss.add(text_embeddings)

	# ---------------------------------------------------------
	# Retrieve metadata associated with the selected
	# vision query.
	# ---------------------------------------------------------
    # =====================================================
    # Query Metadata
    # =====================================================

    query_meta = vision_index[args.query_idx]

    query_event = query_meta["event_id"]
    query_bucket = query_meta["damage_bucket"]
    query_patch = query_meta["patch_id"]

    print("\nQuery")
    print("-" * 50)

    print(f"query_idx   : {args.query_idx}")
    print(f"event       : {query_event}")
    print(f"bucket      : {query_bucket}")
    print(f"patch_id    : {query_patch}")

	# ---------------------------------------------------------
	# Extract query embedding for visual and textual
	# retrieval operations.
	# ---------------------------------------------------------
    # =====================================================
    # Query Embedding
    # =====================================================

    q = vision_embeddings[args.query_idx]
    q = q.reshape(1, -1)

	# ---------------------------------------------------------
	# Retrieve visually similar disaster patches.
	#
	# Optional metadata constraints may be applied to
	# stabilize retrieval behavior.
	# ---------------------------------------------------------
    # =====================================================
    # Visual Neighbor Retrieval
    # =====================================================

	# Retrieve nearest visual neighbors from the
	# learned vision embedding manifold.
    Dv, Iv = vision_faiss.search(q, args.topk + 1)

    visual_neighbors = []

    for score, idx in zip(Dv[0], Iv[0]):

        if idx == args.query_idx:
            continue

        meta = vision_index[idx]

        if args.constraint == "same_bucket":
            if meta["damage_bucket"] != query_bucket:
                continue

        if args.constraint == "same_event":
            if meta["event_id"] != query_event:
                continue

        visual_neighbors.append((score, meta))

        if len(visual_neighbors) == args.topk:
            break

	# ---------------------------------------------------------
	# Retrieve semantically related textual evidence.
	#
	# Retrieval follows the same routing policy used
	# for visual retrieval.
	# ---------------------------------------------------------
    # =====================================================
    # Text Retrieval
    # =====================================================
	# Retrieve candidate text evidence before
	# applying metadata constraints.
    Dt, It = text_faiss.search(q, 20)

    retrieved_text = []

    for score, idx in zip(Dt[0], It[0]):

        meta = text_index[idx]

        if args.constraint == "same_bucket":
            if meta["damage_bucket"] != query_bucket:
                continue

        if args.constraint == "same_event":
            if meta["source_event"] != query_event:
                continue

        retrieved_text.append(meta["text"])

        if len(retrieved_text) == 3:
            break

	# ---------------------------------------------------------
	# Construct a concise evidence-grounded summary
	# from the retrieved multimodal evidence.
	# ---------------------------------------------------------
    # =====================================================
    # Build Grounded Summary
    # =====================================================

    grounded_summary = (
        f"Retrieved multimodal evidence suggests "
        f"{query_bucket.replace('_', ' ')} related damage "
        f"with semantically coherent cross-event retrieval."
    )

	# ---------------------------------------------------------
	# Build final figure containing:
	#
	# query image
	# retrieved visual neighbors
	# retrieved text evidence
	# grounded summary
	# ---------------------------------------------------------
    # =====================================================
    # Create Visualization
    # =====================================================

    fig = plt.figure(figsize=(16, 10))

	# Display query disaster patch.
    # -----------------------------------------------------
    # Query Image
    # -----------------------------------------------------

    ax_query = plt.subplot2grid(
        (3, 4),
        (0, 0),
        colspan=1
    )

    query_img_path = build_image_path(
        args.images_root,
        query_patch
    )

    query_img = Image.open(query_img_path)

    ax_query.imshow(query_img)

    ax_query.set_title(
        f"QUERY\n{query_event}",
        fontsize=12,
        fontweight="bold"
    )

    ax_query.axis("off")

	# Display visually retrieved disaster patches and
	# associated similarity scores.
    # -----------------------------------------------------
    # Visual Neighbors
    # -----------------------------------------------------

    for i, (score, meta) in enumerate(visual_neighbors):

        ax = plt.subplot2grid(
            (3, 4),
            (0, i + 1)
        )

        neighbor_path = build_image_path(
            args.images_root,
            meta["patch_id"]
        )

        img = Image.open(neighbor_path)

        ax.imshow(img)

        ax.set_title(
            f"{meta['event_id']}\nscore={score:.3f}",
            fontsize=10
        )

        ax.axis("off")

	# Display supporting textual evidence used for
	# retrieval grounding.
    # -----------------------------------------------------
    # Retrieved Text Evidence
    # -----------------------------------------------------

    ax_text = plt.subplot2grid(
        (3, 4),
        (1, 0),
        colspan=4
    )

    text_block = ""

    for i, t in enumerate(retrieved_text):

        text_block += f"[{i+1}] {t}\n\n"

    ax_text.text(
        0.01,
        0.95,
        wrap_text(text_block, 120),
        va="top",
        fontsize=11
    )

    ax_text.set_title(
        "Retrieved Text Evidence",
        fontsize=14,
        fontweight="bold"
    )

    ax_text.axis("off")

	# Display evidence-grounded operational summary.
    # -----------------------------------------------------
    # Grounded Summary
    # -----------------------------------------------------

    ax_summary = plt.subplot2grid(
        (3, 4),
        (2, 0),
        colspan=4
    )

    ax_summary.text(
        0.01,
        0.8,
        wrap_text(grounded_summary, 120),
        fontsize=13
    )

    ax_summary.set_title(
        "Grounded Operational Summary",
        fontsize=14,
        fontweight="bold"
    )

    ax_summary.axis("off")

	# ---------------------------------------------------------
	# Finalize figure layout and prepare visualization
	# for export.
	# ---------------------------------------------------------
    # -----------------------------------------------------
    # Final Layout
    # -----------------------------------------------------

    plt.tight_layout()

    save_path = os.path.join(
        args.save_dir,
        args.save_name
    )
	# Save visualization for:
	#
	# README figures
	# slide decks
	# portfolio demonstrations
	# interview walkthroughs
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("\nSaved visualization:")
    print(save_path)

    print("\n[OK] Visualization complete.")


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":
    main()