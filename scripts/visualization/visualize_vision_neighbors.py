"""
Project 2B — Vision Neighbor Retrieval Visualization
=======================================================

Purpose
-------
Visualizes nearest-neighbor retrieval directly within the
vision embedding manifold learned by Project 2A.

For a selected disaster patch, the script displays:

    1. Query image
    2. Top-K retrieved visual neighbors
    3. Neighbor metadata
    4. Similarity scores

This provides a qualitative inspection of the learned
visual embedding space.

Scientific Motivation
---------------------

Project 2B diagnostics revealed:

    vision->vision same_bucket_ratio ≈ 0.9790

indicating that visually similar disaster regions
naturally cluster together in embedding space.

This script allows direct visual verification of
that finding.

Rather than inspecting numerical metrics alone,
the user can visually determine whether retrieved
neighbors share:

    • disaster characteristics
    • structural damage patterns
    • flooding signatures
    • wildfire signatures
    • volcanic damage patterns

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

This script focuses exclusively on:

    Vision → Vision Retrieval

and serves as the qualitative counterpart to:

    debug_retrieval_stats.py

Pipeline Position
-----------------

2A/artifacts/final
            ↓
retrieve_topk_vision.py
            ↓
debug_retrieval_stats.py
            ↓
visualize_vision_neighbors.py
            ↓
README Figures
            ↓
Slide Deck Figures

Inputs
------

From 2A/artifacts/final/

    vision_embeddings.npy
    vision_index.json

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

Retrieval Method
----------------

FAISS Backend
~~~~~~~~~~~~~

Uses:

    faiss.IndexFlatIP

with L2-normalized vision embeddings.

Therefore:

    cosine_similarity(a,b)
        ==
    inner_product(a,b)

allowing exact nearest-neighbor retrieval.

Visualization Layout
--------------------

+---------+---------+---------+---------+---------+---------+
| Query   | Nbr #1  | Nbr #2  | Nbr #3  | Nbr #4  | Nbr #5  |
+---------+---------+---------+---------+---------+---------+

Each neighbor displays:

    event
    bucket
    similarity score

Outputs
-------

Visualization Figure

Default output directory:

    2B/visualization/vision_neighbors/

Example outputs:

    flooding_neighbors.png
    wildfire_neighbors.png
    volcano_neighbors.png

Figure Content
--------------

Query Image
~~~~~~~~~~~

Displays:

    query event
    query damage bucket

Neighbor Images
~~~~~~~~~~~~~~~

Displays:

    retrieved event
    retrieved damage bucket
    retrieval score

Footer
~~~~~~

Summarizes the primary Project 2B finding:

    highly coherent visual semantics

Usage
-----

Flooding Example

PYTHONPATH=2B python \
2B/scripts/visualization/visualize_vision_neighbors.py \
    --query_idx 100 \
    --topk 5 \
    --save_name flooding_neighbors.png

Wildfire Example

PYTHONPATH=2B python \
2B/scripts/visualization/visualize_vision_neighbors.py \
    --query_idx 600 \
    --topk 5 \
    --save_name wildfire_neighbors.png

Volcanic Damage Example

PYTHONPATH=2B python \
2B/scripts/visualization/visualize_vision_neighbors.py \
    --query_idx 20 \
    --topk 5 \
    --save_name volcano_neighbors.png

Example Output
--------------

======================================================================
Query
======================================================================

query_idx : 100
event     : hurricane-florence
bucket    : flooding
patch_id  : hurricane-florence_00000465

======================================================================
Retrieved Visual Neighbors
======================================================================

Neighbor #1

event       : hurricane-harvey
bucket      : flooding
score       : 0.9984

Neighbor #2

event       : hurricane-harvey
bucket      : flooding
score       : 0.9984

Files Written
-------------

Default:

    2B/visualization/vision_neighbors/

Example:

    flooding_neighbors.png

Role in Project 2B
---------------------

This script serves as:

    the primary visual-manifold inspection tool

used to:

    validate visual clustering
    generate README figures
    generate slide figures
    support qualitative retrieval analysis

It provides visual evidence supporting the strong
vision-manifold statistics reported elsewhere in
the project.

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


# ==========================================================
# Utilities
# ==========================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

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
    Convert patch_id into image filename.

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


# ==========================================================
# Main
# ==========================================================

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
        default=5,
        help="Number of visual neighbors."
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to embedding artifacts."
    )

    parser.add_argument(
        "--images_root",
        type=str,
        default="data/images/hold/post_disaster",
        help="Path to post-disaster images."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="2B/visualization/vision_neighbors",
        help="Directory for saving visualizations."
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="vision_neighbors.png",
        help="Output figure filename."
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

	# ---------------------------------------------------------
	# Load frozen vision embeddings and metadata exported
	# from Project 2A_v2.
	#
	# Each embedding represents a disaster image patch.
	# ---------------------------------------------------------
    # ======================================================
    # Load Artifacts
    # ======================================================

    print("=" * 70)
    print("Vision Neighbor Visualization")
    print("=" * 70)

    vision_embeddings = np.load(
        os.path.join(args.artifact_dir, "vision_embeddings.npy")
    ).astype(np.float32)

    vision_index = load_json(
        os.path.join(args.artifact_dir, "vision_index.json")
    )

    print("\nLoaded vision embeddings:")
    print(vision_embeddings.shape)

	# ---------------------------------------------------------
	# Construct exact nearest-neighbor retrieval backend
	# for visual manifold inspection.
	#
	# Retrieval uses:
	#
	#     IndexFlatIP
	#
	# over L2-normalized embeddings.
	# ---------------------------------------------------------
    # ======================================================
    # Build FAISS Index
    # ======================================================

    vision_faiss = faiss.IndexFlatIP(
        vision_embeddings.shape[1]
    )

    vision_faiss.add(vision_embeddings)

	# ---------------------------------------------------------
	# Retrieve metadata associated with the selected
	# query image.
	# ---------------------------------------------------------
    # ======================================================
    # Query
    # ======================================================

    query_meta = vision_index[args.query_idx]

    query_event = query_meta["event_id"]
    query_bucket = query_meta["damage_bucket"]
    query_patch = query_meta["patch_id"]

    print("\nQuery")
    print("-" * 50)

    print(f"query_idx : {args.query_idx}")
    print(f"event     : {query_event}")
    print(f"bucket    : {query_bucket}")
    print(f"patch_id  : {query_patch}")

    q = vision_embeddings[args.query_idx]
    q = q.reshape(1, -1)

    # ======================================================
    # Retrieval
    # ======================================================

	# Retrieve nearest visual neighbors.
	#
	# The query image itself will be returned first and
	# is removed from the final visualization.
    D, I = vision_faiss.search(
        q,
        args.topk + 1
    )

    neighbors = []

    for score, idx in zip(D[0], I[0]):

        if idx == args.query_idx:
            continue

        meta = vision_index[idx]

        neighbors.append({
            "score": float(score),
            "event": meta["event_id"],
            "bucket": meta["damage_bucket"],
            "patch_id": meta["patch_id"]
        })

        if len(neighbors) == args.topk:
            break

	# ---------------------------------------------------------
	# Create visualization canvas containing:
	#
	# query image
	# +
	# top-k retrieved visual neighbors
	# ---------------------------------------------------------
    # ======================================================
    # Figure Setup
    # ======================================================

    total_images = args.topk + 1

    fig, axes = plt.subplots(
        1,
        total_images,
        figsize=(4 * total_images, 5)
    )

    fig.suptitle(
        "Vision Neighbor Retrieval",
        fontsize=20,
        fontweight="bold"
    )

	# ---------------------------------------------------------
	# Display query image and associated metadata.
	# ---------------------------------------------------------
    # ======================================================
    # Query Image
    # ======================================================

    query_img_path = build_image_path(
        args.images_root,
        query_patch
    )

    query_img = Image.open(query_img_path)

    axes[0].imshow(query_img)

    axes[0].set_title(
        f"QUERY\n{query_event}\n({query_bucket})",
        fontsize=11,
        fontweight="bold"
    )

    axes[0].axis("off")

	# ---------------------------------------------------------
	# Display retrieved visual neighbors together with:
	#
	# event
	# bucket
	# similarity score
	# ---------------------------------------------------------
    # ======================================================
    # Neighbor Images
    # ======================================================

    for i, neighbor in enumerate(neighbors):

        ax = axes[i + 1]

        img_path = build_image_path(
            args.images_root,
            neighbor["patch_id"]
        )

        img = Image.open(img_path)

        ax.imshow(img)

        ax.set_title(
            f"{neighbor['event']}\n"
            f"{neighbor['bucket']}\n"
            f"score={neighbor['score']:.4f}",
            fontsize=10
        )

        ax.axis("off")

	# ---------------------------------------------------------
	# Summarize the primary visual-manifold finding of
	# Project 2B_v2.
	# ---------------------------------------------------------
    # ======================================================
    # Footer
    # ======================================================

    footer = (
        "Project 2B Finding:\n"
        "The visual embedding manifold learned highly coherent "
        "cross-event disaster semantics."
    )

    plt.figtext(
        0.5,
        0.02,
        footer,
        ha="center",
        fontsize=12
    )

	# ---------------------------------------------------------
	# Save figure for:
	#
	# README visuals
	# slide decks
	# portfolio demonstrations
	# interview walkthroughs
	# ---------------------------------------------------------
    # ======================================================
    # Save
    # ======================================================

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    save_path = os.path.join(
        args.save_dir,
        args.save_name
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("\nSaved visualization:")
    print(save_path)

    print("\n[OK] Visualization complete.")


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    main()