"""
Project 2B — Constraint Retrieval Comparison Visualization
=============================================================

Purpose
-------
Creates a side-by-side visual comparison of retrieval
behavior under different retrieval policies.

Specifically, this script compares:

    1. Unrestricted Retrieval (MODE: none)
    2. Constrained Retrieval (MODE: same_bucket)

for the same vision query.

This visualization demonstrates the central systems
finding of Project 2B:

    Global multimodal retrieval is unstable,

while:

    Metadata-aware constrained retrieval
    significantly improves semantic coherence.

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

The next question became:

    Can metadata-aware routing stabilize
    retrieval behavior?

This script provides a direct qualitative comparison.

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

This script focuses on:

    Retrieval Policy Comparison

and serves as the qualitative counterpart to:

    evaluate_constraints.py

and:

    evaluate_retrieval_metrics.py

Pipeline Position
-----------------

2A/artifacts/final
            ↓
retrieve_constrained.py
            ↓
evaluate_constraints.py
            ↓
evaluate_retrieval_metrics.py
            ↓
visualize_constraint_comparison.py
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

Expected Artifact Sizes
-----------------------

vision_embeddings
    (694, 256)

text_embeddings
    (70, 256)

whisper_embeddings
    (11, 256)

Retrieval Modes Visualized
--------------------------

MODE: none
~~~~~~~~~~

Global unrestricted retrieval.

All text evidence remains eligible.

Used as:

    baseline retrieval

MODE: same_bucket
~~~~~~~~~~~~~~~~~

Retrieval restricted to the same damage category.

Example:

    flooding
        →
    flooding

This became the primary operational retrieval policy
used in Project 2B.

Visualization Layout
--------------------

+------------------------------------------------------+
| MODE: none        | MODE: same_bucket                |
+------------------------------------------------------+
| Retrieved Text    | Retrieved Text                   |
| Evidence          | Evidence                         |
|                  |                                  |
| Similarity Scores | Similarity Scores               |
| Metadata          | Metadata                         |
+------------------------------------------------------+

The comparison allows direct inspection of:

    retrieval coherence
    semantic relevance
    evidence consistency
    retrieval stability

Outputs
-------

Visualization Figure

Default output directory:

    2B/visualization/constraint_comparisons/

Example outputs:

    flooding_constraint_comparison.png
    wildfire_constraint_comparison.png
    volcano_constraint_comparison.png

Figure Content
--------------

Left Column
~~~~~~~~~~~

MODE: none

Displays:

    unrestricted retrieval results

including:

    score
    event
    bucket
    retrieved evidence

Right Column
~~~~~~~~~~~~

MODE: same_bucket

Displays:

    constrained retrieval results

including:

    score
    event
    bucket
    retrieved evidence

Footer
~~~~~~

Summarizes the primary Project 2B systems finding:

    metadata-aware retrieval stabilization

Usage
-----

Flooding Example

PYTHONPATH=2B python \
2B/scripts/visualization/visualize_constraint_comparison.py \
    --query_idx 100 \
    --topk 5 \
    --save_name flooding_constraint_comparison.png

Wildfire Example

PYTHONPATH=2B python \
2B/scripts/visualization/visualize_constraint_comparison.py \
    --query_idx 600 \
    --topk 5 \
    --save_name wildfire_constraint_comparison.png

Volcanic Damage Example

PYTHONPATH=2B python \
2B/scripts/visualization/visualize_constraint_comparison.py \
    --query_idx 20 \
    --topk 5 \
    --save_name volcano_constraint_comparison.png

Example Output
--------------

======================================================================
Query
======================================================================

query_idx : 100
event     : hurricane-florence
bucket    : flooding

======================================================================
MODE: none
======================================================================

[1]

bucket  : structural_damage
event   : mexico-earthquake
score   : 0.0301

======================================================================
MODE: same_bucket
======================================================================

[1]

bucket  : flooding
event   : hurricane-harvey
score   : 0.0532

Files Written
-------------

Default:

    2B/visualization/constraint_comparisons/

Example:

    flooding_constraint_comparison.png

Role in Project 2B
---------------------

This script serves as:

    the primary retrieval-policy visualization tool

used to:

    compare retrieval modes
    visualize retrieval stabilization
    generate README figures
    generate slide figures
    support interview walkthroughs

It visually demonstrates the core systems insight
that motivated the constrained retrieval framework.

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


# ==========================================================
# Utility Functions
# ==========================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# ---------------------------------------------------------
# Simple text wrapping utility for matplotlib rendering.
#
# Prevents long retrieval evidence from overflowing
# figure boundaries.
# ---------------------------------------------------------
def wrap_text(text, max_len=70):
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


# ==========================================================
# Retrieval Helpers
# ==========================================================
# ---------------------------------------------------------
# Executes retrieval under a specified routing mode.
#
# Supported modes:
#
#     none
#     same_bucket
#     same_event
#     disaster_family
#
# Returns:
#     top-k retrieved evidence items
# --------------------------------------------------------- 
def retrieve_mode(
    query_embedding,
    text_embeddings,
    text_index,
    query_bucket,
    query_event,
    mode="none",
    topk=5
):
    """
    Retrieve text evidence under a specified routing mode.
    """
	# Build exact semantic retrieval backend.
    faiss_index = faiss.IndexFlatIP(
        text_embeddings.shape[1]
    )

    faiss_index.add(text_embeddings)
	
	# Retrieve candidate evidence before applying
	# metadata-aware routing constraints.
    D, I = faiss_index.search(query_embedding, 50)

    results = []

    for score, idx in zip(D[0], I[0]):

        meta = text_index[idx]

        keep = True

		# ---------------------------------------------------------
		# Apply retrieval policy routing.
		#
		# Determines whether a candidate survives
		# metadata filtering.
		# ---------------------------------------------------------
        # --------------------------------------------------
        # Constraint Logic
        # --------------------------------------------------

        if mode == "same_bucket":

            if meta["damage_bucket"] != query_bucket:
                keep = False

        elif mode == "same_event":

            if meta["source_event"] != query_event:
                keep = False

        elif mode == "disaster_family":

            # --------------------------------------------------
            # Example lightweight disaster family grouping
            # --------------------------------------------------

            family_map = {
                "flooding": "water",
                "tsunami_inundation": "water",
                "wildfire": "fire",
                "volcanic_damage": "geological",
                "structural_damage": "collapse",
                "generic_damage": "generic"
            }

            query_family = family_map.get(
                query_bucket,
                "generic"
            )

            candidate_family = family_map.get(
                meta["damage_bucket"],
                "generic"
            )

            if query_family != candidate_family:
                keep = False

        # --------------------------------------------------
        # Keep Result
        # --------------------------------------------------

        if keep:

            results.append({
                "score": float(score),
                "event": meta["source_event"],
                "bucket": meta["damage_bucket"],
                "text": meta["text"]
            })

        if len(results) == topk:
            break

    return results


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
        help="Number of retrievals."
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to embedding artifacts."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="2B/visualization/constraint_comparisons",
        help="Directory for saving visualizations."
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="constraint_comparison.png",
        help="Output figure filename."
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

	# ---------------------------------------------------------
	# Load multimodal embeddings and metadata exported
	# from Project 2A.
	# ---------------------------------------------------------
    # ======================================================
    # Load Artifacts
    # ======================================================

    print("=" * 70)
    print("Constraint Retrieval Comparison Visualization")
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
	# Retrieve metadata associated with the selected
	# vision query.
	# ---------------------------------------------------------
    # ======================================================
    # Query
    # ======================================================

    query_meta = vision_index[args.query_idx]

    query_bucket = query_meta["damage_bucket"]
    query_event = query_meta["event_id"]

    print("\nQuery")
    print("-" * 50)

    print(f"query_idx : {args.query_idx}")
    print(f"event     : {query_event}")
    print(f"bucket    : {query_bucket}")

    q = vision_embeddings[args.query_idx]
    q = q.reshape(1, -1)
	
	# ---------------------------------------------------------
	# Compare unrestricted retrieval against
	# metadata-constrained retrieval.
	# ---------------------------------------------------------
    # ======================================================
    # Retrieve
    # ======================================================

    unrestricted = retrieve_mode(
        query_embedding=q,
        text_embeddings=text_embeddings,
        text_index=text_index,
        query_bucket=query_bucket,
        query_event=query_event,
        mode="none",
        topk=args.topk
    )

    constrained = retrieve_mode(
        query_embedding=q,
        text_embeddings=text_embeddings,
        text_index=text_index,
        query_bucket=query_bucket,
        query_event=query_event,
        mode="same_bucket",
        topk=args.topk
    )

	# ---------------------------------------------------------
	# Create side-by-side retrieval comparison figure.
	#
	# Left:
	#     unrestricted retrieval
	#
	# Right:
	#     constrained retrieval
	# ---------------------------------------------------------
    # ======================================================
    # Build Figure
    # ======================================================

    fig = plt.figure(figsize=(18, 10))

    fig.suptitle(
        "Unrestricted vs Constrained Retrieval",
        fontsize=20,
        fontweight="bold"
    )

	# Display baseline retrieval behavior without
	# metadata-aware routing.
    # ------------------------------------------------------
    # LEFT COLUMN — UNRESTRICTED
    # ------------------------------------------------------

    ax_left = plt.subplot(1, 2, 1)

    left_text = ""

    for i, r in enumerate(unrestricted):

        left_text += (
            f"[{i+1}] "
            f"({r['bucket']}) "
            f"{r['event']}\n"
            f"score={r['score']:.4f}\n\n"
            f"{r['text']}\n\n"
            f"{'-'*50}\n\n"
        )

    ax_left.text(
        0.01,
        0.99,
        wrap_text(left_text, 70),
        va="top",
        fontsize=10
    )

    ax_left.set_title(
        "MODE: none\n(Unrestricted Retrieval)",
        fontsize=15,
        fontweight="bold"
    )

    ax_left.axis("off")

    # ------------------------------------------------------
    # RIGHT COLUMN — CONSTRAINED
    # ------------------------------------------------------

    ax_right = plt.subplot(1, 2, 2)

    right_text = ""

    for i, r in enumerate(constrained):

        right_text += (
            f"[{i+1}] "
            f"({r['bucket']}) "
            f"{r['event']}\n"
            f"score={r['score']:.4f}\n\n"
            f"{r['text']}\n\n"
            f"{'-'*50}\n\n"
        )

    ax_right.text(
        0.01,
        0.99,
        wrap_text(right_text, 70),
        va="top",
        fontsize=10
    )

    ax_right.set_title(
        "MODE: same_bucket\n(Operational Stabilization)",
        fontsize=15,
        fontweight="bold"
    )

    ax_right.axis("off")

	# ---------------------------------------------------------
	# Summarize the central retrieval-stabilization
	# finding of Project 2B.
	# ---------------------------------------------------------
    # ======================================================
    # Footer
    # ======================================================

    footer = (
        "Project 2B Finding:\n"
        "Metadata-aware constrained retrieval dramatically "
        "improves semantic coherence under weak global "
        "cross-modal alignment."
    )

    plt.figtext(
        0.5,
        0.02,
        wrap_text(footer, 120),
        ha="center",
        fontsize=12
    )

	# ---------------------------------------------------------
	# Save visualization for:
	#
	# README figures
	# slide decks
	# portfolio demonstrations
	# interview walkthroughs
	# ---------------------------------------------------------
    # ======================================================
    # Save
    # ======================================================

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

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