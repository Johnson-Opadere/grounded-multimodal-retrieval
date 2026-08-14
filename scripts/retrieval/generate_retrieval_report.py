"""
Project 2B — Retrieval Report Generator
==========================================

Purpose
-------
Generates qualitative retrieval reports for Project 2B.

This script compares retrieval behavior across multiple
retrieval policies and produces human-readable reports
for retrieval auditing.

For each vision query, the system generates side-by-side
retrieval results for:

    1. unrestricted retrieval
    2. same_bucket retrieval
    3. disaster_family retrieval
    4. same_event retrieval

allowing direct inspection of retrieval stabilization.

Scientific Motivation
---------------------

Quantitative metrics explain:

    retrieval performance

but do not explain:

    retrieval behavior.

This script provides qualitative evidence showing:

    • how retrieval changes under constraints
    • how candidate filtering affects evidence quality
    • how retrieval coherence improves
    • how operational retrieval becomes more stable

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

This script provides a qualitative inspection layer
for the retrieval policies evaluated in:

    evaluate_constraints.py

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

Unrestricted retrieval.

All text evidence remains eligible.

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

Allows retrieval among related disaster families.

Example:

    hurricane-florence
        →
    hurricane-harvey

same_event
~~~~~~~~~~

Only candidates originating from the same disaster
event survive.

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
Qualitative Report

Report Contents
---------------

Query Metadata
~~~~~~~~~~~~~~

Displays:

    query index
    event
    damage bucket
    patch identifier

Retrieval Results
~~~~~~~~~~~~~~~~~

For each retrieval mode:

    rank
    similarity score
    source event
    source bucket
    evidence text

This enables direct side-by-side comparison of
retrieval policies.

Outputs
-------

Console Report

Displays:

    qualitative retrieval comparisons

for all selected query indices.

Optional Report File

When:

    --save_path

is provided,

the generated report is written to disk.

Usage
-----

Single Query

PYTHONPATH=2B python \
2B/scripts/retrieval/generate_retrieval_report.py \
    --query_indices 100

Multiple Queries

PYTHONPATH=2B python \
2B/scripts/retrieval/generate_retrieval_report.py \
    --query_indices 100 300 600

Save Report

PYTHONPATH=2B python \
2B/scripts/retrieval/generate_retrieval_report.py \
    --query_indices 100 600 \
    --save_path reports/retrieval_report.txt

Custom Top-K

PYTHONPATH=2B python \
2B/scripts/retrieval/generate_retrieval_report.py \
    --query_indices 100 \
    --topk 5

Example Output
--------------

======================================================================
QUERY 100
======================================================================

event       : hurricane-florence
bucket      : flooding
patch_id    : hurricane-florence_00000465

----------------------------------------------------------------------
RETRIEVAL MODE: none
----------------------------------------------------------------------

[1]

score       : 0.0301
event       : palu-tsunami
bucket      : earthquake_collapse

"...retrieved evidence..."

----------------------------------------------------------------------
RETRIEVAL MODE: same_bucket
----------------------------------------------------------------------

[1]

score       : 0.0532
event       : hurricane-harvey
bucket      : flooding

"...retrieved evidence..."

Files Written
-------------

Optional:

    retrieval_report.txt

when:

    --save_path

is specified.

Otherwise:

    no files are written.

Key Findings
------------

Project 2B demonstrated that:

    retrieval constraints

produce retrieval results that are substantially
more interpretable than unrestricted retrieval.

This script provides qualitative evidence supporting
the quantitative improvements observed in:

    evaluate_constraints.py

Role in Project 2B
---------------------

This script serves as:

    the qualitative auditing layer

used to:

    inspect retrieval behavior
    compare retrieval policies
    generate README examples
    generate demo outputs
    support interview walkthroughs

before introducing:

    interactive retrieval demos
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


def separator():
    return "-" * 70


# =========================================================
# Constraint Logic
# =========================================================
# ---------------------------------------------------------
# Constraint-gating function.
#
# Determines whether a retrieval candidate survives
# metadata filtering under the selected retrieval mode.
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
# Retrieval Function
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
	
	# Iterate over all candidate text evidence items.
    for idx in range(len(text_embeddings)):

        meta = text_metadata[idx]
		
		# Apply selected retrieval constraint.
        passes = candidate_passes_constraint(
            query_meta=query_meta,
            candidate_meta=meta,
            constraint_type=constraint_type
        )

        if not passes:
            continue
		
		# Compute semantic similarity for surviving candidates.
        score = cosine_similarity(
            query_embedding,
            text_embeddings[idx]
        )

        results.append({
            "score": score,
            "metadata": meta
        })
	
	# Rank candidates by semantic similarity score.
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:topk]


# =========================================================
# Report Generation
# =========================================================
# ---------------------------------------------------------
# Generates a complete qualitative retrieval report
# for a single vision query.
#
# Compares all retrieval modes side-by-side.
# ---------------------------------------------------------
def generate_query_report(
    query_idx,
    vision_embeddings,
    vision_metadata,
    text_embeddings,
    text_metadata,
    topk
):

    lines = []

    query_embedding = vision_embeddings[
        query_idx
    ]

    query_meta = vision_metadata[
        query_idx
    ]

    query_event = query_meta["event_id"]
    query_bucket = query_meta["damage_bucket"]

	# ---------------------------------------------------------
	# Display query metadata for retrieval auditing.
	# ---------------------------------------------------------
    # =====================================================
    # Query Header
    # =====================================================

    lines.append("=" * 70)
    lines.append(f"QUERY {query_idx}")
    lines.append("=" * 70)

    lines.append("")
    lines.append(f"event       : {query_event}")
    lines.append(f"bucket      : {query_bucket}")
    lines.append(
        f"patch_id    : "
        f"{query_meta['patch_id']}"
    )

    lines.append("")

	# ---------------------------------------------------------
	# Evaluate all supported retrieval policies for the
	# current query.
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

        lines.append(separator())
        lines.append(
            f"RETRIEVAL MODE: {mode}"
        )
        lines.append(separator())
        lines.append("")

        results = run_retrieval(
            query_embedding=query_embedding,
            query_meta=query_meta,
            text_embeddings=text_embeddings,
            text_metadata=text_metadata,
            constraint_type=mode,
            topk=topk
        )

        if len(results) == 0:

            lines.append(
                "No candidates survived constraints."
            )

            lines.append("")
            continue

        for rank, result in enumerate(
            results,
            start=1
        ):

            meta = result["metadata"]

            lines.append(f"[{rank}]")

            lines.append(
                f"score       : "
                f"{result['score']:.4f}"
            )

            lines.append(
                f"event       : "
                f"{meta['source_event']}"
            )

            lines.append(
                f"bucket      : "
                f"{meta['damage_bucket']}"
            )

            lines.append("")

            lines.append(meta["text"])

            lines.append("")
            lines.append(separator())
            lines.append("")

    return "\n".join(lines)


# =========================================================
# Main
# =========================================================

def main(args):

    print("=" * 60)
    print("2B — Retrieval Report Generator")
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
	
	# ---------------------------------------------------------
	# Generate qualitative reports for all requested
	# query indices.
	# ---------------------------------------------------------
    # =====================================================
    # Generate Reports
    # =====================================================

    reports = []

    for query_idx in args.query_indices:

        report = generate_query_report(
            query_idx=query_idx,
            vision_embeddings=vision_embeddings,
            vision_metadata=vision_metadata,
            text_embeddings=text_embeddings,
            text_metadata=text_metadata,
            topk=args.topk
        )

        reports.append(report)

    final_report = "\n\n".join(reports)

    # =====================================================
    # Print Report
    # =====================================================

    print("\n" + "=" * 60)
    print("RETRIEVAL REPORT")
    print("=" * 60)

    print("\n")
    print(final_report)

	# ---------------------------------------------------------
	# Optionally persist retrieval reports to disk for:
	#
	#     README examples
	#     slide generation
	#     qualitative analysis
	#     interview demonstrations
	# ---------------------------------------------------------
    # =====================================================
    # Save Report
    # =====================================================

    if args.save_path is not None:

        os.makedirs(
            os.path.dirname(args.save_path),
            exist_ok=True
        )

        with open(args.save_path, "w") as f:
            f.write(final_report)

        print("\n" + "=" * 60)
        print(
            f"[OK] Report saved to:"
        )
        print(args.save_path)
        print("=" * 60)

    # =====================================================
    # Final
    # =====================================================

    print("\n" + "=" * 60)
    print("[OK] Retrieval report generation complete.")
    print("=" * 60)


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query_indices",
        type=int,
        nargs="+",
        required=True,
        help="List of query indices"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-k retrieval results"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional report save path"
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final",
        help="Path to 2A artifacts"
    )

    args = parser.parse_args()

    main(args)