"""
Project 2B — Grounded LLM Report Generation
==============================================

Purpose
-------
Generates grounded disaster assessment reports using:

    1. multimodal retrieval
    2. metadata-aware routing
    3. evidence grounding
    4. GPT-4o or Claude report generation

This script extends the Project 2B retrieval pipeline by
connecting grounded retrieval evidence to a large language
model (LLM).

The objective is not autonomous reasoning or agentic
planning.

Instead, the objective is:

    retrieval
        +
    grounding
        +
    evidence-based generation

Usage
-----

OpenAI

PYTHONPATH=2B python \
2B/scripts/llm/generate_grounded_report.py \
    --provider openai \
    --query_idx 100

Anthropic

PYTHONPATH=2B python \
2B/scripts/llm/generate_grounded_report.py \
    --provider anthropic \
    --query_idx 100

Custom Retrieval

PYTHONPATH=2B python \
2B/scripts/llm/generate_grounded_report.py \
    --provider openai \
    --query_idx 100 \
    --constraint disaster_family \
    --visual_topk 5 \
    --text_topk 5

Environment Variables
---------------------

OpenAI

    export OPENAI_API_KEY="..."

Anthropic

    export ANTHROPIC_API_KEY="..."

Dependencies
------------

pip install openai anthropic

Outputs
-------

2B/results/llm_reports/

    query_<idx>_<provider>_report.txt

Author
------
Project 2B

Multimodal Retrieval Systems Engineering
"""

import os
import argparse
import numpy as np

from openai import OpenAI
from anthropic import Anthropic

from scripts.retrieval.grounded_multimodal_summary import (
    load_json,
    retrieve_visual_neighbors,
    retrieve_text_evidence,
    build_grounded_prompt,
)

# =========================================================
# Prompt Builder
# =========================================================

def build_llm_prompt(
    query_meta,
    visual_results,
    text_results
):
    """
    Build a structured grounded prompt for LLM generation.

    Purpose
    -------
    Converts retrieval outputs into an evidence package
    suitable for GPT-4o or Claude.

    The generated prompt represents the final grounding
    stage of Project 2B and serves as the interface
    between retrieval and generation.

    Pipeline Position
    -----------------

    Query Patch
            ↓
    Visual Retrieval
            ↓
    Text Retrieval
            ↓
    Grounded Prompt
            ↓
    GPT-4o / Claude

    Inputs
    ------

    query_meta : dict

        Query image metadata.

        Expected fields:

            event_id
            damage_bucket

    visual_results : list

        Retrieved visual neighbors produced by:

            retrieve_visual_neighbors()

    text_results : list

        Retrieved textual evidence produced by:

            retrieve_text_evidence()

    Prompt Structure
    ----------------

    QUERY EVENT

        Event associated with query image.

    QUERY DAMAGE BUCKET

        Damage category associated with query image.

    VISUAL EVIDENCE

        Retrieved visual neighbors.

    TEXTUAL EVIDENCE

        Retrieved textual evidence.

    TASK

        Instructions for grounded report generation.

    Grounding Policy
    ----------------

    The model is instructed to:

        use only retrieved evidence

    and:

        avoid unsupported claims

    Returns
    -------

    str

        Fully formatted prompt suitable for
        GPT-4o or Claude report generation.
    """

    lines = []

    lines.append(
        "You are an expert disaster-response analyst."
    )

    lines.append("")

    lines.append(
        "Use ONLY the retrieved evidence."
    )

    lines.append(
        "Do not invent unsupported facts."
    )

    lines.append(
        "If evidence is insufficient, explicitly state uncertainty."
    )

    lines.append("")
    lines.append("=" * 60)

    lines.append("QUERY EVENT")
    lines.append(query_meta["event_id"])

    lines.append("")

    lines.append("QUERY DAMAGE BUCKET")
    lines.append(query_meta["damage_bucket"])

    lines.append("")
    lines.append("=" * 60)

    lines.append("VISUAL EVIDENCE")

    for i, result in enumerate(
        visual_results,
        start=1
    ):

        meta = result["metadata"]

        lines.append(
            f"{i}. "
            f"{meta['event_id']} | "
            f"{meta['damage_bucket']}"
        )

    lines.append("")
    lines.append("=" * 60)

    lines.append("TEXTUAL EVIDENCE")

    for i, result in enumerate(
        text_results,
        start=1
    ):

        meta = result["metadata"]

        lines.append(
            f"{i}. {meta['text']}"
        )

    lines.append("")
    lines.append("=" * 60)

    lines.append("TASK")

    lines.append(
        "Generate a grounded disaster assessment report."
    )

    lines.append("")

    lines.append(
        "Required Sections:"
    )

    lines.append(
        "1. Likely Disaster Type"
    )

    lines.append(
        "2. Observed Damage Patterns"
    )

    lines.append(
        "3. Infrastructure Impact"
    )

    lines.append(
        "4. Operational Implications"
    )

    return "\n".join(lines)


# =========================================================
# OpenAI
# =========================================================

def generate_openai_report(prompt):
    """
    Generate a grounded disaster report using GPT-4o.

    Purpose
    -------
    Sends a grounded retrieval prompt to OpenAI
    GPT-4o and returns the generated report.

    Pipeline Position
    -----------------

    Grounded Prompt
            ↓
        GPT-4o
            ↓
    Grounded Report

    Parameters
    ----------

    prompt : str

        Grounded prompt generated from
        retrieval evidence.

    Environment Variable
    --------------------

    OPENAI_API_KEY

    Expected Model
    --------------

    gpt-4o

    Returns
    -------

    str

        Generated disaster assessment report.
    """

    client = OpenAI()

    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )

    return response.output_text


# =========================================================
# Anthropic
# =========================================================

def generate_anthropic_report(prompt):
    """
    Generate a grounded disaster report using Claude.

    Purpose
    -------
    Sends a grounded retrieval prompt to
    Anthropic Claude Sonnet and returns
    the generated report.

    Pipeline Position
    -----------------

    Grounded Prompt
            ↓
    Claude Sonnet 4.6
            ↓
    Grounded Report

    Parameters
    ----------

    prompt : str

        Grounded prompt generated from
        retrieval evidence.

    Environment Variable
    --------------------

    ANTHROPIC_API_KEY

    Expected Model
    --------------

    claude-sonnet-4-6

    Returns
    -------

    str

        Generated disaster assessment report.
    """

    client = Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text

# =========================================================
# Main
# =========================================================

def main(args):
    """
    Execute grounded report generation pipeline.

    Purpose
    -------
    Loads Project 2A artifacts, performs
    multimodal retrieval, constructs a
    grounded prompt, generates a report
    using GPT-4o or Claude, and saves
    the resulting assessment.

    Pipeline
    --------

    Query Image
            ↓
    Visual Retrieval
            ↓
    Text Retrieval
            ↓
    Grounded Prompt
            ↓
    GPT-4o / Claude
            ↓
    Report Generation
            ↓
    Save Report

    Retrieval Components
    --------------------

    Visual Retrieval

        retrieve_visual_neighbors()

    Text Retrieval

        retrieve_text_evidence()

    Inputs
    ------

    args.query_idx

        Query image index.

    args.constraint

        Retrieval policy.

        Supported values:

            none
            same_bucket
            same_event
            disaster_family

    args.visual_topk

        Number of visual neighbors.

    args.text_topk

        Number of textual evidence items.

    args.provider

        LLM provider.

        Supported values:

            openai
            anthropic

    Output Files
    ------------

    results/llm_reports/

        query_<idx>_openai_report.txt

        query_<idx>_anthropic_report.txt

    Returns
    -------

    None
    """

    print("=" * 70)
    print("2B_v2 — Grounded LLM Report Generation")
    print("=" * 70)

    artifact_dir = args.artifact_dir

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

    query_idx = args.query_idx

    query_embedding = vision_embeddings[
        query_idx
    ]

    query_meta = vision_metadata[
        query_idx
    ]

    # =====================================================
    # Visual Retrieval Stage
    # =====================================================
    #
    # Retrieve nearest visual neighbors from
    # the Project 2A embedding space.
    #
    # Similarity Metric:
    #
    #     cosine similarity
    #
    # because all exported embeddings are
    # L2-normalized.
    #
    # Output:
    #
    #     top-k visually similar disaster
    #     examples.
    #
    # =====================================================

    visual_results = retrieve_visual_neighbors(
        query_idx=query_idx,
        vision_embeddings=vision_embeddings,
        vision_metadata=vision_metadata,
        topk=args.visual_topk
    )

    # =====================================================
    # Text Retrieval Stage
    # =====================================================
    #
    # Retrieve textual evidence constrained
    # by the selected routing policy.
    #
    # Supported Policies:
    #
    #     none
    #     same_bucket
    #     same_event
    #     disaster_family
    #
    # Output:
    #
    #     grounded textual evidence used
    #     for downstream report generation.
    #
    # =====================================================

    text_results = retrieve_text_evidence(
        query_embedding=query_embedding,
        query_meta=query_meta,
        text_embeddings=text_embeddings,
        text_metadata=text_metadata,
        constraint_type=args.constraint,
        topk=args.text_topk
    )

    # =====================================================
    # Grounded Prompt Construction
    # =====================================================
    #
    # Convert retrieval outputs into an
    # LLM-consumable evidence package.
    #
    # This stage separates:
    #
    #     retrieval quality
    #
    # from:
    #
    #     generation quality
    #
    # enabling provider-agnostic report
    # generation.
    #
    # =====================================================

    prompt = build_llm_prompt(
        query_meta=query_meta,
        visual_results=visual_results,
        text_results=text_results
    )

    print("\nGenerating report...")

    # =====================================================
    # Provider-Agnostic Generation Layer
    # =====================================================
    #
    # The retrieval and grounding pipeline
    # is independent of the selected LLM.
    #
    # Supported Providers:
    #
    #     GPT-4o
    #     Claude Sonnet 4.6
    #
    # Both providers consume the same
    # grounded evidence package.
    #
    # =====================================================

    if args.provider == "openai":

        report = generate_openai_report(
            prompt
        )

    elif args.provider == "anthropic":

        report = generate_anthropic_report(
            prompt
        )

    else:

        raise ValueError(
            f"Unknown provider: {args.provider}"
        )

    # =====================================================
    # Report Persistence
    # =====================================================
    #
    # Save generated report for:
    #
    #     reproducibility
    #     qualitative evaluation
    #     README examples
    #     provider comparison
    #
    # Output Directory:
    #
    #     results/llm_reports/
    #
    # =====================================================

    output_dir = (
        "2B/results/llm_reports"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        f"query_{query_idx}_{args.provider}_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    print("\nReport saved:")
    print(output_path)

    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)

    print("\n[OK] Complete.")


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=[
            "openai",
            "anthropic"
        ]
    )

    parser.add_argument(
        "--query_idx",
        type=int,
        required=True
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
        ]
    )

    parser.add_argument(
        "--visual_topk",
        type=int,
        default=3
    )

    parser.add_argument(
        "--text_topk",
        type=int,
        default=3
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="2A/artifacts/final"
    )

    args = parser.parse_args()

    main(args)