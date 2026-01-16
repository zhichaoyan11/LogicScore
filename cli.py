"""Command-line interface for :mod:`logical_attribution_module`.

This module provides a small wrapper around :func:`pipeline.main` so that the
multi-stage logical attribution pipeline can be launched from the shell.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .pipeline import main as pipeline_main


# Dataset-specific defaults relative to the project root.
#
# ``file_path`` now points to the **Stage 1 input file**, which is expected to
# already be in the unified pre-processed format consumed by Stage 1 (see the
# README for details). ``save_path`` is where intermediate and final results
# will be stored.
_DATASET_DEFAULTS = {
	"musique": {
	    "file_path": "data/musique_stage1_input.json",
	    "save_path": "results/musique",
	},
	"hotpotqa": {
	    "file_path": "data/hotpotqa_stage1_input.json",
	    "save_path": "results/hotpotqa",
	},
	"2wiki": {
	    "file_path": "data/2wiki_stage1_input.json",
	    "save_path": "results/2wiki",
	},
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Create an :class:`argparse.ArgumentParser` for the CLI."""

    parser = argparse.ArgumentParser(
        description="Logical attribution multi-stage pipeline",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(_DATASET_DEFAULTS.keys()),
        default="2wiki",
        help="Dataset name (hotpotqa/musique/2wiki).",
    )
    parser.add_argument(
        "--generate_model",
        type=str,
        default="Qwen-8B",
        help="Model used to generate answers with citations.",
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default="gpt-4o-mini",
        help=(
            "Evaluation model used in Stage 3 for question typing, triplet "
            "extraction and consistency checks."
        ),
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="o4-mini-2025-04-16",
        help=(
            "Reasoning model used in Stage 3 to regenerate answers from "
            "reasoning steps."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=40,
        help="Number of worker threads for parallel stages.",
    )
    parser.add_argument(
        "--start_stage",
        type=int,
	        choices=[1, 2, 3, 4, 5],
	        default=1,
	        help=(
	            "Start stage (1=stage_1, 2=stage_2, 3=stage_3, 4=stage_4, 5=stage_5). "
	            "Stage 4 is handled by separate tooling; this flag mainly controls "
	            "which JSON checkpoints to load or recompute."
	        ),
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="Optional: override input dataset file path.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional: override directory for results.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point used by ``python -m logical_attribution_module.cli``."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    defaults = _DATASET_DEFAULTS.get(args.dataset)
    if defaults is None:
        raise SystemExit(f"Unknown dataset: {args.dataset}")

    # Fill in defaults only when caller did not supply explicit paths.
    if args.file_path is None:
        args.file_path = defaults["file_path"]
    if args.save_path is None:
        args.save_path = defaults["save_path"]

    pipeline_main(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

