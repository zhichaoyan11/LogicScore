"""Compatibility wrapper for the logical attribution pipeline.

This file used to contain the full implementation of the multi-stage logical
attribution pipeline. The logic has now been split into the
``logical_attribution_module`` package. To preserve backwards compatibility,
this module simply re-exports the key entrypoints and Stage 4 helper
functions.

Typical usage patterns that continue to work:

* ``from logical_attribution import main`` – runs the full pipeline
  (now backed by :func:`logical_attribution_module.run_pipeline`).
* ``from logical_attribution import stage_3_processing`` – access Stage 3.
* ``from logical_attribution import is_support, is_relevant, need_citation`` –
  Stage 4 evaluation helpers used by the standalone Stage 4 scripts.

For new code, prefer importing directly from :mod:`logical_attribution_module`.
"""

from __future__ import annotations

from LogicScore import (
    stage_1_processing,
    stage_2_processing,
    stage_3_processing,
    stage_4_processing,
    stage_5_processing,
    is_support,
    is_relevant,
    need_citation,
    run_pipeline as main,
)
from LogicScore.cli import main as _cli_main


__all__ = [
    "stage_1_processing",
    "stage_2_processing",
    "stage_3_processing",
    "stage_4_processing",
    "stage_5_processing",
    "is_support",
    "is_relevant",
    "need_citation",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _cli_main()
