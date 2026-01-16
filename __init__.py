"""Public package API for :mod:`logical_attribution_module`.

This package splits the original monolithic ``logical_attribution.py`` script
into a set of focused modules, and exposes convenient entrypoints for both
programmatic and CLI use.
"""

from .preprocess import pre_process
from .stage1 import stage_1_processing
from .stage2 import stage_2_processing
from .stage3 import stage_3_processing
from .stage4 import stage_4_processing, is_support, is_relevant, need_citation
from .stage5 import stage_5_processing
from .pipeline import main as run_pipeline
from .cli import main as cli_main, build_arg_parser

__all__ = [
    "pre_process",
    "stage_1_processing",
    "stage_2_processing",
    "stage_3_processing",
    "stage_4_processing",
    "stage_5_processing",
    "is_support",
    "is_relevant",
    "need_citation",
    "run_pipeline",
    "cli_main",
    "build_arg_parser",
]

