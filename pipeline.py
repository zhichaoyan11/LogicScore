"""End-to-end logical attribution pipeline.

This module wires together the Stage 1 to Stage 5 helpers that live in the
other modules in this package. The public entrypoint is ``main(args)``.

The pipeline now assumes that the input to Stage 1 is already pre-processed
into the unified format described in :mod:`logical_attribution_module.stage1`.
The older ``pre_process`` step is no longer invoked automatically here; if you
still need it, you can call :func:`logical_attribution_module.pre_process`
yourself and save the result as the Stage 1 input file.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .stage1 import stage_1_processing
from .stage2 import stage_2_processing
from .stage3 import stage_3_processing
from .stage5 import stage_5_processing


def main(args: Any) -> None:
    """Run the logical attribution pipeline.

    The ``args`` object is expected to have at least the following
    attributes (the CLI in :mod:`logical_attribution_module.cli` creates a
    compatible object):

    - ``dataset``: dataset name (used mainly for logging)
    - ``file_path``: Stage 1 input dataset path (already pre-processed)
    - ``save_path``: directory used to store intermediate / final JSON files
    - ``generate_model``: model name used in output file names
    - ``num_workers``: optional, worker threads for parallel stages
    - ``start_stage``: optional, integer in ``[1, 5]`` indicating the
      first stage to run (1 = Stage 1)
    """

    start_stage = getattr(args, "start_stage", 1)

    eval_model = getattr(args, "eval_model", "gpt-4o-mini")
    reasoning_model = getattr(args, "reasoning_model", "o4-mini-2025-04-16")

    print(f"Starting pipeline, start_stage={start_stage}")
    print(f"  save_path: {args.save_path}")
    print(f"  generate_model: {args.generate_model}")
    print(f"  eval_model: {eval_model}")
    print(f"  reasoning_model: {reasoning_model}")
    print()

    # ========== Stage 1 input loading and processing ==========
    if start_stage <= 1:
        print("Loading Stage 1 input data...")
        if not os.path.exists(args.file_path):
            print(f"ERROR: Stage 1 input file not found: {args.file_path}")
            return
        with open(args.file_path, "r", encoding="utf-8") as f:
            stage_1_input = json.load(f)
        print(f"Stage 1 input loaded, num examples: {len(stage_1_input)}")

        print("Running Stage 1...")
        stage_1_data = stage_1_processing(stage_1_input, args)
        print(f"Stage 1 done, num examples: {len(stage_1_data)}")

        path = os.path.join(args.save_path, f"{args.generate_model}_stage_1.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stage_1_data, f, indent=4, ensure_ascii=False)
        print(f"Stage 1 results saved to: {path}")
    else:
        print("Loading Stage 1 data from file...")
        path = os.path.join(args.save_path, f"{args.generate_model}_stage_1.json")
        if not os.path.exists(path):
            print(f"ERROR: Stage 1 file not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            stage_1_data = json.load(f)
        print(f"Stage 1 data loaded, num examples: {len(stage_1_data)}")

    # ========== Stage 2 ==========
    if start_stage <= 2:
        print("Running Stage 2...")
        workers = getattr(args, "num_workers", 10)
        stage_2_data = stage_2_processing(stage_1_data, args=args, num_workers=workers)
        path = os.path.join(args.save_path, f"{args.generate_model}_stage_2.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stage_2_data, f, indent=4, ensure_ascii=False)
        print(f"Stage 2 done, num examples: {len(stage_2_data)}")
    else:
        print("Loading Stage 2 data from file...")
        path = os.path.join(args.save_path, f"{args.generate_model}_stage_2.json")
        if not os.path.exists(path):
            print(f"ERROR: Stage 2 file not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            stage_2_data = json.load(f)
        print(f"Stage 2 data loaded, num examples: {len(stage_2_data)}")

    # ========== Stage 3 ==========
    if start_stage <= 3:
        print("Running Stage 3...")
        stage_3_data = stage_3_processing(stage_2_data, args)
        path = os.path.join(args.save_path, f"{args.generate_model}_stage_3.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stage_3_data, f, indent=4, ensure_ascii=False)
        print(f"Stage 3 done, num examples: {len(stage_3_data)}")
    else:
        print("Loading Stage 3 data from file...")
        path = os.path.join(args.save_path, f"{args.generate_model}_stage_3.json")
        if not os.path.exists(path):
            print(f"ERROR: Stage 3 file not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            stage_3_data = json.load(f)
        print(f"Stage 3 data loaded, num examples: {len(stage_3_data)}")

    # Stage 4 is handled by separate tooling and scripts.

    # ========== Stage 5 ==========
    if start_stage <= 5:
        print("Running Stage 5...")
        workers = getattr(args, "num_workers", 4)
        subset = stage_3_data[:1000]  # keep original behavior: only first 1000
        stage_5_data = stage_5_processing(args, subset, num_workers=workers)
        path = os.path.join(args.save_path, f"{args.generate_model}_stage_5.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stage_5_data, f, indent=4, ensure_ascii=False)
        print(f"Stage 5 done, num examples: {len(stage_5_data)}")

    print("=" * 80)
    print("Pipeline finished.")
    print("=" * 80)
