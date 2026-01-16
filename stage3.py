"""Stage 3: reasoning-path analysis and scoring.

This module implements the minimum-step reasoning path extraction and the
faithfulness / reasonableness / conciseness scoring, with checkpointed
multi-threaded execution.
"""

import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple

import numpy as np  # noqa: F401 - kept for backwards compatibility
from tqdm import tqdm

from .qa import (
    decsided_by_LLM,
    reasoning_generation,
    check_answer_by_llm,
    extract_triplets,
    extract_entity_from_question,
    check_entity_by_llm,
)


def parse_logic_answer(logic_answer_str: str) -> Dict:
    """Parse the logic answer string into propositions and expression.

    The original data format sometimes stores propositions under the key
    ``"\xa0"``; we mirror that behaviour but fall back to ``"propositions"``.
    """

    logical_json = eval(logic_answer_str)  # noqa: S307 - mirrors original behaviour
    try:
        logic_attribution = logical_json["\xa0"]
    except KeyError:
        logic_attribution = logical_json.get("propositions", {})
    logical_expression = logical_json.get("logical_expression", "")
    return {"propositions": logic_attribution, "logical_expression": logical_expression}


def get_minimum_reasoning_steps(question, gold_answer, propositions, args):
    """Filter propositions to the minimum necessary reasoning steps.

    The evaluation model is controlled by ``args.eval_model`` (defaulting to
    ``gpt-4o-mini``).
    """

    minimum_steps: List[str] = []
    eval_model = getattr(args, "eval_model", "gpt-4o-mini")
    for _, step in propositions.items():
        if decsided_by_LLM(question, gold_answer, step, model=eval_model):
            minimum_steps.append(step)
    return minimum_steps


def evaluate_faithfulness(question, predict_answer, steps, args):
    """Evaluate faithfulness of reasoning steps to the predicted answer.

    Uses ``args.reasoning_model`` to regenerate an answer from the steps, and
    ``args.eval_model`` (defaulting to ``gpt-4o-mini``) to check semantic
    equivalence when needed.
    """

    reasoning_again_answer = reasoning_generation(question, steps, args)
    if isinstance(predict_answer, list):
        predict_answer = predict_answer[0]

    eval_model = getattr(args, "eval_model", "gpt-4o-mini")
    if predict_answer.strip() in reasoning_again_answer:
        return 1
    if check_answer_by_llm(
        question, predict_answer, reasoning_again_answer, model=eval_model
    ):
        return 1
    return 0


def identify_key_triplets(minimum_reasoning_steps, entities, gold_answer, args):
    """Identify start/end triplets and necessary step indices.

    Returns ``(start_triplets, end_triplets_list, necessary_statement_index)``.
    """

    start_triplets: List[Tuple[str, str, str]] = []
    end_triplets_list: List[Tuple[int, List[Tuple[str, str, str]]]] = []
    necessary_statement_index: Set[int] = set()

    eval_model = getattr(args, "eval_model", "gpt-4o-mini")

    for i, step in enumerate(minimum_reasoning_steps):
        triplets = extract_triplets(step, args=args)

        for entity in entities:
            for triple in triplets:
                if None in triple:
                    continue
                if (
                    entity["text"].lower() == triple[0].lower()
                    or entity["text"].lower() == triple[2].lower()
                ):
                    if triplets not in [start_triplets]:
                        necessary_statement_index.add(i)
                    if triple not in start_triplets:
                        start_triplets.append(triple)
                elif check_entity_by_llm(entity["text"], triple, model=eval_model):
                    if triplets not in [start_triplets]:
                        necessary_statement_index.add(i)
                    if triple not in start_triplets:
                        start_triplets.append(triple)

        if gold_answer.lower().strip() in step.lower() or check_entity_by_llm(
            gold_answer, step, model=eval_model
        ):
            necessary_statement_index.add(i)
            end_triplets_list.append((i, triplets))

    for idx in necessary_statement_index:
        if idx < 0 or idx >= len(minimum_reasoning_steps):
            raise ValueError(
                f"索引不一致错误：necessary_statement_index 中的索引 {idx} "
                f"超出 minimum_reasoning_steps 的范围 [0, {len(minimum_reasoning_steps)-1}]"
            )

    return start_triplets, end_triplets_list, necessary_statement_index


def find_matching_start_triplet(bridge_entity, start_triplets):
    for start_triple in start_triplets:
        if (
            bridge_entity.lower() in start_triple[0].lower()
            or bridge_entity.lower() in start_triple[2].lower()
        ):
            return start_triple
    return None


def find_intermediate_triplets_with_bridge(
    bridge_entity,
    minimum_reasoning_steps,
    necessary_statement_index,
    used_indices,
    args,
):
    """Find intermediate triplets that contain the current bridge entity."""

    results = []
    for idx in range(len(minimum_reasoning_steps)):
        if idx in used_indices:
            continue
        step = minimum_reasoning_steps[idx]
        triplets = extract_triplets(step, args=args)
        for tr in triplets:
            if (
                bridge_entity.lower() in tr[0].lower()
                or bridge_entity.lower() in tr[2].lower()
            ):
                if bridge_entity.lower() == tr[0].lower():
                    next_bridge = tr[2]
                else:
                    next_bridge = tr[0]
                results.append((idx, tr, next_bridge))
    return results


def search_path_recursively(
    current_bridge,
    start_triplets,
    minimum_reasoning_steps,
    necessary_statement_index,
    used_indices,
    args,
    path_indices=None,
    max_depth: int = 10,
):
    """Depth-first search for a reasoning path connecting start and end triplets."""

    if path_indices is None:
        path_indices = set()
    if max_depth <= 0:
        return False, path_indices

    # If the current bridge already matches a start triplet, we have a path.
    if find_matching_start_triplet(current_bridge, start_triplets):
        for start_idx in necessary_statement_index:
            if start_idx in used_indices:
                continue
            step = minimum_reasoning_steps[start_idx]
            triplets = extract_triplets(step, args=args)
            for triplet in triplets:
                for start_triple in start_triplets:
                    if (
                        triplet[0].lower() == start_triple[0].lower()
                        or triplet[0].lower() == start_triple[2].lower()
                        or triplet[2].lower() == start_triple[0].lower()
                        or triplet[2].lower() == start_triple[2].lower()
                    ):
                        path_indices.add(start_idx)
                        return True, path_indices
        return True, path_indices

    # Otherwise, search via intermediate bridge entities.
    intermediate_results = find_intermediate_triplets_with_bridge(
        current_bridge,
        minimum_reasoning_steps,
        necessary_statement_index,
        used_indices,
        args,
    )

    for step_idx, _, next_bridge_entity in intermediate_results:
        new_used_indices = used_indices.copy()
        new_used_indices.add(step_idx)
        new_path_indices = path_indices.copy()
        new_path_indices.add(step_idx)

        found, final_path_indices = search_path_recursively(
            next_bridge_entity,
            start_triplets,
            minimum_reasoning_steps,
            necessary_statement_index,
            new_used_indices,
            args,
            new_path_indices,
            max_depth - 1,
        )
        if found:
            return True, final_path_indices

    return False, path_indices


def build_reasoning_path(
    start_triplets,
    end_triplets_list,
    minimum_reasoning_steps,
    necessary_statement_index,
    gold_answer,
    args,
):
    """Build a reasoning path from start to end triplets, if possible."""

    if not start_triplets or not end_triplets_list:
        return False, set()

    for end_step_idx, end_triplets in end_triplets_list:
        for end_triple in end_triplets:
            if (
                gold_answer.lower() in end_triple[0].lower()
                or end_triple[0].lower() in gold_answer.lower()
            ):
                bridge_entity = end_triple[2]
            elif (
                gold_answer.lower() in end_triple[2].lower()
                or end_triple[2].lower() in gold_answer.lower()
            ):
                bridge_entity = end_triple[0]
            else:
                bridge_entity = ""

            if not bridge_entity:
                continue

            initial_path_indices = {end_step_idx}
            initial_used_indices = {end_step_idx}
            found, path_indices = search_path_recursively(
                bridge_entity,
                start_triplets,
                minimum_reasoning_steps,
                necessary_statement_index,
                initial_used_indices,
                args,
                initial_path_indices,
            )
            if found:
                return True, path_indices

    return False, set()


def calculate_final_step_indices(path_indices):
    return path_indices


def calculate_scores(final_step_indices, propositions, path_found):
    conciseness_score = len(final_step_indices) / len(propositions) if propositions else 0
    reasonable_score = 1 if path_found else 0
    return conciseness_score, reasonable_score


def process_single_data(data, args):
    question = data["query"]
    gold_answer = data["gold_answer"]
    predict_answer = data["predict_answer"]

    logic_data = parse_logic_answer(data["logic_answer"])
    propositions = logic_data["propositions"]

    minimum_reasoning_steps = get_minimum_reasoning_steps(
        question, gold_answer, propositions, args
    )
    data["minimum_reasoning_steps"] = minimum_reasoning_steps

    # For simple yes/no questions we skip the path-finding metrics.
    if gold_answer.lower() in ["yes", "no"]:
        return None, None

    entities, entity_type = extract_entity_from_question(question, args=args)
    if entity_type != 1:
        return None, None

    steps = ["Step: " + value for _, value in propositions.items()]
    faithful_score = evaluate_faithfulness(question, predict_answer, steps, args)

    start_triplets, end_triplets_list, necessary_statement_index = identify_key_triplets(
        minimum_reasoning_steps, entities, gold_answer, args
    )

    path_found, path_indices = build_reasoning_path(
        start_triplets,
        end_triplets_list,
        minimum_reasoning_steps,
        necessary_statement_index,
        gold_answer,
        args,
    )

    final_step_indices = calculate_final_step_indices(path_indices)
    conciseness_score, reasonable_score = calculate_scores(
        final_step_indices, propositions, path_found
    )

    scores = {
        "faithful_score": faithful_score,
        "reasonable_score": reasonable_score,
        "conciseness_score": conciseness_score,
    }
    data["final_step_indices"] = sorted(list(final_step_indices))
    return data, scores


def _load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return [], set(), {
            "faithful_score": 0.0,
            "reasonable_score": 0.0,
            "conciseness_score": 0.0,
            "data_count": 0,
        }
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        return (
            checkpoint.get("data", []),
            set(checkpoint.get("processed_indices", [])),
            checkpoint.get(
                "stats",
                {
                    "faithful_score": 0.0,
                    "reasonable_score": 0.0,
                    "conciseness_score": 0.0,
                    "data_count": 0,
                },
            ),
        )
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error loading checkpoint: {e}")
        return [], set(), {
            "faithful_score": 0.0,
            "reasonable_score": 0.0,
            "conciseness_score": 0.0,
            "data_count": 0,
        }


def _save_checkpoint(checkpoint_path, new_data, processed_indices, stats):
    try:
        checkpoint = {
            "data": new_data,
            "processed_indices": list(processed_indices),
            "stats": stats,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error saving checkpoint: {e}")


def _process_single_data_wrapper(args_tuple):
    index, data, args = args_tuple
    if not data["predict_answer"]:
        return index, None, None
    try:
        processed_data, scores = process_single_data(data, args)
        return index, processed_data, scores
    except Exception as e:  # pylint: disable=broad-except
        import traceback

        print(f"Error processing data at index {index}: {traceback.print_exc()}")
        return index, None, None


def stage_3_processing(
    total_data,
    args=None,
    num_workers: int = 4,
    checkpoint_dir: str = None,
    enable_checkpoint: bool = True,
):
    """Run Stage 3 over ``total_data`` with optional checkpointing.

    Args:
        total_data: Iterable of data dicts (output from Stage 2).
        args: Argument namespace carrying configuration.
        num_workers: Thread-pool size.
        checkpoint_dir: Directory to store checkpoints.
        enable_checkpoint: If ``True``, periodically save progress.
    """

    if checkpoint_dir is None:
        checkpoint_dir = "./checkpoints"
    if enable_checkpoint and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    if enable_checkpoint:
        model_name = args.generate_model
        dataset_name = args.dataset
        checkpoint_filename = f"{model_name}_{dataset_name}_stage3_checkpoint.json"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    else:
        checkpoint_path = None

    if enable_checkpoint and checkpoint_path:
        new_data, processed_indices, stats = _load_checkpoint(checkpoint_path)
        print(
            f"✓ Loaded checkpoint: {len(new_data)} items processed, {len(processed_indices)} indices"
        )
    else:
        new_data = []
        processed_indices = set()
        stats = {
            "faithful_score": 0.0,
            "reasonable_score": 0.0,
            "conciseness_score": 0.0,
            "data_count": 0,
        }

    pending_data = [
        (i, data, args) for i, data in enumerate(total_data) if i not in processed_indices
    ]
    if not pending_data:
        print("✓ All data has been processed!")
        return new_data

    print(f"Processing {len(pending_data)} remaining items (total: {len(total_data)})")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_single_data_wrapper, args_tuple): args_tuple[0]
            for args_tuple in pending_data
        }
        with tqdm(total=len(pending_data), desc="Stage 3 Processing") as pbar:
            for future in as_completed(futures):
                try:
                    index, processed_data, scores = future.result()
                    if processed_data is not None and scores is not None:
                        processed_data.update(scores)
                        new_data.append(processed_data)
                        stats["faithful_score"] += scores["faithful_score"]
                        stats["reasonable_score"] += scores["reasonable_score"]
                        stats["conciseness_score"] += scores["conciseness_score"]
                        stats["data_count"] += 1
                    processed_indices.add(index)
                    if (
                        enable_checkpoint
                        and checkpoint_path
                        and len(processed_indices) % 10 == 0
                    ):
                        _save_checkpoint(checkpoint_path, new_data, processed_indices, stats)
                    pbar.update(1)
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error in future result: {e}")
                    pbar.update(1)

    if enable_checkpoint and checkpoint_path:
        _save_checkpoint(checkpoint_path, new_data, processed_indices, stats)

    if stats["data_count"] > 0:
        print(f"\n✓ Processed {stats['data_count']} data points.")
        print(f"  faithful_score: {stats['faithful_score'] / stats['data_count']:.4f}")
        print(f"  reasonable_score: {stats['reasonable_score'] / stats['data_count']:.4f}")
        print(
            f"  conciseness_score: {stats['conciseness_score'] / stats['data_count']:.4f}"
        )

    return new_data
