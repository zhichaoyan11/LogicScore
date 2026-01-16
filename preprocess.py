"""Dataset pre-processing utilities for the logical attribution pipeline.

This module encapsulates the `pre_process` function and its helpers, which
prepare musique, hotpotqa and 2wiki style datasets into a unified format.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from prompt import question_type_check
from .qa import check_question_type


def _process_single_hotpotqa_item(data):
    """Filter a single HotpotQA / 2Wiki record.

    Returns (passed: bool, processed_data or None).
    """
    try:
        answer = data["answer"]
        question = data["query"]

        # 1) Skip yes/no answers
        if "yes" in answer or "no" in answer:
            return False, None

        # 2) Skip answers that contain numbers
        if re.findall(r"\d+", answer):
            return False, None

        # 3) Ensure the question is a sequential-chain type question
        if not check_question_type(question):
            return False, None

        return True, data
    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ 处理数据时出错: {e}")
        return False, None


def _process_hotpotqa_wrapper(args_tuple):
    """Thread-pool wrapper for `_process_single_hotpotqa_item`.

    Args:
        args_tuple: (idx, data)
    Returns:
        (idx, passed, processed_data)
    """
    idx, data = args_tuple
    passed, processed_data = _process_single_hotpotqa_item(data)
    return idx, passed, processed_data


def pre_process(args):
    """Entry point used by the main pipeline.

    It normalises different datasets to a shared structure:
    `{id, query, context, answer}` where context is a list of
    `[title, [sent_1, sent_2, ...]]`.
    """
    if "musique" in args.file_path:
        with open(args.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        all_data = []
        for line in tqdm(lines, desc="Pre-processing musique"):
            json_data = json.loads(line)
            answer = json_data["answer"]
            if "yes" in answer or "no" in answer:
                continue
            if re.findall(r"\d+", answer):
                continue

            paragraphs = json_data["paragraphs"]
            context = []
            for doc in paragraphs:
                content = doc["paragraph_text"]
                sentences = sent_tokenize(content)
                context.append([doc["title"], sentences])

            all_data.append(
                {
                    "id": json_data["id"],
                    "query": json_data["question"],
                    "context": context,
                    "answer": answer,
                }
            )
        return all_data

    if "hotpot" in args.file_path or "2wiki" in args.file_path:
        with open(args.file_path, "r", encoding="utf-8") as f:
            total_data = json.load(f)[:3000]

        all_data = []
        num_workers = getattr(args, "num_workers", 10)

        pending_data = [(i, data) for i, data in enumerate(total_data)]
        results = {}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_hotpotqa_wrapper, args_tuple): args_tuple[0]
                for args_tuple in pending_data
            }

            with tqdm(total=len(pending_data), desc="Pre-processing hotpotqa/2wiki") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, passed, processed_data = future.result()
                        if passed:
                            results[idx] = processed_data
                        pbar.update(1)
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"❌ 处理任务时出错: {e}")
                        pbar.update(1)

        all_data = [results[i] for i in sorted(results.keys())]
        return all_data

    raise NotImplementedError("Unsupported dataset type in file_path: " + args.file_path)

