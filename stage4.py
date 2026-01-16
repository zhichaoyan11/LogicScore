"""Stage 4: citation-level recall/precision evaluation."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from prompt import (
    support_prompt_template,
    need_citation_prompt_template,
    relevant_prompt_template,
)

from .config import GPT_MODEL, query_llm


def cat_question_statement_context(question, statement, context):
    return (
        f"<question>\n{question.strip()}\n</question>\n\n"
        f"<statement>\n{statement.strip()}\n</statement>\n\n"
        f"<snippet>\n{context.strip()}\n</snippet>\n\n"
    )


def cat_qa_and_statement(question, answer, statement):
    return (
        f"<question>\n{question.strip()}\n</question>\n\n"
        f"<response>\n{answer.strip()}\n</response>\n\n"
        f"<sentence>\n{statement.strip()}\n</sentence>"
    )


def need_citation_to_score(s):
    l = re.findall(r"\[\[([ /a-zA-Z]+)\]\]", s)
    if l:
        v = l[0]
        if "yes" in v.lower():
            return 1
        return 0
    return None


def need_citation(question, answer, sentence):
    prompt = need_citation_prompt_template.format(
        cat_qa_and_statement(question, answer, sentence)
    )
    for t in range(5):
        msg = [{"role": "user", "content": prompt}]
        try:
            output = query_llm(
                msg,
                model=GPT_MODEL,
                temperature=0 if t == 0 else 1,
                max_new_tokens=10,
                stop="Analysis:",
                return_usage=True,
            )
            if isinstance(output, tuple):
                output, usage = output
                score = need_citation_to_score(output)
            else:
                score, usage = None, None
            if score is None:
                print("Unexcept need_citation output: ", output)
                if "Trigger" in str(output):
                    break
                continue
            break
        except Exception as e:  # pylint: disable=broad-except
            if "502" in str(e) or "Bad Gateway" in str(e) or "SERVICE_UNAVAILABLE" in str(e):
                wait_time = 2 ** t
                print(
                    f"⚠️  need_citation 收到 502 错误，等待 {wait_time} 秒后重试... (尝试 {t+1}/5)"
                )
                import time as _time

                _time.sleep(wait_time)
                if t < 4:
                    continue
                raise
            raise
    return score, output, usage


def support_level_to_score(s):
    l = re.findall(r"\[\[([ /a-zA-Z]+)\]\]", s)
    if l:
        v = l[0]
        if "fully" in v.lower():
            return 1
        if "partially" in v.lower():
            return 0.5
        return 0
    return None


def is_support(question, statement, context):
    if context == "":
        return 0, "No matched citation", None
    prompt = support_prompt_template.format(
        cat_question_statement_context(question, statement, context)
    )
    for t in range(5):
        msg = [{"role": "user", "content": prompt}]
        try:
            output = query_llm(
                msg,
                model=GPT_MODEL,
                temperature=0 if t == 0 else 1,
                max_new_tokens=10,
                stop="Analysis:",
                return_usage=True,
            )
            if isinstance(output, tuple):
                output, usage = output
                score = support_level_to_score(output)
            else:
                score, usage = None, None
            if score is None:
                print("Unexcept support output: ", output)
                if "Trigger" in str(output):
                    break
                continue
            break
        except Exception as e:  # pylint: disable=broad-except
            if "502" in str(e) or "Bad Gateway" in str(e) or "SERVICE_UNAVAILABLE" in str(e):
                wait_time = 2 ** t
                print(
                    f"⚠️  is_support 收到 502 错误，等待 {wait_time} 秒后重试... (尝试 {t+1}/5)"
                )
                import time as _time

                _time.sleep(wait_time)
                if t < 4:
                    continue
                raise
            raise
    return score, output, usage


def relevant_level_to_score(s):
    l = re.findall(r"\[\[([ /a-zA-Z]+)\]\]", s)
    if l:
        v = l[0]
        if "unrelevant" in v.lower():
            return 0
        return 1
    return None


def is_relevant(question, statement, citation):
    prompt = relevant_prompt_template.format(
        cat_question_statement_context(question, statement, citation)
    )
    for t in range(5):
        msg = [{"role": "user", "content": prompt}]
        try:
            output = query_llm(
                msg,
                model=GPT_MODEL,
                temperature=0 if t == 0 else 1,
                max_new_tokens=10,
                stop="Analysis:",
                return_usage=True,
            )
            if isinstance(output, tuple):
                output, usage = output
                score = relevant_level_to_score(output)
            else:
                score, usage = None, None
            if score is None:
                print("Unexcept relevant output: ", output)
                if "Trigger" in str(output):
                    break
                continue
            break
        except Exception as e:  # pylint: disable=broad-except
            if "502" in str(e) or "Bad Gateway" in str(e) or "SERVICE_UNAVAILABLE" in str(e):
                wait_time = 2 ** t
                print(
                    f"⚠️  is_relevant 收到 502 错误，等待 {wait_time} 秒后重试... (尝试 {t+1}/5)"
                )
                import time as _time

                _time.sleep(wait_time)
                if t < 4:
                    continue
                raise
            raise
    return score, output, usage


def _process_single_data_stage4(data):
    try:
        context = data["context"]
        statements = data["statements"]
        question = data["query"]
        answer = data["answer"]

        recall_scores = []
        recall_evaluation_state = []
        precision_scores = []
        precision_evaluation_state = []

        for state in statements:
            pairs = re.findall(r"\[(\d+)\]\s*,?\s*<[a-zA-Z]*(\d+)>", state)
            citation = ""
            if pairs:
                for doc_id, sent_id in pairs:
                    if int(doc_id) - 1 < len(context):
                        if int(sent_id) - 1 < len(context[int(doc_id) - 1][1]):
                            citation += context[int(doc_id) - 1][1][int(sent_id) - 1] + " "
                            precision_score, output, usage = is_relevant(
                                question,
                                state,
                                context[int(doc_id) - 1][1][int(sent_id) - 1],
                            )
                            if precision_score is not None:
                                precision_scores.append(precision_score)
                            else:
                                print(
                                    f"Warning: is_relevant returned None for statement: {state[:50]}..."
                                )
                                precision_scores.append(0.0)
                            precision_evaluation_state.append(
                                {
                                    "statement": state,
                                    "citation": context[int(doc_id) - 1][1][
                                        int(sent_id) - 1
                                    ],
                                    "relevant_output": output,
                                    "relevant_score": precision_score
                                    if precision_score is not None
                                    else 0.0,
                                }
                            )
            else:
                doc_num = re.findall(r"(?<=\[)\d+(?=\])", state)
                for num in doc_num:
                    if int(num) - 1 < len(context):
                        citation += " ".join(context[int(num) - 1][1]) + " "
                        precision_score, output, usage = is_relevant(
                            question, state, citation
                        )
                        if precision_score is not None:
                            precision_scores.append(precision_score)
                        else:
                            print(
                                f"Warning: is_relevant returned None for statement: {state[:50]}..."
                            )
                            precision_scores.append(0.0)
                        precision_evaluation_state.append(
                            {
                                "statement": state,
                                "citation": context[int(num) - 1][1],
                                "relevant_output": output,
                                "relevant_score": precision_score
                                if precision_score is not None
                                else 0.0,
                            }
                        )
            if citation:
                score, output, usage = is_support(question, state, citation)
                if score is not None:
                    recall_scores.append(score)
                else:
                    print(
                        f"Warning: is_support returned None for statement: {state[:50]}..."
                    )
                    recall_scores.append(0.0)
                recall_evaluation_state.append(
                    {
                        "statement": state,
                        "citation": citation,
                        "support_output": output,
                        "support_score": score if score is not None else 0.0,
                    }
                )
            if not citation:
                score, output, usage = need_citation(question, answer, state)
                if score is not None:
                    recall_scores.append(1 - score)
                else:
                    print(
                        f"Warning: need_citation returned None for statement: {state[:50]}..."
                    )
                    recall_scores.append(0.0)
                recall_evaluation_state.append(
                    {
                        "statement": state,
                        "citation": citation,
                        "support_output": output,
                        "support_score": (1 - score) if score is not None else 0.0,
                    }
                )

        recall_scores_valid = [s for s in recall_scores if s is not None]
        precision_scores_valid = [s for s in precision_scores if s is not None]
        recall_score = np.mean(recall_scores_valid) if recall_scores_valid else 0.0
        precision_score = (
            np.mean(precision_scores_valid) if precision_scores_valid else 0.0
        )
        if np.isnan(recall_score):
            print(
                f"Warning: recall_score is NaN, recall_scores_valid: {recall_scores_valid}"
            )
            recall_score = 0.0
        if np.isnan(precision_score):
            print(
                "Warning: precision_score is NaN, precision_scores_valid: "
                f"{precision_scores_valid}"
            )
            precision_score = 0.0

        data["recall_scores"] = float(recall_score)
        data["recall_evaluation_state"] = recall_evaluation_state
        data["precision_scores"] = float(precision_score)
        data["precision_evaluation_state"] = precision_evaluation_state

        return data, float(recall_score), float(precision_score)
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error processing data in stage 4: {e}")
        return data, 0.0, 0.0


def _process_single_data_stage4_wrapper(args_tuple):
    index, data = args_tuple
    processed_data, recall_score, precision_score = _process_single_data_stage4(data)
    return index, processed_data, recall_score, precision_score


def stage_4_processing(total_data, args=None, num_workers: int = 10):
    if not total_data:
        return []

    pending_data = [(i, data) for i, data in enumerate(total_data)]
    results = {}
    total_recall_score = 0.0
    total_precision_score = 0.0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_single_data_stage4_wrapper, args_tuple): args_tuple[0]
            for args_tuple in pending_data
        }
        with tqdm(total=len(pending_data), desc="Stage 4 Processing") as pbar:
            for future in as_completed(futures):
                try:
                    index, processed_data, recall_score, precision_score = future.result()
                    results[index] = processed_data
                    total_recall_score += recall_score
                    total_precision_score += precision_score
                    pbar.update(1)
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error in future result: {e}")
                    pbar.update(1)

    new_data = [results[i] for i in range(len(total_data))]
    avg_recall_score = total_recall_score / len(total_data) if total_data else 0.0
    avg_precision_score = (
        total_precision_score / len(total_data) if total_data else 0.0
    )
    if np.isnan(avg_recall_score):
        print(
            "Warning: avg_recall_score is NaN, total_recall_score: "
            f"{total_recall_score}, len(total_data): {len(total_data)}"
        )
        avg_recall_score = 0.0
    if np.isnan(avg_precision_score):
        print(
            "Warning: avg_precision_score is NaN, total_precision_score: "
            f"{total_precision_score}, len(total_data): {len(total_data)}"
        )
        avg_precision_score = 0.0
    if (avg_recall_score + avg_precision_score) > 0:
        avg_f1_score = (
            2 * avg_recall_score * avg_precision_score /
            (avg_recall_score + avg_precision_score)
        )
    else:
        avg_f1_score = 0.0
    if np.isnan(avg_f1_score):
        print(
            "Warning: avg_f1_score is NaN, avg_recall_score: "
            f"{avg_recall_score}, avg_precision_score: {avg_precision_score}"
        )
        avg_f1_score = 0.0

    print(
        f"recall_score: {avg_recall_score:.4f}, precision_score: {avg_precision_score:.4f}, f1_score: {avg_f1_score:.4f}"
    )
    return new_data

