"""LLM-based QA, reasoning and extraction helpers.

This module groups together the higher-level LLM calls used across stages:
question answering, reasoning generation, entity and triplet extraction, and
various small classification helpers.
"""

import re
from typing import Any, Dict, List, Tuple, Optional

from prompt import (
    citation_generate_prompt,
    named_entity_extract_prompt,
    extract_key_information_as_pattern,
    Step_decision_rpompt,
    triplet_extract,
    step_reasoning_prompt,
    answer_check_prompt,
    entity_check_prompt,
)

from .config import client, vllm_client, deepseek_client


def parse_question_answering_response(response: str) -> Tuple[str, str]:
    """Parse the reasoning + answer from an LLM response string."""
    response_lines = response.split("\n")
    answer = None
    for line in response_lines:
        if "Answer" in line:
            answer = line.split("Answer:")[1].strip().replace("*", "")
    reasoning = response
    if reasoning and answer:
        return reasoning, answer
    return None, None


def question_answering(question: str, documents: str, args) -> Tuple[str, str]:
    """Run the citation-generation style QA prompt with different backends."""
    prompts = citation_generate_prompt.format(question, documents)
    # prompts = citation_generate_prompt_alce.format(question=question, documents=documents)

    model_name = args.generate_model.lower()

    if "deepseek" in model_name or "qwen3" in model_name:
        backend = deepseek_client
    elif "llama" in model_name or "qwen" in model_name:
        backend = vllm_client
    else:
        backend = client

    for _ in range(5):
        response = backend.chat.completions.create(
            model=args.generate_model,
            messages=[{"role": "user", "content": prompts}],
            temperature=1.0,
            max_completion_tokens=8240,
        )
        reasoning, answer = parse_question_answering_response(
            response.choices[0].message.content
        )
        if reasoning and answer:
            return reasoning, answer
    return [], []


def extract_path(statements: List[str], question: str, args) -> str:
	"""Extract key information / pattern from reasoning statements.

	This uses the evaluation model specified in ``args.eval_model`` when
	available, otherwise it falls back to ``gpt-4o-mini``.
	"""
	reasoning_chain = "".join(f"Step {i+1}:{s}\n" for i, s in enumerate(statements))
	prompts = extract_key_information_as_pattern.format(
	    question=question, reasoning="".join(statements)
	)

	# Model is now configurable via args.eval_model (Stage 3 eval model).
	eval_model = getattr(args, "eval_model", "gpt-4o-mini")

	response = client.chat.completions.create(
	    model=eval_model,
	    messages=[{"role": "user", "content": prompts}],
	    temperature=1.0,
	    max_completion_tokens=4096,
	    response_format={"type": "json_object"},
	)
	return response.choices[0].message.content


def reasoning_generation(question: str, answer: List[str], args) -> str:
	"""Generate step-by-step reasoning for a question / answer pair.

	The reasoning model is controlled by ``args.reasoning_model``; if that
	attribute is missing, we fall back to ``o4-mini-2025-04-16``.
	"""
	prompts = step_reasoning_prompt.format(question=question, answer=answer)
	reasoning_model = getattr(args, "reasoning_model", "o4-mini-2025-04-16")

	response = client.chat.completions.create(
	    model=reasoning_model,
	    messages=[{"role": "user", "content": prompts}],
	    temperature=1.0,
	    max_completion_tokens=4096,
	)
	return response.choices[0].message.content


def decsided_by_LLM(
	question: str,
	gold_answer: str,
	step: str,
	model: Optional[str] = None,
) -> bool:
	"""Use an LLM to decide whether a reasoning step is necessary.

	The ``model`` argument lets callers control which evaluation model is used.
	If omitted, it defaults to ``gpt-4o-mini``.
	"""
	prompt = Step_decision_rpompt.format(
	    question=question, gold_answer=gold_answer, reasoning_step=step
	)
	response = call_general_format_llm(prompt, model=model)
	return "<TRUE>" in response


def call_json_format_llm(prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Call an LLM that returns JSON-formatted output.

    The ``model`` parameter allows callers to override the default evaluation
    model (``gpt-4o-mini``).
    """

    if model is None:
        model = "gpt-4o-mini"

    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=4096,
                response_format={"type": "json_object"},
            ).choices[0].message.content
            return eval(  # noqa: S307 - mirrors original behaviour
                response.replace("true", "True")
                .replace("false", "False")
                .replace("null", "None")
            )
        except Exception:  # pylint: disable=broad-except
            continue


def call_general_format_llm(prompt: str, model: Optional[str] = None) -> str:
    if model is None:
        model = "gpt-4o-mini"
    response = client.chat.completions.create(
	    model=model,
	    messages=[{"role": "user", "content": prompt}],
	    temperature=0.7,
	    max_completion_tokens=4096,
	).choices[0].message.content
    return response
 



def check_answer_by_llm(
	question: str,
	answer1: str,
	answer2: str,
	model: Optional[str] = None,
) -> bool:
	"""Use an LLM to check whether two answers are equivalent."""
	prompt = answer_check_prompt.format(
	    question=question, answer1=answer1, answer2=answer2
	)
	response = call_general_format_llm(prompt, model=model)
	return "<TRUE>" in response


def check_entity_by_llm(
	entity: str,
	triple,
	model: Optional[str] = None,
) -> bool:
	"""Use an LLM to check whether an entity appears in a triple/sentence."""
	prompt = entity_check_prompt.format(entity=entity, sentence=" ".join(triple))
	response = call_general_format_llm(prompt, model=model)
	return "<TRUE>" in response


def extract_triplets(step: str, args=None) -> List[List[str]]:
    """Extract (subject, predicate, object) triplets from a reasoning step.

    The extraction model is driven by ``args.eval_model`` when supplied,
    otherwise it defaults to ``gpt-4o-mini``.
    """
    prompt = triplet_extract.format(step=step)
    if args is not None:
        eval_model = getattr(args, "eval_model", "gpt-4o-mini")
    else:
        eval_model = "gpt-4o-mini"
    while True:
        try:
            response = call_json_format_llm(prompt, model=eval_model)
            if "triples" in response.keys():
                response = response["triples"]
            if isinstance(response, dict):
                return [[response["subject"], response["predicate"], response["object"]]]
            triplets = [
                [t["subject"].strip(), t["predicate"].strip(), t["object"].strip()]
                for t in response
            ]
            return triplets
        except Exception:  # pylint: disable=broad-except
            continue


def extract_entity_from_question(question: str, args=None):
    """Extract named entities from the question.

    Uses ``args.eval_model`` when provided; otherwise defaults to
    ``gpt-4o-mini``.
    """
    named_prompt = named_entity_extract_prompt.format(question=question)
    if args is not None:
        eval_model = getattr(args, "eval_model", "gpt-4o-mini")
    else:
        eval_model = "gpt-4o-mini"
    response = call_json_format_llm(named_prompt, model=eval_model)
    if "entities" in response.keys():
        entities = response["entities"]
        if entities:
            return entities, 1
        return [], 2
    return [], 2

