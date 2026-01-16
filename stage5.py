"""Stage 5: FActScore-style atomic fact evaluation."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from factscore.atomic_facts import AtomicFactGenerator
from factscore.openai_lm import OpenAIModel

from .config import api_key, api_url


def verify_atom(fact, context, model):
    prompt = f"""
    Answer the question based on the context below. 
    Context:
    {context}
    
    Statement: {fact}
    
    Is the statement supported by the context? Answer only  True or False?\nAnswer:
    """
    is_supported = model.generate(prompt)
    return "True" in is_supported


def _process_single_data_stage5(index, item, evaluator_model, AFG):
    try:
        generation = "".join(item["statements"])
        context = item["context"]

        atomic_facts = AFG.run(generation)
        if not atomic_facts:
            print(f"No facts found for {item.get('query', 'unknown')}")
            item["fact_score"] = 0
            return index, item, 0

        supported_count = 0
        for fact in atomic_facts[0][0][1]:
            if verify_atom(fact, context, evaluator_model):
                supported_count += 1

        score = (
            supported_count / len(atomic_facts[0][0][1])
            if len(atomic_facts[0][0][1]) > 0
            else 0
        )
        item["fact_score"] = score
        return index, item, score
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error processing item at index {index}: {e}")
        item["fact_score"] = 0
        return index, item, 0


def _process_stage5_wrapper(args_tuple):
    index, item, evaluator_model, AFG = args_tuple
    return _process_single_data_stage5(index, item, evaluator_model, AFG)


def stage_5_processing(args, total_data, num_workers: int = 4):
    documents_file_path = args.file_path
    with open(documents_file_path, "r", encoding="utf-8") as f:
        documents_file = f.readlines()

    if "musique" in args.file_path:
        for data in total_data:
            question = data["query"]
            for d in documents_file:
                documents = json.loads(d)
                if question == documents["question"]:
                    gold_context = "\n".join(
                        [
                            p["paragraph_text"]
                            for p in documents["paragraphs"]
                            if p["is_supporting"]
                        ]
                    )
                    data["context"] = gold_context
                    break

    evaluator_model = OpenAIModel(
        model_name="gpt-4o-2024-08-06",
        api_key=api_key,
        base_url=api_url,
        cache_file=f"{args.save_path}/{args.generate_model}_stage_5_cache.json",
    )
    AFG = AtomicFactGenerator(
        demon_dir="demos",
        api_key=api_key,
        base_url=api_url,
        gpt3_cache_file=f"{args.save_path}/{args.generate_model}_stage_5_cache.json",
    )

    pending_data = [
        (i, item, evaluator_model, AFG) for i, item in enumerate(total_data)
    ]
    results = {}
    final_scores = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_stage5_wrapper, args_tuple): args_tuple[0]
            for args_tuple in pending_data
        }
        with tqdm(total=len(pending_data), desc="Stage 5 Processing") as pbar:
            for future in as_completed(futures):
                try:
                    index, processed_item, score = future.result()
                    results[index] = processed_item
                    final_scores.append(score)
                    pbar.update(1)
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error in future result: {e}")
                    pbar.update(1)

    new_data = [results[i] for i in range(len(total_data))]

    if final_scores:
        avg_score = np.mean(final_scores)
        print(f"Average FActScore (Faithfulness): {avg_score:.4f}")

    return new_data

