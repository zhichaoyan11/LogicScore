"""Stage 1: question answering and reasoning extraction.

This stage consumes the pre-processed dataset and runs the QA model to
produce `statements`, `predict_answer` and `gold_answer` fields.
"""

import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm

from .qa import question_answering


def process_single_item(idx, data, args):
    """处理单条数据的函数，用于多线程处理。"""
    try:
        data_id = data.get("_id", idx)
        question = data["query"]
        documents = data["context"]
        gold_answer = data["answer"]

        # 格式化文档
        format_document = ""
        for i, doc in enumerate(documents):
            format_document += (
                f"Document[{i+1}] (Title: {doc[0]}): " + "".join(doc[1])
            )

        passage, answer = question_answering(question, format_document, args)
        if not passage:
            return None

        data["statements"] = passage
        data["predict_answer"] = answer
        data["gold_answer"] = gold_answer

        return {
            "idx": idx,
            "data_id": data_id,
            "data": data,
            "success": True,
            "error": None,
        }
    except Exception as e:  # pylint: disable=broad-except
        return {
            "idx": idx,
            "data_id": data.get("_id", idx),
            "data": None,
            "success": False,
            "error": str(e),
        }


def stage_1_processing(total_data, args):
    """Run Stage 1 with checkpointing and multi-threading.

    The logic mirrors the original implementation in logical_attribution.py
    but is isolated into its own module.
    """
    new_data = []
    output_file = f"{args.save_path}/{args.generate_model}_stage_1_test_prompt.json"
    checkpoint_file = f"{args.save_path}/{args.generate_model}_stage_1_checkpoint.json"

    start_idx = 0
    processed_ids = set()

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            if not isinstance(checkpoint_data, dict):
                raise ValueError("断点文件格式错误: 期望字典")

            start_idx = checkpoint_data.get("last_processed_idx", 0)
            processed_ids_list = checkpoint_data.get("processed_ids", [])

            if not isinstance(processed_ids_list, list):
                raise ValueError("processed_ids 格式错误: 期望列表")

            processed_ids = set(processed_ids_list)
            print(f"✅ 找到断点文件，从第 {start_idx} 条数据继续处理")
            print(f"   已处理数据数: {len(processed_ids)}")
        except Exception as e:  # pylint: disable=broad-except
            print(f"⚠️ 读取断点文件失败: {e}，将从头开始处理")
            start_idx = 0
            processed_ids = set()

    if start_idx == 0:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("[\n")
    else:
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
            if content.rstrip().endswith("]"):
                content = content.rstrip()[:-1]
                if not content.rstrip().endswith(","):
                    content = content.rstrip() + ","
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(content + "\n")

    num_workers = getattr(args, "num_workers", 4)
    file_lock = Lock()
    processed_count = 0

    items_to_process = []
    for idx, data in enumerate(total_data):
        if idx < start_idx:
            continue
        data_id = data.get("_id", idx)
        if data_id in processed_ids:
            continue
        items_to_process.append((idx, data))

    print(f"📊 开始多线程处理，共 {len(items_to_process)} 条数据，使用 {num_workers} 个线程")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_item, idx, data, args): (idx, data)
            for idx, data in items_to_process
        }

        with tqdm(total=len(items_to_process), desc="处理进度") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()

                    if result is None:
                        pbar.update(1)
                        continue

                    if result["success"]:
                        with file_lock:
                            data_id = result["data_id"]
                            processed_ids.add(data_id)
                            new_data.append(result["data"])
                            processed_count += 1

                            with open(output_file, "a", encoding="utf-8") as f:
                                json.dump(result["data"], f, ensure_ascii=False)
                                f.write(",\n")

                            if processed_count % 10 == 0:
                                checkpoint_data = {
                                    "last_processed_idx": result["idx"] + 1,
                                    "processed_ids": list(processed_ids),
                                    "timestamp": str(datetime.datetime.now()),
                                }
                                with open(checkpoint_file, "w", encoding="utf-8") as f:
                                    json.dump(checkpoint_data, f, indent=4, ensure_ascii=False)
                    else:
                        with file_lock:
                            checkpoint_data = {
                                "last_processed_idx": result["idx"],
                                "processed_ids": list(processed_ids),
                                "timestamp": str(datetime.datetime.now()),
                                "error": result["error"],
                            }
                            with open(checkpoint_file, "w", encoding="utf-8") as f:
                                json.dump(checkpoint_data, f, indent=4, ensure_ascii=False)

                    pbar.update(1)
                except Exception as e:  # pylint: disable=broad-except
                    print(f"❌ 线程执行出错: {e}")
                    pbar.update(1)

    # Close JSON array
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()
    if content.rstrip().endswith(","):
        content = content.rstrip()[:-1]
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n]")

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("✅ 处理完成，断点文件已删除")

    return new_data

