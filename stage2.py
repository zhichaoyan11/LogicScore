"""Stage 2: extract logical answers from reasoning statements."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .qa import extract_path


def _process_single_data_stage2(data, args):
    """处理单条数据 - Stage 2。

    提取逻辑答案并写入 `logic_answer`、`gold_answer`、`predict_answer` 字段。
    """
    try:
        statements = data["statements"]
        question = data["query"]

        extraced_key_path = extract_path(statements, question, args)
        data["logic_answer"] = extraced_key_path
        data["gold_answer"] = data["answer"]
        if "prediction" in data.keys():
            data["predict_answer"] = data["prediction"]
        return data, True
    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ 处理数据时出错: {e}")
        return data, False


def _process_stage2_wrapper(args_tuple):
    """Thread-pool wrapper around `_process_single_data_stage2`."""
    idx, data, args = args_tuple
    processed_data, success = _process_single_data_stage2(data, args)
    return idx, processed_data, success


def stage_2_processing(total_data, args=None, num_workers: int = 10):
    """第二阶段处理：提取逻辑答案（支持并发执行）。"""
    if not total_data:
        return []

    pending_data = [(i, data, args) for i, data in enumerate(total_data)]
    results = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_stage2_wrapper, args_tuple): args_tuple[0]
            for args_tuple in pending_data
        }

        with tqdm(total=len(pending_data), desc="Stage 2 Processing") as pbar:
            for future in as_completed(futures):
                try:
                    idx, processed_data, success = future.result()
                    if success:
                        results[idx] = processed_data
                    else:
                        results[idx] = processed_data
                    pbar.update(1)
                except Exception as e:  # pylint: disable=broad-except
                    print(f"❌ 处理任务时出错: {e}")
                    pbar.update(1)

    new_data = [results[i] for i in range(len(total_data))]
    return new_data

