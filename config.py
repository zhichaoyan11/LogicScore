"""Configuration and shared LLM clients for the logical attribution pipeline.

This module centralizes all API endpoints, keys, and shared OpenAI client
instances, plus low-level utilities such as `query_llm` and `call_nli_api`.
"""

import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI




# Standard chat-completions HTTP endpoint
req_api_url = ""
req_api_url_deepseek = ""

# Default OpenAI-compatible HTTP endpoint
api_url = ""
api_key = ""



# Default GPT model used in evaluation prompts
GPT_MODEL = "gpt-4o-2024-05-13"

# Main client used by most calls in this project
client = OpenAI(base_url=api_url, api_key=api_key)


def query_llm(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    stop: Optional[str] = None,
    return_usage: bool = False,
):
    """Low-level helper that calls the HTTP chat-completions endpoint with
    built-in retry logic and basic error handling.

    This is extracted from the original logical_attribution.py so it can be
    reused across modules.
    """
    tries = 0
    while tries < 5:
        tries += 1
        try:
            if "claude" not in model:
                headers = {"Authorization": f"Bearer {api_key}"}
            else:
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                }

            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
            }
            # Different key name for Anthropic-style APIs
            if "claude" not in model:
                payload["stop"] = stop
            else:
                payload["stop_sequences"] = stop

            resp = requests.post(req_api_url, json=payload, headers=headers, timeout=600)

            # Retry on transient 502s
            if resp.status_code == 502:
                wait_time = 2 ** (tries - 1)
                print(f"⚠️  收到 502 错误，等待 {wait_time} 秒后重试... (尝试 {tries}/5)")
                time.sleep(wait_time)
                continue

            if resp.status_code != 200:
                raise Exception(resp.text)
            data = resp.json()
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:  # pylint: disable=broad-except
            msg = str(e)
            if "maximum context length" in msg:
                raise
            if "triggering" in msg:
                return "Trigger OpenAI's content management policy."
            if "502" in msg or "Bad Gateway" in msg or "SERVICE_UNAVAILABLE" in msg:
                wait_time = 2 ** (tries - 1)
                print(f"⚠️  收到 502 错误: {msg[:100]}，等待 {wait_time} 秒后重试... (尝试 {tries}/5)")
                time.sleep(wait_time)
                continue
            print(f"Error Occurs: \"{msg}\"        Retry ...")
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return None

    try:
        choice = data["choices"][0]["message"]
        if "content" not in choice and "content_filter_results" in data["choices"][0]:
            choice["content"] = "Trigger OpenAI's content management policy."
        if return_usage:
            return choice["content"], data.get("usage")
        return choice["content"]
    except Exception:  # pylint: disable=broad-except
        return None
