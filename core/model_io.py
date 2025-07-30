"""
封装与本地LLM的调用,以及知识库检索(RAG检索、embedding搜索等)

call_language_model_stream(): 负责最终LLM问答,流式返回答案给UI

rag_search(): 用于检索补充知识（如手册/操作指导/FAQ片段)

由controller.py调用

可以灵活组合prompt/上下文/背景资料等
"""

from __future__ import annotations

import json
import queue
from typing import Dict, List, Generator

import requests

from globals import GLOBAL  
from rag.RAG_background import answer_from_rag_with_lang  

def _stream_generate(payload: Dict) -> Generator[str, None, None]:

    url: str = "http://127.0.0.1:11434/api/generate"
    with requests.post(url, json=payload, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token: str = data.get("response", "")
            yield token
            if data.get("done"):
                break


def call_language_model(
    full_prompt: str,
    model_name: str,
    options: Dict,
) -> str:
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,
        "options": options,
    }
    return "".join(_stream_generate(payload))


def call_language_model_stream(
    full_prompt: str,
    model_name: str,
    options: Dict,
    beginner_tip: str | None,
    q: queue.Queue,
) -> None:
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": True,
        "options": options,
    }

    tip_added = False
    for token in _stream_generate(payload):
        if beginner_tip and not tip_added:
            token += beginner_tip
            tip_added = True
        q.put(token)

    q.put(None)


def rag_search(query: str, lang: str) -> str:
    try:
        return answer_from_rag_with_lang(query, lang) or ""
    except Exception as exc:  
        print("[WARN] RAG search failed:", exc)
        return ""
