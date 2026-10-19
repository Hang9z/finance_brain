import sys
import os
import asyncio
import json
import time
from pathlib import Path
import requests
from openai import api_key

# 确保在直接独立运行测试时能正确找到项目根路径模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.config.bailian_mcp_config import web_search_config
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_done_task, add_running_task

@node_log("node_web_search_mcp")
def node_web_search_mcp(state: QueryGraphState):
    """
    节点功能：调用外部搜索引擎补充信息
    注意：并行节点只返回增量字段，不直接 return state
    """
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)

    # 1. 校验并获取检索问句
    rewritten_query = step_1_validate_and_get_data(state)
    if not rewritten_query:
        logger.warning(f"{[session_id]} 改写问句为空, 跳过联网检索")
        add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
        return {"web_search_docs": []}

    # 2. 调用联网搜索
    provider = web_search_config.provider
    logger.info(f"[{session_id}] 启动联网搜索, 引擎提供方: {provider}")

    web_search_docs = []
    if provider == "deepseek":
        web_search_docs = step_2_search_deepseek(rewritten_query, count=5)
    elif provider == "brave":
        web_search_docs = step_2_search_brave(rewritten_query, count=5)
    else:
        # 默认优先 deepseek，其次 brave 兜底
        web_search_docs = step_2_search_deepseek(rewritten_query, count=5) or step_2_search_brave(rewritten_query, count=5)

    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
    return {"web_search_docs": web_search_docs}

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState) -> str:
    """
    提取 rewritten_query, 回退使用 original_query
    :param state:
    :return:
    """
    rewritten_query = state.get("rewritten_query", "").strip()
    if not rewritten_query:
        rewritten_query = state.get("original_query", "").strip()
    return rewritten_query

@step_log("step_2_search_deepseek")
def step_2_search_deepseek(query: str, count: int = 5) -> list[dict]:
    """
    调用 DeepSeek Search API 进行原生高质量金融与网页检索
    :param query: 关键词
    :param count: 期望召回网页数量
    :return: [{"title": ..., "url": ..., "snippet": ...}]
    """
    api_key = web_search_config.deepseek_api_key
    base_url = web_search_config.deepseek_base_url

    if not api_key:
        logger.warning("[step_2_search_deepseek] 未配置 DEEPSEEK_API_KEY，跳过 DeepSeek 联网检索")
        return []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "count": min(count, 50),
        "need_content": False
    }

    # 3 次网络重试
    for attempt in range(1, 4):
        try:
            res = requests.post(base_url, headers=headers, json=payload, timeout=25)
            if res.status_code != 200:
                logger.warning(f"[step_2_search_deepseek] 第 {attempt} 次请求返回状态码 {res.status_code}: {res.text[:200]}")
                time.sleep(attempt * 0.8)
                continue

            data = res.json()
            raw_results = data.get("results", [])
            clean_docs: list[dict] = []

            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                title = (item.get("title") or "").strip()
                url = (item.get("url") or "").strip()
                # 优先使用专门面向模型理解的 summary 字段，兜底使用 snippet
                snippet_text = (item.get("summary") or item.get("snippet") or "").strip()

                if snippet_text:
                    clean_docs.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet_text[:800],
                        "rerank_score": item.get("rerank_score", 0.0)
                    })

            logger.info(f"[step_2_search_deepseek] 成功从 DeepSeek 检索到 {len(clean_docs)} 条优质结果")
            return clean_docs[:count]

        except Exception as e:
            logger.warning(f"[step_2_search_deepseek] 第 {attempt} 次请求异常: {e}")
            time.sleep(attempt * 0.8)

    logger.error("[step_2_search_deepseek] DeepSeek Search 请求最终失败，降级返回空列表")
    return []

@step_log("step_2_search_brave")
def step_2_search_brave(query: str, count: int = 5) -> list[dict]:
    """
    调用 Brave Search LLM Context API 进行外网语义检索
    :param query: 关键词
    :param count: 期望召回网页数量
    :return: [{"title": ..., "url": ..., "snippet": ...}]
    """
    api_key = web_search_config.brave_api_key
    base_url = web_search_config.brave_base_url

    if not api_key:
        logger.warning("[step_2_search_brave] 未配置 BRAVE_SEARCH_API_KEY, 跳过联网检索")
        return []

    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json"
    }
    params = {
        "q": query,
        "count": count
    }

    # 3 次重试防御网络抖动
    for attempt in range(1, 4):
        try:
            res = requests.get(base_url, headers=headers, params=params, timeout=20)
            res.raise_for_status()
            data = res.json()

            # 解析 grounding.generic 列表
            generic_items = data.get("grounding", {}).get("generic", [])
            clean_docs: list[dict] = []

            for item in generic_items:
                if not isinstance(item, dict):
                    continue
                title = (item.get("title") or "").strip()
                url = (item.get("url") or "").strip()
                snippets_list = item.get("snippets", [])

                # 提取前 3-4 个高相关摘要段落并拼接
                snippet_text = ""
                if isinstance(snippets_list, list) and snippets_list:
                    selected_snippets = [s.strip() for s in snippets_list[:4] if isinstance(s, str) and s.strip()]
                    snippet_text = "\n".join(selected_snippets)
                elif item.get("snippet"):
                    snippet_text = str(item.get("snippet")).strip()

                if snippet_text:
                    clean_docs.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet_text[:600]
                    })

            logger.info(f"[step_2_search_brave] 成功检索到 {len(clean_docs)} 条结果")
            return clean_docs[:count]

        except Exception as e:
            logger.warning(f"[step_2_search_brave] 第 {attempt} 次请求失败: {e}")
            time.sleep(attempt * 0.8)

    logger.error("[step_2_search_brave] Brave Search 请求最终失败, 降级返回空列表")
    return []


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("\n========== 开始测试 node_web_search_mcp 节点 ==========")
    test_state: QueryGraphState = {
        "session_id": "test_web_search_session_001",
        "is_stream": False,
        "original_query": "HAK 180 烫金机如何设置局部烫金",
        "rewritten_query": "HAK 180 烫金机在纸张上设置顶部 50-170mm 局部转印的操作方法",
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "web_search_docs": [],
        "rrf_chunks": [],
        "reranked_docs": [],
        "prompt": "",
        "answer": "",
        "item_names": ["HAK 180 烫金机"],
        "history": [],
        "image_urls": [],
    }
    res = node_web_search_mcp(test_state)
    print("\n========== 节点运行结果 ==========")
    print(f"返回字段: {list(res.keys())}")
    docs = res.get("web_search_docs", [])
    print(f"召回网页数量: {len(docs)}")
    for idx, doc in enumerate(docs, 1):
        print(f"\n[{idx}] 标题: {doc.get('title')}")
        print(f"    链接: {doc.get('url')}")
        snippet_preview = doc.get('snippet', '').replace('\n', ' ')[:120]
        print(f"    摘要: {snippet_preview}...")