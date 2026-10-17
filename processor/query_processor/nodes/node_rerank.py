import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

# 确保在直接独立运行测试时能正确找到项目根路径模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState
from utils.lm.lm_utils import get_llm_client
from utils.lm.reranker_utils import compute_token_number, get_reranker_model
from utils.load_prompt import load_prompt
from utils.task_utils import add_done_task, add_running_task
# -----------------------------
# Rerank / TopK 常量配置
# -----------------------------
RERANK_MAX_TOPK: int = 10               # 动态截取硬上限
RERANK_MIN_TOPK: int = 1                # 至少保留条数
RERANK_GAP_RATIO: float = 0.25          # 相对降幅断崖阈值（25%）
RERANK_GAP_ABS: float = 0.5             # 绝对分值断崖阈值（0.5 分）
RERANK_MAX_INPUT_TOKENS: int = 512      # 重排模型的最大上下文窗口
RERANK_SUMMARY_CHAR_RATIO: float = 1.3  # Token 到中文字符换算系数
RERANK_MIN_SUMMARY_CHARS: int = 50      # 文本压缩最低字数下限

@node_log("node_rerank")
def node_rerank(state: QueryGraphState) -> dict:
    """
    节点功能：使用 Cross-Encoder 模型对本地 RRF 切片与外网检索网页进行统一打分，
    并应用动态断崖规则截取 Top-K 最优参考文档，存入 reranked_docs。
    """
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)

    # 1. 获取并校验参数
    rrf_chunks, web_search_docs, rewritten_query = step_1_validate_and_get_data(state)

    # 若两路候选全为空，安全返回空列表
    if not rrf_chunks and not web_search_docs:
        logger.warning(f"[{session_id}] 本地与外网候选文档全为空，Rerank 提前返回空列表")
        add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
        return {"reranked_docs": []}

    # 2. 规范化两路候选格式, 合并为统一待排列表
    merged_list = step_2_merge_rrf_and_web(rrf_chunks, web_search_docs)

    # 3. 构造问答对
    question_answer_pair = step_3_create_question_answer_pair(merged_list, rewritten_query)

    # 4. 调用 Reranker 模型计算精准语义相似度并降序排列
    step_4_list_score_and_rank(merged_list, question_answer_pair)

    # 5. 动态断崖阈值截取 Top-K
    reranked_docs = step_5_dynamic_topk(merged_list)

    logger.info(f"[{session_id}] Rerank 重排完成，候选总数: {len(merged_list)} -> 截取输出: {len(reranked_docs)}")

    state["reranked_docs"] = reranked_docs
    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
    return {"reranked_docs": reranked_docs}

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState) -> tuple[list, list, str]:
    """提取两路候选数据与检索问句，具备安全容错能力"""
    rrf_chunks = state.get("rrf_chunks") or []
    web_search_docs = state.get("web_search_docs") or []
    rewritten_query = state.get("rewritten_query", "").strip() or state.get("original_query", "").strip()
    return rrf_chunks, web_search_docs, rewritten_query

@step_log("step_2_merge_rrf_and_web")
def step_2_merge_rrf_and_web(rrf_chunks: list, web_search_docs: list) -> List[Dict]:
    """

    :param rrf_chunks:
    :param web_search_docs:
    :return:
    """
    merged_list: list[dict] = []

    # 1. 校对 RRF 格式
    for chunk in rrf_chunks:
        text = chunk.get("content")
        merged_list.append({
            "chunk_id": str(chunk.get("chunk_id", "")),
            "title": chunk.get("title", ""),
            "text": text,
            "score": 0.0,
            "type": chunk.get("type", "milvus"),
            "url": chunk.get("url", "")
        })

    # 2. 校对网页数据格式
    for doc in web_search_docs:
        snippet = (doc.get("snippet") or doc.get("text") or "").strip()
        if not snippet:
            continue
        merged_list.append({
            "chunk_id": "",
            "title": doc.get("title", ""),
            "text": snippet,
            "score": 0.0,
            "type": "web",
            "url": doc.get("url", "")
        })

    logger.info(f"[step_2_merge_rrf_and_web] 本地切片={len(rrf_chunks)}, 外网网页={len(web_search_docs)} -> 合并候选数={len(merged_list)}")
    return merged_list

@step_log("step_3_create_question_answer_pair")
def step_3_create_question_answer_pair(merged_list: list[dict], rewritten_query: str) -> List[List[str]]:
    """
    构造 [问题, 候选文本] 重排对, 为了避免文本过长无法送入模型, 对问题和文本进行压缩
    :param merged_list:
    :param rewritten_query:
    :return:
    """
    question_answer_pair: List[List[str]] = []
    question_token_len: int = compute_token_number(rewritten_query)

    for item in merged_list:
        answer = item.get("text", "")
        token_len = compute_token_number(answer)

        # 判断是否超出模型上下文窗口长度
        available_token = RERANK_MAX_INPUT_TOKENS - 4- question_token_len
        if token_len > available_token and available_token > 0:
            char_length = max(
                int(available_token / RERANK_SUMMARY_CHAR_RATIO),
                RERANK_MIN_SUMMARY_CHARS
            )
            # 调用模型压缩文本
            try:
                llm_model = get_llm_client()
                prompt_content = load_prompt(
                    "rerank_text_refine",
                    question=rewritten_query,
                    answer=answer,
                    limit=char_length
                )
                message = HumanMessage(content=prompt_content)
                chain = llm_model | StrOutputParser()
                compressed_answer = chain.invoke([message]).strip()
                if compressed_answer:
                    logger.debug(f"[step_3_create_question_answer_pair] 文本超长触发压缩: {len(answer)}字 -> {len(compressed_answer)}字")
                    answer = compressed_answer
            except Exception as e:
                logger.warning(f"[step_3_create_question_answer_pair] 大模型压缩失败, 回退物理截断: {e}")
                answer = answer[:char_length]

        question_answer_pair.append([rewritten_query, answer])

    return question_answer_pair


@step_log("step_4_list_score_and_rank")
def step_4_list_score_and_rank(merged_list: list[dict], question_answer_pair: list[list[str]]):
    """
    调用 Reranker 模型进行打分, 并按得分从高到低就地排序
    :param merged_list:
    :param question_answer_pair:
    :return:
    """
    if not merged_list or not question_answer_pair:
        return

    reranker_model = get_reranker_model()
    score_list = reranker_model.compute_score(question_answer_pair, normalize=True)

    # 分数写回字典
    for score, item in zip(score_list, merged_list):
        item["score"] = float(score)

    # 按分数降序排列
    merged_list.sort(key=lambda c: c.get("score", 0.0), reverse=True)

@step_log("step_5_dynamic_topk")
def step_5_dynamic_topk(merged_list: list[dict]) -> list[dict]:
    """
    动态断崖截取: 当相邻 chunk 出现断崖时截断
    :param merged_list:
    :return:
    """
    if not merged_list:
        return []

    max_topk = min(RERANK_MAX_TOPK, len(merged_list))
    min_topk = min(RERANK_MIN_TOPK, max_topk)
    top_k = max_topk

    if max_topk > min_topk:
        for index in range(min_topk - 1, max_topk - 1):
            current = merged_list[index]
            nxt = merged_list[index + 1]

            curr_score = float(current.get("score") or 0.0)
            next_score = float(nxt.get("score") or 0.0)
            score_diff = curr_score - next_score
            # 防御除以零
            gap_ratio = (score_diff / curr_score) if curr_score > 1e-6 else 0.0

            # 触发断崖: 绝对落差 > 0.5 或 相对降低幅度 > 25%
            if score_diff > RERANK_GAP_ABS or gap_ratio > RERANK_GAP_RATIO:
                top_k = index + 1
                logger.info(f"[step_5_dynamic_topk] 索引 {index} 处检测到断崖: 当前得分={curr_score:.4f}, 下一得分={next_score:.4f}, 截取前 {top_k} 条")
                break

    reranked_docs = merged_list[:top_k]
    return reranked_docs


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("\n========== 开始测试 node_rerank 节点 ==========")
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种多路召回倒排排名融合算法", "title": "算法介绍", "type": "milvus"},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "type": "milvus"},
        {"chunk_id": "local_3", "content": "路漫漫其修远兮，吾将上下而求索", "title": "励志诗句", "type": "milvus"},
        {"chunk_id": "local_4", "content": "尚硅谷-让天下没有难学的技术", "title": "广告推广", "type": "milvus"},
    ]
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "https://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段对粗排结果精排"},
    ]
    test_state: QueryGraphState = {
        "session_id": "test_rerank_session_001",
        "is_stream": False,
        "original_query": "什么是RRF和Rerank？",
        "rewritten_query": "什么是RRF和Rerank？二者在RAG中的作用是什么？",
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "web_search_docs": mock_web_docs,
        "rrf_chunks": mock_rrf_chunks,
        "reranked_docs": [],
        "prompt": "",
        "answer": "",
        "item_names": [],
        "history": [],
        "image_urls": [],
    }
    res = node_rerank(test_state)
    print("\n========== 节点运行结果 ==========")
    final_docs = res.get("reranked_docs", [])
    print(f"输入候选: 本地={len(mock_rrf_chunks)}条, 外网={len(mock_web_docs)}条")
    print(f"精排截取输出: {len(final_docs)} 条")
    for idx, d in enumerate(final_docs, 1):
        print(f"\n[{idx}] 来源: {d.get('type')} | 得分: {d.get('score'):.6f} | 标题: {d.get('title')}")
        print(f"    内容: {d.get('text')[:100]}...")

