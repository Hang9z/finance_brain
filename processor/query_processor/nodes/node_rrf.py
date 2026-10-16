import time
import sys
import copy
from pathlib import Path
from typing import List, Dict, Tuple

# 确保在直接独立运行测试时能正确找到项目根路径模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task
from common.logging.logger import logger, node_log, step_log


@node_log("node_rrf")
def node_rrf(state: QueryGraphState) -> dict:
    """
    节点功能：Reciprocal Rank Fusion (倒排排名加权融合与去重)
    将两路向量召回切片（常规向量检索 + HyDE 假设性问答检索）按 RRF 算法进行融合加权排序，
    输出 Top-N 个最优质切片至 rrf_chunks，供下游 Rerank 精排使用。
    """
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)

    # 1. 获取并校验数据
    embedding_chunks, hyde_embedding_chunks = step_1_validate_and_get_data(state)

    # 若两路数据全为空，防御性返回空列表，避免下游空转或主图崩溃
    if not embedding_chunks and not hyde_embedding_chunks:
        logger.warning(f"[{session_id}] 本地两路召回切片均为空，RRF 输出空列表")
        add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
        return {"rrf_chunks": []}

    # 2. 组装待融合数据源列表: [(), ()]
    data_list: list[tuple[float, list]] = [(0.5, embedding_chunks), (0.5, hyde_embedding_chunks)]

    # 3. 执行 RRF 融合排序 (k = 60, 截取 Top 6)
    rrf_chunks = step_2_use_rrf_rank(data_list, k=60, top_k=6)

    logger.info(
        f"[{session_id}] RRF 融合完成，输入切片数: 常规={len(embedding_chunks)}, HyDE={len(hyde_embedding_chunks)} -> 输出={len(rrf_chunks)}")

    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
    return {"rrf_chunks": rrf_chunks}

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState) -> Tuple[List, List]:
    """
    从全局状态中安全获取两路切片, 缺失时回退为空列表
    """
    # 1. 获取普通向量检索的 chunk 和假设性搜索的 chunk
    embedding_chunks = state.get("embedding_chunks") or []
    hyde_embedding_chunks = state.get("hyde_embedding_chunks") or []

    return embedding_chunks, hyde_embedding_chunks

@step_log("step_2_use_rrf_rank")
def step_2_use_rrf_rank(data_list: list[tuple[float, list]], k: int = 60, top_k: int = 6) -> list[dict]:
    """
    执行带权重的倒数排名融合算法（RRF）
    :param data_list: 各路数据元组列表，格式 [(weight, chunks), ...]
    :param k: 平滑超参数（默认 60）
    :param top_k: 最终保留的最优切片数量上限
    :return: 融合去重后按 RRF 得分降序排列的切片列表
    """
    score_map: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for weight, chunks in data_list:
        if not chunks:
            continue
        for rank, chunk in enumerate(chunks, start=1):
            if not isinstance(chunk, dict):
                continue

            chunk_id = chunk.get("chunk_id")
            rrf_score = weight * (1.0 / (k + rank))

            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + rrf_score

            # 存入 chunk_map，深拷贝防止污染原始对象
            if chunk_id not in chunk_map:
                chunk_copy = copy.deepcopy(chunk)
                chunk_copy["original_score"] = chunk.get("score")
                chunk_map[chunk_id] = chunk_copy

    # 赋值 RRF 综合得分并组装输出（移出两层循环体）
    merged_chunks: List[dict] = []
    for cid, c in chunk_map.items():
        c["score"] = score_map.get(cid, 0.0)
        merged_chunks.append(c)

    # 按融合得分从高到低排序
    merged_chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)

    # 截取 Top-K
    final_rrf_chunks = merged_chunks[:top_k]
    return final_rrf_chunks


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("\n========== 开始测试 node_rrf 节点 ==========")
    # 模拟两路召回的测试数据（包含部分重合切片）
    mock_embedding_chunks = [
        {"chunk_id": "c_001", "title": "万用表说明书", "content": "测量电阻必须先断电调零", "score": 0.88, "type": "milvus"},
        {"chunk_id": "c_002", "title": "万用表说明书", "content": "测量直流电压红表笔接正极", "score": 0.82, "type": "milvus"},
        {"chunk_id": "c_003", "title": "万用表说明书", "content": "蜂鸣档用于检测线路通断", "score": 0.79, "type": "milvus"},
    ]
    mock_hyde_chunks = [
        {"chunk_id": "c_003", "title": "万用表说明书", "content": "蜂鸣档用于检测线路通断", "score": 0.85, "type": "hyde"},
        {"chunk_id": "c_001", "title": "万用表说明书", "content": "测量电阻必须先断电调零", "score": 0.81, "type": "hyde"},
        {"chunk_id": "c_004", "title": "万用表说明书", "content": "电流档必须串联在电路中", "score": 0.75, "type": "hyde"},
    ]
    test_state: QueryGraphState = {
        "session_id": "test_rrf_session_001",
        "is_stream": False,
        "original_query": "万用表怎么测电阻和通断",
        "rewritten_query": "数字万用表测量电阻与电路通断的操作步骤",
        "embedding_chunks": mock_embedding_chunks,
        "hyde_embedding_chunks": mock_hyde_chunks,
        "web_search_docs": [],
        "rrf_chunks": [],
        "reranked_docs": [],
        "prompt": "",
        "answer": "",
        "item_names": ["数字万用表"],
        "history": [],
        "image_urls": [],
    }
    res = node_rrf(test_state)
    print("\n========== 节点运行结果 ==========")
    rrf_res = res.get("rrf_chunks", [])
    print(f"输入切片: 常规={len(mock_embedding_chunks)}条, HyDE={len(mock_hyde_chunks)}条")
    print(f"融合输出: {len(rrf_res)} 条（已去重合并）")
    for idx, c in enumerate(rrf_res, 1):
        print(f"\n[{idx}] 切片ID: {c.get('chunk_id')} | RRF综合得分: {c.get('score'):.6f} (原始分: {c.get('original_score')})")
        print(f"    内容: {c.get('content')}")


