import sys
from typing import List, Dict, Any, Tuple
from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task
from utils.lm.embedding_utils import generate_embeddings
from utils.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search

@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState) -> Dict[str, Any]:
    """
    常规切片双路向量召回节点：
    注意：并发节点绝不返回整个 state，仅返回增量字典 {"embedding_chunks": [...]}
    """
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)
    try:
        # 1. 获取并校验参数
        item_names, rewritten_query = step_1_validate_and_get_data(state)
        # 2. Milvus 切片库双向量混合检索
        raw_hits = step_2_select_chunks_in_milvus(item_names, rewritten_query)
        # 3. 结果格式化与打标
        embedding_chunks = step_3_after_deal_milvus_result(raw_hits)
        # 4. 增量返回，由 LangGraph 自动合并入全局状态
        return {
            "embedding_chunks": embedding_chunks
        }
    finally:
        # 无论成功或异常，必须标记当前节点已完成
        add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState) -> Tuple[List[str], str]:
    """
    提取商品名称与查询语句, 具备缺失回退机制
    :param state:
    :return:
    """
    item_names: List[str] = state.get("item_names") or []
    # 优先使用改写后的独立问题, 若无则回退使用原始提问
    rewritten_query: str = (state.get("rewritten_query") or state.get("original_query") or "").strip()

    if not rewritten_query:
        logger.warning("检索语句为空, 无法进行向量检索")
        return item_names, ""

    if not item_names:
        logger.warning("未检测到商品主体 item_names, 将在整个知识库切片中进行无过滤检索")

    return item_names, rewritten_query


def step_2_select_chunks_in_milvus(item_names: List[str], rewritten_query: str) -> List[Dict[str, Any]]:
    """
    生成 dense、sparse, 建立索引, 进行检索
    :param item_names:
    :param rewritten_query:
    :return: chunks
    """
    if not rewritten_query:
        return []

    # 1. 调用 BGE-M3 生成双向量嵌入
    embeddings = generate_embeddings([rewritten_query])
    # {"dense":[[第一个item_name的稠密], [第二个item_name的稠密]]}, 因为这里只有一个
    dense_vec = embeddings["dense"][0]
    sparse_vec = embeddings["sparse"][0]

    # 2. 构造 Milvus 过滤表达式 (仅在有商品名时过滤)
    expr = None
    if item_names:
        expr = f"item_name in {item_names}"
        logger.info(f"Milvus 切片检索过滤表达式: {expr}")

    # 3. 构建双向量 ANN 检索请求
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        expr=expr,
        limit=10
    )

    # 4. 执行混合检索
    client = get_milvus_client()

    response = hybrid_search(
        client=client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        limit=5,
        output_fields=[
            "chunk_id",
            "file_title",
            "item_name",
            "title",
            "parent_title",
            "part",
            "content"
        ]
    )

    # 防御性判空
    if not response or not response[0]:
        logger.warning("Milvus 切片检索未命中任何结果")
        return []

    # [[{},{},{}]]
    return response[0]

@step_log("step_3_after_deal_milvus_result")
def step_3_after_deal_milvus_result(raw_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    扁平提取 entity 字段, 挂载 score 相似度 与 "type": 'milvus' 来源
    :param raw_hits:
    :return:
    """
    embedding_chunks: List[Dict] = []

    for hit in raw_hits:
        # 防御性浅拷贝, 避免意外修改
        chunk = dict(hit.get("entity", {}))
        chunk["score"] = float(hit.get("distance", 0.0))
        chunk["type"] = "milvus"
        embedding_chunks.append(chunk)

    logger.info(f"常规向量检索完成, 有效召回切片数: {len(embedding_chunks)}")
    return embedding_chunks


if __name__ == "__main__":
    # 构造模拟输入状态（使用您知识库中已录入的真实商品进行测试）
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "万用表怎么测电阻？",
        "item_names": ["RS PRO RS-12数字万用表"],  # 知识库实际录入的标准品名
        "is_stream": False
    }
    print("\n========== 开始测试 node_search_embedding 节点 ==========")
    output = node_search_embedding(test_state)
    chunks = output.get("embedding_chunks", [])
    print(f"\n[OK] 检索完成！成功召回 {len(chunks)} 条切片数据")
    if chunks:
        top1 = chunks[0]
        print("\nTop 1 召回详情:")
        print(f"  - chunk_id: {top1.get('chunk_id')}")
        print(f"  - item_name: {top1.get('item_name')}")
        print(f"  - file_title: {top1.get('file_title')}")
        print(f"  - score: {top1.get('score')}")
        print(f"  - type: {top1.get('type')}")
        print(f"  - content 片段: {top1.get('content', '')[:100]}...\n")
    else:
        print("[WARN] 未检索到切片，请确认知识库 kb_chunks 中是否录入了该商品说明书。")
    print("========================================================\n")