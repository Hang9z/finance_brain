import sys
from typing import List, Dict, Any, Tuple

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task
from utils.load_prompt import load_prompt
from utils.lm.lm_utils import get_llm_client
from utils.lm.embedding_utils import generate_embeddings
from utils.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search

@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state: QueryGraphState) -> Dict[str, Any]:
    """
    HyDE 假设性回答检索节点主入口：
    注意：并发节点绝不返回整个 state，仅返回增量字典 {"hyde_embedding_chunks": [...]}
    """
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)
    try:
        # 1. 获取并校验数据
        item_names, rewritten_query = step_1_validate_and_get_data(state)
        # 2. 调用 LLM 生成假设性回答
        hyde_answer = step_2_call_llm_by_rewritten_query(rewritten_query)
        # 3. 联合向量化并在 Milvus 中检索
        raw_hits = step_3_select_chunks_in_milvus(item_names, rewritten_query, hyde_answer)
        # 4. 格式化切片并打标
        hyde_embedding_chunks = step_4_after_deal_milvus_result(raw_hits)
        return {
            "hyde_embedding_chunks": hyde_embedding_chunks
        }
    finally:
        # 无论成功或异常，必须标记当前节点已完成
        add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState) -> Tuple[List, str]:
    item_names = state.get("item_names") or []
    rewritten_query = (state.get("rewritten_query") or state.get("original_query") or "").strip()

    if not item_names:
        logger.warning("未检测到商品主体 item_names, 将在整个知识库切片中进行无过滤 HyDE 检索")


    if not rewritten_query:
        logger.warning("HyDE 检索语句为空, 跳过检索")
        return item_names, ""

    return item_names, rewritten_query

@step_log("step_2_call_llm_by_rewritten_query")
def step_2_call_llm_by_rewritten_query(rewritten_query: str) -> str:
    """
    让大模型针对问题先生成一段简短、专业的假想答案
    :param rewritten_query:
    :return:
    """
    if not rewritten_query:
        return ""

    llm_client = get_llm_client(json_mode=False)
    # 加载已有的 hyde_prompt.prompt 模板并渲染
    hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)

    chains = llm_client | StrOutputParser()
    try:
        hyde_answer = chains.invoke([HumanMessage(content=hyde_prompt)])
        clean_answer = str(hyde_answer).strip()
        logger.info(f"HyDE 假设性回答生成完成 (长度: {len(clean_answer)} 字): {clean_answer[:80]}...")
        return clean_answer
    except Exception as e:
        logger.error(f"HyDE 假设性回答生成失败: {e}, 回退使用原问题")
        return rewritten_query

@step_log("step_3_select_chunks_in_milvus")
def step_3_select_chunks_in_milvus(
        item_names: List[str],
        rewritten_query: str,
        hyde_answer: str
) -> List[Dict[str, Any]]:
    """
    将"问题 + 假设性答案"拼接后生成 BGE-M3 双向量, 在 kb_chunks 集合中执行混合检索
    :param item_names:
    :param rewritten_query:
    :param hyde_answer:
    :return:
    """
    if not rewritten_query:
        return []


    # 1. 组合更饱满的语义信号文本
    combined_text = f"问题: {rewritten_query}\n假设性回答: {hyde_answer}"

    # 2. 生成稠密向量与稀疏向量
    embeddings = generate_embeddings([combined_text])
    dense_vec = embeddings["dense"][0]
    sparse_vec = embeddings["sparse"][0]

    # 3. 构造过滤表达式 (仅在有商品名时过滤, 防语法报错)
    expr = None
    if item_names:
        expr = f"item_name in {item_names}"
        logger.info(f"Milvus HyDE 切片检索过滤表达式: {expr}")

    # 4. 构建双向量 ANN 检索请求
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        expr=expr,
        limit=10
    )

    # 5. 执行混合检索
    collection_name = milvus_config.chunks_collection
    client = get_milvus_client()

    response = hybrid_search(
        client=client,
        collection_name=collection_name,
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

    # 避坑点: 防御性判空, 避免 NoneType 下标报错
    if not response or not response[0]:
        logger.warning("Milvus HyDE 切片检索未命中任何结果")
        return []

    return response[0]

@step_log("step_4_after_deal_milvus_result")
def step_4_after_deal_milvus_result(raw_hits: List[Dict[str, Any]]) -> List[Dict]:
    """
    扁平化提取 entity 字段, 打上 'type': 'hyde' 来源标识
    :param raw_hits:
    :return:
    """
    hyde_embedding_chunks: List[Dict[str, Any]] = []

    for hit in raw_hits:
        chunk = dict(hit.get("entity", {}))
        chunk["score"] = float(hit.get("distance", 0.0))
        chunk["type"] = "hyde" # 避坑点: 正确标记为 hyde 渠道来源, 与 milvus 通道区分开
        hyde_embedding_chunks.append(chunk)

    logger.info(f"HyDE 向量检索完成, 有效召回切片数: {len(hyde_embedding_chunks)}")
    return hyde_embedding_chunks


if __name__ == "__main__":
    test_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "万用表怎么测电阻？",
        "rewritten_query": "RS PRO RS-12数字万用表如何测量电阻？",
        "item_names": ["RS PRO RS-12数字万用表"],
        "is_stream": False
    }
    print("\n========== 开始测试 node_search_embedding_hyde 节点 ==========")
    output = node_search_embedding_hyde(test_state)
    chunks = output.get("hyde_embedding_chunks", [])
    print(f"\n[OK] HyDE 检索完成！成功召回 {len(chunks)} 条切片数据")
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
    print("=============================================================\n")