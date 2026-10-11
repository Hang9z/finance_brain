from typing import List, TypedDict
import copy
class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    使用字典式访问（如 state["session_id"]、state.get("answer")）。
    """
    session_id: str # 会话唯一标识
    original_query: str # 用户原始问题

    # --- 检索阶段的中间数据 ---
    embedding_chunks: list # 基础双路向量检索召回的切片
    hyde_embedding_chunks: list # HyDE 假设性回答检索召回的切片
    web_search_docs: list # 网络搜索召回的文档

    # --- 排序阶段的中间数据 ---
    rrf_chunks: list # RRF 融合去重与粗排后的切片
    reranked_docs: list # BGE-Reranker 交叉编码器精排后的 Top-K 文档

    # --- 生成阶段的数据 ---
    prompt: str # 组装好的完整 Prompt
    answer: str # 最终生成的答案内容

    # --- 辅助元数据 ---
    item_names: List[str]       # 识别出的商品型号列表（如 ['RS-12数字万用表']）
    rewritten_query: str        # 结合上下文改写后的完整独立问题
    history: list               # 历史对话记录
    is_stream: bool             # 是否开启 SSE 流式打字机输出标记
    image_urls: List[str]       # 答案中引用的图片直链

# ========================
# 默认状态（初始全部为空）
# ========================
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "image_urls": []
}

def create_query_default_state(**overrides) -> QueryGraphState:
    """创建查询流程的默认状态，支持传参覆盖字段"""
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state

def get_query_default_state() -> QueryGraphState:
    """获取一个全新的干净默认状态字典，防止全局状态污染"""
    return copy.deepcopy(query_graph_default_state)

def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """深拷贝复制现有状态并支持增量覆盖"""
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == "__main__":
    # 快速自检测试
    state = create_query_default_state(
        session_id="test_001",
        original_query="万用表RS-12如何测电压?",
        is_stream=True
    )
    print("初始状态:", state)