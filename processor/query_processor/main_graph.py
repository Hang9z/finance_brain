import json
from pathlib import Path
import sys
# 确保脚本直接运行时正确加载工程根目录模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from langgraph.graph import StateGraph, END
from common.logging.logger import logger
from processor.query_processor.state import QueryGraphState, create_query_default_state
from processor.query_processor.nodes.node_item_name_confirm import node_item_name_confirm
from processor.query_processor.nodes.node_search_embedding import node_search_embedding
from processor.query_processor.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from processor.query_processor.nodes.node_web_search_mcp import node_web_search_mcp
from processor.query_processor.nodes.node_rrf import node_rrf
from processor.query_processor.nodes.node_rerank import node_rerank
from processor.query_processor.nodes.node_answer_output import node_answer_output

# 1. 实例化状态图容器
builder = StateGraph(QueryGraphState)

# 2. 挂载 7 个核心检索节点
builder.add_node("node_item_name_confirm", node_item_name_confirm)
builder.add_node("node_search_embedding", node_search_embedding)
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_rrf", node_rrf)
builder.add_node("node_rerank", node_rerank)
builder.add_node("node_answer_output", node_answer_output)

# 3. 设置工作流起点
builder.set_entry_point("node_item_name_confirm")

# 4. 定义意图确认后的条件分支决策函数
def route_after_item_confirm(state: QueryGraphState):
    """
    意图确认后的条件路由分支：
    - 若 state 中已经产生 answer（例如多型号反问用户、或查无此商品拒绝回答），直接流转至 node_answer_output 输出；
    - 正常确认出商品型号后，同时并发触发 3 路搜索召回：
      1. node_search_embedding（本地常规双路向量检索）
      2. node_search_embedding_hyde（HyDE 假设性回答向量检索）
      3. node_web_search_mcp（联网搜索）
    """
    if state.get("answer"):
        return "node_answer_output"
    # 并发进入 3 路搜索
    return ["node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"]

# 5. 绑定条件边
builder.add_conditional_edges(
    "node_item_name_confirm",
    route_after_item_confirm,
    {
        "node_answer_output": "node_answer_output",
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp",
    }
)

# 6. 三路召回汇聚到 RRF 倒排融合节点
builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_web_search_mcp", "node_rrf")

# 7. 顺序静态边: RRF -> Rerank -> 答案生成 -> 结束
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)

# 8. 编译工作图
query_app = builder.compile()


# 9. 本地测试执行块
if __name__ == "__main__":
    logger.info("===== 开始测试检索流程主图 (query_processor/main_graph) =====")
    # 构造测试初始状态
    initial_state = create_query_default_state(
        session_id="test_session_001",
        original_query="万用表RS-12如何测直流电压?",
        is_stream=False
    )
    final_state = None
    for event in query_app.stream(initial_state):
        for node_name, result in event.items():
            logger.success(f"节点执行完成: {node_name}")
            final_state = result
    logger.info(f"最终输出: {json.dumps(final_state, indent=2, ensure_ascii=False)}")
    # 打印 ASCII 拓扑图（grandalf 支持）
    try:
        logger.info("工作流 ASCII 拓扑结构:")
        query_app.get_graph().print_ascii()
    except Exception as e:
        logger.warning(f"打印 ASCII 图结构跳过: {e}")
    logger.info("===== 检索流程主图拓扑测试通过 =====")
