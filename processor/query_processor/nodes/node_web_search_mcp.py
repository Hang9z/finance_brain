import time
import sys
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_done_task, add_running_task
from common.logging.logger import logger, node_log

@node_log("node_web_search_mcp")
def node_web_search_mcp(state: QueryGraphState):
    """
    节点功能：调用外部搜索引擎补充信息
    注意：并行节点只返回增量字段，不直接 return state
    """
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)
    logger.info("--- 联网搜索 MCP 开始处理 ---")
    time.sleep(0.5)
    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
    
    logger.info("--- 联网搜索 MCP 处理结束 ---")
    return {"web_search_docs": ["网络参考: 万用表使用注意事项"]}