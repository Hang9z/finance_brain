import time
import sys
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_done_task, add_running_task

@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState):
    """
    节点功能：常规双路向量内容检索
    注意：并行节点只返回增量字段，不直接 return state
    """
    logger.info("--- 向量内容检索 开始处理 ---")
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)
    time.sleep(0.5)
    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)

    logger.info("--- 向量内容检索 处理结束 ---")
    return {"embedding_chunks": ["切片A: 电压测量步骤", "切片B: 电流测量规范"]}