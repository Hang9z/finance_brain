import time
import sys
from common.logging.logger import node_log, logger
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_done_task, add_running_task

@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state: QueryGraphState):
    """
    节点功能：HyDE (Hypothetical Document Embedding) 假设检索
    注意：并行节点只返回增量字段，不直接 return state
    """
    logger.info("--- HyDE 假设检索 开始处理 ---")
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)
    time.sleep(0.5)
    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)

    logger.info("--- HyDE 假设检索 处理结束 ---")
    return {"hyde_embedding_chunks": ["HyDE切片C: 假设答案关联内容"]}