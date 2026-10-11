import time
import sys
from common.logging.logger import node_log, logger
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task

@node_log("node_rerank")
def node_rerank(state: QueryGraphState):
    """
    节点功能：使用 Cross-Encoder 模型对召回结果进行精细打分重排
    """
    logger.info("--- Rerank 重排序 开始处理 ---")
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)
    time.sleep(0.5)
    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)

    logger.info("--- Rerank 重排序 处理结束 ---")
    return {"reranked_docs": ["精排最优切片Top1"]}