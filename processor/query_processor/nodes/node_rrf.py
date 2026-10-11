import time
import sys
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task
from common.logging.logger import logger, node_log

@node_log("node_rrf")
def node_rrf(state: QueryGraphState):
    """
    节点功能：Reciprocal Rank Fusion (倒排排序融合)
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """
    logger.info("--- RRF 融合排序 开始处理 ---")
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)
    time.sleep(0.5)
    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)

    logger.info("--- RRF 融合排序 处理结束 ---")
    return {"rrf_chunks": ["RRF融合切片Top1", "RRF融合切片Top2"]}