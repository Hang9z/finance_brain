import time
import sys
from common.logging.logger import node_log, logger
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task

@node_log("node_item_name_confirm")
def node_item_name_confirm(state: QueryGraphState):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    logger.info("--- node_item_name_confirm 开始处理 ---")
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)
    time.sleep(0.5)  # 模拟执行
    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)

    logger.info("--- node_item_name_confirm 处理结束 ---")
    # 模拟确认出的商品型号（如果需要测试反问，可在此返回 {"answer": "请问您问的是哪个型号？"}）
    return {"item_names": ["RS-12数字万用表"]}