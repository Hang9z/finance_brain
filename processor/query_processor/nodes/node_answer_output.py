import time
import sys
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.sse_utils import push_to_session, SSEEvent
from utils.task_utils import add_running_task, add_done_task
@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState):
    """
    节点功能：组装 Prompt，调用大模型生成答案，支持流式与非流式输出
    """
    logger.info("--- node_answer_ouput 节点开始处理 ---")
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)

    base_answer = state.get("answer") or f"关于「{state.get('original_query', '您的问题')}」的回答演示：请按照说明书标准接线使用。"
    final_text = ""

    if is_stream and session_id:
        for ch in base_answer:
            final_text += ch
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
            time.sleep(0.02)
        push_to_session(session_id, SSEEvent.FINAL, {"answer": final_text, "status": "completed"})
    else:
        final_text = base_answer

    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
    logger.info("--- node_answer_output 节点处理结束 ---")
    return {"answer": final_text}