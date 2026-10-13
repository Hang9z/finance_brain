import time
import sys
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.clients.mongo_history_utils import save_chat_message
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
    # 使用知识库 MinIO 中真实的万用表正面结构面板图（替代测试用的百度图片）
    image_urls = state.get("image_urls") or [
        "http://192.168.100.133:9000/knowledge-base-files/upload-images/万用表RS-12的使用/d6946861c4592804bd8d7e75b58029565712d4dc58f855e374bf0fcf370c91dd.jpg"
    ]
    final_text = ""

    if is_stream and session_id:
        for ch in base_answer:
            final_text += ch
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
            time.sleep(0.02)
    else:
        final_text = base_answer

    save_chat_message(
        session_id=state.get("session_id"),
        role="assistant",
        text=final_text,
        rewritten_query="这是重写问题",
        item_names=["商品a", "商品b"],
        image_urls=image_urls
    )

    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
    logger.info("--- node_answer_output 节点处理结束 ---")
    return {
        "answer": final_text,
        "image_urls": image_urls
    }