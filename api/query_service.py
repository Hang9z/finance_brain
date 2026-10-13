import json
from pathlib import Path
import sys
import uuid
from datetime import datetime

# 确保直接运行脚本时能正确找到项目根目录
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from processor.query_processor.state import create_query_default_state

from utils.task_utils import *
from utils.sse_utils import create_sse_queue, get_sse_queue, SSEEvent, sse_generator
from utils.clients.mongo_history_utils import *
from processor.query_processor.main_graph import query_app


class QueryRequest(BaseModel):
    """
     query 接口接收的数据结构
    """
    # 用户的提问、是否选择流式输出、会话 id
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否选择流式返回")


"""
 接口实现: 返回前端页面 1. chat
 用户请求 2. query
 建立SSE连接 3. sse
 主图运行 4. run_graph
"""
# 定义 fastAPI
app = FastAPI()

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat.html")
def chat():
    """
    返回前端页面接口
    :return: FileResponse(path)
    """
    chat_html_path = Path(__file__).absolute().parent.parent / "page" / "chat.html"
    if not chat_html_path.exists():
        logger.error(f"页面不存在")
        raise HTTPException(status_code=404, detail=f"没有查询到页面, 地址为: {chat_html_path}")
    logger.info("前端页面加载成功")
    return FileResponse(chat_html_path)

@app.post("/query")
def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    用户提问接口
    1. 获取当前会话id、提问内容、是否流式返回
    2. 开启一个后台任务调用主图
    :return: {session_id: str, data: {}}
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream
    logger.info(f"用户提问: {user_query}, 当前会话id: {session_id}, 流式状态: {is_stream}")
    if is_stream:
        # 提前同步创建 SSE 队列，确保前端收到响应后立刻连接 /stream/{session_id} 时队列已就绪
        create_sse_queue(session_id)
        background_tasks.add_task(run_query_graph, session_id=session_id, is_stream=is_stream, user_query=user_query)
        logger.info(f"[{session_id}] 流式任务已提交至后台执行")
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        state = run_query_graph(session_id, user_query, is_stream)
        done_list = get_done_task_list(session_id)

        return {
            "session_id": session_id,
            "message": f"{session_id} 问题解析完毕",
            "answer": state.get("answer"),
            "done_list": done_list,
            "image_urls": state.get("image_urls", [])
        }

def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    logger.info(f"[{session_id}] 开始执行查询, 流式模式: {is_stream}")
    try:
        clear_task(session_id)  # 清空上一次提问产生的数据

        # 1. 如果是流式响应且队列尚未创建, 创建会话专属队列
        if is_stream and get_sse_queue(session_id) is None:
            create_sse_queue(session_id)

        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

        # 2. 初始化图状态（无论是否流式，均需初始化）
        state = create_query_default_state(
            session_id=session_id,
            is_stream=is_stream,
            original_query=user_query
        )
        logger.info(f"查询图开始执行,传入参数:\n{json.dumps(state, indent=4, ensure_ascii=False)}")

        # 3. 执行 LangGraph 图
        state = query_app.invoke(state)
        logger.info(f"查询图执行结束,返回结果:\n{json.dumps(state, indent=4, ensure_ascii=False)}")

        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)

        # 4. 流式模式下推送最终交付与关闭事件
        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": state.get("answer"),
                    "status": "completed",
                    "image_urls": state.get("image_urls", [])
                }
            )
            push_to_session(session_id, SSEEvent.CLOSE, {})

        return state
    except Exception as e:
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        logger.exception(f"{user_query}: 执行图流程报错!")
        if is_stream:
            push_to_session(session_id=session_id, event=SSEEvent.ERROR, data={
                "error": str(e)
            })
            push_to_session(session_id=session_id, event=SSEEvent.CLOSE, data={})
        raise

@app.get("/stream/{session_id}")
def stream(session_id: str, request: Request):
    """
    SSE 流式响应接口
    :param session_id:
    :param request:
    :return:
    """
    logger.info(f"{datetime.now()}完成{session_id}流式响应")
    return StreamingResponse(
        # 生成器函数
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )

@app.get("/health")
def health():
    """服务健康检查"""
    logger.info(f"{datetime.now()}: 完成本次健康状态检查")
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)