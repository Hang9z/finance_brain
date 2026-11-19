import json
import os
from pathlib import Path
import sys

# 确保直接运行脚本时能正确找到项目根目录
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import uuid
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from processor.query_processor.state import create_query_default_state
from processor.query_processor.nodes.node_search_embedding import node_search_embedding
from processor.query_processor.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from processor.query_processor.nodes.node_web_search_mcp import node_web_search_mcp
from processor.query_processor.nodes.node_rrf import node_rrf
from processor.query_processor.nodes.node_rerank import node_rerank

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


class RetrieveRequest(BaseModel):
    """纯检索接口请求参数."""

    query: str = Field(..., description="检索问句")
    top_k: int = Field(4, ge=1, le=10, description="返回切片数量上限")
    item_names: List[str] = Field(default_factory=list, description="可选金融产品或主体过滤词")
    enable_hyde: bool = Field(False, description="是否启用 HyDE 假设回答增强检索")
    enable_web: bool = Field(False, description="是否同时启用联网搜索")


class RetrievedChunk(BaseModel):
    """单条检索命中的精排切片."""

    chunk_id: str = ""
    content: str
    title: str = ""
    file_title: str = ""
    parent_title: str = ""
    item_name: str = ""
    source_type: str = "milvus"
    url: str = ""
    score: float = 0.0


class RetrieveResponse(BaseModel):
    """纯检索接口返回体."""

    query: str
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    degraded: bool = False
    reason: str = ""

class HistoryItem(BaseModel):
    """
    接收 mongoDB 返回的数据
    """
    id: str = Field(alias="_id", default="")
    session_id: str
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    item_names: Optional[List[str]] = []
    image_urls: Optional[List[str]] = []
    ts: Optional[float] = None

class HistoryResponse(BaseModel):
    session_id: str
    items: List[HistoryItem]

class ClearHistoryResponse(BaseModel):
    message: str = "History cleared"
    deleted_count: int = Field(..., description="成功删除的消息条数", ge=0)

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

@app.get("/")
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

@app.get("/history/{session_id}", response_model=HistoryResponse)
def history(session_id: str, limit: int = 50):
    f"""
    从 mongoDB 中获取历史消息, 返回给前端
    :param session_id: 会话id
    :param limit: 消息条数
    :return: {"session_id":}
    """
    records = get_recent_messages(session_id=session_id, limit=limit)
    # 因为 mongoDB 中有主键 _id 类型是 ObjectId, 跟 python 中的定义冲突, 会被认为是私有化向量, 重命名
    for r in records:
        r["_id"] = str(r["_id"])
    return {"session_id": session_id, "items": records}

@app.delete("/history/{session_id}",
            response_model=ClearHistoryResponse,
            summary="清空指定会话的聊天历史",
            description="根据 session_id 彻底物理删除该会话的所有历史消息")
def clear_chat_history(session_id: str):
    # 假设 clear_history 在数据库出错时会抛出异常, 由全局异常处理器接管
    deleted_count = clear_history(session_id)

    logger.info(f"会话 [{session_id}] 聊天记录已清空, 删除数量: {deleted_count}")

    return {
        "message": "History cleared",
        "deleted_count": deleted_count
    }

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

        # 3. 执行 LangGraph 查询图 (图中各节点已内置消息持久化与状态更新)
        state = query_app.invoke(state)
        logger.info(f"查询图执行结束,返回结果:\n{json.dumps(state, indent=4, ensure_ascii=False)}")

        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)

        # 6. 流式模式下推送最终交付与关闭事件
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

@app.post("/api/v1/retrieve", response_model=RetrieveResponse, summary="纯检索知识切片接口")
def retrieve_knowledge_chunks(req: RetrieveRequest):
    """复用 BGE-M3 双路向量检索 + RRF 融合 + BGE-Reranker 精排返回 Top-K 切片（无状态，不写会话历史）."""
    clean_query = (req.query or "").strip()
    if not clean_query:
        return RetrieveResponse(query=req.query, chunks=[], degraded=True, reason="检索问句为空")

    session_id = f"retrieve_{uuid.uuid4().hex[:12]}"
    state = create_query_default_state(
        session_id=session_id,
        original_query=clean_query,
        rewritten_query=clean_query,
        item_names=list(req.item_names or []),
        is_stream=False,
    )
    reasons: List[str] = []

    try:
        try:
            emb_res = node_search_embedding(state)
            state["embedding_chunks"] = emb_res.get("embedding_chunks") or []
        except Exception as exc:
            logger.warning(f"[{session_id}] 向量检索异常，降级继续: {exc}")
            reasons.append(str(exc))

        if req.enable_hyde:
            try:
                hyde_res = node_search_embedding_hyde(state)
                state["hyde_embedding_chunks"] = hyde_res.get("hyde_embedding_chunks") or []
            except Exception as exc:
                logger.warning(f"[{session_id}] HyDE 检索异常，降级继续: {exc}")
                reasons.append(str(exc))

        if req.enable_web:
            try:
                web_res = node_web_search_mcp(state)
                state["web_search_docs"] = web_res.get("web_search_docs") or []
            except Exception as exc:
                logger.warning(f"[{session_id}] 联网检索异常，降级继续: {exc}")
                reasons.append(str(exc))

        rrf_res = node_rrf(state)
        state["rrf_chunks"] = rrf_res.get("rrf_chunks") or []

        if not state["rrf_chunks"] and not state["web_search_docs"]:
            clear_task(session_id)
            reason_text = "; ".join(r for r in reasons if r) or "知识库未命中或向量库未连接"
            return RetrieveResponse(query=clean_query, chunks=[], degraded=True, reason=reason_text)

        rerank_res = node_rerank(state)
        docs = rerank_res.get("reranked_docs") or []
        limit = max(1, min(req.top_k, 10))
        chunks: List[RetrievedChunk] = []
        for doc in docs[:limit]:
            content = str(doc.get("text") or doc.get("content") or "").strip()
            if not content:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(doc.get("chunk_id") or ""),
                    content=content,
                    title=str(doc.get("title") or ""),
                    file_title=str(doc.get("file_title") or ""),
                    parent_title=str(doc.get("parent_title") or ""),
                    item_name=str(doc.get("item_name") or ""),
                    source_type=str(doc.get("type") or "milvus"),
                    url=str(doc.get("url") or ""),
                    score=float(doc.get("score") or 0.0),
                )
            )
        clear_task(session_id)
        return RetrieveResponse(query=clean_query, chunks=chunks, degraded=len(chunks) == 0)
    except Exception as exc:
        logger.warning(f"[{session_id}] /api/v1/retrieve 执行异常，安全降级返回空切片: {exc}")
        clear_task(session_id)
        return RetrieveResponse(query=clean_query, chunks=[], degraded=True, reason=str(exc))


@app.get("/health")
def health():
    """服务健康检查"""
    logger.info(f"{datetime.now()}: 完成本次健康状态检查")
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("QUERY_SERVICE_PORT", "8001")))