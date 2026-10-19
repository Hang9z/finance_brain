import sys
import os
import re
import time
from pathlib import Path

# 确保在直接独立运行测试时能正确找到项目根路径模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langchain_core.messages import HumanMessage
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.mongo_history_utils import get_recent_messages, save_chat_message
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.sse_utils import push_to_session, SSEEvent
from utils.task_utils import add_done_task, add_running_task


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState) -> dict:
    """
    节点功能：生成最终回答并交付给用户（支持流式/非流式），
    提取参考切片中的图片直链，并持久化到 MongoDB 历史会话。
    """
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)

    # 1. 检查 state 中是否已有答案（如 Node 1 的多型号反问或拒绝回答）
    answer = state.get("answer")
    image_urls = state.get("image_urls") or []

    if not answer:
        # 2. 正常生成流程：获取并校验前序上下文参数
        reranked_docs, session_id, item_names, rewritten_query, is_stream = step_2_validate_and_get_data(state)

        # 3. 读取有效的历史聊天记录（按时间正序）
        history_list = step_3_get_history_by_session_id(session_id)

        # 4. 组装最终提示词
        answer_prompt = step_4_create_answer_prompt(reranked_docs, history_list, item_names, rewritten_query)

        # 5. 调用大模型生成回答（支持流式实时 SSE 推送）
        answer = step_5_call_llm_create_answer(answer_prompt, is_stream, session_id)

        # 6. 从参考切片中正则提取可靠图片 URL
        image_urls = step_6_extract_chunk_and_url_image(reranked_docs, answer)
    else:
        # 已有答案（如意图消歧反问）：若开启流式，则打字机推送已有文本
        if is_stream and session_id:
            for ch in answer:
                push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
                time.sleep(0.015)

    # 7. 更新 state
    state["answer"] = answer
    state["image_urls"] = image_urls

    # 8. 保存助手回答到 MongoDB 会话历史
    step_7_save_assistant_message(state)

    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
    logger.info(f"[{session_id}] 答案生成节点完成交付，图片数: {len(image_urls)}")

    return {
        "answer": answer,
        "image_urls": image_urls,
    }


@step_log("step_2_validate_and_get_data")
def step_2_validate_and_get_data(state: QueryGraphState) -> tuple[list, str, list, str, bool]:
    """提取生成答案所需的上下文参数，具备宽容防御能力"""
    reranked_docs = state.get("reranked_docs") or []
    session_id = state.get("session_id", "")
    item_names = state.get("item_names") or []
    rewritten_query = state.get("rewritten_query", "").strip() or state.get("original_query", "").strip()
    is_stream = state.get("is_stream", False)

    if not rewritten_query:
        logger.warning(f"[{session_id}] 问句为空，使用默认提问兜底")
        rewritten_query = "请根据已有资料提供产品使用帮助。"

    return reranked_docs, session_id, item_names, rewritten_query, is_stream


@step_log("step_3_get_history_by_session_id")
def step_3_get_history_by_session_id(session_id: str) -> list[dict]:
    """获取会话最近有效聊天记录，并反转为时间正序（从旧到新）"""
    if not session_id:
        return []

    try:
        raw_history = get_recent_messages(session_id=session_id, limit=6)
        # 过滤掉空消息
        valid_history = [item for item in raw_history if item.get("text")]
        # MongoDB 查出的是 ts DESC（降序），此处反转为时序正序以供模型阅读上下文
        valid_history.reverse()
        return valid_history
    except Exception as e:
        logger.warning(f"[{session_id}] 获取会话历史异常 (降级忽略): {e}")
        return []


MANDATORY_DISCLAIMER = (
    "\n\n【合规免责声明】\n"
    "本回答基于公开资料与知识库检索结果生成，仅供金融知识科普与资料查询参考，"
    "不构成任何投资建议、买卖要约或收益保证。市场有风险，投资需谨慎。"
    "投资者在作出投资决策前应认真阅读正式法律文件、产品合同、招募说明书及风险揭示书，"
    "并根据自身风险承受能力独立审慎决策。"
)


@step_log("step_4_create_answer_prompt")
def step_4_create_answer_prompt(
    reranked_docs: list[dict],
    history_message: list[dict],
    item_names: list[str],
    rewritten_query: str,
) -> str:
    """拼装参考切片、历史上下文与金融专属合规提示词"""
    # 1. 组装参考切片上下文
    if reranked_docs:
        context_parts = []
        for idx, doc in enumerate(reranked_docs, start=1):
            source_type = "联网搜索" if doc.get("type") == "web" else "内部金融知识库"
            doc_file = doc.get("file_title") or doc.get("title") or "参考资料"
            entity_tag = f" | 产品/主体: {doc.get('item_name')}" if doc.get("item_name") else ""
            parent_tag = f" | 章节/条款: {doc.get('parent_title')}" if doc.get("parent_title") else ""
            score = doc.get("score", 0.0)
            text = doc.get("text", "").strip()
            context_parts.append(
                f"[{idx}] 来源: {source_type} | 资料: 《{doc_file}》{entity_tag}{parent_tag} | 置信度: {score:.4f}\n{text}"
            )
        context_str = "\n\n".join(context_parts)
    else:
        context_str = "（知识库中暂未检索到直接相关的金融产品资料、公告或外网网页）"

    # 2. 组装历史会话
    if history_message:
        history_parts = []
        for idx, h in enumerate(history_message, start=1):
            role_label = "用户" if h.get("role") == "user" else "金融智脑"
            text = h.get("text", "")[:150]
            history_parts.append(f"{role_label}: {text}")
        history_str = "\n".join(history_parts)
    else:
        history_str = "（无前序历史对话）"

    # 3. 关联实体
    item_names_str = "、".join(item_names) if item_names else "通用金融概念/未指定具体产品"

    # 4. 加载并渲染外部提示词模版
    answer_prompt = load_prompt(
        "answer_out",
        context=context_str,
        history=history_str,
        item_names=item_names_str,
        question=rewritten_query,
    )
    return answer_prompt


@step_log("step_5_call_llm_create_answer")
def step_5_call_llm_create_answer(answer_prompt: str, is_stream: bool, session_id: str) -> str:
    """调用大模型生成答案，流式模式下实时向前端推送 DELTA 事件，并执行合规兜底保障"""
    llm_client = get_llm_client()
    answer_text = ""

    message = HumanMessage(content=answer_prompt)

    if is_stream and session_id:
        logger.info(f"[{session_id}] 启动 LLM 流式输出与 SSE 推送")
        for chunk in llm_client.stream([message]):
            delta = chunk.content if hasattr(chunk, "content") else str(chunk)
            if delta:
                answer_text += delta
                push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
    else:
        logger.info(f"[{session_id}] 调用 LLM 非流式完整生成")
        response = llm_client.invoke([message])
        answer_text = response.content if hasattr(response, "content") else str(response)

    answer_text = answer_text.strip()

    # 合规安全兜底检查：若大模型回答缺失【合规免责声明】区块，自动补齐
    if "【合规免责声明】" not in answer_text:
        logger.warning(f"[{session_id}] 答案缺失【合规免责声明】，自动追加法定合规提示")
        if is_stream and session_id:
            push_to_session(session_id, SSEEvent.DELTA, {"delta": MANDATORY_DISCLAIMER})
        answer_text += MANDATORY_DISCLAIMER

    return answer_text


@step_log("step_6_extract_chunk_and_url_image")
def step_6_extract_chunk_and_url_image(reranked_docs: list[dict], answer: str = "") -> list[str]:
    """
    双保险提取图片直链：
    1. 从参考切片 markdown 文本中正则匹配真实 MinIO 图片链接；
    2. 从模型回答末尾的【图片】区块匹配补充链接；
    3. 去重且保持原有出现顺序。
    """
    img_regex = re.compile(r"!\[.*?\]\((https?://[^\s\)]+)\)")
    raw_url_regex = re.compile(r"(https?://[^\s\)]+\.(?:png|jpg|jpeg|gif|svg|webp))", re.IGNORECASE)

    extracted_urls: list[str] = []
    seen = set()

    def _add_url(u: str):
        u_clean = u.strip().rstrip("。，,；;)")
        if u_clean and u_clean not in seen:
            seen.add(u_clean)
            extracted_urls.append(u_clean)

    # 1. 优先从精排切片源文本中提取真实的 Markdown 图片链接
    for doc in reranked_docs:
        text = doc.get("text", "")
        # Markdown 语法图片
        for match in img_regex.findall(text):
            _add_url(match)
        # 单独存在的图片 url 字段
        url = doc.get("url", "")
        if url and url.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")):
            _add_url(url)

    # 2. 从大模型生成的回答中补充提取（防范模型输出了图片列表）
    if answer:
        for match in img_regex.findall(answer):
            _add_url(match)
        for match in raw_url_regex.findall(answer):
            _add_url(match)

    logger.info(f"[step_6_extract_chunk_and_url_image] 提取出有效图片数量: {len(extracted_urls)}")
    return extracted_urls


@step_log("step_7_save_assistant_message")
def step_7_save_assistant_message(state: QueryGraphState):
    """将助手回答落库至 MongoDB"""
    session_id = state.get("session_id")
    if not session_id:
        return

    try:
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=state.get("answer", ""),
            rewritten_query=state.get("rewritten_query", ""),
            item_names=state.get("item_names", []),
            image_urls=state.get("image_urls", []),
        )
        logger.info(f"[{session_id}] 助手回答与图片链接已成功入库 MongoDB")
    except Exception as e:
        logger.warning(f"[{session_id}] 写入助手回答至 MongoDB 异常 (降级忽略): {e}")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n========== 开始测试 node_answer_output 节点 ==========")
    mock_reranked_docs = [
        {
            "chunk_id": "c_local_01",
            "type": "milvus",
            "title": "HAK 180 烫金机使用说明书",
            "score": 0.985,
            "text": (
                "HAK 180 烫金机局部烫金设置步骤：\n"
                "1. 打开前面板，按下【转印参数】按键；\n"
                "2. 选择【区域设置】，输入顶部起始 50mm，结束 170mm；\n"
                "3. 面板结构请参考下图：\n"
                "![HAK180控制面板图](http://192.168.100.133:9000/knowledge-base-files/upload-images/hak180_panel.jpg)"
            ),
        }
    ]

    test_state: QueryGraphState = {
        "session_id": "test_answer_session_001",
        "is_stream": False,
        "original_query": "HAK 180 怎么设置顶部 50-170mm 局部烫金",
        "rewritten_query": "HAK 180 烫金机设置纸张顶部 50-170mm 局部烫金转印的方法",
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "web_search_docs": [],
        "rrf_chunks": [],
        "reranked_docs": mock_reranked_docs,
        "prompt": "",
        "answer": "",
        "item_names": ["HAK 180 烫金机"],
        "history": [],
        "image_urls": [],
    }

    res = node_answer_output(test_state)
    print("\n========== 节点运行结果 ==========")
    print("【生成的最终回答】:")
    print(res.get("answer"))
    print("\n【提取到的图片列表】:")
    for img in res.get("image_urls", []):
        print(f" - {img}")