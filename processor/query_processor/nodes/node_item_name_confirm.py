import sys
import re
import json
from typing import Any, List, Dict, Tuple

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pyparsing import original_text_for

from common.logging.logger import logger, node_log, step_log
from common.config.milvus_config import milvus_config
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task
from utils.load_prompt import load_prompt
from utils.lm.lm_utils import get_llm_client
from utils.lm.embedding_utils import generate_embeddings
from utils.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from utils.clients.mongo_history_utils import get_recent_messages, save_chat_message


@node_log("node_item_name_confirm")
def node_item_name_confirm(state: QueryGraphState) -> Dict[str, Any]:
    """
    解决用户提问中指代的问题 -> rewritten_query
    确保这个主体是真的存在的 -> 去 milvus 再查一遍进行确认 -> extracted_names
    节点功能：
    1. 校验输入并读取历史对话
    2. LLM 改写问题 + 抽取实体
    3. Milvus 混合检索与双阈值裁决
    4. 依据判定结果返回对应的增量字典
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)

    # 1. 获取并校验数据
    original_query, session_id = step_1_validate_and_get_data(state)

    # 2. 获取上下文历史 (按时序正序)
    history_list = step_2_get_history_by_session_id(session_id)

    # 3. 大模型重写与抽取
    llm_dict = step_3_call_llm_rewritten_and_extract_item_names(history_list, original_query)
    extracted_names = llm_dict.get("item_names", [])
    rewritten_query = llm_dict.get("rewritten_query", original_query)

    add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    return {
        "item_names": extracted_names,
        "rewritten_query": rewritten_query
    }


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState) -> Tuple[str, str]:
    # 1. 获取 session_id 和 original_query
    session_id = state.get("session_id")
    original_query = state.get("original_query")
    if not session_id or not original_query:
        logger.error("session_id 或 original_query 为空, 无法继续执行")
        raise ValueError("session_id 或 original_query 为空")
    return original_query.strip(), session_id

@step_log("step_2_get_history_by_session_id")
def step_2_get_history_by_session_id(session_id) -> List[Dict[str, Any]]:
    # [{"role": "user", "text": "123"},
    #  {"role": "assistant", "text": "ok"}]
    # 1. 获取历史消息队列, 从 MongoDB 查询最近 10 条 (MongoDB 默认是降序, 最新的在前)
    raw_history = get_recent_messages(session_id=session_id, limit=10)
    if not raw_history:
        return []

    # 关键点 1. 反转为时间正序, 也就是最新的在最下面, 符合人类对话与大模型阅读顺序
    history_list = list(reversed(raw_history))
    return history_list

@step_log("step_3_call_llm_rewritten_and_extract_item_names")
def step_3_call_llm_rewritten_and_extract_item_names(history_list: List[Dict[str, Any]], original_query: str) -> Dict[str, Any]:
    # 1. 调用大模型对象
    llm_model = get_llm_client(json_mode=True)

    # 格式化历史消息文本: 本质是对数据进行清洗
    # [1] 用户提问: 原始='你们有什么手机', 改写='请问你们店里有哪些型号的手机？', 关联商品='手机'
    # [2] 助手回答: 原始='我们有iPhone 15和华为Mate 60', 改写='我们店里有iPhone 15和华为Mate 60。', 关联商品='iPhone 15,华为Mate 60'
    # [3] 用户提问: 原始='它俩哪个拍照好', 改写='iPhone 15和华为Mate 60哪个拍照更好？', 关联商品='iPhone 15,华为Mate 60'
    # 这里的 rewritten_query 在第一轮进来的时候是没有值的
    # 因为第一轮之前没有历史消息, 没有调用大模型进行改写
    if history_list:
        history_text_list = []
        for idx, h in enumerate(history_list, start=1):
            role_cn = "用户提问" if h.get("role") == "user" else "助手回答"
            items_str = ",".join(h.get("item_names") or [])
            text_str = (h.get("text") or "").strip()[:80]
            rewritten_str = (h.get("rewritten_query") or "").strip()

            history_text_list.append(
                f"[{idx}] {role_cn}: 原始='{text_str}', 改写='{rewritten_str}', 关联商品='{items_str}'"
            )
        history_text = "\n".join(history_text_list)
    else:
        history_text = "暂无历史对话"

    # 加载提示词模板并渲染
    prompt_text = load_prompt("rewritten_query_and_itemnames", query=original_query, history_text=history_text)

    # 关键点 2. 增强的容错调用与 JSON 解析
    response = llm_model.invoke([HumanMessage(content=prompt_text)])
    # content 可能长这样
    # 请直接返回JSON格式结果，格式如下：
    # {{
    #     "item_names": ["商品A", "商品B"],
    #     "rewritten_query": "关于商品A和商品B，..."
    # }}
    content = response.content if hasattr(response, "content") else str(response)

    # 2. 精准提取 JSON (不管前后有多少废话, 只抓 ```json 和 ``` 之间的内容)
    # DOTALL 标志意味着让 . 匹配包括换行符在内的所有字符
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)

    if match:
        # 提取到了代码块里面的内容
        clean_content = match.group(1).strip()
    else:
        # 如果大模型连 ``` 都没写，那就尝试直接寻找大括号 {...}
        match_braces = re.search(r'\{.*\}', content, re.DOTALL)
        if match_braces:
            clean_content = match_braces.group(0).strip()
        else:
            # 啥也没匹配到，死马当活马医
            clean_content = content.strip()

    # 3. 解析与兜底
    try:
        parsed = json.loads(clean_content)
    except Exception as e:
        logger.warning(f"LLM 输出 JSON 解析失败: {e}, 回退使用原始问题。原始内容: {content}")
        # 【修复点】这里直接构造一个兜底的完整结构，真正实现“回退”
        parsed = {
            "item_names": [],
            "rewritten_query": original_query  # original_query 假设是你传进来的参数
        }
    # 4. 安全提取最终结果
    item_names = parsed.get("item_names") or []
    rewritten_query = parsed.get("rewritten_query") or original_query

    return {
        "item_names": item_names,
        "rewritten_query": rewritten_query
    }

