import sys
import re
import json
from typing import Any, List, Dict, Tuple

from langchain_core.messages import HumanMessage

from common.logging.logger import logger, node_log, step_log
from common.config.milvus_config import milvus_config
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task
from utils.load_prompt import load_prompt
from utils.lm.lm_utils import get_llm_client
from utils.lm.embedding_utils import generate_embeddings
from utils.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from utils.clients.mongo_history_utils import get_recent_messages, save_chat_message

ITEM_NAME_CONFIRM_THRESHOLD = 0.70 # 高于该分数: 加入确认队列
ITEM_NAME_CANDIDATE_THRESHOLD = 0.60 # 介于 0.60 ~ 0.70: 视为候选型号, 反问用户
ITEM_NAME_OPTIONS_TOP_K = 3 # 反问用户时最多展示 3 个候选
ITEM_NAME_CONFIRMED_TOP_K = 1 # 每个抽取词最多绑定 1 个标准商品


@node_log("node_item_name_confirm")
def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    主体确认节点主入口：
    1. 校验参数并读取历史会话
    2. LLM 改写问题并抽取潜在商品词
    3. Milvus 混合检索与双阈值裁决
    4. 显式更新 state 并持久化用户提问
    """
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)

    try:
        # 步骤 1：获取并校验数据
        original_query, session_id = step_1_validate_and_get_data(state)

        # 步骤 2：获取上下文历史（按时序正序）
        history_list = step_2_get_history_by_session_id(session_id)

        # 步骤 3：大模型重写与商品抽取
        llm_dict = step_3_call_llm_rewritten_and_extract_item_names(history_list, original_query)
        extracted_names = llm_dict.get("item_names", [])
        rewritten_query = llm_dict.get("rewritten_query", original_query)

        # 若大模型未抽取出金融实体，且用户输入简短（如直接输入产品简称或代码），回退检索
        # 若用户输入包含疑问词或属于通用金融概念（如“什么是最大回撤？”、“如何开户？”），保持 extracted_names 为空
        if not extracted_names and original_query.strip():
            clean_q = re.sub(r'[？?！!。，,\s]+$', '', original_query.strip())
            is_generic_question = any(w in clean_q for w in ["什么是", "如何", "怎么", "区别", "哪些", "多少", "是否", "为什么", "规则", "流程", "规定", "介绍一下"])
            if clean_q and len(clean_q) <= 20 and not is_generic_question:
                logger.info(f"大模型未抽取出金融主体，属于简短主体提问，触发兜底实体检索: '{clean_q}'")
                extracted_names = [clean_q]

        # 步骤 4：在 Milvus 的 finance_item_names 中检索标准名称
        milvus_dict = {}
        if extracted_names:
            milvus_dict = step_4_select_item_names_milvus(extracted_names)

        # 步骤 5：依据分数阈值分离确信与候选列表
        confirmed_dict = step_5_select_confirmed_and_option_item_names(milvus_dict)

        # 步骤 6：显式更新 state 属性（确信实体过滤、歧义反问或放行全量检索）
        step_6_change_state_property(state, rewritten_query, confirmed_dict)

        # 步骤 7：保存当前提问到 MongoDB 历史
        step_7_save_user_chat_message(state)

        return state

    finally:
        # 保证无论成功还是异常，节点进度均标记完成
        add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)


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

@step_log("step_4_select_item_names_milvus")
def step_4_select_item_names_milvus(item_names: List[str]) -> Dict[str, List[Dict]]:
    """
    根据大模型提取出的口语词，在 Milvus 的 kb_item_names 集合中做双向量混合搜索
    :param item_names: 大模型抽取出的商品词列表，如 ['万用表']
    :return: 字典结构，格式如 {'万用表': [{'item_name': 'RS-12数字万用表', 'score': 0.85}, ...]}
    """
    if not item_names:
        return {}

    milvus_search_result = {}
    milvus_client = get_milvus_client()

    # 1. 批量计算 item_names 的稠密向量与稀疏向量
    # "dense":[[], []] "sparse": [[], []]
    embeddings = generate_embeddings(item_names)

    # 2. 遍历 dense 列表和 sparse 列表进行混合检索
    for idx, item_name in enumerate(item_names):
        dense_vector = embeddings["dense"][idx]
        sparse_vector = embeddings["sparse"][idx]

        # 构建混合搜索
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=10
        )

        # 在 kb_item_names 集合中检索，商品名实体对齐以稠密语义相似度为主(0.8, 0.2)
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.8, 0.2),
            norm_score=True,
            limit=5,
            output_fields=["item_name"]
        )


        # 防御性判空, 且未命中单项使用 continue 而不是 break
        if not response or not response[0]:
            logger.warning(f"商品名称 [{item_name}] 在 Milvus 中未检索到任何匹配项")
            continue

        item_candidates = []
        for hit in response[0]:
            # [    [{"id", "distance", "entity":{"item_name": xxx, } }, {}, {}]    ]
            name = hit.get("entity", {}).get("item_name")
            score = hit.get("distance", 0.0)
            if name:
                item_candidates.append({"item_name": name, "score": score})

        logger.info(f"商品名称 [{item_name}] Milvus 检索命中候选: {item_candidates}")
        milvus_search_result[item_name] = item_candidates

    return milvus_search_result

@step_log("step_5_select_confirmed_and_option_item_names")
def step_5_select_confirmed_and_option_item_names(milvus_search_result: Dict[str, List[Dict]]) -> Dict[str, List]:
    # 入参 "item_name":[{"item_name": "", "score": ""}, {item_name}]
    # 准备两个队列, confirmed_list, option_list
    confirmed_list = []
    option_list = []

    for query_item, candidates in milvus_search_result.items():
        high_scored = [c["item_name"] for c in candidates if c.get("score", 0.0) >= ITEM_NAME_CONFIRM_THRESHOLD]
        middle_scored = [c["item_name"] for c in candidates if ITEM_NAME_CANDIDATE_THRESHOLD <= c.get("score", 0.0) < ITEM_NAME_CONFIRM_THRESHOLD]

        if high_scored:
            confirmed_list.extend(high_scored[:ITEM_NAME_CONFIRMED_TOP_K])
        elif middle_scored:
            option_list.extend(middle_scored[:ITEM_NAME_OPTIONS_TOP_K])

    return {
        "confirmed": confirmed_list,
        "option": option_list
    }

@step_log("step_6_change_state_property")
def step_6_change_state_property(state: QueryGraphState, rewritten_query: str, confirmed_and_option: Dict[str, List[str]]) -> None:
    confirmed_list = confirmed_and_option.get("confirmed", [])
    option_list = confirmed_and_option.get("option", [])

    if confirmed_list:
        # 场景 1: 确认成功
        state["item_names"] = confirmed_list
        state["rewritten_query"] = rewritten_query
        logger.info(
            f"主体确认成功，已更新 state['item_names']: {confirmed_list}, state['rewritten_query']: {rewritten_query}")
        return

    if option_list:
        # 场景 2: 候选反问（当用户提问存在多个相近金融产品/机构时）
        options_text = "、".join(option_list)
        state["answer"] = f"请问您指的是以下哪项金融产品或资料呢：{options_text}？请明确后再次提问。"
        state["rewritten_query"] = rewritten_query
        logger.info(f"金融主体存在歧义，已更新 state['answer'] 为候选反问: {options_text}")
        return

    # 场景 3: 未锁定单一具体金融产品（如通用概念、法规政策、投教FAQ、宏观分析或跨产品泛问）
    # 核心设计：不设 answer，保持 item_names 为空，放行进入全量知识库切片混合检索与联网搜索
    state["item_names"] = []
    state["rewritten_query"] = rewritten_query
    logger.info(f"未锁定特定金融产品实体，放行至全知识库向量检索与外网搜索: rewritten_query='{rewritten_query}'")

def step_7_save_user_chat_message(state: QueryGraphState) -> None:
    """
    保存用户当前提问到 MongoDB 历史对话集合
    """
    save_chat_message(
        session_id=state.get("session_id"),
        role="user",
        text=state.get("original_query"),
        rewritten_query=state.get("rewritten_query", ""),
        item_names=state.get("item_names", []),
        image_urls=[]
    )
    logger.info(f"用户提问消息已持久化至 MongoDB (session_id: {state.get('session_id')})")


if __name__ == "__main__":
    # 测试案例：模拟用户提问“万用表怎么测电阻”
    test_state = {
        "session_id": "test_confirm_node_01",
        "original_query": "RS PRO RS-12数字万用表",
        "is_stream": False,
    }

    print("\n========== 开始测试 node_item_name_confirm 完整流程 ==========")
    final_output = node_item_name_confirm(test_state)
    print("\n节点最终输出增量字典:", json.dumps(final_output, ensure_ascii=False, indent=2))
    print("============================================================\n")