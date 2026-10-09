import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType
from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

# 上下文切片数量太多,取前k个切片用于主体识别
ITEM_NAME_CONTEXT_CHUNK_K = 5
# 控制上下文的字符数量,防止 Prompt 太长
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 2000


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    业务定位: 识别文档核心描述的物品/商品名称 (Item Name)，并完成向量化注册。
    """
    # 1. 任务状态追踪
    if state.get("task_id"):
        add_running_task(state["task_id"], "node_item_name_recognition")

    # 2. 步骤一: 校验参数 chunks 与 file_title
    chunks, file_title = step_1_validate_and_get_data(state)

    # 3. 步骤二: 调用大模型提取核心主体名称 item_name
    item_name: str = step_2_call_llm_return_item_name(chunks, file_title)

    # 4. 步骤三: 回填 item_name 到所有切片数据中
    step_3_padding_item_name_to_chunks(chunks, item_name)

    # 5. 步骤四: 检查并创建存储 item_name 的 Milvus 集合及索引
    step_4_prepared_item_name_collection()

    # 6. 步骤五: 生成混合向量，幂等插入 Milvus 集合
    step_5_insert_item_name_data(item_name, file_title)

    # 7. 更新全局状态并标记任务完成
    state["item_name"] = item_name
    state["chunks"] = chunks
    if state.get("task_id"):
        add_done_task(state["task_id"], "node_item_name_recognition")

    return state

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: ImportGraphState) -> Tuple[List[Dict[str, Any]], str]:
    """
    步骤一: 校验参数 chunks 与 file_title
    chunks 是列表, 格式是字典, 如果 chunks 为空, 从上一步生成的.json文件中恢复
    :param state: 全局状态
    :return: (切片列表 chunks, 文档标题 file_title)
    """
    chunks: List[Dict[str, Any]] = state.get("chunks")
    file_title: str = state.get("file_title")
    md_path: str = state.get("md_path")

    # 1. 如果 chunks 为空, 从磁盘中恢复
    if not chunks:
        if not md_path or not Path(md_path).is_file():
            logger.error("[node_item_name_recognition] state 中缺少 chunks 且 md_path 无效")
            raise ValueError("[node_item_name_recognition] 缺少有效的 chunks 数据或 md_path 路径")

        md_path_obj = Path(md_path)
        json_path_obj = md_path_obj.parent / f"{md_path_obj.stem}.json"
        if json_path_obj.is_file():
            chunks = json.loads(json_path_obj.read_text(encoding="utf-8"))
            state["chunks"] = chunks
            logger.warning(f"[node_item_name_recognition] 内存中 chunks 为空，已从磁盘备份恢复: {json_path_obj}")
        else:
            logger.error(f"[node_item_name_recognition] chunks 为空且未找到磁盘备份: {json_path_obj}")
            raise ValueError(f"[node_item_name_recognition] 未找到切片数据或备份文件: {json_path_obj}")

        # 2. 如果没有 file_title
        if not file_title:
            file_title = md_path_obj.stem if md_path else "default_title"
            state["file_title"] = file_title
            logger.warning(f"[node_item_name_recognition] file_title 为空，使用兜底标题: {file_title}")

    return chunks, file_title

@step_log("step_2_call_llm_return_item_name")
def step_2_call_llm_return_item_name(chunks: List[Dict[str, Any]], file_title: str) -> str:
    """

    :param chunks:
    :param file_title:
    :return:
    """
    llm_model = get_llm_client()

    # 先获取前 k 个 chunk 中的标题和内容并进行拼接
    context = ""
    for chunk in chunks[:ITEM_NAME_CONTEXT_CHUNK_K]:
        context += f"标题: {chunk.get("parent_title")}, 内容: {chunk.get("content")}"
    # 超过最大字数限制的部分截断
    context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]

    # 导入本地的 prompt 模板
    prompt_text: str = load_prompt("item_name_recognition", file_title=file_title, context=context)
    message = HumanMessage(content=prompt_text)

    # 链式调用
    chain = llm_model | StrOutputParser()
    item_name = ""
    for attempt in range(1, 4):
        try:
            item_name = chain.invoke([message])
            break
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "RateLimit" in err_msg:
                wait_s = 20 * attempt
                logger.warning(f"[node_item_name_recognition] 触发 LLM 频率限制 (429)，等待 {wait_s} 秒后重试 (尝试 {attempt}/3)...")
                time.sleep(wait_s)
            else:
                logger.error(f"[node_item_name_recognition] LLM 调用失败: {e}")
                break

    # 清洗与兜底校验: 若识别为空, 兜底使用file_title
    item_name = item_name.strip().strip('"').strip("'") if item_name else ""
    if not item_name:
        item_name = file_title
        logger.warning(f"[node_item_name_recognition] LLM 未能识别出主体，兜底使用 file_title: {item_name}")
    else:
        logger.info(f"[node_item_name_recognition] LLM 成功识别主体 item_name: 【{item_name}】")

    return item_name

@step_log("step_3_padding_item_name_to_chunks")
def step_3_padding_item_name_to_chunks(chunks: List[Dict[str, Any]], item_name: str) -> None:
    """
    步骤三: 将提取出的 item_name 回填补充到所有切片字典中
    建立文档切片与业务实体的强绑定关系，便于后续向量化与元数据过滤
    :param chunks: 切片列表
    :param item_name: 产品/设备主体名称
    """
    for chunk in chunks:
        chunk["item_name"] = item_name
    logger.info(f"[step_3_padding_item_name_to_chunks] 已将 item_name=【{item_name}】回填至全量 {len(chunks)} 个 Chunk 中")

@step_log("step_4_prepared_item_name_collection")
def step_4_prepared_item_name_collection():
    """
    步骤四: 检查并创建存储 item_name 业务实体的 Milvus 集合以及对应索引
    集合名称: milvus_config.item_name_collection (kb_item_names)
    pk：主键，INT64 自增（auto_id=True）；
    file_title：文件名标识（便于后续关联与按文档幂等清理）；
    item_name：标量字符串（如 "RS-12数字万用表"）；
    dense_vector：稠密向量（维度 dim=1024，对齐 BGE-M3）；
    sparse_vector：稀疏向量（SPARSE_FLOAT_VECTOR，存储关键字特征权重）。
    """
    # 获取 milvus 客户端
    milvus_client = get_milvus_client()
    collection_name = milvus_config.item_name_collection

    # 1. 幂等性校验: 若集合已存在则直接返回复用
    if milvus_client.has_collection(collection_name=collection_name):
        logger.info(f"[step_4_prepared_item_name_collection] 集合 [{collection_name}] 已存在，直接使用")
        return
    logger.info(f"[step_4_prepared_item_name_collection] 集合 [{collection_name}] 不存在，开始构建 Schema 与索引...")

    # 建表
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 建立双向量索引
    index_params = milvus_client.prepare_index_params()
    # 稠密向量索引 (HNSW + COSINE)
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        params={"M": 64, "efConstruction": 100},
        metric_type="COSINE"
    )
    # 稀疏向量索引 (SPARSE_INVERTED_INDEX + IP)
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
        metric_type="IP"
    )

    # 创建集合
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    logger.info(f"[step_4_prepared_item_name_collection] 集合 [{collection_name}] 及其双路索引创建成功")

@step_log("step_5_insert_item_name_data")
def step_5_insert_item_name_data(item_name: str, file_title: str) -> None:
    """
    步骤五: 为 item_name 生成混合向量并在 Milvus 中幂等入库
    1. 调用 BGE-M3 生成稠密向量 (dense) 与稀疏向量 (sparse)
    2. 按 file_title 幂等清理旧记录
    3. 插入最新向量数据
    """
    # 1. 生成混合向量嵌入 (BGE-M3: 稠密 1024 维 + 稀疏关键词权重字典)
    result = generate_embeddings([item_name])
    dense_vector = result["dense"][0]
    sparse_vector = result["sparse"][0]

    milvus_client = get_milvus_client()
    collection_name = milvus_config.item_name_collection

    # 2. 幂等性清理: 依据 file_title 删除历史旧数据 (注意: Milvus 过滤表达式等于为 ==)
    milvus_client.delete(
        collection_name=collection_name,
        filter=f"file_title == '{file_title}'"
    )

    # 3. 插入实体向量记录
    milvus_client.insert(
        collection_name=collection_name,
        data=[
            {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
        ]
    )
    logger.info(f"[step_5_insert_item_name_data] 成功将 item_name=【{item_name}】向量数据存入 Milvus [{collection_name}]")


if __name__ == '__main__':
    """
    主体识别节点 (node_item_name_recognition) 本地单元测试
    """
    from utils.path_util import PROJECT_ROOT
    # 1. 指定测试用的 Markdown 文件路径
    test_md_path = PROJECT_ROOT / "output" / "万用表RS-12的使用" / "万用表RS-12的使用_new.md"
    if not test_md_path.exists():
        logger.error(f"测试文件不存在: {test_md_path}")
    else:
        logger.info("=== 开始运行 node_item_name_recognition 单元测试 ===")
        # 2. 构造模拟 State (将 chunks 设为空，验证 step_1 从 _new.json 自动恢复的能力)
        test_state: ImportGraphState = {
            "task_id": "test_item_name_task_001",
            "md_path": str(test_md_path),
            "file_title": "万用表RS-12的使用",
            "chunks": [],
        }
        # 3. 执行主体识别节点
        result_state = node_item_name_recognition(test_state)
        # 4. 打印验证结果
        print("\n" + "=" * 50)
        print(f" 主体识别成功: item_name = 【{result_state.get('item_name')}】")
        print(f" 切片元数据回填验证 (抽样前 3 个 Chunk):")
        for i, chunk in enumerate(result_state.get("chunks", [])[:3], 1):
            print(f"   [{i}] title: {chunk.get('title')}, item_name: {chunk.get('item_name')}")
        print("=" * 50 + "\n")
