from pathlib import Path
import sys
from typing import Any, Dict, List

from torchgen.gen_functionalization_type import return_from_mutable_noop_redispatch

# 确保脚本独立运行可正确识别项目根目录
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pymilvus import DataType
from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.task_utils import add_running_task, add_done_task

# 从配置中读取切片集合名称 (kb_chunks)
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    业务定位: 将全量携带向量与元数据的 Chunks 批量持久化写入 Milvus 切片集合中。
    核心流程:
    1. 获取并校验已向量化的切片数据；
    2. 确保 Milvus 中 kb_chunks 集合与双路索引就绪；
    3. 按 file_title 幂等清理旧切片并批量插入新切片。
    """
    # 1. 任务状态跟踪
    if state.get("task_id"):
        add_running_task(state["task_id"], "node_import_milvus")

    # 2. 步骤一: 获取并校验向量切片数据
    embeddings_content = step_1_validate_and_get_data(state)

    # 3. 步骤二: 检查并准备 Milvus 切片集合结构以及索引
    step_2_prepared_chunks_collection()

    # 4. 步骤三: 幂等清理旧记录并批量插入新切片
    step_3_insert_chunks_data(embeddings_content)

    file_title = embeddings_content[0].get("file_title", "未知文档")
    logger.info(f"[node_import_milvus] 文档【{file_title}】全量切片成功导入 Milvus，总记录数: {len(embeddings_content)}")

    if state.get("task_id"):
        add_done_task(state["task_id"], "node_import_milvus")

    return state

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: ImportGraphState) -> List[Dict[str, Any]]:
    """
    步骤一: 获取并校验向量切片数据
    :param state: 全局图状态
    :return: 向量切片列表 embeddings_content
    """
    # 1. 优先读取已完成向量化的切片集合
    embeddings_content: List[Dict[str, Any]] = state.get("embeddings_content")

    # 2. 容错防卫: 若 embeddings_content 缺失, 回退读取 chunks
    if not embeddings_content:
        embeddings_content = state.get("chunks")

    if not embeddings_content:
        logger.error("[node_import_milvus] state 中缺失有效切片数据 (embeddings_content 为空)")
        raise ValueError("[node_import_milvus] 缺少有效的切片数据，无法执行 Milvus 入库")

    # 3. 校验切片是否携带必要的向量字段
    first_chunk = embeddings_content[0]
    if "dense_vector" not in first_chunk or "sparse_vector" not in first_chunk:
        logger.error("[node_import_milvus] 切片数据中缺失向量字段 (dense_vector 或 sparse_vector)")
        raise ValueError("[node_import_milvus] 切片数据未完成向量化，无法入库")

    return embeddings_content

@step_log("step_2_prepared_chunks_collection")
def step_2_prepared_chunks_collection() -> None:
    """
    步骤二: 检查并创建 Milvus 切片核心集合 (kb_chunks) 及其双路索引
    集合结构:
      - chunk_id (主键, INT64, 自增)
      - content (切片正文, VARCHAR, 最大 65535 字符)
      - file_title, item_name, title, parent_title (元数据标量, VARCHAR)
      - part (子块序号, INT16)
      - dense_vector (1024 维稠密向量, HNSW 索引, COSINE)
      - sparse_vector (关键词稀疏权重, 倒排索引, IP)
    """
    milvus_client = get_milvus_client()
    collection_name = CHUNKS_COLLECTION_NAME

    # 1. 幂等性校验: 若集合已存在无需重复创建
    if milvus_client.has_collection(collection_name=collection_name):
        logger.info(f"[step_2_prepared_chunks_collection] 集合 [{collection_name}] 已存在, 直接使用")
        return

    logger.info(f"[step_2_prepared_chunks_collection] 集合 [{collection_name}] 不存在，开始构建 Schema 与索引...")

    # 2. 构建 Schema
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="part", datatype=DataType.INT16)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 3. 构建双向量索引参数
    index_params = milvus_client.prepare_index_params()
    # 稠密向量索引 (HNSW + COSINE)
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="chunks_dense_index",
        params={"M": 64, "efConstruction": 100},
        metric_type="COSINE"
    )
    # 稀疏向量索引 (倒排 + IP)
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="chunks_sparse_index",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
        metric_type="IP"
    )

    # 4. 创建集合
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

@step_log("step_3_insert_chunks_data")
def step_3_insert_chunks_data(embeddings_content: List[Dict[str, Any]]) -> None:
    """
    步骤三: 依据 file_title 幂等清理旧切片并批量插入最新切片
    :param embeddings_content: 已完成向量化的全量切片列表
    """
    if not embeddings_content:
        return

    file_title: str = embeddings_content[0].get("file_title")
    milvus_client = get_milvus_client()
    collection_name = CHUNKS_COLLECTION_NAME

    # 1. 幂等性清理: 依据 file_title 物理删除历史同名文档切片 (注意: 过滤条件等于为 ==)
    if file_title:
        milvus_client.delete(
            collection_name=collection_name,
            filter=f"file_title == '{file_title}'"
        )
        logger.info(f"[step_3_insert_chunks_data] 已幂等清理文档【{file_title}】在集合 [{collection_name}] 中的历史切片")

    # 2. 批量插入最新全量切片数据
    milvus_client.insert(
        collection_name=collection_name,
        data=embeddings_content
    )
    logger.info(f"[step_3_insert_chunks_data] 成功将 {len(embeddings_content)} 条切片记录存入 Milvus [{collection_name}]")


if __name__ == '__main__':
    """
    向量库导入节点 (node_import_milvus) 本地单元测试
    """
    dim = 1024
    # 构造轻量测试 State (模拟上游节点输出的高维双向量切片)
    test_state: ImportGraphState = {
        "task_id": "test_import_milvus_task_001",
        "file_title": "万用表RS-12的使用",
        "embeddings_content": [
            {
                "content": "## 安全手册\n使用说明书 RS-12 数字万用表",
                "file_title": "万用表RS-12的使用",
                "item_name": "RS-12数字万用表",
                "title": "## 安全手册_1",
                "parent_title": "## 安全手册",
                "part": 1,
                "dense_vector": [0.01] * dim,
                "sparse_vector": {1173: 0.15, 3895: 0.2, 5873: 0.3},
            },
            {
                "content": "## 控制与端口\n1.LCD液晶显示 2.功能转盘 3.10A端口",
                "file_title": "万用表RS-12的使用",
                "item_name": "RS-12数字万用表",
                "title": "## 控制与端口",
                "parent_title": "## 控制与端口",
                "part": 1,
                "dense_vector": [0.02] * dim,
                "sparse_vector": {1173: 0.12, 4012: 0.25},
            }
        ]
    }
    logger.info("=== 开始运行 node_import_milvus 单元测试 ===")
    result_state = node_import_milvus(test_state)
    print("\n" + "=" * 50)
    print(" Milvus 向量切片入库单元测试成功！数据已成功持久化。")
    print("=" * 50 + "\n")