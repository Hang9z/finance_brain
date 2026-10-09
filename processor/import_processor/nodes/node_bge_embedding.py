import json
from pathlib import Path
import sys
from typing import Any, Dict, List

# 确保单脚本直接运行时能正确解析根目录模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.lm.embedding_utils import generate_embeddings
from utils.task_utils import add_running_task, add_done_task


# ====================== 向量化配置常量 ======================
# 向量化批次大小：每批处理 5 条切片，兼顾吞吐量与稳定性
EMBEDDING_BATCH_SIZE = 5


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    业务定位: 使用 BGE-M3 模型将全量文本切片转化为稠密 (Dense) + 稀疏 (Sparse) 混合向量。
    核心流程:
    1. 获取并校验切片数据（支持磁盘备份自动恢复）；
    2. 分批次进行实体强化拼接，调用云端 GPU 并发生成双路向量；
    3. 更新 state['embeddings_content'] 与 state['chunks']。
    """
    # 1. 任务状态追踪
    if state.get("task_id"):
        add_running_task(state["task_id"], "node_bge_embedding")

    # 2. 步骤一: 获取并校验切片参数 chunks
    chunks = step_1_validate_and_get_data(state)

    # 3. 步骤二: 分批次为切片生成 Dense 与 Sparse 向量
    step_2_batch_generate_vector(chunks)

    # 4. 更新全局状态并标记任务完成
    state["embeddings_content"] = chunks
    state["chunks"] = chunks
    if state.get("task_id"):
        add_done_task(state["task_id"], "node_bge_embedding")

    return state

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: ImportGraphState) -> List[Dict[str, Any]]:
    """
    步骤一: 获取数据并进行容错校验
    防御性设计: 若内存中 chunks 为空，自动从同名 .json 备份中回退加载恢复
    :param state: 全局图状态
    :return: 切片列表 chunks
    """
    # 1. 从 State 中提取切片数据
    chunks: List[Dict[str, Any]] = state.get("chunks")

    # 2. 健壮性处理: 若内存中 chunks 为空, 从磁盘备份文件恢复
    if not chunks:
        md_path: str = state.get("md_path")
        if not md_path or not Path(md_path).is_file():
            logger.error("[node_bge_embedding] state 中 chunks 为空且 md_path 无效，无法继续")
            raise ValueError("[node_bge_embedding] 缺少有效的 chunks 数据或 md_path 路径")

        md_path_obj = Path(md_path)
        json_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}.json")
        if json_path_obj.is_file():
            chunks = json.loads(json_path_obj.read_text(encoding="utf-8"))
            # 若恢复出的切片缺少 item_name, 则尝试从 state 或文件名补齐
            item_name = state.get("item_name") or md_path_obj.stem
            for chunk in chunks:
                if "item_name" not in chunk:
                    chunk["item_name"] = item_name
            state["chunks"] = chunks
            logger.warning(f"[node_bge_embedding] 内存中 chunks 为空，已从磁盘恢复: {json_path_obj}")
        else:
            logger.error(f"[node_bge_embedding] chunks 为空且未找到对应 json 备份文件: {json_path_obj}")
            raise ValueError(f"[node_bge_embedding] 未找到对应切片备份文件: {json_path_obj}")

    return chunks

@step_log("step_2_batch_generate_vector")
def step_2_batch_generate_vector(chunks: List[Dict[str, Any]]) -> None:
    """
    步骤二: 分批次为切片生成 Dense 与 Sparse 向量，并回填至切片对象中
    核心增强: content 向量化前，自动前置拼接 item_name (如: "RS-12数字万用表_## 安全手册\n...")
    这样能让向量包含明确的产品实体上下文，显著提升 RAG 检索时的实体匹配精度
    :param chunks: 全量切片列表
    """
    total_chunks = len(chunks)
    logger.info(f"[step_2_batch_generate_vector] 开始全量向量化，切片总数: {total_chunks}，批次大小: {EMBEDDING_BATCH_SIZE}")

    # 按批次切分处理
    for start_idx in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
        batch_chunks = chunks[start_idx : start_idx + EMBEDDING_BATCH_SIZE]

        # 实体前缀增强拼接: item_name + "_" + content
        content_list: List[str] = [
            f"{chunk.get('item_name', '')}_{chunk.get('content', '')}"
            for chunk in batch_chunks
        ]

        # 调用向量生成工具（已自动路由至 Google Colab T4 GPU）
        result = generate_embeddings(content_list)
        dense_list = result["dense"]
        sparse_list = result["sparse"]

        # 原地回填到切片字典中
        for idx, chunk in enumerate(batch_chunks):
            chunk["dense_vector"] = dense_list[idx]
            chunk["sparse_vector"] = sparse_list[idx]

    logger.info(f"[step_2_batch_generate_vector] 全量 {total_chunks} 个切片向量化完成！")


if __name__ == '__main__':
    """
    向量化节点 (node_bge_embedding) 本地单元测试
    """
    from utils.path_util import PROJECT_ROOT

    # 1. 指定测试用的 Markdown 文件路径
    test_md_path = PROJECT_ROOT / "output" / "万用表RS-12的使用" / "万用表RS-12的使用_new.md"

    if not test_md_path.exists():
        logger.error(f"测试文件不存在: {test_md_path}")
    else:
        logger.info("=== 开始运行 node_bge_embedding 单元测试 ===")

        # 2. 构造模拟 State (chunks 留空，验证自动从 _new.json 恢复并补齐)
        test_state: ImportGraphState = {
            "task_id": "test_embedding_task_001",
            "md_path": str(test_md_path),
            "file_title": "万用表RS-12的使用",
            "item_name": "RS-12数字万用表",
            "chunks": [],
        }

        # 3. 执行向量化节点
        result_state = node_bge_embedding(test_state)
        res_chunks = result_state.get("embeddings_content", [])

        # 4. 打印验证结果
        print("\n" + "=" * 50)
        print(f" 向量化测试完成！共处理切片数: {len(res_chunks)}")
        if res_chunks:
            sample = res_chunks[0]
            print(f" 抽样第 1 块 [Chunk 1] 向量结构:")
            print(f"   - title        : {sample.get('title')}")
            print(f"   - item_name    : {sample.get('item_name')}")
            print(f"   - dense_vector 维度 : {len(sample.get('dense_vector', []))} (标准为 1024)")
            print(f"   - sparse_vector 样本: {list(sample.get('sparse_vector', {}).items())[:3]}")
        print("=" * 50 + "\n")