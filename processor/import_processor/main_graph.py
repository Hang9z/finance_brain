import json
from pathlib import Path
import sys
from dotenv import load_dotenv

# 确保脚本直接运行时正确加载工程根目录模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 核心图组件和内置常量
from langgraph.graph import StateGraph, START, END

# 日志与状态管理对象
from common.logging.logger import logger
from processor.import_processor.state import ImportGraphState, create_default_state

# 引入业务节点
from processor.import_processor.nodes.node_entry import node_entry # 入口节点，判断文件类型
from processor.import_processor.nodes.node_pdf_to_md import node_pdf_to_md # 格式转化
from processor.import_processor.nodes.node_md_img import node_md_img # 图片处理
from processor.import_processor.nodes.node_document_split import node_document_split # 文件切分
from processor.import_processor.nodes.node_item_name_recognition import node_item_name_recognition # 内容摘要
from processor.import_processor.nodes.node_bge_embedding import node_bge_embedding # 向量化
from processor.import_processor.nodes.node_import_milvus import node_import_milvus # 导入向量数据库

# 加载环境变量(读取MinIO/milvus相关服务的连接地址)
load_dotenv()

# 1. 实例化状态图容器
# 每个节点都要维护这个状态的副本,返回字典增量更新状态
workflow = StateGraph(ImportGraphState)

# 2. 挂载 7 个核心流水线节点
# 节点名称是每个节点在图中的唯一标识,用于后续边的添加
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# 3. 设置工作流起点,告诉 LangGraph 启动时先执行哪个节点
workflow.set_entry_point("node_entry")

# 4. 定义入口执行后文件分流函数
def route_after_entry(state: ImportGraphState) -> str:
    """
    根据 node_entry 探测出的文件类型标记，动态分流后续路径：
    - 若为 Markdown 文件：跳过格式转换，直接处理图片与内容
    - 若为 PDF 手册：先流转到 MinerU 节点进行解析与提取
    - 其它无效情况：安全退出流程
    """
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    elif state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    else:
        return END

# 5. 绑定条件边: 把node_entry和决策函数关联起来
workflow.add_conditional_edges(
    # 决策来自哪个节点
    "node_entry",
    # 使用哪个函数判断路径
    route_after_entry,
    # 字典映射,节点返回值实际对应哪个节点
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END
    }
)

# 6. 静态边: 顺序执行
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# 7. 编译工作流,落地执行
# 编译后,外部通过 kb_import_app.invoke(state)即可启动导入流程
kb_import_app = workflow.compile()

# 8. 本地测试执行块
if __name__ == "__main__":
    logger.info("===== 开始测试文档导入完整工作流 (main_graph) =====")

    # 构建测试用初始状态 (测试 PDF 端到端链路)
    test_state = create_default_state(
        task_id="task_demo_main_001",
        local_file_path="doc/万用表RS-12的使用.pdf"
    )

    # 触发状态图执行
    result = kb_import_app.invoke(test_state)

    logger.info("===== 文档导入工作流全拓扑执行成功 =====")
    logger.info(f"  - 任务 ID: {result.get('task_id')}")
    logger.info(f"  - 文档标题 (file_title): {result.get('file_title')}")
    logger.info(f"  - 识别商品实体 (item_name): {result.get('item_name')}")
    logger.info(f"  - 转换规整 MD (md_path): {result.get('md_path')}")
    logger.info(f"  - 增强后正文长度 (字符数): {len(result.get('md_content', ''))}")
    logger.info(f"  - 生成切片总数: {len(result.get('chunks', []))}")
    logger.info(f"  - 向量化与入库数: {len(result.get('embeddings_content', []))}")
    logger.info(f"  - 分流标记: is_pdf={result.get('is_pdf_read_enabled')}, is_md={result.get('is_md_read_enabled')}")