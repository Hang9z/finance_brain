import sys
from pathlib import Path

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.path_util import PROJECT_ROOT
from utils.task_utils import add_running_task, add_done_task

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    核心职责:
    1. 任务进度追踪: 上报节点运行状态至前端 SSE
    2. 接收文件路径，兼容相对/绝对路径并校验实体文件存在性
    3. 提取文档标题 (file_title) 供下游商品识别使用
    4. 判断文件类型 (PDF/MD)，互斥设置路由开关与目标路径
    5. 异常场景软退出，平滑配合 LangGraph 路由至 END
    """
    # 0. 任务状态追踪: 记录当前节点正在运行
    if state.get("task_id"):
        add_running_task(state["task_id"], "node_entry")

    # 1. 从上游接收文件路径
    raw_path_str = state.get("local_file_path", "").strip()

    # 2. 判空防守：空路径软退出
    if not raw_path_str:
        logger.warning("[node_entry] 文件路径为空, 直接终止当前导入流程")
        return state

    # 3. 兼容相对路径与绝对路径
    file_path = Path(raw_path_str)  # 包装为 Path 对象
    if not file_path.is_absolute():
        file_path = (PROJECT_ROOT / file_path).resolve()
    else:
        file_path = file_path.resolve()

    # 4. 判断物理文件是否存在：不存在则软退出
    if not file_path.exists() or not file_path.is_file():
        logger.warning(f"[node_entry] 待处理文件不存在: {file_path}, 直接终止当前导入流程")
        return state

    # 5. 更新为系统规范绝对路径
    state["local_file_path"] = str(file_path)

    # 6. 提取文件标题 (去除后缀), 存入 state
    file_title = file_path.stem
    state["file_title"] = file_title
    logger.info(f"[node_entry] 识别文件标题: {file_title}")

    # 7. 根据文件后缀判断类型并进行路由标记与互斥赋值
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        state["is_pdf_read_enabled"] = True
        state["is_md_read_enabled"] = False
        state["is_doc_read_enabled"] = False
        state["pdf_path"] = str(file_path)
        state["doc_path"] = None
        state["md_path"] = None
        logger.info(f"[node_entry] 检测到 PDF 文件, 已激活 PDF 处理流程: {file_path.name}")
    elif suffix in (".docx", ".doc"):
        state["is_doc_read_enabled"] = True
        state["is_pdf_read_enabled"] = False
        state["is_md_read_enabled"] = False
        state["doc_path"] = str(file_path)
        state["pdf_path"] = None
        state["md_path"] = None
        logger.info(f"[node_entry] 检测到 Word 文档 ({suffix}), 已激活 Word 转换流程: {file_path.name}")
    elif suffix == ".md":
        state["is_md_read_enabled"] = True
        state["is_pdf_read_enabled"] = False
        state["is_doc_read_enabled"] = False
        state["md_path"] = str(file_path)
        state["pdf_path"] = None
        state["doc_path"] = None
        logger.info(f"[node_entry] 检测到 Markdown 文件, 已激活 Markdown 处理流程: {file_path.name}")
    else:
        logger.warning(f"[node_entry] 不支持的文件类型: '{suffix}', 当前支持 .pdf, .docx, .doc, .md 文件, 终止流程")
        return state

    # 8. 任务状态追踪: 记录当前节点已处理完成
    if state.get("task_id"):
        add_done_task(state["task_id"], "node_entry")

    return state


if __name__ == "__main__":
    logger.info("=== 测试 node_entry 节点 ===")

    # 场景 1: 测试合法的 PDF 文件
    test_pdf_state = create_default_state(
        task_id="task_test_pdf_001",
        local_file_path="doc/万用表RS-12的使用.pdf"
    )
    result_pdf = node_entry(test_pdf_state)
    logger.info(f"PDF 测试结果:")
    logger.info(f" - 标题 (file_title): {result_pdf['file_title']}")
    logger.info(f" - 规整路径 (local_file_path): {result_pdf['local_file_path']}")
    logger.info(f" - pdf_path: {result_pdf['pdf_path']}")
    logger.info(f" - md_path: {result_pdf['md_path']}")
    logger.info(f" - is_pdf_read_enabled: {result_pdf['is_pdf_read_enabled']}")
    logger.info(f" - is_md_read_enabled: {result_pdf['is_md_read_enabled']}")

    # 场景 2: 测试不存在的文件 (软退出)
    test_missing_state = create_default_state(
        task_id="task_test_missing_002",
        local_file_path="doc/不存在的文件.pdf"
    )
    result_missing = node_entry(test_missing_state)
    logger.info(f"不存在文件测试结果 -> PDF开关: {result_missing['is_pdf_read_enabled']}, MD开关: {result_missing['is_md_read_enabled']}")
