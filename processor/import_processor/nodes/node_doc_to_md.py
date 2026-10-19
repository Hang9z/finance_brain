import os
import re
import sys
from pathlib import Path
from typing import Tuple

# 确保单脚本直接运行时能正确解析根目录模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.path_util import PROJECT_ROOT
from utils.task_utils import add_running_task, add_done_task


@node_log("node_doc_to_md")
def node_doc_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: DOC/DOCX 转 Markdown (node_doc_to_md)
    业务定位: 将金融问答/资料中的 Word 文件 (.docx, .doc) 解析并规整为 Markdown 格式，
    精准保留金融问答对 (Q&A)、层级标题与正文段落，免去云端 OCR 转换成本与格式失真。
    """
    task_id = state.get("task_id", "")
    if task_id:
        add_running_task(task_id, "node_doc_to_md")

    # 1. 步骤一: 校验输入路径并初始化输出目录
    doc_path_obj, local_dir_obj = step_1_validate_and_get_data(state)

    # 2. 步骤二: 解析 Word 内容并结构化为 Markdown
    md_content = step_2_convert_doc_to_markdown(doc_path_obj)

    # 3. 步骤三: 保存 Markdown 文件并更新状态
    final_md_path = step_3_save_markdown_file(md_content, local_dir_obj, doc_path_obj.stem)

    state["md_path"] = str(final_md_path)
    state["md_content"] = md_content

    if task_id:
        add_done_task(task_id, "node_doc_to_md")

    logger.info(f"[node_doc_to_md] Word 文档成功转换为 Markdown: {final_md_path}")
    return state


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: ImportGraphState) -> Tuple[Path, Path]:
    """
    步骤一: 校验路径与目录初始化
    """
    doc_path = state.get("doc_path") or state.get("local_file_path")
    local_dir = state.get("local_dir")

    if not doc_path:
        logger.error("[node_doc_to_md] doc_path 参数为空，无法读取文件")
        raise ValueError("[node_doc_to_md] doc_path 参数为空，无法读取文件")

    if not local_dir:
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)

    doc_path_obj = Path(doc_path)
    local_dir_obj = Path(local_dir)

    if not doc_path_obj.is_file():
        logger.error(f"[node_doc_to_md] 文件不存在: {doc_path_obj}")
        raise ValueError(f"[node_doc_to_md] 文件不存在: {doc_path_obj}")

    local_dir_obj.mkdir(parents=True, exist_ok=True)
    return doc_path_obj, local_dir_obj


@step_log("step_2_convert_doc_to_markdown")
def step_2_convert_doc_to_markdown(doc_path_obj: Path) -> str:
    """
    步骤二: 使用 Word 自动化引擎解析 .doc / .docx 为结构化 Markdown
    """
    import win32com.client
    import pythoncom

    # 确保在多线程或后台任务中正确初始化 COM 环境
    pythoncom.CoInitialize()
    word = None
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = False

        abs_path = str(doc_path_obj.resolve())
        doc = word.Documents.Open(abs_path, ReadOnly=True)
        raw_text = doc.Content.Text
        doc.Close(False)

        # 文本规整化与问答/章节标记强化
        lines = [line.strip() for line in re.split(r"[\r\n\x0b\x0c]+", raw_text) if line.strip()]
        md_lines = []

        # 主标题
        file_stem = doc_path_obj.stem
        md_lines.append(f"# {file_stem}\n")

        # 匹配大写序号、问答序号或章节标题
        heading_pattern = re.compile(
            r"^(第[一二三四五六七八九十\d]+[章节篇]|"
            r"[一二三四五六七八九十]+[、.]|"
            r"Q\d+[:：]|【问】|"
            r"\d+[\.、]\s*[\u4e00-\u9fa5A-Za-z])"
        )

        for line in lines:
            if line == file_stem:
                continue

            # 检测是否为问答题或分级标题
            if heading_pattern.match(line):
                md_lines.append(f"\n## {line}\n")
            elif re.match(r"^(\([一二三四五六七八九十\d]+\)|（[一二三四五六七八九十\d]+）|\d+\))", line):
                md_lines.append(f"\n### {line}\n")
            else:
                md_lines.append(line)

        md_content = "\n\n".join(md_lines)
        return md_content

    except Exception as e:
        logger.error(f"[node_doc_to_md] 解析 Word 文档失败: {e}", exc_info=True)
        raise RuntimeError(f"解析 Word 文档失败: {e}")
    finally:
        if word:
            try:
                word.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()


@step_log("step_3_save_markdown_file")
def step_3_save_markdown_file(md_content: str, local_dir_obj: Path, stem: str) -> Path:
    """
    步骤三: 将解析后的 Markdown 写入磁盘规范路径: output/{stem}/{stem}.md
    """
    doc_out_dir = local_dir_obj / stem
    doc_out_dir.mkdir(parents=True, exist_ok=True)

    final_md_path = doc_out_dir / f"{stem}.md"
    final_md_path.write_text(md_content, encoding="utf-8")
    return final_md_path


if __name__ == "__main__":
    logger.info("=== 测试 node_doc_to_md 节点 ===")
    test_docx = Path(r"D:\尚硅谷笔记\阶段八：尚硅谷大模型技术之掌柜智库项目\尚硅谷大模型项目实战之掌柜智库实战\1.资料\金融\数据\用户FAQ\私募基金投资者教育知识问答.docx")
    if test_docx.is_file():
        from processor.import_processor.state import create_default_state
        test_state = create_default_state(
            task_id="test_doc_001",
            doc_path=str(test_docx),
            local_file_path=str(test_docx),
            local_dir=str(PROJECT_ROOT / "output")
        )
        res = node_doc_to_md(test_state)
        logger.info(f"转换成功, 输出 md 路径: {res['md_path']}, 正文长度: {len(res['md_content'])}")
