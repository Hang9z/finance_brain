import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.task_utils import add_running_task, add_done_task


# ====================== 全局切分配置常量 ======================
# 文本切块最大长度：单个文本块硬上限为 1000 字符（防止过长导致向量失真）
CHUNK_MAX_SIZE = 1000
# 文本切块基准长度：单个文本块理想目标为 600 字符（兼顾语义独立性与检索命中率）
CHUNK_SIZE = 600
# 文本块重叠长度：相邻块之间重叠 50 字符，保证语义交界处不丢失上下文
CHUNK_OVERLAP = 50
# 最小碎片阈值：低于 400 字符判定为短碎片，需触发同父标题下的合并机制
CHUNK_MIN = 400


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    业务定位: 将清洗与图片增强后的长篇 Markdown，切分为适配向量检索的微观语义单元 (Chunk)。
    核心流程: 先按标题层级做粗粒度语义切分，再对超长块递归细切、过短碎片按同父级智能合并。
    """
    # 1. 任务状态追踪
    if state.get("task_id"):
        add_running_task(state["task_id"], "node_document_split")

    # 2. 步骤一: 获取并校验参数与 Markdown 文本
    md_content, file_title, md_path = step_1_validate_get_data(state)

    # 3. 步骤二: 依据 Markdown 标题语法 (#) 进行粗粒度语义分层切分
    chunks: List[Dict[str, Any]] = step_2_split_document_by_title(md_content, file_title)

    # 4. 步骤三: 针对超长块进行递归细切 (>600)，针对短碎片进行同级合并 (<400)
    chunks = step_3_refine_split_and_merge_chunks(chunks)

    # 5. 步骤四: 补齐未切分短块的 parent_title 与 part 元数据
    step_4_padding_chunks_metadata(chunks)

    # 6. 步骤五: 将 Chunks 序列化备份为同名 JSON 文件，便于审计与调试
    step_5_backup_chunks_json(chunks, md_path)

    # 7. 更新全局状态并标记任务完成
    state["chunks"] = chunks
    if state.get("task_id"):
        add_done_task(state["task_id"], "node_document_split")

    return state

@step_log("step_1_validate_get_data")
def step_1_validate_get_data(state: ImportGraphState) -> Tuple[str, str, str]:
    """
    步骤一: 获取并校验参数, 兼容空值容错并统一换行符
    :param state: 全局图状态
    :return: 清洗后的 Markdown 文本, 文件主标题, Markdown 文件绝对路径
    """
    # 1. 从 State 中提取参数
    md_content: str = state.get("md_content")
    file_title: str = state.get("file_title")
    md_path: str = state.get("md_path")

    # 2. 若 md_content 为空, 尝试从磁盘 md_path 恢复读取
    if not md_content:
        if not md_path or not Path(md_path).is_file():
            logger.error("[node_document_split] md_content 与 md_path 均无效, 无法进行切分")
            raise ValueError("[node_document_split] 缺少有效的 Markdown 内容或文件路径")
        md_content = Path(md_path).read_text(encoding="utf-8")
        state["md_content"] = md_content

    # 3. 若文件名为空, 兜底取 md_path 的 stem
    if not file_title:
        file_title = Path(md_path).stem if md_path else "default"
        logger.warning(f"[node_document_split] file_title 为空, 使用兜底文件名: {file_title}")
        state["file_title"] = file_title

    # 4. 跨平台文本换行符清洗: 将 Windows CR-LF (\r\n) 和 Mac CR (\r) 替换为标准 LF (\n)
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    return md_content, file_title, md_path

@step_log("step_2_split_document_by_title")
def step_2_split_document_by_title(md_content: str, file_title: str) -> List[Dict[str, Any]]:
    """
    步骤二: 基于 Markdown 标题语法 (#) 进行粗粒度语义分层切分
    1. 代码块保护(is_code): 在```区域内, 忽略所有 # 开头的行
    2. 多级标题层级栈(heading_stack)
    3. 孤儿行累积(orphan_lines)
    :param md_content:
    :param file_title:
    :return:
    """
    chunks: List[Dict[str, Any]] = []

    # 1. 当前激活的完整嵌套标题 (例如: "#控制与端口_##面板图示")
    current_title: Optional[str] = None

    # 2. 当前标题下的正文缓存
    current_content_lines: List[str] = []

    # 3. 孤儿数据缓存
    orphan_lines: List[str] = []

    # 4. 标记位: 孤儿行是否被加入首个 chunk
    has_flushed_first_chunk: bool = False

    # 5. 状态机标记: 当前行是否处于代码块内部
    is_code: bool = False

    # 6. 多级标题栈
    heading_stack: List[Optional[str]] = []

    # 7. 正则表达式
    title_reg = re.compile(r"^#{1,6}\s+.+")

    # 8. 按行扫描
    document_lines = md_content.splitlines()

    for line in document_lines:
        line_strip = line.strip()

        # 跳过纯空行
        if not line_strip:
            continue

        # 1. 遇到代码块围栏表示, 反转代码块状态
        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            is_code = not is_code
            if not current_title:
                orphan_lines.append(line_strip)
            else:
                current_content_lines.append(line_strip)
            continue

        # 2. 处于代码块内部时, 不论是否包含 #, 一律视作普通代码文本
        if is_code:
            if not current_title:
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)
            continue

        # 3. 标题行识别与多级切分
        if title_reg.match(line_strip):
            # 计算标题级别: # 数量即为 level (1-6)
            heading_level = len(line_strip) - len(line_strip.lstrip("#"))

            # 如果前面已有激活标题且有内容, 先结算上一个 chunk
            if current_title and len(current_content_lines) > 0:
                # 若孤儿数据未合并, 优先将前言吸附至首个 chunk
                if not has_flushed_first_chunk and orphan_lines:
                    current_content_lines = orphan_lines + current_content_lines
                    has_flushed_first_chunk = True
                    orphan_lines  = []

                full_content = f"{current_title}\n" + "\n".join(current_content_lines)
                chunks.append({
                    "title": current_title,
                    "content": full_content,
                    "file_title": file_title
                })
                # 清空正文缓冲区, 准备接收新标题下的内容
                current_content_lines = []

            # 维护多级标题栈
            while len(heading_stack) < heading_level:
                heading_stack.append(None)
            # 截断当前级别
            heading_stack = heading_stack[:heading_level]
            heading_stack[heading_level - 1] = line_strip

            # 拼接当前标题层级路径
            current_title = "_".join([h for h in heading_stack if h])
            continue
        else:
            # 4. 正文归档
            if not current_title:
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)

    # 5. 循环结束后的尾部收尾处理
    if current_title and (len(current_content_lines) > 0 or (not has_flushed_first_chunk and len(orphan_lines) > 0)):
        if not has_flushed_first_chunk and len(orphan_lines) > 0:
            current_content_lines = orphan_lines + current_content_lines
            has_flushed_first_chunk = True
            orphan_lines = []
        full_content = f"{current_title}\n" + "\n".join(current_content_lines)
        chunks.append({
            "title": current_title,
            "content": full_content,
            "file_title": file_title
        })
    elif not current_title and len(orphan_lines) > 0:
        # 极端场景: 全文无任何 # 标题
        chunks.append({
            "title": current_title,
            "content": "\n".join(orphan_lines),
            "file_title": file_title
        })

    logger.info(f"[step_2_split_document_by_title] 标题多级粗分完成, 共生成 {len(chunks)} 个粗分语义块")
    return chunks

@step_log("_split_chunk_content")
def _split_chunk_content(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    内部辅助函数: 对超长 chunk (>600字符) 进行递归切分
    核心保护: 切分前剥离标题前缀, 切分后为每个子切片重新注入标题前缀
    :param chunk:
    :return:
    """
    sub_chunks: List[Dict[str, Any]] = []
    content: str = chunk.get("content", "")
    title: str = chunk.get("title", "")
    prefix = title + "\n"

    # 1. 剥离正文开头的标题行, 避免标题参与纯文本字符切分
    deal_content = content[len(prefix):] if content.startswith(prefix) else content

    # 2. 递归字符且分期: 切分目标扣除前缀长度
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - len(prefix),
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    )

    # 3. 逐块封装并补齐上下文前缀
    split_texts = splitter.split_text(deal_content)
    for index, text in enumerate(split_texts, start=1):
        sub_chunks.append({
            "file_title": chunk.get("file_title"),
            "parent_title": title,
            "title": f"{title}_{index}",
            "part": index,
            "content": prefix + text
        })

    return sub_chunks

@step_log("_merge_chunk_content")
def _merge_chunk_content(refine_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
     内部辅助函数: 合并同一 parent_title 下小于 400 字符的短碎片
     合并限制:
     1. 必须属于同一父标题 (is_same_parent)
     2. base 块长度 <= CHUNK_MIN (400)
     3. 合并后总长度 <= CHUNK_MAX_SIZE (1000)
    :param refine_chunks:
    :return:
    """
    merge_refine_chunks: List[Dict[str, Any]] = []
    if not refine_chunks:
        return merge_refine_chunks

    # 基准块, 从第 0 个开始
    base_chunk: Optional[Dict[str, Any]] = None

    for next_chunk in refine_chunks:
        if base_chunk is None:
            base_chunk = next_chunk.copy()
            continue

        base_content = base_chunk.get("content", "")
        next_content = next_chunk.get("content", "")
        parent_title = base_chunk.get("parent_title")

        # 判断是否属于同一父标题 (排除 None 的情况)
        is_same_parent = bool(parent_title) and (parent_title == next_chunk.get("parent_title"))
        is_base_short = len(base_content) <= CHUNK_MIN

        if is_same_parent and is_base_short:
            # 剥离 next_chunk 开头的 parent_title 前缀, 防止合并后标题重复
            prefix = parent_title + "\n"
            next_cleared_content = next_content[len(prefix):] if next_content.startswith(prefix) else next_content

            # 校验合并后是否超限 (1000 字符)
            if len(base_content) + len(next_cleared_content) + 1 <= CHUNK_MAX_SIZE:
                base_chunk["content"] = f"{base_content}\n{next_cleared_content}"
                continue

        # 无法合并, 将当前 base_chunk 归档, 并将 next_chunk 设为新的基准块
        merge_refine_chunks.append(base_chunk)
        base_chunk = next_chunk.copy()

    # 循环结束后收尾最后一个 base_chunk
    if base_chunk:
        merge_refine_chunks.append(base_chunk)

    logger.info(f"[_merge_chunk_content] 完成短碎片合并，最终块数量: {len(merge_refine_chunks)}")
    return merge_refine_chunks

@step_log("step_3_refine_split_and_merge_chunks")
def step_3_refine_split_and_merge_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    步骤三: 对粗切后的 Chunks 进行精细化二次处理 (超长切割 + 短碎片合并)
    :param chunks:
    :return:
    """
    refine_chunks: List[Dict[str, Any]] = []

    # 1. 扫描所有块, 超过 CHUNK_SIZE (600) 的触发二次切分
    for chunk in chunks:
        if len(chunk.get("content", "")) > CHUNK_SIZE:
            refine_chunks.extend(_split_chunk_content(chunk))
        else:
            refine_chunks.append(chunk)

    logger.info(f"[step_3_refine_split_and_merge_chunks] 超长细切后块数量: {len(refine_chunks)}")
    
    # 2. 对切分后的结果执行同父标题下的短碎片合并
    final_chunks = _merge_chunk_content(refine_chunks)
    return final_chunks

@step_log("step_4_padding_chunks_metadata")
def step_4_padding_chunks_metadata(chunks: List[Dict[str, Any]]) -> None:
    """
    步骤四: 补齐未切分短块的元数据 (parent_title 与 part)
    对于未触发超长细切的独立块，补全 parent_title 为自身的 title，part 设为 1
    :param chunks: 处理后的切块列表
    :return:
    """
    for chunk in chunks:
        if "parent_title" not in chunk:
            chunk["parent_title"] = chunk.get("title")
        if "part" not in chunk:
            chunk["part"] = 1
    logger.info(f"[step_4_padding_chunks_metadata] 完成 chunks 元数据补齐, 共 {len(chunks)} 个 Chunk")

@step_log("step_5_backup_chunks_json")
def step_5_backup_chunks_json(chunks: List[Dict[str, Any]], md_path: str) -> None:
    """
    步骤五: 将切分后的 chunks 列表持久化备份为 JSON 文件，便于审计与调试
    文件路径: output/<文档目录>/<文档名>.json (例如 万用表RS-12的使用_new.json)
    :param chunks: 切分后的 Chunk 列表
    :param md_path: 对应的 Markdown 文件路径
    """
    if not md_path:
        logger.warning("[step_5_backup_chunks_json] md_path 为空, 跳过磁盘备份")
        return
    md_path_obj = Path(md_path)
    json_path_obj = md_path_obj.parent / f"{md_path_obj.stem}.json"
    json_path_obj.write_text(data=json.dumps(chunks, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info(f"[step_5_backup_chunks_json] 完成 chunks 磁盘备份: {json_path_obj}")


if __name__ == '__main__':
    """
    文档切分节点 (node_document_split) 本地单元测试
    基于前序节点 (node_md_img) 产出的实际增强 Markdown 文件进行切分验证
    """
    from utils.path_util import PROJECT_ROOT
    # 1. 指定测试用的 Markdown 文件路径
    test_md_path = PROJECT_ROOT / "output" / "万用表RS-12的使用" / "万用表RS-12的使用_new.md"
    if not test_md_path.exists():
        logger.error(f"测试文件不存在: {test_md_path}")
    else:
        logger.info(f"=== 开始运行 node_document_split 单元测试 ===")
        logger.info(f"输入测试文件: {test_md_path}")
        # 2. 构造模拟 State
        test_state: ImportGraphState = {
            "task_id": "test_split_task_001",
            "md_path": str(test_md_path),
            "file_title": "万用表RS-12的使用",
            "md_content": "",
        }
        # 3. 执行切分节点
        result_state = node_document_split(test_state)
        chunks = result_state.get("chunks", [])
        # 4. 打印切分统计与抽样预览
        logger.info(f" 切分完成！共生成 {len(chunks)} 个微观 Chunk 切片")
        for idx, chunk in enumerate(chunks[:3], start=1):
            print(f"\n--- [Chunk {idx}] ---")
            print(f"file_title  : {chunk.get('file_title')}")
            print(f"parent_title: {chunk.get('parent_title')}")
            print(f"title       : {chunk.get('title')}")
            print(f"part        : {chunk.get('part')}")
            print(f"length      : {len(chunk.get('content', ''))} 字符")
            print(f"content     :\n{chunk.get('content')[:120]}...\n")
        if len(chunks) > 3:
            print(f"... (中间省略 {len(chunks) - 4} 个 chunks) ...\n")
            last_chunk = chunks[-1]
            print(f"--- [Chunk {len(chunks)}] ---")
            print(f"file_title  : {last_chunk.get('file_title')}")
            print(f"parent_title: {last_chunk.get('parent_title')}")
            print(f"title       : {last_chunk.get('title')}")
            print(f"part        : {last_chunk.get('part')}")
            print(f"length      : {len(last_chunk.get('content', ''))} 字符")
            print(f"content     :\n{last_chunk.get('content')[:120]}...\n")
        # 5. 校验 JSON 备份文件
        json_backup_path = test_md_path.parent / f"{test_md_path.stem}.json"
        if json_backup_path.exists():
            logger.info(f" 备份 JSON 文件成功生成: {json_backup_path}")
        else:
            logger.warning(f" 备份 JSON 文件未找到: {json_backup_path}")