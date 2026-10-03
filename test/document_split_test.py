import logging
from typing import Any
from langchain_text_splitter import RecursiveCharacterTextSplitter

# ==========================================
# 1. 配置与模拟日志装饰器
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
CHUNK_MIN = 400
CHUNK_MAX_SIZE = 1000


def step_log(step_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ==========================================
# 2. 核心业务代码 (完全使用我们优化后的版本)
# ==========================================
@step_log("_split_chunk_content")
def _split_chunk_content(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    sub_chunks: list[dict[str, Any]] = []
    content = chunk.get("content")
    deal_content = content[len(chunk.get("title")) + 1:]
    prefix = chunk.get("title") + "\n"

    spliter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - len(prefix),
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    )

    for index, text in enumerate(spliter.split_text(deal_content), start=1):
        sub_chunks.append({
            "file_title": chunk.get("file_title"),
            "parent_title": chunk.get("title"),
            "title": f"{chunk.get('title')}_{index}",
            "part": index,
            "content": prefix + text
        })
    return sub_chunks


@step_log("_merge_chunk_content")
def _merge_chunk_content(refine_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merge_refine_chunks: list[dict[str, Any]] = []
    if not refine_chunks: return merge_refine_chunks

    base_chunk = refine_chunks[0].copy()

    for next_chunk in refine_chunks[1:]:
        base_content = base_chunk.get("content", "")
        next_content = next_chunk.get("content", "")
        parent_title = base_chunk.get("parent_title")

        is_same_parent = bool(parent_title) and (parent_title == next_chunk.get("parent_title"))
        is_base_short = len(base_content) <= CHUNK_MIN

        if is_same_parent and is_base_short:
            next_cleared_content = next_content[len(parent_title) + 1:]
            is_merged_short = (len(base_content) + len(next_cleared_content) + 1) <= CHUNK_MAX_SIZE

            if is_merged_short:
                base_chunk['content'] = f"{base_content}\n{next_cleared_content}"
                continue

        merge_refine_chunks.append(base_chunk)
        base_chunk = next_chunk.copy()

    if base_chunk:
        merge_refine_chunks.append(base_chunk)

    logger.info(f"合并完成，最终数量:{len(merge_refine_chunks)}")
    return merge_refine_chunks


@step_log("step_3_refine_split_and_merge_chunks")
def step_3_refine_split_and_merge_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refine_chunks: list[dict[str, Any]] = []

    for chunk in chunks:
        if len(chunk.get("content", "")) > 600:
            refine_chunks.extend(_split_chunk_content(chunk))
        else:
            refine_chunks.append(chunk)

    logger.info(f"切割完成，切割后包含的碎片数量:{len(refine_chunks)}")
    refine_chunks = _merge_chunk_content(refine_chunks)
    return refine_chunks


# ==========================================
# 3. 漂亮的可视化打印方法
# ==========================================
def print_chunks(chunks: list[dict[str, Any]], stage_name: str):
    print(f"\n{'=' * 25} {stage_name} {'=' * 25}")
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        # 截断展示，避免满屏全是文字
        if len(content) > 100:
            preview = f"{content[:50]} ......(省略)...... {content[-30:]}"
        else:
            preview = content

        print(f"🟢 [Chunk {i}]")
        print(f"   ├─ file_title  : {chunk.get('file_title')}")
        print(f"   ├─ parent_title: {chunk.get('parent_title', 'None (未被切分)')}")
        print(f"   ├─ title       : {chunk.get('title', 'None')}")
        print(f"   ├─ 文本总长度   : {len(content)} 字符")
        print(f"   └─ 文本预览     : {preview.replace(chr(10), r'\n')}\n")


# ==========================================
# 4. 运行 Demo
# ==========================================
if __name__ == '__main__':
    # 构造模拟数据：包含一个超大文本(触发切分) 和一个短文本(不触发切分)
    raw_chunks = [
        {
            "file_title": "测试文档.pdf",
            "title": "#警告信息",
            # 造一个 800字 的大段落 (会触发 >600 的切割)
            "content": "#警告信息\n" + "我们遇到了一些严重的问题。" * 60 + "最后这是结尾部分。"
        },
        {
            "file_title": "测试文档.pdf",
            "title": "#普通小结",
            # 造一个 200字 的小段落 (不会触发切割)
            "content": "#普通小结\n" + "一切正常，不需要处理。" * 20
        }
    ]

    # 1. 打印处理前
    print_chunks(raw_chunks, "切分与合并前 (原始输入)")

    # 2. 执行核心链路
    print("\n⏳ 正在执行切分与合并...\n")
    final_chunks = step_3_refine_split_and_merge_chunks(raw_chunks)

    # 3. 打印处理后
    print_chunks(final_chunks, "切分与合并后 (最终输出)")