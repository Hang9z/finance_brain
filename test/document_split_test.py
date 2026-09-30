import re
from typing import Any

def step_2_split_document_by_title(md_content: str, file_title: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current_title: str | None = None
    current_content_lines: list[str] = []
    orphan_lines: list[str] = []
    has_flushed_first_chunk: bool = False
    is_code: bool = False
    heading_stack: list[str] = []

    document_lines = md_content.splitlines()
    title_reg = re.compile(r"^#{1,6}\s.+")

    for line in document_lines:
        line_strip = line.strip()
        if not line_strip:
            continue

        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            is_code = not is_code
            (orphan_lines if not current_title else current_content_lines).append(line_strip)
            continue

        if is_code:
            (orphan_lines if not current_title else current_content_lines).append(line)
            continue

        if title_reg.match(line_strip):
            heading_level = len(line_strip) - len(line_strip.lstrip("#"))

            if current_title and len(current_content_lines) > 0:
                if not has_flushed_first_chunk and orphan_lines:
                    current_content_lines = orphan_lines + current_content_lines
                    has_flushed_first_chunk = True
                    orphan_lines = []

                full_content = f"{current_title}\n" + "\n".join(current_content_lines)
                chunks.append({
                    "title": current_title,
                    "content": full_content,
                    "file_title": file_title
                })
                current_content_lines = []

            while len(heading_stack) < heading_level:
                heading_stack.append(None)

            heading_stack = heading_stack[:heading_level]
            heading_stack[heading_level - 1] = line_strip
            current_title = "_".join([h for h in heading_stack if h])
            continue
        else:
            if not current_title:
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)

    if current_title and (len(current_content_lines) > 0 or (not has_flushed_first_chunk and len(orphan_lines) > 0)):
        if not has_flushed_first_chunk and len(orphan_lines):
            current_content_lines = orphan_lines + current_content_lines
            has_flushed_first_chunk = True
            orphan_lines = []
        full_content = f"{current_title}\n" + "\n".join(current_content_lines)
        chunks.append({
            "title": current_title,
            "content": full_content,
            "file_title": file_title
        })
    elif (not current_title) and len(orphan_lines):
        chunks.append({
            "title": file_title,
            "content": "\n".join(orphan_lines),
            "file_title": file_title
        })

    return chunks

# ----------------- 漏洞验证用例 -----------------

print("=== 漏洞1：嵌套/多反引号代码块状态反转错误 ===")
test_code_block = """# 编程文档
````markdown
```python
# 这只是代码块内的普通注释
print("hello")
```
````
正文结尾。
"""
res1 = step_2_split_document_by_title(test_code_block, "test1")
for i, c in enumerate(res1):
    print(f"Chunk {i+1} title: {c['title']}")
    print(f"Content:\n{c['content']}\n")

print("=== 漏洞2：空行全被抹杀，导致代码格式和段落严重毁损 ===")
test_blank_lines = """# 规范
段落一内容。

段落二内容。

```python
def add(a, b):
    # 中间有空行

    return a + b
```
"""
res2 = step_2_split_document_by_title(test_blank_lines, "test2")
print("Content with blank lines stripped:")
print(repr(res2[0]['content']))

print("\n=== 漏洞3：孤儿前言与首个有内容章节的语义污染 & 幽灵漂移 ===")
test_orphan_drift = """免责声明：本内容仅供参考，不负法律责任。
# 第一章：引言 (无正文)
# 第二章：核心算法
这是第二章算法正文。
"""
res3 = step_2_split_document_by_title(test_orphan_drift, "test3")
for i, c in enumerate(res3):
    print(f"Chunk {i+1} title: {c['title']}")
    print(f"Content:\n{c['content']}\n")

print("=== 漏洞4：文末空标题被完全静默丢弃 ===")
test_last_empty_heading = """# 第一章
正文一
# 第二章 (末尾空标题)
"""
res4 = step_2_split_document_by_title(test_last_empty_heading, "test4")
print(f"Total chunks: {len(res4)}")
for i, c in enumerate(res4):
    print(f"Chunk {i+1} title: {c['title']}")
