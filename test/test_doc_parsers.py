import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from processor.import_processor.nodes.node_doc_to_md import node_doc_to_md
from processor.import_processor.nodes.node_entry import node_entry
from processor.import_processor.state import create_default_state
from utils.path_util import PROJECT_ROOT

DATA_DIR = Path(r"D:\尚硅谷笔记\阶段八：尚硅谷大模型技术之掌柜智库项目\尚硅谷大模型项目实战之掌柜智库实战\1.资料\金融\数据")


class TestDocumentParsers(unittest.TestCase):

    def test_01_node_entry_routing(self):
        """测试 node_entry 对 PDF/DOCX/DOC 的正确识别与状态标记"""
        # 1. 测试 PDF
        pdf_state = create_default_state(local_file_path=str(DATA_DIR / "用户FAQ" / "基金基础知识.pdf"))
        res_pdf = node_entry(pdf_state)
        self.assertTrue(res_pdf["is_pdf_read_enabled"])
        self.assertFalse(res_pdf["is_doc_read_enabled"])
        self.assertFalse(res_pdf["is_md_read_enabled"])

        # 2. 测试 DOCX
        docx_state = create_default_state(local_file_path=str(DATA_DIR / "用户FAQ" / "私募基金投资者教育知识问答.docx"))
        res_docx = node_entry(docx_state)
        self.assertTrue(res_docx["is_doc_read_enabled"])
        self.assertFalse(res_docx["is_pdf_read_enabled"])
        self.assertFalse(res_docx["is_md_read_enabled"])

        # 3. 测试 DOC
        doc_state = create_default_state(local_file_path=str(DATA_DIR / "用户FAQ" / "ETF基金投资者教育问答.doc"))
        res_doc = node_entry(doc_state)
        self.assertTrue(res_doc["is_doc_read_enabled"])
        self.assertFalse(res_doc["is_pdf_read_enabled"])
        self.assertFalse(res_doc["is_md_read_enabled"])

    def test_02_node_doc_to_md_docx(self):
        """测试 DOCX 文件转换为结构化 Markdown"""
        docx_file = DATA_DIR / "用户FAQ" / "私募基金投资者教育知识问答.docx"
        if not docx_file.exists():
            self.skipTest("DOCX测试文件不存在")

        test_state = create_default_state(
            task_id="test_docx_001",
            doc_path=str(docx_file),
            local_file_path=str(docx_file),
            local_dir=str(PROJECT_ROOT / "output")
        )

        result_state = node_doc_to_md(test_state)
        md_path = Path(result_state["md_path"])
        md_content = result_state["md_content"]

        self.assertTrue(md_path.exists())
        self.assertEqual(md_path.suffix, ".md")
        self.assertGreater(len(md_content), 500)
        self.assertIn("私募基金", md_content)
        self.assertIn("## ", md_content)

    def test_03_node_doc_to_md_doc(self):
        """测试 DOC 二进制文件转换为结构化 Markdown"""
        doc_file = DATA_DIR / "用户FAQ" / "ETF基金投资者教育问答.doc"
        if not doc_file.exists():
            self.skipTest("DOC测试文件不存在")

        test_state = create_default_state(
            task_id="test_doc_002",
            doc_path=str(doc_file),
            local_file_path=str(doc_file),
            local_dir=str(PROJECT_ROOT / "output")
        )

        result_state = node_doc_to_md(test_state)
        md_path = Path(result_state["md_path"])
        md_content = result_state["md_content"]

        self.assertTrue(md_path.exists())
        self.assertEqual(md_path.suffix, ".md")
        self.assertGreater(len(md_content), 1000)
        self.assertIn("ETF", md_content)
        self.assertIn("## ", md_content)


if __name__ == "__main__":
    unittest.main()
