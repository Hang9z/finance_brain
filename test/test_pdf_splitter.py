import os
import shutil
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from processor.import_processor.nodes.node_pdf_to_md import (
    get_pdf_page_count,
    step_split_upload_and_merge,
    node_pdf_to_md
)
from processor.import_processor.state import create_default_state
from utils.path_util import PROJECT_ROOT


@pytest.fixture
def dummy_5page_pdf(tmp_path: Path) -> Path:
    """生成一个 5 页的测试 PDF 文件"""
    pdf_path = tmp_path / "long_report_test.pdf"
    doc = fitz.open()
    for i in range(1, 6):
        page = doc.new_page()
        page.insert_text((50, 100), f"Financial Annual Report - Page {i}")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_get_pdf_page_count(dummy_5page_pdf: Path):
    """测试 PDF 页数精准检测"""
    pages = get_pdf_page_count(dummy_5page_pdf)
    assert pages == 5


def test_step_split_upload_and_merge(dummy_5page_pdf: Path, tmp_path: Path):
    """
    测试超过页数限制时的自动切分、分卷上传与合并逻辑
    将 max_pages 设为 2，5 页 PDF 应切为 3 个分卷 (2 + 2 + 1)
    """
    local_dir = tmp_path / "output"
    local_dir.mkdir(parents=True, exist_ok=True)

    calls = []

    def mock_step_2_upload(part_pdf: Path) -> str:
        calls.append(part_pdf.name)
        return f"http://mock.zip/{part_pdf.stem}.zip"

    def mock_step_3_download(zip_url: str, local_dir_obj: Path, stem: str) -> Path:
        # 模拟 MinerU 下载解压得到的分卷 markdown
        part_md_dir = local_dir_obj / stem
        part_md_dir.mkdir(parents=True, exist_ok=True)
        img_dir = part_md_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / f"{stem}_chart.png").write_text("fake_png_data")

        md_file = part_md_dir / f"{stem}.md"
        md_file.write_text(f"# 解析内容 来自分卷 {stem}\n正文说明...", encoding="utf-8")
        return md_file

    with patch("processor.import_processor.nodes.node_pdf_to_md.step_2_upload_and_poll", side_effect=mock_step_2_upload), \
         patch("processor.import_processor.nodes.node_pdf_to_md.step_3_download_and_extract", side_effect=mock_step_3_download):

        final_md_path = step_split_upload_and_merge(
            pdf_path_obj=dummy_5page_pdf,
            local_dir_obj=local_dir,
            total_pages=5,
            max_pages=2
        )

        # 1. 验证切分调用为 3 次
        assert len(calls) == 3
        assert calls == [
            "long_report_test_part1.pdf",
            "long_report_test_part2.pdf",
            "long_report_test_part3.pdf"
        ]

        # 2. 验证合并结果文件存在且包含 3 个分卷内容
        assert final_md_path.is_file()
        content = final_md_path.read_text(encoding="utf-8")
        assert "long_report_test_part1" in content
        assert "long_report_test_part2" in content
        assert "long_report_test_part3" in content

        # 3. 验证合并后的图片目录存在提取的图表
        images_dir = final_md_path.parent / "images"
        assert images_dir.is_dir()
        assert (images_dir / "long_report_test_part1_chart.png").exists()

        # 4. 验证临时分卷目录已被清理
        temp_dir = local_dir / "long_report_test_temp_parts"
        assert not temp_dir.exists()


def test_node_pdf_to_md_routing(dummy_5page_pdf: Path, tmp_path: Path):
    """测试节点层级自动感知超限并切换分卷路由"""
    state = create_default_state(
        task_id="test_routing_001",
        pdf_path=str(dummy_5page_pdf),
        local_dir=str(tmp_path / "output"),
        is_pdf_read_enabled=True
    )

    with patch("processor.import_processor.nodes.node_pdf_to_md.MAX_PAGES_PER_BATCH", 2), \
         patch("processor.import_processor.nodes.node_pdf_to_md.step_split_upload_and_merge") as mock_split:
        
        mock_split.return_value = tmp_path / "output" / "long_report_test" / "long_report_test.md"

        new_state = node_pdf_to_md(state)
        assert mock_split.called
        assert "long_report_test.md" in new_state["md_path"]
