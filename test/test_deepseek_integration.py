import base64
from pathlib import Path
import pytest
import fitz
from langchain_core.messages import HumanMessage

from common.config.lm_config import lm_config
from common.config.bailian_mcp_config import web_search_config
from utils.lm.lm_utils import get_llm_client
from processor.query_processor.nodes.node_web_search_mcp import node_web_search_mcp
from processor.query_processor.state import QueryGraphState


def test_deepseek_config_loaded():
    """验证 DeepSeek 相关配置已成功加载"""
    assert "deepseek" in lm_config.llm_model.lower()
    assert "deepseek" in lm_config.vl_model.lower()
    assert "api.deepseek.com" in lm_config.base_url
    assert lm_config.api_key.startswith("sk-")
    assert web_search_config.provider == "deepseek"
    assert "api.deepseek.com" in web_search_config.deepseek_base_url


def test_deepseek_llm_text_generation():
    """验证 DeepSeek 对话文本生成"""
    llm = get_llm_client()
    resp = llm.invoke("请回复八个字：金融智脑调用成功")
    text = resp.content.strip()
    assert len(text) > 0
    print(f"\n[DeepSeek Text Output]: {text}")


def test_deepseek_vision_multimodal():
    """验证 DeepSeek-flash 多模态图像解析"""
    # 动态生成一张 200x200 矩形测试图
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.draw_rect(fitz.Rect(20, 20, 180, 180), color=(1, 0, 0), fill=(0, 0, 1))
    pix = page.get_pixmap()
    b64_str = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    doc.close()

    vl_client = get_llm_client(model=lm_config.vl_model)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "简要描述此图像的主要颜色和形状（不超过30字）："},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}}
        ]
    )
    resp = vl_client.invoke([msg])
    text = resp.content.strip()
    assert len(text) > 0
    print(f"\n[DeepSeek Vision Output]: {text}")


def test_deepseek_web_search():
    """验证 DeepSeek 网页搜索接口集成"""
    state: QueryGraphState = {
        "session_id": "test_deepseek_search_session",
        "is_stream": False,
        "original_query": "贵州茅台 2024 一季度 营收 净利润",
        "rewritten_query": "贵州茅台 2024 一季度报告 营收 归母净利润",
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "web_search_docs": [],
        "rrf_chunks": [],
        "reranked_docs": [],
        "prompt": "",
        "answer": "",
        "item_names": ["贵州茅台"],
        "history": [],
        "image_urls": [],
    }

    result = node_web_search_mcp(state)
    docs = result.get("web_search_docs", [])
    assert len(docs) > 0, "DeepSeek 搜索未召回网页文档"
    first = docs[0]
    assert "title" in first and len(first["title"]) > 0
    assert "url" in first and first["url"].startswith("http")
    assert "snippet" in first and len(first["snippet"]) > 0
    print(f"\n[DeepSeek Search Docs Count]: {len(docs)}")
    print(f"[First Result Title]: {first['title']}")
    print(f"[First Result URL]: {first['url']}")
    print(f"[First Result Snippet Preview]: {first['snippet'][:100]}...")


if __name__ == "__main__":
    test_deepseek_config_loaded()
    print("test_deepseek_config_loaded: PASSED")
    test_deepseek_llm_text_generation()
    print("test_deepseek_llm_text_generation: PASSED")
    test_deepseek_vision_multimodal()
    print("test_deepseek_vision_multimodal: PASSED")
    test_deepseek_web_search()
    print("test_deepseek_web_search: PASSED")
