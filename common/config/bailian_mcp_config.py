# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class WebSearchConfig:
    # 联网搜索服务提供方: "deepseek" (首选推荐，原生专业搜索), "brave" 或 "bailian_mcp"
    provider: str

    # DeepSeek 网页搜索配置
    deepseek_api_key: str
    deepseek_base_url: str

    # Brave Search 配置
    brave_api_key: str
    brave_base_url: str

    # 阿里百炼 MCP 配置（保留向后兼容）
    mcp_base_url: str
    mcp_api_key: str


# 统一实例化联网搜索配置
web_search_config = WebSearchConfig(
    provider=os.getenv("WEB_SEARCH_PROVIDER", "deepseek").strip().lower(),
    deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip(),
    deepseek_base_url=os.getenv("DEEPSEEK_SEARCH_BASE_URL", "https://api.deepseek.com/v1/tools/search").strip(),
    brave_api_key=os.getenv("BRAVE_SEARCH_API_KEY", "").strip(),
    brave_base_url=os.getenv("BRAVE_SEARCH_BASE_URL", "https://api.search.brave.com/res/v1/llm/context").strip(),
    mcp_base_url=os.getenv("MCP_DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp").strip(),
    mcp_api_key=os.getenv("DASHSCOPE_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip(),
)

# 向后兼容 McpConfig 与 mcp_config
McpConfig = WebSearchConfig
mcp_config = web_search_config
