# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件
load_dotenv()


@dataclass
class RerankerConfig:
    # 服务提供方: "cloudflare" (默认，推荐) 或 "local" (本地 FlagEmbedding)
    provider: str
    cf_account_id: str
    cf_api_token: str
    cf_reranker_model: str

    # 本地模型配置（保留向后兼容）
    bge_reranker_large: str
    bge_reranker_device: str
    bge_reranker_fp16: bool


# 实例化配置对象
reranker_config = RerankerConfig(
    provider=os.getenv("RERANKER_PROVIDER", "cloudflare").strip().lower(),
    cf_account_id=os.getenv("CF_ACCOUNT_ID", "").strip(),
    cf_api_token=os.getenv("CF_API_TOKEN", "").strip() or os.getenv("OPENAI_API_KEY", "").strip(),
    cf_reranker_model=os.getenv("CF_RERANKER_MODEL", "@cf/baai/bge-reranker-base").strip(),
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE", "").strip(),
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE", "cpu").strip(),
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true", 1),
)