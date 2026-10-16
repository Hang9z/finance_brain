import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import requests
import tiktoken
from common.config.reranker_config import reranker_config
from common.logging.logger import logger


class _TiktokenTokenizerProxy:
    """轻量 Tokenizer 代理，提供与 FlagReranker 相同的 encode 接口"""
    def __init__(self):
        self._enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return []
        return self._enc.encode(text)


class CloudflareReranker:
    """
    基于 Cloudflare Workers AI Serverless 的 BGE Reranker 客户端
    API 接口对齐 FlagReranker，支持 compute_score 与 tokenizer.encode
    """
    def __init__(self, account_id: str, api_token: str, model: str = "@cf/baai/bge-reranker-base"):
        self.account_id = account_id
        self.api_token = api_token
        self.model = model
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.tokenizer = _TiktokenTokenizerProxy()

    def compute_score(self, sentence_pairs: list[list[str]], normalize: bool = True) -> list[float]:
        """
        计算 [问题, 候选切片] 对的语义相似度得分
        :param sentence_pairs: 形如 [[query, doc1], [query, doc2], ...]
        :param normalize: 接口占位，Cloudflare 原生返回即为 [0, 1] 归一化得分
        :return: 与输入列表顺序严格一一对应的得分列表
        """
        if not sentence_pairs:
            return []

        # 在 RAG 场景中，全部问答对共享同一个问题 query
        query = sentence_pairs[0][0]
        contexts = [{"text": pair[1]} for pair in sentence_pairs]
        payload = {"query": query, "contexts": contexts}

        # 3 次指数退避重试，防御跨境网络抖动
        last_exception = None
        for attempt in range(1, 4):
            try:
                res = requests.post(self.url, headers=self.headers, json=payload, timeout=25)
                res.raise_for_status()
                data = res.json()

                if not data.get("success"):
                    errors = data.get("errors", [])
                    raise RuntimeError(f"Cloudflare Reranker API 业务报错: {errors}")

                # Cloudflare 响应已按 score 降序排列，按原始 id 将分数放回对应位置
                response_items = data.get("result", {}).get("response", [])
                scores = [0.0] * len(sentence_pairs)
                for item in response_items:
                    idx = item.get("id")
                    if idx is not None and 0 <= idx < len(scores):
                        scores[idx] = float(item.get("score", 0.0))

                return scores
            except Exception as e:
                last_exception = e
                logger.warning(f"[CloudflareReranker] 第 {attempt} 次请求失败: {e}，正在重试...")
                time.sleep(attempt * 0.8)

        logger.error(f"[CloudflareReranker] 请求最终失败: {last_exception}")
        # 降级返回全 0.0，避免阻断上层流程
        return [0.0] * len(sentence_pairs)


_reranker_instance = None


def get_reranker_model():
    """获取单例 Reranker 实例，根据配置自动切换 Cloudflare 或本地 FlagReranker"""
    global _reranker_instance
    if _reranker_instance is None:
        if reranker_config.provider == "cloudflare":
            logger.info(f"[Reranker] 初始化 Cloudflare Workers AI 重排模型: {reranker_config.cf_reranker_model}")
            if not reranker_config.cf_account_id or not reranker_config.cf_api_token:
                raise ValueError("未配置 CF_ACCOUNT_ID 或 CF_API_TOKEN，无法使用 Cloudflare Reranker")
            _reranker_instance = CloudflareReranker(
                account_id=reranker_config.cf_account_id,
                api_token=reranker_config.cf_api_token,
                model=reranker_config.cf_reranker_model,
            )
        else:
            logger.info(f"[Reranker] 初始化本地 FlagReranker 模型: {reranker_config.bge_reranker_large}")
            from FlagEmbedding import FlagReranker
            _reranker_instance = FlagReranker(
                model_name_or_path=reranker_config.bge_reranker_large,
                device=reranker_config.bge_reranker_device,
                use_fp16=reranker_config.bge_reranker_fp16,
            )
    return _reranker_instance


def compute_token_number(content: str) -> int:
    """计算文本 Token 长度，用于文本超长压缩判断"""
    reranker_model = get_reranker_model()
    return len(reranker_model.tokenizer.encode(content, add_special_tokens=False))


if __name__ == "__main__":
    print("\n========== 测试 Cloudflare Reranker ==========")
    model = get_reranker_model()
    test_query = "什么是 RRF 融合？"
    test_docs = [
        "RRF (Reciprocal Rank Fusion) 是一种多路召回倒数排名融合算法",
        "今天天气真好，去公园散步",
        "重排序 Rerank 使用交叉编码器对文本进行细粒度语义打分",
    ]
    test_pairs = [[test_query, doc] for doc in test_docs]
    scores = model.compute_score(test_pairs)

    for doc, score in zip(test_docs, scores):
        print(f"得分: {score:.6f} | 文本: {doc}")

    print("\nToken 统计测试:", compute_token_number("测试中文 Token 数量"))