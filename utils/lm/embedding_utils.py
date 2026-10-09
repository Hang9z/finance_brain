import hashlib
import os
import requests
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from common.logging.logger import logger
from common.config.embedding_config import embedding_config

# 模型单例对象，避免重复初始化
_bge_m3_ef = None


def _text_to_sparse_fallback(text: str) -> dict:
    """
    当外部服务未暴露稀疏向量时，基于文本特征生成 L2 归一化的轻量稀疏向量字典。
    维度采用非负整数 uint32 哈希，确保与 Milvus SPARSE_FLOAT_VECTOR 倒排索引完全兼容。
    """
    words = [w for w in text.split() if w.strip()]
    if not words:
        words = list(text[:30])
    freq = {}
    for w in words:
        dim_id = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:6], 16) % 30000
        freq[dim_id] = freq.get(dim_id, 0.0) + 1.0
    norm = sum(v ** 2 for v in freq.values()) ** 0.5
    if norm > 0:
        return {k: round(v / norm, 4) for k, v in freq.items()}
    return {1: 1.0}

def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    # 单例模式：已初始化则直接返回，避免重复加载模型
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return _bge_m3_ef

    # 从环境变量加载配置，无配置则使用默认值
    # 本地有可以使用本地地址！ 没有使用 "BAAI/bge-m3" 会自动下载！ 如果云端部署也可以使用url地址！
    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False

    # 打印模型初始化配置，便于问题排查
    logger.info(
        "开始初始化BGE-M3模型",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True
        }
    )

    try:
        # 初始化BGE-M3模型，开启原生L2归一化（适配Milvus IP内积检索）
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True  # 模型原生对稠密向量做L2归一化
        )
        logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
        raise  # 向上抛出异常，由调用方处理


def generate_embeddings(texts):
    """
    为文本列表生成稠密+稀疏混合向量嵌入（模型原生L2归一化）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    # 1. 优先检查是否配置了 Cloudflare Workers AI (Serverless 高可用)
    cf_account_id = os.getenv("CF_ACCOUNT_ID")
    cf_api_token = os.getenv("CF_API_TOKEN")
    cf_model = os.getenv("CF_EMBEDDING_MODEL", "@cf/baai/bge-m3")

    if cf_account_id and cf_api_token:
        logger.info(f"使用 Cloudflare Workers AI ({cf_model}) 生成 {len(texts)} 条文本向量")
        try:
            cf_url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/{cf_model}"
            headers = {"Authorization": f"Bearer {cf_api_token}"}
            response = requests.post(cf_url, headers=headers, json={"text": texts}, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success", False):
                raise RuntimeError(f"Cloudflare Workers AI 错误: {payload.get('errors')}")

            dense_vectors = payload.get("result", {}).get("data", [])
            # 为保证与 Milvus 双向量 Schema 完全兼容，配套生成轻量稀疏向量
            processed_sparse = [_text_to_sparse_fallback(t) for t in texts]

            result = {
                "dense": dense_vectors,
                "sparse": processed_sparse
            }
            logger.success(f"{len(texts)}条文本向量通过 Cloudflare Workers AI 生成完成")
            return result
        except Exception as e:
            logger.error(f"Cloudflare Workers AI 向量生成失败: {str(e)}", exc_info=True)
            raise

    # 2. 检查是否配置了自建云端/远程 Embedding 接口 (如 Google Colab FastAPI)
    bge_api_url = os.getenv("BGE_API_URL")
    if bge_api_url:
        logger.info(f"使用云端/远程 BGE-M3 接口 ({bge_api_url}) 生成 {len(texts)} 条文本向量")
        try:
            response = requests.post(f"{bge_api_url.rstrip('/')}/embeddings", json={"texts": texts}, timeout=60)
            response.raise_for_status()
            payload = response.json()
            processed_sparse = [
                {int(k): float(v) for k, v in item.items()}
                for item in payload.get("sparse", [])
            ]
            result = {
                "dense": payload.get("dense", []),
                "sparse": processed_sparse
            }
            logger.success(f"{len(texts)}条文本向量通过云端接口生成完成")
            return result
        except Exception as e:
            logger.error(f"云端 BGE-M3 接口调用失败: {str(e)}", exc_info=True)
            raise

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3模型单例
        model = get_bge_m3_ef()
        # 模型编码生成向量，返回dense（稠密向量）+sparse（CSR格式稀疏向量）
        embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
        processed_sparse = []
        for i in range(len(texts)):
            # 提取第i个文本的稀疏向量索引：np.int64 → Python int（满足字典key可哈希要求）
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 提取第i个文本的稀疏向量权重：np.float32 → Python float（适配JSON序列化/接口返回）
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 构造{特征索引: 归一化权重}的稀疏向量字典
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # 构造最终返回结果，稠密向量转列表（解决numpy数组不可序列化问题）
        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # 嵌套列表，与输入文本一一对应
            "sparse": processed_sparse  # 字典列表，模型已做L2归一化
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理


"""
核心设计亮点&适配说明：
1. 模型原生归一化：开启normalize_embeddings = True，自动对稠密+稀疏向量做L2归一化，完美适配Milvus IP内积检索（单位化后IP等价于余弦，计算更快）；
2. 彻底解决NumPy类型做key问题：sparse_indices加.tolist()，将np.int64转为Python原生int，满足字典key的可哈希要求，无报错风险；
3. 稀疏值适配序列化：sparse_data加.tolist()，将np.float32转为Python原生float，支持JSON写入/接口返回/Milvus入库等所有场景；
4. 单例模式优化：模型仅初始化一次，避免重复加载耗时耗资源，提升批量处理效率；
5. 格式匹配业务调用：返回dense嵌套列表、sparse字典列表，与vector_result["dense"][0]/sparse_vector["sparse"][0]取值逻辑完美契合；
6. 分级日志覆盖：从模型初始化、向量生成到异常报错，全流程日志记录，便于生产环境问题排查；
7. 入参合法性校验：防止空列表/非列表入参导致的内部报错，提升工具类健壮性。
"""