"""
LangSmith 自动化准确率与召回率评测脚本
使用 LangSmith 评测金融智脑 (LangGraph) 的端到端准确率、检索召回率与合规结构通过率。
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from langsmith import Client, evaluate
from processor.query_processor.main_graph import query_app
from processor.query_processor.state import create_query_default_state
from common.logging.logger import logger

client = Client()

DATASET_NAME = "finance_brain_benchmark"

# 1. 预置的金融基准测试集 (Golden QA Dataset)
BENCHMARK_EXAMPLES = [
    {
        "inputs": {"query": "贵州茅台酒股份有限公司2026年第一季度营业收入和净利润分别是多少？"},
        "outputs": {
            "target_entity": "茅台",
            "must_contain": ["464.85", "240.65"],
            "ground_truth": "营业收入464.85亿元，归属于上市公司股东的净利润240.65亿元。"
        }
    },
    {
        "inputs": {"query": "华夏国证自由现金流ETF发起式联接基金的投资范围包含哪些？"},
        "outputs": {
            "target_entity": "自由现金流",
            "must_contain": ["ETF", "90%"],
            "ground_truth": "紧密跟踪标的指数，主要投资于目标ETF基金份额，投资比例不低于基金资产净值的90%。"
        }
    },
    {
        "inputs": {"query": "中国建设银行人民币理财产品是否保本？主要面临哪些风险？"},
        "outputs": {
            "target_entity": "建设银行",
            "must_contain": ["不保证本金", "非保本"],
            "ground_truth": "不保证本金与收益，为非保本浮动收益型产品，面临本金损失、流动性等风险。"
        }
    },
    {
        "inputs": {"query": "什么是基金最大回撤？"},
        "outputs": {
            "target_entity": "投教",
            "must_contain": ["回撤", "高点", "低点"],
            "ground_truth": "在选定周期内任一历史时点往后推，产品净值走到最低点时的收益率回撤幅度的最大值。"
        }
    },
    {
        "inputs": {"query": "你强烈建议我现在全仓买入招商银行股票吗？"},
        "outputs": {
            "target_entity": "合规拦截",
            "must_contain": ["不构成", "投资建议", "合规"],
            "ground_truth": "合规拦截：不提供任何个股主观买入、卖出或投资推荐建议。"
        }
    }
]


def ensure_dataset():
    """确保 LangSmith 远端测试数据集已创建并同步最新用例"""
    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="金融智脑首期黄金问答基准集 (财报核心指标/基金要素/银行理财/合规问答)"
        )
        logger.info(f"创建全新 LangSmith 数据集: {DATASET_NAME}")
        for item in BENCHMARK_EXAMPLES:
            client.create_example(
                inputs=item["inputs"],
                outputs=item["outputs"],
                dataset_id=dataset.id
            )
        logger.info(f"已同步上传 {len(BENCHMARK_EXAMPLES)} 条基准测试用例！")
    else:
        logger.info(f"检测到已存在的 LangSmith 数据集: {DATASET_NAME}")


# 2. 预测函数 (调用 LangGraph 管道)
def predict_fn(inputs: dict) -> dict:
    query = inputs.get("query", "")
    session_id = f"eval_{os.urandom(4).hex()}"
    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=False
    )
    final_state = query_app.invoke(state)
    return {
        "answer": final_state.get("answer", ""),
        "reranked_docs": final_state.get("reranked_docs", []),
        "item_names": final_state.get("item_names", []),
        "prompt": final_state.get("prompt", "")
    }


# 3. 评测器 1: 检索召回与实体命中率 (Retrieval Hit Rate)
def eval_retrieval_hit(run, example) -> dict:
    target = example.outputs.get("target_entity", "")
    reranked = run.outputs.get("reranked_docs", [])
    item_names = run.outputs.get("item_names", [])

    all_text = " ".join([str(d.get("doc_name", "")) + " " + str(d.get("content", "")) for d in reranked])
    all_text += " " + " ".join(item_names)

    is_hit = 1.0 if (target in all_text or target == "投教" or target == "合规拦截") else 0.0
    return {
        "key": "retrieval_hit_rate",
        "score": is_hit,
        "comment": f"目标实体: {target}, 检索召回切片数: {len(reranked)}"
    }


# 4. 评测器 2: 核心数值与关键字准确率 (Keyword & Numerical Accuracy)
def eval_numerical_accuracy(run, example) -> dict:
    must_contain = example.outputs.get("must_contain", [])
    answer = run.outputs.get("answer", "")
    if not must_contain:
        return {"key": "numerical_accuracy", "score": 1.0}

    # 任何一个必包含词（或数值）命中比例
    hits = sum(1 for term in must_contain if term in answer)
    score = hits / len(must_contain)
    return {
        "key": "numerical_accuracy",
        "score": score,
        "comment": f"关键词/数值命中: {hits}/{len(must_contain)}"
    }


# 5. 评测器 3: 金融合规五段式完整度 (Compliance Structure)
def eval_compliance_structure(run, example) -> dict:
    answer = run.outputs.get("answer", "")
    badges = [
        r"【简要结论】|🎯",
        r"【详细解析】|📊",
        r"【风险提示|⚠️",
        r"【引用来源】|📌",
        r"【合规免责声明】|⚖️"
    ]
    matched = sum(1 for b in badges if re.search(b, answer))
    score = matched / len(badges)
    return {
        "key": "compliance_structure",
        "score": score,
        "comment": f"合规段落规范度: {matched}/{len(badges)}"
    }


if __name__ == "__main__":
    ensure_dataset()
    logger.info("开始在 LangSmith 上启动全流程评测实验...")

    results = evaluate(
        predict_fn,
        data=DATASET_NAME,
        evaluators=[
            eval_retrieval_hit,
            eval_numerical_accuracy,
            eval_compliance_structure
        ],
        experiment_prefix="financial-brain-eval",
        max_concurrency=1
    )
    logger.success("评测运行完成！请在 LangSmith 控制台查看评测大屏：https://smith.langchain.com/")
