import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from processor.query_processor.main_graph import route_after_item_confirm
from processor.query_processor.nodes.node_item_name_confirm import (
    step_5_select_confirmed_and_option_item_names,
    step_6_change_state_property,
    step_3_call_llm_rewritten_and_extract_item_names
)
from processor.query_processor.state import create_query_default_state


class TestFinancialIntent(unittest.TestCase):

    def test_01_confirmed_item_route(self):
        """测试已命中高置信度金融实体时，正确绑定 item_names 并进入搜索分支"""
        state = create_query_default_state(
            session_id="test_sess_001",
            original_query="贵州茅台2026年一季度净利润是多少？"
        )
        confirmed_dict = {
            "confirmed": ["贵州茅台酒股份有限公司"],
            "option": []
        }
        rewritten_query = "贵州茅台酒股份有限公司2026年第一季度净利润是多少？"

        step_6_change_state_property(state, rewritten_query, confirmed_dict)

        # 断言: 不设 answer (为空字符串)，绑定实体
        self.assertFalse(bool(state.get("answer")))
        self.assertEqual(state.get("item_names"), ["贵州茅台酒股份有限公司"])
        self.assertEqual(state.get("rewritten_query"), rewritten_query)

        # 断言路由: 并发进入 3 路检索
        next_nodes = route_after_item_confirm(state)
        self.assertIsInstance(next_nodes, list)
        self.assertIn("node_search_embedding", next_nodes)
        self.assertIn("node_search_embedding_hyde", next_nodes)
        self.assertIn("node_web_search_mcp", next_nodes)

    def test_02_option_disambiguation_route(self):
        """测试命中多候选金融实体时，输出反问消歧并直接路由至输出节点"""
        state = create_query_default_state(
            session_id="test_sess_002",
            original_query="华夏基金赎回费用怎么算？"
        )
        confirmed_dict = {
            "confirmed": [],
            "option": ["华夏债券C", "华夏国证自由现金流ETF发起式联接基金"]
        }
        rewritten_query = "华夏基金赎回费率及费用如何计算？"

        step_6_change_state_property(state, rewritten_query, confirmed_dict)

        # 断言: 存在反问答案
        self.assertIsNotNone(state.get("answer"))
        self.assertIn("请问您指的是以下哪项金融产品或资料呢", state["answer"])
        self.assertIn("华夏债券C", state["answer"])

        # 断言路由: 直接流转到 node_answer_output 输出反问
        next_node = route_after_item_confirm(state)
        self.assertEqual(next_node, "node_answer_output")

    def test_03_generic_financial_concept_route(self):
        """核心断言: 通用金融概念与FAQ提问时，不因未找到特定商品而短路拒绝，放行进入全局检索"""
        state = create_query_default_state(
            session_id="test_sess_003",
            original_query="什么是最大回撤？"
        )
        confirmed_dict = {
            "confirmed": [],
            "option": []
        }
        rewritten_query = "什么是基金或金融投资中的最大回撤？"

        step_6_change_state_property(state, rewritten_query, confirmed_dict)

        # 断言: 绝不应该被兜底提示拒绝，answer 保持空字符串
        self.assertFalse(bool(state.get("answer")), "通用金融概念不应被设置为拒绝回答或商品未收录")
        self.assertEqual(state.get("item_names"), [])
        self.assertEqual(state.get("rewritten_query"), rewritten_query)

        # 断言路由: 必须正常流转到 3 路搜索
        next_nodes = route_after_item_confirm(state)
        self.assertIsInstance(next_nodes, list)
        self.assertIn("node_search_embedding", next_nodes)
        self.assertIn("node_search_embedding_hyde", next_nodes)
        self.assertIn("node_web_search_mcp", next_nodes)

    def test_04_llm_query_rewrite_live(self):
        """调用实际大模型测试金融语境下的改写与实体抽取"""
        # 测试通用概念
        res_concept = step_3_call_llm_rewritten_and_extract_item_names(
            history_list=[],
            original_query="什么是最大回撤？"
        )
        self.assertIsInstance(res_concept, dict)
        self.assertIn("rewritten_query", res_concept)
        # 针对通用概念，item_names 应为空或不含无关实体
        self.assertIsInstance(res_concept.get("item_names"), list)

        # 测试具体产品
        res_product = step_3_call_llm_rewritten_and_extract_item_names(
            history_list=[],
            original_query="招商银行2026年一季度净利润是多少？"
        )
        self.assertIsInstance(res_product, dict)
        extracted = res_product.get("item_names", [])
        self.assertTrue(any("招商银行" in name for name in extracted) or "招商银行" in res_product.get("rewritten_query", ""))


if __name__ == "__main__":
    unittest.main()
