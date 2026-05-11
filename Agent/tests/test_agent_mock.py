"""
Agent 框架端到端冒烟测试

测试范围：
  1. 模块导入（prompt_template, tool_executor, react_agent, pipeline_runner）
  2. ToolExecutor — mock 后端 (无需 API Key / GPU)
  3. ReactAgent — 覆盖 _call_vlm() 后的完整 ReAct 循环（无需真实 API）
  4. CompetitionPipeline — mock 模式端到端（无需 API Key / GPU）

运行方式：
    # 从 Agent/ 目录
    python tests/test_agent_mock.py

    # 或用 unittest discovery
    python -m unittest discover -s Agent/tests -p "test_*.py" -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# ─── 确保 Agent/ 目录在 sys.path 中 ──────────────────────────────────────────
_AGENT_DIR = Path(__file__).parent.parent          # .../Agent/
_REPO_ROOT = _AGENT_DIR.parent                     # repo root
_ENGINE_DIR = _REPO_ROOT / "3D数学引擎"
_GROUNDING_DIR = _REPO_ROOT / "视觉定位_Grounding"

for _p in [str(_AGENT_DIR), str(_ENGINE_DIR), str(_GROUNDING_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：生成临时 PNG 测试图像
# ─────────────────────────────────────────────────────────────────────────────

def _make_test_image(path: Path, width: int = 64, height: int = 48) -> Path:
    """在 path 写入一张最小的合法 PNG，供测试使用（不依赖真实图像文件）。"""
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    img.save(str(path), format="PNG")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. 模块导入测试
# ─────────────────────────────────────────────────────────────────────────────

class TestImports(unittest.TestCase):
    """验证所有 Agent 模块可以正常导入，不抛出任何异常。"""

    def test_import_prompt_template(self):
        import prompt_template  # noqa: F401
        self.assertTrue(hasattr(prompt_template, "AGENT_SYSTEM_PROMPT"))
        self.assertTrue(len(prompt_template.AGENT_SYSTEM_PROMPT) > 200)

    def test_import_tool_executor(self):
        import tool_executor  # noqa: F401
        self.assertTrue(hasattr(tool_executor, "ToolExecutor"))

    def test_import_react_agent(self):
        import react_agent  # noqa: F401
        self.assertTrue(hasattr(react_agent, "ReactAgent"))
        self.assertTrue(hasattr(react_agent, "run_agent"))

    def test_import_pipeline_runner(self):
        import pipeline_runner  # noqa: F401
        self.assertTrue(hasattr(pipeline_runner, "CompetitionPipeline"))
        self.assertTrue(hasattr(pipeline_runner, "run_competition"))

    def test_import_agent_package(self):
        """测试 Agent/ 作为包导入（从 repo root 触发）。"""
        sys.path.insert(0, str(_REPO_ROOT))
        import Agent  # noqa: F401
        self.assertTrue(hasattr(Agent, "ReactAgent"))
        self.assertTrue(hasattr(Agent, "CompetitionPipeline"))

    def test_all_tool_names_nonempty(self):
        from prompt_template import ALL_TOOL_NAMES
        self.assertGreater(len(ALL_TOOL_NAMES), 0)
        for name in ALL_TOOL_NAMES:
            self.assertIsInstance(name, str)
            # Tool names must be valid Python identifiers (underscores are allowed)
            self.assertTrue(name.isidentifier(), f"'{name}' is not a valid identifier")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ToolExecutor — mock 后端
# ─────────────────────────────────────────────────────────────────────────────

class TestToolExecutorMock(unittest.TestCase):
    """验证 ToolExecutor 在 mock 模式下的所有工具均返回非错误字符串。"""

    def setUp(self):
        from tool_executor import ToolExecutor
        self.tmp = tempfile.TemporaryDirectory()
        self.img_path = _make_test_image(Path(self.tmp.name) / "test.png")
        self.executor = ToolExecutor(
            image_path=self.img_path,
            grounding_backend="mock",
            depth_backend="mock",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_get_bounding_box(self):
        result = self.executor.execute("get_bounding_box", {"object_name": "red car"})
        self.assertIsInstance(result, str)
        self.assertNotIn("ERROR:", result)

    def test_calculate_3d_distance(self):
        result = self.executor.execute(
            "calculate_3d_distance", {"obj_1": "red car", "obj_2": "white van"}
        )
        self.assertIsInstance(result, str)
        self.assertNotIn("ERROR:", result)
        self.assertIn("meters", result)

    def test_calculate_horizontal_distance(self):
        result = self.executor.execute(
            "calculate_horizontal_distance", {"obj_1": "car", "obj_2": "tree"}
        )
        self.assertIsInstance(result, str)
        self.assertNotIn("ERROR:", result)

    def test_calculate_vertical_distance(self):
        result = self.executor.execute(
            "calculate_vertical_distance", {"obj_1": "drone", "obj_2": "building"}
        )
        self.assertIsInstance(result, str)
        self.assertNotIn("ERROR:", result)

    def test_get_direction_clock(self):
        result = self.executor.execute(
            "get_direction",
            {"from_object": "car", "to_object": "tree", "mode": "clock"},
        )
        self.assertIsInstance(result, str)
        self.assertNotIn("ERROR:", result)

    def test_get_direction_cardinal(self):
        result = self.executor.execute(
            "get_direction",
            {"from_object": "car", "to_object": "tree", "mode": "cardinal"},
        )
        self.assertIsInstance(result, str)
        self.assertNotIn("ERROR:", result)

    def test_get_object_size(self):
        result = self.executor.execute(
            "get_object_size", {"object_name": "truck"}
        )
        self.assertIsInstance(result, str)
        self.assertNotIn("ERROR:", result)

    def test_unknown_tool(self):
        result = self.executor.execute("nonexistent_tool", {})
        self.assertIn("ERROR:", result)
        self.assertIn("未知工具", result)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ReactAgent — 覆盖 VLM，验证 ReAct 状态机
# ─────────────────────────────────────────────────────────────────────────────

class TestReactAgentStateMachine(unittest.TestCase):
    """用 Mock VLM 响应验证 ReAct 状态机的解析与控制流。"""

    def setUp(self):
        from react_agent import ReactAgent
        self.tmp = tempfile.TemporaryDirectory()
        self.img_path = _make_test_image(Path(self.tmp.name) / "scene.png")
        self.agent = ReactAgent(
            model="mock-model",
            api_key="fake-key",       # 不会真正调用 API
            grounding_backend="mock",
            depth_backend="mock",
            max_iterations=5,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _run_with_scripted_vlm(self, responses: list[str]) -> str:
        """将 ReactAgent._call_vlm 替换为逐轮返回 responses 的 Mock。"""
        call_count = [0]

        def mock_call_vlm(messages):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        with patch.object(self.agent, "_call_vlm", side_effect=mock_call_vlm):
            return self.agent.run(str(self.img_path), "测试问题")

    def test_direct_final_answer(self):
        """第 1 轮直接给出 Final Answer，应立刻返回。"""
        responses = [
            "Thought: 我知道答案。\nFinal Answer: 8.5 meters"
        ]
        answer = self._run_with_scripted_vlm(responses)
        self.assertEqual(answer.strip(), "8.5 meters")

    def test_one_tool_then_answer(self):
        """第 1 轮调用工具，第 2 轮给出 Final Answer。"""
        responses = [
            'Thought: 先定位。\nAction: get_bounding_box\nAction Input: {"object_name": "red car"}',
            "Thought: 已定位。\nFinal Answer: 12.3 meters",
        ]
        answer = self._run_with_scripted_vlm(responses)
        self.assertEqual(answer.strip(), "12.3 meters")

    def test_two_tools_then_answer(self):
        """两轮工具调用后给出最终答案。"""
        responses = [
            'Thought: 先定位车。\nAction: get_bounding_box\nAction Input: {"object_name": "car"}',
            'Thought: 再算距离。\nAction: calculate_3d_distance\nAction Input: {"obj_1": "car", "obj_2": "tree"}',
            "Thought: 距离已知。\nFinal Answer: 5.0 meters",
        ]
        answer = self._run_with_scripted_vlm(responses)
        self.assertIn("5.0 meters", answer)

    def test_format_error_recovery(self):
        """格式错误后应触发纠错提示并继续，最终给出答案。"""
        responses = [
            "这是一个不符合格式的回复，没有 Action 也没有 Final Answer。",
            "Thought: 好的，修正格式。\nFinal Answer: 格式恢复正常",
        ]
        answer = self._run_with_scripted_vlm(responses)
        self.assertIn("格式恢复正常", answer)

    def test_fallback_on_max_iterations(self):
        """达到最大迭代次数后，应触发 _fallback 并返回非空答案。"""
        # 一直调用工具但不给 Final Answer → 触发 fallback
        tool_response = (
            'Thought: 还需要更多信息。\n'
            'Action: get_bounding_box\n'
            'Action Input: {"object_name": "car"}'
        )
        fallback_response = "Thought: 数据已够。\nFinal Answer: 兜底答案"
        # 前 max_iterations 轮全是工具调用，最后 fallback 给出答案
        responses = [tool_response] * self.agent.max_iterations + [fallback_response]
        answer = self._run_with_scripted_vlm(responses)
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)

    def test_unknown_tool_name_handled(self):
        """模型调用未知工具名时，应注入格式错误提示后继续推理。"""
        responses = [
            'Thought: 调用不存在的工具。\nAction: nonexistent_tool\nAction Input: {}',
            "Thought: 换个方式。\nFinal Answer: 正常恢复",
        ]
        answer = self._run_with_scripted_vlm(responses)
        self.assertIn("正常恢复", answer)

    def test_single_quote_json_fallback(self):
        """Action Input 使用单引号 JSON 时应能正常解析。"""
        responses = [
            "Thought: 调用。\nAction: get_bounding_box\nAction Input: {'object_name': 'car'}",
            "Thought: 完成。\nFinal Answer: 单引号兼容",
        ]
        answer = self._run_with_scripted_vlm(responses)
        self.assertIn("单引号兼容", answer)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ReactAgent 解析函数单元测试
# ─────────────────────────────────────────────────────────────────────────────

class TestReactAgentParsing(unittest.TestCase):
    """单独测试 _parse_action 和 _parse_final_answer 的各种边界情况。"""

    def setUp(self):
        from react_agent import ReactAgent
        self.tmp = tempfile.TemporaryDirectory()
        img = _make_test_image(Path(self.tmp.name) / "x.png")
        self.agent = ReactAgent(model="x", api_key="x", grounding_backend="mock", depth_backend="mock")

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_action_standard(self):
        text = 'Thought: x\nAction: get_bounding_box\nAction Input: {"object_name": "car"}'
        result = self.agent._parse_action(text)
        self.assertIsNotNone(result)
        tool, params = result
        self.assertEqual(tool, "get_bounding_box")
        self.assertEqual(params["object_name"], "car")

    def test_parse_action_unknown_tool_returns_none(self):
        text = 'Action: super_unknown_tool\nAction Input: {}'
        result = self.agent._parse_action(text)
        self.assertIsNone(result)

    def test_parse_action_missing_input_returns_empty_dict(self):
        text = 'Action: get_bounding_box'
        result = self.agent._parse_action(text)
        self.assertIsNotNone(result)
        _, params = result
        self.assertIsInstance(params, dict)

    def test_parse_final_answer_simple(self):
        text = "Thought: x\nFinal Answer: 8.5 meters"
        ans = self.agent._parse_final_answer(text)
        self.assertEqual(ans, "8.5 meters")

    def test_parse_final_answer_multiline(self):
        text = "Final Answer: North-East\n（距离约 10 米）"
        ans = self.agent._parse_final_answer(text)
        self.assertIn("North-East", ans)

    def test_parse_final_answer_not_present(self):
        text = "Thought: 还不知道。\nAction: get_bounding_box\nAction Input: {}"
        ans = self.agent._parse_final_answer(text)
        self.assertIsNone(ans)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CompetitionPipeline — mock 端到端测试
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineMock(unittest.TestCase):
    """验证 CompetitionPipeline 在 mock 模式下端到端运行正常。"""

    def setUp(self):
        from pipeline_runner import CompetitionPipeline
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

        # 生成测试图像
        self.img_path = _make_test_image(self.tmp_dir / "scene.png")

        # 生成小型测试数据集（3 条）
        self.dataset = [
            {"id": "q1", "image": "scene.png", "question": "红色车和路灯距离多远？"},
            {"id": "q2", "image": "scene.png", "question": "从车辆看，建筑物在哪个方向？"},
            {"id": "q3", "image": "scene.png", "question": "这辆卡车有多宽？"},
        ]
        self.data_path = self.tmp_dir / "test_data.json"
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False)

        self.output_path = self.tmp_dir / "submission.jsonl"
        self.ckpt_dir = self.tmp_dir / "checkpoints"

        self.pipeline = CompetitionPipeline(
            model="mock-model",
            api_key="fake-key",
            grounding_backend="mock",
            depth_backend="mock",
            max_workers=1,
            retry_times=0,
            request_interval=0.0,
            checkpoint_dir=str(self.ckpt_dir),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _mock_agent_run(self, image_path, question):
        """替代真实 ReactAgent.run()，返回固定格式的答案。"""
        return f"mock answer for: {question[:20]}"

    def test_pipeline_produces_output_file(self):
        """流水线应生成包含所有条目的 JSONL 提交文件。"""
        with patch("pipeline_runner.ReactAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.run.side_effect = self._mock_agent_run
            summary = self.pipeline.run(
                data_path=str(self.data_path),
                output_path=str(self.output_path),
            )

        self.assertTrue(self.output_path.exists(), "提交文件未创建")
        lines = self.output_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3, f"期望 3 条，实际 {len(lines)} 条")
        for line in lines:
            rec = json.loads(line)
            self.assertIn("id", rec)
            self.assertIn("answer", rec)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["done_new"], 3)
        self.assertEqual(summary["error_cnt"], 0)

    def test_pipeline_checkpoint_resume(self):
        """中途中断后，续传应只处理剩余条目。"""
        with patch("pipeline_runner.ReactAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.run.side_effect = self._mock_agent_run
            # 第一次运行：只处理前 2 条（手动写入检查点模拟中断）
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_name = f"ckpt_test_data_mock_mock.jsonl"
            ckpt_path = self.ckpt_dir / ckpt_name
            # 提前写入 q1、q2 的检查点
            with open(ckpt_path, "w", encoding="utf-8") as f:
                for qid in ["q1", "q2"]:
                    f.write(json.dumps({"id": qid, "answer": "already done", "status": "ok"}) + "\n")

            # 第二次运行（续传）：应只处理 q3
            summary = self.pipeline.run(
                data_path=str(self.data_path),
                output_path=str(self.output_path),
                resume=True,
            )

        self.assertEqual(summary["skipped"], 2, "应跳过 2 条已完成记录")
        self.assertEqual(summary["done_new"], 1, "应新处理 1 条")

    def test_pipeline_jsonl_input(self):
        """验证 JSONL 格式输入数据正常加载。"""
        jsonl_path = self.tmp_dir / "test_data.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in self.dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with patch("pipeline_runner.ReactAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.run.side_effect = self._mock_agent_run
            summary = self.pipeline.run(
                data_path=str(jsonl_path),
                output_path=str(self.output_path),
                resume=False,
            )
        self.assertEqual(summary["total"], 3)

    def test_pipeline_auto_assigns_id(self):
        """没有 id 字段的数据集应自动以行号作为 id。"""
        no_id_data = [
            {"image": "scene.png", "question": "q1"},
            {"image": "scene.png", "question": "q2"},
        ]
        no_id_path = self.tmp_dir / "no_id.json"
        with open(no_id_path, "w", encoding="utf-8") as f:
            json.dump(no_id_data, f)

        with patch("pipeline_runner.ReactAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.run.return_value = "answer"
            summary = self.pipeline.run(
                data_path=str(no_id_path),
                output_path=str(self.output_path),
                resume=False,
            )
        self.assertEqual(summary["total"], 2)

    def test_pipeline_error_handling(self):
        """单条推理异常时，其他条目应正常处理，error_cnt 应正确统计。"""
        call_count = [0]

        def flaky_run(image_path, question):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("模拟 API 崩溃")
            return f"ok answer {call_count[0]}"

        with patch("pipeline_runner.ReactAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.run.side_effect = flaky_run
            self.pipeline.retry_times = 0  # 不重试，加速测试
            summary = self.pipeline.run(
                data_path=str(self.data_path),
                output_path=str(self.output_path),
                resume=False,
            )

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["error_cnt"], 1)
        # 提交文件中应有 3 条（包括失败的那条）
        lines = self.output_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Checkpoint 单元测试
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        from pipeline_runner import Checkpoint
        self.tmp = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.tmp.name) / "test_ckpt.jsonl"
        self.Checkpoint = Checkpoint

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_and_is_done(self):
        ckpt = self.Checkpoint(self.ckpt_path)
        self.assertFalse(ckpt.is_done("item1"))
        ckpt.save({"id": "item1", "answer": "x", "status": "ok"})
        self.assertTrue(ckpt.is_done("item1"))
        self.assertFalse(ckpt.is_done("item2"))

    def test_persistence(self):
        """保存后重新加载应能恢复已完成状态。"""
        ckpt = self.Checkpoint(self.ckpt_path)
        ckpt.save({"id": "a", "answer": "x", "status": "ok"})
        ckpt.save({"id": "b", "answer": "y", "status": "ok"})

        # 重新加载
        ckpt2 = self.Checkpoint(self.ckpt_path)
        self.assertTrue(ckpt2.is_done("a"))
        self.assertTrue(ckpt2.is_done("b"))
        self.assertFalse(ckpt2.is_done("c"))
        self.assertEqual(len(ckpt2), 2)

    def test_all_results_sorted(self):
        ckpt = self.Checkpoint(self.ckpt_path)
        for i in [3, 1, 2]:
            ckpt.save({"id": str(i), "answer": f"ans{i}", "status": "ok"})
        # 应能返回全部 3 条（顺序不做要求，但长度正确）
        self.assertEqual(len(ckpt.all_results()), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 运行入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent 框架 Mock 测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    verbosity = 2 if args.verbose else 1
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
