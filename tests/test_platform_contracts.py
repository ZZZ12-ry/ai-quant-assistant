import sys
import unittest
import shutil
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class IntentRoutingTests(unittest.TestCase):
    def test_intent_classification_contract(self):
        from app.chat import _classify_intent, _deterministic_non_persistent_reply, _should_persist_intent

        cases = {
            "均线策略思路：短期均线上穿长期均线做多，下穿做空": "strategy_material",
            "请根据阶段1整理出的策略说明生成策略代码。": "stage_command",
            "请执行回测。": "stage_command",
            "请检查策略说明、代码和回测结果是否一致。": "stage_command",
            "请基于真实回测结果生成分析报告。": "stage_command",
            "怎么打开测试": "platform_help",
            "夏普比率怎么理解": "quant_qa",
            "权益曲线是什么意思": "quant_qa",
            "增加一个ADX过滤条件": "strategy_revision",
            "加一个成交量过滤": "strategy_revision",
            "今天吃什么": "off_topic",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(_classify_intent(text), expected)

        self.assertTrue(_should_persist_intent("strategy_material"))
        self.assertTrue(_should_persist_intent("stage_command"))
        self.assertFalse(_should_persist_intent("platform_help"))
        self.assertFalse(_should_persist_intent("quant_qa"))
        self.assertFalse(_should_persist_intent("strategy_revision"))
        self.assertFalse(_should_persist_intent("off_topic"))

        revision_reply = _deterministic_non_persistent_reply("strategy_revision", "增加一个ADX过滤条件")
        self.assertIsNotNone(revision_reply)
        self.assertIn("不会直接覆盖", revision_reply)
        self.assertIn("不会写入策略工作台", revision_reply)

        startup_reply = _deterministic_non_persistent_reply("platform_help", "怎么打开测试")
        self.assertIsNotNone(startup_reply)
        self.assertIn("uvicorn", startup_reply)
        self.assertIn("8015", startup_reply)
        self.assertIsNone(_deterministic_non_persistent_reply("off_topic", "今天吃什么"))

    def test_strategy_qa_focus_note_for_stop_loss_questions(self):
        from app.chat import _strategy_qa_focus_note

        note = _strategy_qa_focus_note("止损退出占比100%，策略可能没有充分让趋势奔跑这是为什么")
        self.assertIn("question_focus", note)
        self.assertIn("exit_signals_seen", note)
        self.assertIn("model_py_contains_exit_signal", note)


class StrategySpecTests(unittest.TestCase):
    def test_strategy_spec_contains_current_signal_contract(self):
        from engine.strategy_spec import build_strategy_spec

        spec = build_strategy_spec(
            """
            ## 策略核心思想
            双均线趋势跟踪。

            ## 入场规则
            短期均线上穿长期均线做多，下穿做空。

            ## 出场规则
            反向信号平仓或反手。
            """
        )
        constraints = spec["implementation_constraints"]
        self.assertIn("signal_raw", constraints)
        self.assertIn("exit_signal", constraints)
        self.assertIn("反手", constraints["signal_raw"])
        self.assertIn("只平仓", constraints["exit_signal"])

    def test_strategy_spec_uses_defaults_for_timing_but_blocks_missing_entry(self):
        from engine.strategy_spec import build_strategy_spec

        timing_default_spec = build_strategy_spec(
            """
            ## 策略核心思想
            均线斜率趋势策略。

            ## 指标计算公式
            计算20日均线和斜率。

            ## 入场规则
            斜率向上且价格突破高点做多。

            ## 出场规则
            使用移动止损。
            """
        )
        gate = timing_default_spec["stage2_gate"]
        self.assertTrue(all("成交" not in item and "时点" not in item for item in gate["blocking_questions"]))

        missing_entry_spec = build_strategy_spec(
            """
            ## 策略核心思想
            均线斜率趋势策略。

            ## 指标计算公式
            计算20日均线和斜率。

            ## 出场规则
            跌破均线出场。
            """
        )
        gate = missing_entry_spec["stage2_gate"]
        self.assertFalse(gate["stage2_allowed"])
        self.assertTrue(any("入场" in item for item in gate["blocking_questions"]))


class Stage2PersistenceTests(unittest.TestCase):
    def test_stage2_artifact_saves_model_for_active_draft(self):
        from app import chat
        from engine import strategy_manager as sm

        sid = "test_stage2_persistence_contract"
        draft_dir = sm.DRAFT_DIR / sid
        previous_active = sm.ACTIVE_FILE.read_text(encoding="utf-8") if sm.ACTIVE_FILE.exists() else ""
        if draft_dir.exists():
            shutil.rmtree(draft_dir)
        draft_dir.mkdir(parents=True)
        try:
            sm.set_active(sid, "draft")
            code = '''
```python
import pandas as pd


class PersistedTestStrategy:
    def __init__(self, params: dict):
        self.N = params.get("N", 20)
        self.contract_multiplier = params.get("contract_multiplier", 10)
        self.commission_rate = params.get("commission_rate", 0.0001)
        self.slippage = params.get("slippage", 1)
        self.margin_rate = params.get("margin_rate", 0.08)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ma"] = df["close"].rolling(self.N).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal_raw"] = 0
        df["exit_signal"] = 0
        df.loc[df["close"] > df["ma"], "signal_raw"] = 1
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generate_signals(self.compute_indicators(df))
```
'''
            result = chat.save_stage2_artifact_from_content(code, sid=sid)
            self.assertTrue(result["ok"])
            self.assertTrue((draft_dir / "model.py").exists())
            self.assertTrue((draft_dir / "params.yaml").exists())
            self.assertTrue(chat.is_stage2_ready())
            validation = chat.validate_active_strategy()
            self.assertTrue(validation["ok"])
        finally:
            if draft_dir.exists():
                shutil.rmtree(draft_dir)
            if previous_active:
                sm.ACTIVE_FILE.write_text(previous_active, encoding="utf-8")
            chat.set_stage(1)

    def test_stage2_accepts_contract_class_without_strategy_suffix(self):
        from app import chat
        from engine import strategy_manager as sm

        sid = "test_stage2_plain_class_contract"
        draft_dir = sm.DRAFT_DIR / sid
        previous_active = sm.ACTIVE_FILE.read_text(encoding="utf-8") if sm.ACTIVE_FILE.exists() else ""
        if draft_dir.exists():
            shutil.rmtree(draft_dir)
        draft_dir.mkdir(parents=True)
        try:
            sm.set_active(sid, "draft")
            code = '''
```python
import pandas as pd


class TrendWeaver:
    def __init__(self, params: dict):
        self.N = params.get("N", 20)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ma"] = df["close"].rolling(self.N).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal_raw"] = 0
        df["exit_signal"] = 0
        df.loc[df["close"] > df["ma"], "signal_raw"] = 1
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generate_signals(self.compute_indicators(df))
```
'''
            result = chat.save_stage2_artifact_from_content(code, sid=sid)
            self.assertTrue(result["ok"])
            strategy_class = sm.load_module(sid)
            self.assertEqual(strategy_class.__name__, "TrendWeaver")
        finally:
            if draft_dir.exists():
                shutil.rmtree(draft_dir)
            if previous_active:
                sm.ACTIVE_FILE.write_text(previous_active, encoding="utf-8")
            chat.set_stage(1)

    def test_stage1_gate_blocks_advance_even_when_model_exists(self):
        from app import chat
        from engine import strategy_manager as sm

        sid = "test_stage1_gate_blocks_advance"
        draft_dir = sm.DRAFT_DIR / sid
        previous_active = sm.ACTIVE_FILE.read_text(encoding="utf-8") if sm.ACTIVE_FILE.exists() else ""
        if draft_dir.exists():
            shutil.rmtree(draft_dir)
        draft_dir.mkdir(parents=True)
        try:
            sm.set_active(sid, "draft")
            (draft_dir / "README.md").write_text(
                """
## 策略核心思想
均线斜率策略。

## 指标计算公式
计算均线斜率。

## 出场规则
使用移动止损。
""",
                encoding="utf-8",
            )
            (draft_dir / "model.py").write_text(
                "class DummyStrategy:\n"
                "    def __init__(self, params): pass\n"
                "    def compute_indicators(self, df): return df\n"
                "    def generate_signals(self, df): return df\n"
                "    def run(self, df): return df\n",
                encoding="utf-8",
            )
            from engine.strategy_spec import save_strategy_spec
            save_strategy_spec(draft_dir, (draft_dir / "README.md").read_text(encoding="utf-8"))
            chat.set_stage(1)

            self.assertEqual(chat.infer_active_stage(), 1)
            result = chat.advance_stage(2)
            self.assertFalse(result["ok"])
            self.assertIn("阶段1未确认", result["error"])
        finally:
            if draft_dir.exists():
                shutil.rmtree(draft_dir)
            if previous_active:
                sm.ACTIVE_FILE.write_text(previous_active, encoding="utf-8")
            chat.set_stage(1)

    def test_stage1_confirmation_unlocks_stage2(self):
        from app import chat
        from engine import strategy_manager as sm

        sid = "test_stage1_confirmation_unlocks"
        draft_dir = sm.DRAFT_DIR / sid
        previous_active = sm.ACTIVE_FILE.read_text(encoding="utf-8") if sm.ACTIVE_FILE.exists() else ""
        if draft_dir.exists():
            shutil.rmtree(draft_dir)
        draft_dir.mkdir(parents=True)
        try:
            sm.set_active(sid, "draft")
            (draft_dir / "README.md").write_text(
                """
## 策略核心思想
均线策略。

## 指标计算公式
```python
MA = close.rolling(20).mean()
```

## 入场规则
信号确认时点：当前K线收盘后。
成交时点：下一根K线开盘价。
收盘价突破MA做多。

## 出场与风控
触发条件：收盘价跌破MA时出场。
止损触发后按下一根K线开盘价成交。

## 待确认问题
- 入场条件缺失：多头和空头开仓条件互相冲突，需要确认采用哪一组。
""",
                encoding="utf-8",
            )
            from engine.strategy_spec import save_strategy_spec
            save_strategy_spec(draft_dir, (draft_dir / "README.md").read_text(encoding="utf-8"))
            chat.set_stage(1)

            self.assertFalse(chat.stage1_gate_status()["allowed"])
            confirm = chat.confirm_stage1_gate()
            self.assertTrue(confirm["ok"])
            self.assertTrue(chat.stage1_gate_status()["allowed"])
            result = chat.advance_stage(2)
            self.assertTrue(result["ok"])
        finally:
            if draft_dir.exists():
                shutil.rmtree(draft_dir)
            if previous_active:
                sm.ACTIVE_FILE.write_text(previous_active, encoding="utf-8")
            chat.set_stage(1)

    def test_stage2_rejects_self_persistent_position_state(self):
        from app import chat
        from engine import strategy_manager as sm

        sid = "test_stage2_rejects_self_state"
        draft_dir = sm.DRAFT_DIR / sid
        previous_active = sm.ACTIVE_FILE.read_text(encoding="utf-8") if sm.ACTIVE_FILE.exists() else ""
        if draft_dir.exists():
            shutil.rmtree(draft_dir)
        draft_dir.mkdir(parents=True)
        try:
            sm.set_active(sid, "draft")
            code = '''
```python
import pandas as pd


class BadStateStrategy:
    def __init__(self, params: dict):
        self.N = params.get("N", 20)
        self.position = 0

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ma"] = df["close"].rolling(self.N).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal_raw"] = 0
        df["exit_signal"] = 0
        if len(df) > self.N:
            df.loc[df.index[self.N], "signal_raw"] = 1
            self.position = 1
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generate_signals(self.compute_indicators(df))
```
'''
            result = chat.save_stage2_artifact_from_content(code, sid=sid)
            self.assertFalse(result["ok"])
            self.assertIn("不得用 self 持久化交易状态", result["error"])
        finally:
            if draft_dir.exists():
                shutil.rmtree(draft_dir)
            if previous_active:
                sm.ACTIVE_FILE.write_text(previous_active, encoding="utf-8")
            chat.set_stage(1)

    def test_stage2_rejects_underscored_self_trade_state(self):
        from app import chat
        from engine import strategy_manager as sm

        sid = "test_stage2_rejects_underscored_self_state"
        draft_dir = sm.DRAFT_DIR / sid
        previous_active = sm.ACTIVE_FILE.read_text(encoding="utf-8") if sm.ACTIVE_FILE.exists() else ""
        if draft_dir.exists():
            shutil.rmtree(draft_dir)
        draft_dir.mkdir(parents=True)
        try:
            sm.set_active(sid, "draft")
            code = '''
```python
import pandas as pd


class BadUnderscoreStateStrategy:
    def __init__(self, params: dict):
        self.N = params.get("N", 20)
        self._current_position = 0
        self._entry_price = 0.0
        self._bars_held = 0
        self._stop_price = 0.0
        self._has_added = False

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ma"] = df["close"].rolling(self.N).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal_raw"] = 0
        df["exit_signal"] = 0
        if len(df) > self.N:
            df.loc[df.index[self.N], "signal_raw"] = 1
            self._current_position = 1
            self._entry_price = float(df.loc[df.index[self.N], "open"])
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generate_signals(self.compute_indicators(df))
```
'''
            result = chat.save_stage2_artifact_from_content(code, sid=sid)
            self.assertFalse(result["ok"])
            self.assertIn("_current_position", result["error"])
            self.assertIn("_entry_price", result["error"])
        finally:
            if draft_dir.exists():
                shutil.rmtree(draft_dir)
            if previous_active:
                sm.ACTIVE_FILE.write_text(previous_active, encoding="utf-8")
            chat.set_stage(1)

    def test_stage2_controlled_codegen_uses_stage1_spec_after_ai_code_failure(self):
        from app import chat
        from engine import strategy_manager as sm
        from engine.strategy_spec import save_strategy_spec

        sid = "test_stage2_controlled_codegen"
        draft_dir = sm.DRAFT_DIR / sid
        previous_active = sm.ACTIVE_FILE.read_text(encoding="utf-8") if sm.ACTIVE_FILE.exists() else ""
        if draft_dir.exists():
            shutil.rmtree(draft_dir)
        draft_dir.mkdir(parents=True)
        try:
            sm.set_active(sid, "draft")
            readme = "\n".join([
                "## \u7b56\u7565\u6838\u5fc3\u601d\u60f3",
                "\u5747\u7ebf\u659c\u7387\u8d8b\u52bf\u8ddf\u8e2a\u7b56\u7565\uff0c\u4f7f\u7528\u5747\u7ebf\u659c\u7387\u5224\u65ad\u8d8b\u52bf\u65b9\u5411\uff0c\u5e76\u7ed3\u5408\u901a\u9053\u7a81\u7834\u5165\u573a\u3002",
                "## \u53c2\u6570\u5b9a\u4e49\u8868\u683c",
                "| \u53c2\u6570 | \u542b\u4e49 | \u9ed8\u8ba4\u503c | \u8303\u56f4 |",
                "| --- | --- | --- | --- |",
                "| N | \u5747\u7ebf\u5468\u671f\u4e0e\u901a\u9053\u7a81\u7834\u5468\u671f | 20 | 10-60 |",
                "| initial_stop_pct | \u521d\u59cb\u6b62\u635f\u6bd4\u4f8b | 0.02 | 0.01-0.05 |",
                "## \u6307\u6807\u8ba1\u7b97\u516c\u5f0f",
                "\u8ba1\u7b97N\u5468\u671f\u5747\u7ebf\u3001\u5747\u7ebf\u659c\u7387\u3001N\u5468\u671f\u901a\u9053\u9ad8\u4f4e\u70b9\u3002",
                "## \u5165\u573a\u89c4\u5219",
                "\u659c\u7387\u5411\u4e0a\u4e14\u6536\u76d8\u4ef7\u7a81\u7834N\u5468\u671f\u9ad8\u70b9\u505a\u591a\uff1b\u659c\u7387\u5411\u4e0b\u4e14\u6536\u76d8\u4ef7\u8dcc\u7834N\u5468\u671f\u4f4e\u70b9\u505a\u7a7a\u3002",
                "## \u51fa\u573a\u4e0e\u98ce\u63a7",
                "\u4f7f\u7528\u6301\u4ed3\u65f6\u95f4\u8870\u51cf\u7684\u81ea\u9002\u5e94\u79fb\u52a8\u6b62\u635f\uff0cK\u4ece1\u9012\u51cf\u81f30.3\u3002",
            ])
            (draft_dir / "README.md").write_text(readme, encoding="utf-8")
            save_strategy_spec(draft_dir, readme)
            bad_code = '''
```python
import pandas as pd

class BadGeneratedStrategy:
    def __init__(self, params: dict):
        self.N = params.get("N", 20)
        self._current_position = 0
        self._entry_price = 0
        self._stop_price = 0

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal_raw"] = 0
        df["exit_signal"] = 0
        self._current_position = 1
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generate_signals(self.compute_indicators(df))
```
'''
            result = chat.save_stage2_artifact_from_content(bad_code, sid=sid)
            self.assertTrue(result["ok"])
            self.assertTrue(result["controlled_codegen"])
            self.assertEqual(result["family"], "ma_slope_breakout")
            self.assertTrue((draft_dir / "stage2_codegen_trace.json").exists())
            code = (draft_dir / "model.py").read_text(encoding="utf-8")
            self.assertIn("strategy_stop_price", code)
            self.assertNotIn("self._current_position", code)
            validation = chat.validate_active_strategy()
            self.assertTrue(validation["ok"])
        finally:
            if draft_dir.exists():
                shutil.rmtree(draft_dir)
            if previous_active:
                sm.ACTIVE_FILE.write_text(previous_active, encoding="utf-8")
            chat.set_stage(1)

    def test_stage4_blockers_prevent_stage5_advance(self):
        from app import chat
        from engine import strategy_manager as sm

        sid = "test_stage4_blocks_stage5_contract"
        draft_dir = sm.DRAFT_DIR / sid
        report_dir = ROOT / "reports" / "web"
        report_dir.mkdir(parents=True, exist_ok=True)
        previous_active = sm.ACTIVE_FILE.read_text(encoding="utf-8") if sm.ACTIVE_FILE.exists() else ""
        previous_stage = chat.get_stage()
        if draft_dir.exists():
            shutil.rmtree(draft_dir)
        draft_dir.mkdir(parents=True)
        stats_path = report_dir / f"{sid}_RB0_stats.json"
        html_path = report_dir / f"{sid}_RB0_backtest.html"
        try:
            sm.set_active(sid, "draft")
            (draft_dir / "README.md").write_text(
                """
## 策略核心思想
测试策略。
## 指标计算公式
计算均线。
## 入场规则
收盘价突破均线入场。
## 出场与风控
跌破均线出场。
""",
                encoding="utf-8",
            )
            chat.confirm_stage1_gate()
            (draft_dir / "model.py").write_text(
                "class TestStrategy:\n"
                "    def __init__(self, params): pass\n"
                "    def compute_indicators(self, df): return df\n"
                "    def generate_signals(self, df): return df\n"
                "    def run(self, df): return df\n",
                encoding="utf-8",
            )
            (draft_dir / "check_report.md").write_text("阶段4检查：当前存在阻断项，不能进入阶段5。", encoding="utf-8")
            html_path.write_text("<html></html>", encoding="utf-8")
            stats_path.write_text(
                '{"research_gate":{"blockers":["strategy stop mismatch"]},"total_trades":5}',
                encoding="utf-8",
            )
            chat.set_stage(4)

            result = chat.advance_stage(5)
            self.assertFalse(result["ok"])
            self.assertIn("阶段4", result["error"])
        finally:
            if draft_dir.exists():
                shutil.rmtree(draft_dir)
            if stats_path.exists():
                stats_path.unlink()
            if html_path.exists():
                html_path.unlink()
            if previous_active:
                sm.ACTIVE_FILE.write_text(previous_active, encoding="utf-8")
            chat.set_stage(previous_stage)


class StrategyEvaluationTests(unittest.TestCase):
    def test_evaluation_downgrades_insufficient_or_unprofitable_evidence(self):
        from engine.evaluation import evaluate_strategy

        result = evaluate_strategy({
            "total_trades": 6,
            "net_profit": 1200,
            "total_return_pct": 1.2,
            "sharpe_ratio": 0.8,
            "max_drawdown_pct": -2.0,
            "return_drawdown_ratio": 0.6,
            "win_rate": 0.5,
            "payoff_ratio": 1.5,
            "profit_factor": 1.3,
        })
        self.assertEqual(result["rating"], "D")
        self.assertIn("样本不足", result["decision_state"])
        self.assertTrue(any("少于10笔" in flag for flag in result["hard_flags"]))

        poor = evaluate_strategy({
            "total_trades": 57,
            "net_profit": -500,
            "total_return_pct": -0.5,
            "sharpe_ratio": -0.1,
            "max_drawdown_pct": -4.0,
            "return_drawdown_ratio": 0,
            "win_rate": 0.5,
            "payoff_ratio": 0.9,
            "profit_factor": 0.95,
            "yearly": [{"year": 2024, "return_pct": -0.5}],
        })
        self.assertIn(poor["rating"], {"C", "D"})
        self.assertLessEqual(poor["total_score"], 49)
        self.assertTrue(any("未转正" in flag for flag in poor["hard_flags"]))

    def test_evaluation_rewards_balanced_profitable_evidence(self):
        from engine.evaluation import evaluate_strategy

        result = evaluate_strategy({
            "total_trades": 80,
            "net_profit": 35000,
            "total_return_pct": 35,
            "sharpe_ratio": 1.25,
            "max_drawdown_pct": -9,
            "return_drawdown_ratio": 3.8,
            "win_rate": 0.42,
            "payoff_ratio": 2.3,
            "profit_factor": 1.7,
            "total_slippage_cost": 800,
            "cost_profit_ratio": 6,
            "max_consecutive_losses": 4,
            "yearly": [
                {"year": 2022, "return_pct": 8},
                {"year": 2023, "return_pct": 14},
                {"year": 2024, "return_pct": -3},
                {"year": 2025, "return_pct": 12},
            ],
            "direction": [
                {"direction": "long", "net_profit": 20000},
                {"direction": "short", "net_profit": 15000},
            ],
            "data_rows": {"bars": 1000, "trades": 80, "equity": 1000},
            "diagnostics": {"entries_opened": 80},
        })
        self.assertIn(result["rating"], {"A", "B"})
        self.assertGreaterEqual(result["total_score"], 65)
        self.assertIn("收益能力", [item["name"] for item in result["dimensions"]])


class StrategyIterationTests(unittest.TestCase):
    def test_create_iteration_copies_parent_as_draft_without_downstream_reports(self):
        from engine import strategy_manager as sm

        parent = "test_parent_iteration_contract"
        parent_dir = sm.COMPLETED_DIR / parent
        draft_dirs_before = set(sm.DRAFT_DIR.glob(f"{parent}_iter_*"))
        previous_active = sm.ACTIVE_FILE.read_text(encoding="utf-8") if sm.ACTIVE_FILE.exists() else ""
        if parent_dir.exists():
            shutil.rmtree(parent_dir)
        parent_dir.mkdir(parents=True)
        try:
            (parent_dir / "model.py").write_text("class TestStrategy:\n    pass\n", encoding="utf-8")
            (parent_dir / "params.yaml").write_text("N: 20\n", encoding="utf-8")
            (parent_dir / "README.md").write_text("# 测试父策略\n", encoding="utf-8")
            (parent_dir / "check_report.md").write_text("old check", encoding="utf-8")
            (parent_dir / "analysis_template.md").write_text("old analysis", encoding="utf-8")
            (parent_dir / "meta.yaml").write_text(
                yaml.dump({"complete": True, "name": "测试父策略"}, allow_unicode=True),
                encoding="utf-8",
            )

            result = sm.create_strategy_iteration(parent, "add_filter", "增加ADX过滤", "测试父策略 v2")
            draft_dir = sm.DRAFT_DIR / result["strategy"]
            self.assertTrue(draft_dir.exists())
            self.assertTrue((draft_dir / "iteration.yaml").exists())
            self.assertFalse((draft_dir / "check_report.md").exists())
            self.assertFalse((draft_dir / "analysis_template.md").exists())
            self.assertEqual((parent_dir / "check_report.md").read_text(encoding="utf-8"), "old check")

            meta = yaml.safe_load((draft_dir / "meta.yaml").read_text(encoding="utf-8"))
            iteration = yaml.safe_load((draft_dir / "iteration.yaml").read_text(encoding="utf-8"))
            self.assertFalse(meta["complete"])
            self.assertEqual(meta["parent_strategy"], parent)
            self.assertEqual(iteration["parent_strategy"], parent)
            self.assertEqual(iteration["iteration_type"], "add_filter")
        finally:
            if parent_dir.exists():
                shutil.rmtree(parent_dir)
            for d in set(sm.DRAFT_DIR.glob(f"{parent}_iter_*")) - draft_dirs_before:
                if d.exists():
                    shutil.rmtree(d)
            if previous_active:
                sm.ACTIVE_FILE.write_text(previous_active, encoding="utf-8")


class BacktestEngineContractTests(unittest.TestCase):
    def _synthetic_bars(self):
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=9, freq="D"),
            "open": [100, 101, 102, 103, 104, 103, 102, 101, 100],
            "high": [102, 103, 104, 105, 105, 104, 103, 102, 101],
            "low": [99, 100, 101, 102, 103, 102, 101, 100, 99],
            "close": [101, 102, 103, 104, 103, 102, 101, 100, 99],
            "volume": [1000] * 9,
            # Day 0 signal opens long on day 1; day 1 signal adds on day 2.
            # Day 3 exit signal closes long on day 4.
            # Day 4 short signal opens short on day 5 and is final-closed.
            "signal_raw": [1, 1, 0, 0, -1, 0, 0, 0, 0],
            "exit_signal": [0, 0, 0, 1, 0, 0, 0, 0, 0],
        })

    def test_engine_executes_add_exit_and_slippage_accounting(self):
        from engine.backtest import BacktestEngine

        result = BacktestEngine(100000).run(
            self._synthetic_bars(),
            {
                "contract_multiplier": 10,
                "margin_rate": 0.08,
                "max_position": 2,
                "slippage": 1,
                "init_stop": 0.5,
                "commission_rate": 0.0,
            },
        )
        trades = result["trades"]
        diagnostics = result["stats"]["diagnostics"]

        self.assertEqual(diagnostics["entries_opened"], 2)
        self.assertEqual(diagnostics["add_entries_opened"], 1)
        self.assertEqual(diagnostics["strategy_exits"], 1)
        self.assertEqual(diagnostics["forced_final_close"], 1)
        self.assertEqual(int(trades.iloc[0]["position_size"]), 2)
        self.assertEqual(trades.iloc[0]["exit_reason"], "strategy_exit")
        self.assertGreater(float(trades["slippage_cost"].sum()), 0)

    def test_engine_uses_strategy_supplied_trailing_stop(self):
        from engine.backtest import BacktestEngine

        bars = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "open": [100, 101, 103, 102, 101],
            "high": [101, 104, 105, 103, 102],
            "low": [99, 100, 100, 98, 99],
            "close": [100, 103, 102, 100, 101],
            "volume": [1000] * 5,
            "signal_raw": [1, 0, 0, 0, 0],
            "exit_signal": [0, 0, 0, 0, 0],
            "strategy_stop_price": [0, 0, 100.5, 0, 0],
        })
        result = BacktestEngine(100000).run(
            bars,
            {
                "contract_multiplier": 10,
                "margin_rate": 0.08,
                "slippage": 0,
                "commission_rate": 0,
                "init_stop": 0.5,
            },
        )
        trades = result["trades"]
        diagnostics = result["stats"]["diagnostics"]

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "strategy_trailing_stop")
        self.assertEqual(diagnostics["strategy_stop_exits"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
