"""Deterministic stage 4/5 report builders.

These reports are generated from local artifacts and backtest evidence. They
should be conservative: if the platform cannot prove something from README,
model.py, trades/equity/stats, it should say so instead of filling the gap with
generic AI language.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read(path: Path, limit: int = 12000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:limit]


def _num(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = data.get(key, default)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "0.00%"


def _money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "0.00"


def _plain(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value if value is not None else "")


def _load_spec(strategy_dir: Path) -> Dict[str, Any]:
    path = strategy_dir / "strategy_spec.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _strategy_name(strategy_dir: Path) -> str:
    spec = _load_spec(strategy_dir)
    name = str(spec.get("strategy_name") or "").strip()
    if name:
        return name
    readme = _read(strategy_dir / "README.md", 2000)
    for line in readme.splitlines():
        line = line.strip().lstrip("#").strip()
        if line and len(line) <= 60:
            return line
    return strategy_dir.name


def _bullet_list(items: List[Any]) -> str:
    return "\n".join(f"- {str(item).strip()}" for item in items if str(item).strip())


def _metrics_table(summary: Dict[str, Any]) -> str:
    win_rate = _num(summary, "win_rate") * 100
    rows = [
        ("交易次数", str(int(_num(summary, "total_trades"))), "样本量决定结论可信度"),
        ("净盈利", _money(summary.get("net_profit", 0)), "来自交易明细净盈亏求和"),
        ("总收益率", _pct(summary.get("total_return_pct", 0)), "相对初始资金的收益幅度"),
        ("最大回撤", _pct(summary.get("max_drawdown_pct", 0)), "资金曲线承受的最大压力"),
        ("夏普比率", _plain(summary.get("sharpe_ratio", 0)), "风险调整后收益"),
        ("胜率", f"{win_rate:.1f}%", "盈利交易占比"),
        ("盈亏比", _plain(summary.get("payoff_ratio", 0)), "平均盈利单 / 平均亏损单"),
        ("成本/净利润", _pct(summary.get("cost_profit_ratio", 0)), "成本对收益的侵蚀程度"),
    ]
    text = "| 指标 | 数值 | 解释 |\n| :--- | :---: | :--- |\n"
    for key, value, note in rows:
        text += f"| {key} | {value} | {note} |\n"
    return text


def _exit_label(reason: str) -> str:
    return {
        "strategy_trailing_stop": "策略动态移动止损",
        "strategy_exit": "策略主动离场",
        "stop_loss": "通用止损",
        "signal_reversal": "反向信号平仓",
        "reverse_signal": "反向信号平仓",
        "final_close": "期末强制平仓",
        "exit_signal": "只平仓信号",
    }.get(reason, reason or "未记录")


def _behavior(summary: Dict[str, Any]) -> Dict[str, Any]:
    value = summary.get("behavior_diagnostics")
    return value if isinstance(value, dict) else {}


def _exit_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = _behavior(summary).get("exit_reason")
    return rows if isinstance(rows, list) else []


def _dominant_exit_text(summary: Dict[str, Any]) -> str:
    rows = _exit_rows(summary)
    if not rows:
        return "当前回测没有提供退出原因拆分，需要补充交易明细后再判断出场质量。"
    total = sum(int(row.get("trade_count", 0) or 0) for row in rows) or 1
    main = sorted(rows, key=lambda row: int(row.get("trade_count", 0) or 0), reverse=True)[0]
    reason = str(main.get("reason") or "")
    share = int(main.get("trade_count", 0) or 0) / total * 100
    avg_bars = _plain(main.get("avg_bars_held", 0), 1)
    text = f"主要退出方式是 **{_exit_label(reason)}**，占比 {share:.1f}%，平均持仓 {avg_bars} 根K线。"
    if reason == "strategy_trailing_stop":
        text += " 这说明策略原文中的移动止损已经参与交易闭环，阶段四需要重点复核止损线是否真按开仓后的有利极值移动。"
    elif reason in {"signal_reversal", "reverse_signal"}:
        text += " 这说明交易更多由反向趋势信号结束，需要检查反手时点是否与阶段一规则一致。"
    elif reason == "final_close":
        text += " 期末强平占比高时，结果容易受回测结束日期影响。"
    return text


def _evidence_text(summary: Dict[str, Any]) -> str:
    trades = int(_num(summary, "total_trades"))
    period = summary.get("period") if isinstance(summary.get("period"), dict) else {}
    parts: List[str] = []
    if period:
        parts.append(f"回测区间为 {period.get('start', '')} 至 {period.get('end', '')}。")
    if trades == 0:
        parts.append("当前没有产生交易，不能判断策略有效性，只能回到阶段2检查信号链路。")
    elif trades < 10:
        parts.append("交易样本明显不足，只能证明代码跑通，不能证明策略有效。")
    elif trades < 30:
        parts.append("交易样本偏少，可以作为初筛证据，但不能直接得出长期稳定结论。")
    else:
        parts.append("交易样本达到初步观察门槛，但仍需要多品种、参数稳定性和真实数据口径验证。")
    data_policy = summary.get("data_policy") if isinstance(summary.get("data_policy"), dict) else {}
    if data_policy:
        parts.append(
            "数据口径："
            + str(data_policy.get("provider", ""))
            + " / "
            + str(data_policy.get("contract_mode", ""))
            + "。"
        )
    return "\n\n".join(part for part in parts if part.strip())


def _result_judgement(summary: Dict[str, Any]) -> str:
    trades = int(_num(summary, "total_trades"))
    net_profit = _num(summary, "net_profit")
    sharpe = _num(summary, "sharpe_ratio")
    payoff = _num(summary, "payoff_ratio")
    if trades == 0:
        return "当前版本没有交易，优先检查阶段2信号条件是否过严或字段映射是否错误。"
    if net_profit > 0 and sharpe >= 0.8 and payoff >= 1.5:
        return "当前版本形成了初步正收益证据，值得继续研究，但还不能直接推导为可实盘策略。"
    if net_profit > 0:
        return "当前版本收益为正，但说服力有限，更适合作为下一轮迭代的基线。"
    return "当前版本尚未形成正收益证据，需要先定位信号质量、品种适配和交易成本问题。"


def _risk_points(summary: Dict[str, Any]) -> List[str]:
    trades = int(_num(summary, "total_trades"))
    net_profit = _num(summary, "net_profit")
    max_dd = _num(summary, "max_drawdown_pct")
    slippage = _num(summary, "total_slippage_cost")
    risks: List[str] = []
    if trades < 30:
        risks.append("样本量不足，当前结论只能作为初筛，不能支撑长期稳定判断。")
    if net_profit <= 0:
        risks.append("收益尚未转正，策略有效性没有被本轮回测证据支持。")
    if max_dd < -10:
        risks.append("最大回撤偏高，后续需要优先处理风控和仓位。")
    if slippage == 0 and trades > 0:
        risks.append("滑点成本为 0 或没有充分体现，真实交易中可能高估收益。")
    data_policy = summary.get("data_policy") if isinstance(summary.get("data_policy"), dict) else {}
    if "连续" in str(data_policy.get("contract_mode", "")):
        risks.append("当前仍是连续合约口径，真实换月日期、展期价差和换月滑点需要补充。")
    if not risks:
        risks.append("当前没有单一阻断风险，但仍需补充多品种、参数稳定性和真实数据口径验证。")
    return risks


def _verification_focus(summary: Dict[str, Any]) -> List[str]:
    behavior = _behavior(summary)
    rows = _exit_rows(summary)
    focus = [
        "入场复核：抽取典型交易，检查入场信号日前一根K线是否满足阶段一的趋势、形态、动能和突破条件。",
        "成交复核：确认信号在收盘后产生，回测引擎按下一根K线开盘价成交，而不是用信号K线价格成交。",
        "手数复核：当前 v1 固定 1 手时，应明确它只是复现口径，不代表实盘资金管理。",
        "成本复核：检查手续费、滑点诊断值、保证金口径是否已经在交易明细中可追溯。",
    ]
    if rows:
        main = sorted(rows, key=lambda row: int(row.get("trade_count", 0) or 0), reverse=True)[0]
        if str(main.get("reason") or "") == "strategy_trailing_stop":
            focus.append("出场复核：重点检查每笔交易的最高价/最低价、K值和动态止损价，确认退出确实由移动止损触发。")
    if float(behavior.get("avg_bars_held", 0) or 0) < 5 and int(_num(summary, "total_trades")) > 0:
        focus.append("持仓复核：平均持仓偏短时，需要判断这是策略原文限制导致，还是代码实现让止损过早收紧。")
    return focus


def _direction_table(summary: Dict[str, Any]) -> str:
    rows = summary.get("direction") if isinstance(summary.get("direction"), list) else []
    if not rows:
        return ""
    text = "| 方向 | 净盈利 | 交易次数 | 胜率 | 盈亏比 |\n| :--- | ---: | ---: | ---: | ---: |\n"
    labels = {"long": "多头", "short": "空头"}
    for row in rows:
        direction = labels.get(str(row.get("direction", "")), row.get("direction", ""))
        text += (
            f"| {direction} | {_money(row.get('net_profit', 0))} | {row.get('trade_count', 0)} | "
            f"{_num(row, 'win_rate') * 100:.1f}% | {_plain(row.get('payoff_ratio', 0))} |\n"
        )
    return text


def _yearly_table(summary: Dict[str, Any]) -> str:
    yearly = summary.get("yearly") if isinstance(summary.get("yearly"), list) else []
    if not yearly:
        return ""
    text = "| 年份 | 收益率 | 最大回撤 | 夏普 | 交易次数 |\n| :--- | ---: | ---: | ---: | ---: |\n"
    for row in yearly[:8]:
        text += (
            f"| {row.get('year', '')} | {_pct(row.get('return_pct', 0))} | "
            f"{_pct(row.get('max_drawdown_pct', 0))} | {_plain(row.get('sharpe_ratio', 0))} | "
            f"{row.get('trade_count', 0)} |\n"
        )
    return text


def _scorecard_short(summary: Dict[str, Any]) -> str:
    evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
    dimensions = evaluation.get("dimensions") if isinstance(evaluation.get("dimensions"), list) else []
    if not dimensions:
        return ""
    text = "| 维度 | 分数 | 证据 |\n| :--- | :---: | :--- |\n"
    for item in dimensions[:6]:
        evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
        note = "；".join(str(x) for x in evidence[:2])
        text += f"| {item.get('name', '')} | {item.get('score', 0)}/5 | {note} |\n"
    return text


def _next_steps(summary: Dict[str, Any]) -> List[str]:
    gate = summary.get("research_gate") if isinstance(summary.get("research_gate"), dict) else {}
    configured = gate.get("next_steps") if isinstance(gate.get("next_steps"), list) else []
    if configured:
        return [str(x) for x in configured[:4]]
    trades = int(_num(summary, "total_trades"))
    net_profit = _num(summary, "net_profit")
    if trades == 0:
        return ["回到阶段2检查入场条件是否过严。", "输出每个核心条件的命中次数，定位没有交易的原因。"]
    if net_profit <= 0:
        return ["定位亏损来源：方向、退出原因、交易成本和无效信号密集区。", "保留当前版本作为基线，再做单变量规则修正。"]
    return ["保留当前版本作为 v1 基线。", "先做单品种参数稳定性测试，再扩展多品种验证。", "抽样复核关键交易，确认入场和出场符合阶段1。"]


def _code_audit(strategy_dir: Path) -> Dict[str, Any]:
    code = _read(strategy_dir / "model.py", 80000)
    if not code:
        return {"ok": False, "items": [("代码文件", "缺少 model.py", "阻断", "阶段2没有形成可回测代码")]}

    items: List[tuple[str, str, str, str]] = []
    required_methods = {"__init__", "compute_indicators", "generate_signals", "run"}
    try:
        tree = ast.parse(code)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        method_sets = [{item.name for item in node.body if isinstance(item, ast.FunctionDef)} for node in classes]
        has_strategy_class = any(required_methods.issubset(methods) for methods in method_sets)
    except Exception as exc:
        has_strategy_class = False
        items.append(("语法检查", "Python 代码无法解析", "阻断", str(exc)))

    items.append((
        "策略接口",
        "包含 __init__ / compute_indicators / generate_signals / run" if has_strategy_class else "缺少标准策略接口",
        "通过" if has_strategy_class else "阻断",
        "回测引擎只调用标准 Strategy 接口",
    ))
    items.append((
        "入场信号",
        "输出 signal_raw" if "signal_raw" in code else "未发现 signal_raw",
        "通过" if "signal_raw" in code else "阻断",
        "signal_raw 是阶段3开仓的唯一方向字段",
    ))
    items.append((
        "出场信号",
        "提供 exit_signal 或 strategy_stop_price" if ("exit_signal" in code or "strategy_stop_price" in code) else "未显式输出出场字段",
        "通过" if ("exit_signal" in code or "strategy_stop_price" in code) else "提醒",
        "出场必须能在交易明细中追踪原因",
    ))
    items.append((
        "防未来函数",
        "检测到 shift/rolling 历史窗口" if (".shift(" in code or ".rolling(" in code) else "未检测到 shift/rolling",
        "通过" if (".shift(" in code or ".rolling(" in code) else "提醒",
        "通道、均线状态、突破条件原则上应使用历史窗口",
    ))
    risky_state = [token for token in ["self.position", "self._current_position", "self._entry_price", "self._bars_held", "self._stop_price"] if token in code]
    items.append((
        "交易状态归属",
        "未发现高风险 self 持仓状态" if not risky_state else "发现策略类持久化交易状态",
        "通过" if not risky_state else "阻断",
        "持仓、成交、成本、交易记录应由回测引擎统一处理" if not risky_state else "、".join(risky_state),
    ))
    return {"ok": not any(row[2] == "阻断" for row in items), "items": items}


def _report_root(strategy_dir: Path) -> Path:
    # strategies/<sid> -> project root
    if strategy_dir.parent.name in {"strategies", "strategies_drafts"}:
        return strategy_dir.parent.parent
    return strategy_dir.parents[1] if len(strategy_dir.parents) >= 2 else strategy_dir.parent


def _trade_examples_text(strategy_dir: Path, evidence: Dict[str, Any]) -> str:
    report_url = str(evidence.get("report_url", "") if isinstance(evidence, dict) else "")
    if not report_url.endswith("_backtest.html"):
        return ""
    root = _report_root(strategy_dir)
    stem = Path(report_url).name[:-len("_backtest.html")]
    trades_path = root / "reports" / "web" / f"{stem}_trades.csv"
    if not trades_path.exists():
        return ""
    try:
        with trades_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return ""
    if not rows:
        return ""

    def fnum(row: Dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key, 0) or 0)
        except Exception:
            return 0.0

    selected: List[tuple[str, Dict[str, Any]]] = [("首笔交易", rows[0])]
    selected.append(("最大盈利", max(rows, key=lambda r: fnum(r, "net_pnl"))))
    selected.append(("最大亏损", min(rows, key=lambda r: fnum(r, "net_pnl"))))
    seen = set()
    text = "## 三笔交易逐项验证\n\n"
    text += "下面三笔来自交易明细 CSV，用来快速复核信号、成交、手数、费用和净盈亏，不需要研究员再手动打开表格计算。\n"
    for label, row in selected:
        key = (row.get("entry_date"), row.get("exit_date"), row.get("net_pnl"))
        if key in seen:
            continue
        seen.add(key)
        direction = "多头" if row.get("direction") == "long" else "空头" if row.get("direction") == "short" else row.get("direction", "")
        sign = 1 if row.get("direction") == "long" else -1
        entry = fnum(row, "entry_price")
        exit_price = fnum(row, "exit_price")
        size = fnum(row, "position_size") or 1
        gross = fnum(row, "gross_pnl")
        booked_cost = fnum(row, "total_cost") or fnum(row, "fee")
        economic_cost = fnum(row, "economic_cost") or booked_cost
        slippage = fnum(row, "slippage_cost")
        net = fnum(row, "net_pnl")
        multiplier = abs(gross / ((exit_price - entry) * sign * size)) if exit_price != entry else 0
        text += f"\n### {label}\n\n"
        text += (
            f"- 方向：{direction}；信号日：{row.get('entry_signal_date') or row.get('entry_date')}；"
            f"成交日：{row.get('entry_date')}；成交方式：{row.get('entry_fill_mode', 'next_open')}。\n"
        )
        text += (
            f"- 入场价：{_plain(entry)}；出场日：{row.get('exit_date')}；出场价：{_plain(exit_price)}；"
            f"出场方式：{row.get('exit_fill_mode', '')}；退出原因：{_exit_label(row.get('exit_reason', ''))}；手数：{_plain(size, 0)}。\n"
        )
        text += "```text\n"
        text += "毛盈亏 = 方向 × (出场价 - 入场价) × 合约乘数 × 手数\n"
        text += f"      = {sign:+d} × ({_plain(exit_price)} - {_plain(entry)}) × {_plain(multiplier)} × {_plain(size, 0)}\n"
        text += f"      = {_money(gross)}\n"
        text += "净盈亏 = 毛盈亏 - 引擎入账成本\n"
        text += f"      = {_money(gross)} - {_money(booked_cost)}\n"
        text += f"      = {_money(net)}\n"
        text += f"经济成本观察值 = {_money(economic_cost)}，其中滑点诊断值 = {_money(slippage)}\n"
        text += "```\n"
    return text


def build_stage4_check_report(strategy_dir: Path, evidence: Dict[str, Any]) -> str:
    """Build a trade-audit oriented stage-4 report."""
    summary = evidence.get("summary") if isinstance(evidence, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    gate = summary.get("research_gate") if isinstance(summary.get("research_gate"), dict) else {}
    research_blockers = gate.get("blockers") if isinstance(gate.get("blockers"), list) else []
    code = _code_audit(strategy_dir)
    code_blockers = [row[3] for row in code["items"] if row[2] == "阻断"]
    can_continue = not code_blockers
    name = _strategy_name(strategy_dir)

    text = "## 检查结论\n\n"
    if can_continue:
        text += (
            f"{name} 的阶段1说明、阶段2代码和阶段3回测结果已经形成基本闭环。"
            "阶段4的目标不是证明策略已经能实盘，而是复核：代码是否按阶段1执行，成交时点、手数、成本和退出原因是否能被交易明细解释。\n\n"
        )
        text += f"结论：**可以进入阶段5，但阶段5必须保持保守。** {_result_judgement(summary)}\n"
    else:
        text += (
            f"{name} 当前存在技术阻断项，不能把阶段3结果作为策略有效性证据。"
            "应先修复代码接口、信号生成或回测前置问题，再重新回测。\n"
        )

    text += "\n## 本阶段要验证什么\n\n"
    text += _bullet_list(_verification_focus(summary)) + "\n"

    text += "\n## 规则与代码一致性\n\n"
    text += "| 检查项 | 状态 | 证据 |\n| :--- | :---: | :--- |\n"
    for item, actual, status, evidence_text in code["items"]:
        text += f"| {item} | {status} | {actual}；{evidence_text} |\n"

    text += "\n## 回测证据\n\n"
    text += _metrics_table(summary)
    text += "\n"
    text += _evidence_text(summary) + "\n"
    text += "\n## 退出原因复核\n\n"
    text += _dominant_exit_text(summary) + "\n"

    behavior = _behavior(summary)
    if behavior:
        text += "\n| 行为项 | 数值 |\n| :--- | ---: |\n"
        text += f"| 平均持仓 | {_plain(behavior.get('avg_bars_held', 0), 1)} 根K线 |\n"
        text += f"| 持仓中位数 | {_plain(behavior.get('median_bars_held', 0), 1)} 根K线 |\n"
        text += f"| 最长持仓 | {_plain(behavior.get('max_bars_held', 0), 0)} 根K线 |\n"
        text += f"| 原始信号 | {_plain(behavior.get('signal_count', 0), 0)} 次 |\n"
        text += f"| 实际开仓 | {_plain(behavior.get('entries_opened', 0), 0)} 次 |\n"
        text += f"| 信号转化率 | {_num(behavior, 'signal_to_entry_rate') * 100:.1f}% |\n"

    trade_examples = _trade_examples_text(strategy_dir, evidence)
    if trade_examples:
        text += "\n" + trade_examples

    direction = _direction_table(summary)
    if direction:
        text += "\n## 多空方向拆分\n\n" + direction

    diagnostics = summary.get("diagnostics") if isinstance(summary.get("diagnostics"), dict) else {}
    if diagnostics:
        text += "\n## 执行诊断\n\n"
        labels = {
            "raw_rows": "原始K线",
            "valid_rows": "有效K线",
            "invalid_rows_removed": "清洗剔除",
            "signals_seen": "原始信号",
            "entries_opened": "实际开仓",
            "add_entries_opened": "实际加仓",
            "orders_rejected_insufficient_cash": "资金不足拒单",
            "orders_rejected_volume_participation": "成交量参与率拒单",
            "orders_blocked_price_limit": "涨跌停阻断",
            "bars_skipped_low_volume": "低成交量跳过",
            "forced_final_close": "期末强制平仓",
        }
        text += "| 项目 | 数值 |\n| :--- | ---: |\n"
        for key, label in labels.items():
            if key in diagnostics:
                text += f"| {label} | {diagnostics[key]} |\n"

    risks = [*research_blockers, *_risk_points(summary)]
    text += "\n## 当前不能忽略的风险\n\n" + _bullet_list(list(dict.fromkeys(risks))) + "\n"
    return text


def build_stage5_analysis_report(strategy_dir: Path, evidence: Dict[str, Any]) -> str:
    """Build a concise but evidence-heavy stage-5 analysis report."""
    summary = evidence.get("summary") if isinstance(evidence, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    name = _strategy_name(strategy_dir)
    evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}

    text = "## 研究结论\n\n"
    text += f"{name} 当前的研究结论是："
    text += _result_judgement(summary)
    if evaluation:
        rating = evaluation.get("rating", "")
        decision = evaluation.get("decision", "")
        score = evaluation.get("score", "")
        if score in ("", None):
            dimensions = evaluation.get("dimensions") or []
            if isinstance(dimensions, list) and dimensions:
                try:
                    score = sum(float(item.get("score", 0) or 0) for item in dimensions)
                except Exception:
                    score = ""
        if rating or decision:
            text += f"\n\n平台评分卡给出的结果是 **{rating}**，总分 {score}。{decision}"

    text += "\n\n## 是否值得继续深入\n\n"
    trades = int(_num(summary, "total_trades"))
    net_profit = _num(summary, "net_profit")
    sharpe = _num(summary, "sharpe_ratio")
    payoff = _num(summary, "payoff_ratio")
    if trades == 0:
        text += "暂时不值得讨论优化方向。没有交易意味着当前版本还停留在代码/信号链路验证阶段。\n"
    elif net_profit > 0 and sharpe >= 0.8 and payoff >= 1.5:
        text += "值得继续深入，但下一步不是直接乐观推进，而是先确认结果是否来自稳定规则，而不是单一行情或单一品种。\n"
    elif net_profit > 0:
        text += "可以继续作为基线研究，但目前说服力还不够强。下一步应优先验证参数稳定性和品种适配性。\n"
    else:
        text += "不建议直接进入实盘模拟。应先定位亏损来源，再决定是修正规则、换品种，还是放弃该版本。\n"

    text += "\n## 关键绩效证据\n\n"
    text += _metrics_table(summary)

    yearly = _yearly_table(summary)
    if yearly:
        text += "\n## 年度稳定性\n\n" + yearly
        text += "\n年度表的作用不是看哪一年最好，而是看收益是否依赖少数年份。如果大部分收益集中在一两年，策略稳定性要打折。\n"

    text += "\n## 收益结构解释\n\n"
    win_rate = _num(summary, "win_rate") * 100
    text += (
        f"本轮胜率为 {win_rate:.1f}%，盈亏比为 {_plain(summary.get('payoff_ratio', 0))}，"
        f"最大回撤为 {_pct(summary.get('max_drawdown_pct', 0))}。"
    )
    text += " 对趋势策略来说，关键不是单看胜率，而是看盈利单是否能覆盖亏损单和交易成本。\n\n"
    text += _dominant_exit_text(summary) + "\n"

    direction = _direction_table(summary)
    if direction:
        text += "\n## 多空方向贡献\n\n" + direction

    scorecard = _scorecard_short(summary)
    if scorecard:
        text += "\n## 策略评分卡\n\n" + scorecard

    text += "\n## 继续研究的优先方向\n\n"
    steps = _next_steps(summary)
    text += _bullet_list(steps) + "\n"

    text += "\n## 如果进入模拟盘，需要先处理什么\n\n"
    text += _bullet_list([
        "确认数据口径：连续合约结果不能直接等同于真实可交易合约，换月日期、展期价差和换月滑点需要补充。",
        "确认成交口径：当前按收盘确认、下一根开盘成交；止损按触及止损价成交，需要进一步压力测试跳空和涨跌停。",
        "确认仓位口径：固定 1 手只适合复现和研究，真实模拟盘需要改成基于资金风险的手数计算。",
        "确认成本口径：手续费、滑点、冲击成本要按品种和交易频率重新校准。",
    ]) + "\n"

    text += "\n## 当前不能忽略的风险\n\n"
    text += _bullet_list(_risk_points(summary)) + "\n"
    return text
