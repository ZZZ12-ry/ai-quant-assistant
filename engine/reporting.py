"""Backtest report summarization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from engine.evaluation import evaluate_strategy


def _safe_round(value, digits=2):
    try:
        if pd.isna(value):
            return 0
        return round(float(value), digits)
    except Exception:
        return 0


def _read_csv(path: Path, parse_dates=None) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path, parse_dates=parse_dates)
    except EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _max_consecutive_losses(trades: pd.DataFrame) -> int:
    if trades.empty or "net_pnl" not in trades.columns:
        return 0
    longest = current = 0
    for pnl in trades["net_pnl"].fillna(0):
        if pnl < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _yearly_summary(equity: pd.DataFrame, trades: pd.DataFrame) -> list:
    if equity.empty or "date" not in equity.columns or "equity" not in equity.columns:
        return []
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
    eq = eq.dropna(subset=["date"]).sort_values("date")
    if eq.empty:
        return []
    eq["year"] = eq["date"].dt.year

    tr = trades.copy()
    if not tr.empty and "exit_date" in tr.columns:
        tr["exit_date"] = pd.to_datetime(tr["exit_date"], errors="coerce")
        tr["year"] = tr["exit_date"].dt.year
    else:
        tr["year"] = np.nan

    rows = []
    for year, group in eq.groupby("year"):
        if len(group) < 2:
            continue
        start = float(group["equity"].iloc[0])
        end = float(group["equity"].iloc[-1])
        returns = group["equity"].pct_change().dropna()
        peak = group["equity"].cummax()
        drawdown = group["equity"] / peak - 1
        sharpe = 0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        year_trades = tr[tr["year"] == year] if "year" in tr.columns else pd.DataFrame()
        rows.append({
            "year": int(year),
            "return_pct": _safe_round((end / start - 1) * 100 if start else 0),
            "max_drawdown_pct": _safe_round(drawdown.min() * 100),
            "sharpe_ratio": _safe_round(sharpe),
            "trade_count": int(len(year_trades)),
        })
    return rows


def _direction_summary(trades: pd.DataFrame) -> list:
    if trades.empty or "direction" not in trades.columns or "net_pnl" not in trades.columns:
        return []
    rows = []
    for direction, group in trades.groupby("direction"):
        wins = group[group["net_pnl"] > 0]
        losses = group[group["net_pnl"] < 0]
        avg_loss = abs(losses["net_pnl"].mean()) if len(losses) else 0
        rows.append({
            "direction": str(direction),
            "net_profit": _safe_round(group["net_pnl"].sum()),
            "trade_count": int(len(group)),
            "win_rate": _safe_round(len(wins) / len(group), 4) if len(group) else 0,
            "payoff_ratio": _safe_round(wins["net_pnl"].mean() / avg_loss) if avg_loss else 0,
        })
    return rows


def _exit_reason_summary(trades: pd.DataFrame) -> list:
    if trades.empty or "exit_reason" not in trades.columns:
        return []
    rows = []
    total = len(trades)
    for reason, group in trades.groupby("exit_reason"):
        rows.append({
            "reason": str(reason),
            "trade_count": int(len(group)),
            "share": _safe_round(len(group) / total, 4) if total else 0,
            "net_profit": _safe_round(group["net_pnl"].sum()) if "net_pnl" in group.columns else 0,
            "avg_bars_held": _safe_round(group["bars_held"].mean()) if "bars_held" in group.columns else 0,
        })
    return sorted(rows, key=lambda item: item["trade_count"], reverse=True)


def _behavior_diagnostics(trades: pd.DataFrame, summary: dict, diagnostics: dict) -> dict:
    """Summarize strategy behavior beyond headline PnL."""
    diagnostics = diagnostics or {}
    total_trades = int(summary.get("total_trades", len(trades)) or 0)
    exit_rows = _exit_reason_summary(trades)
    signal_count = int(diagnostics.get("signals_seen", 0) or 0)
    entries_opened = int(diagnostics.get("entries_opened", 0) or 0)
    exit_signals = int(diagnostics.get("exit_signals_seen", 0) or 0)

    avg_bars = _safe_round(trades["bars_held"].mean()) if not trades.empty and "bars_held" in trades.columns else 0
    median_bars = _safe_round(trades["bars_held"].median()) if not trades.empty and "bars_held" in trades.columns else 0
    max_bars = int(trades["bars_held"].max()) if not trades.empty and "bars_held" in trades.columns else 0
    conversion = _safe_round(entries_opened / signal_count, 4) if signal_count else 0

    flags = []
    stop_row = next((row for row in exit_rows if row["reason"] in {"stop_loss", "strategy_trailing_stop"}), None)
    if stop_row and total_trades and stop_row["share"] >= 0.8:
        flags.append(
            f"退出高度集中在{stop_row['reason']}，占比{stop_row['share']:.0%}。"
            "需要重点判断这是策略原文的动态止损特征，还是止损过紧导致趋势被截断。"
        )
    if total_trades >= 10 and avg_bars and avg_bars < 3:
        flags.append(f"平均持仓仅{avg_bars}根K线，行为更接近短线突破试错，需要和原文的中长线定位对照。")

    directions = summary.get("direction") if isinstance(summary.get("direction"), list) else []
    losers = [row for row in directions if float(row.get("net_profit", 0) or 0) < 0]
    if total_trades and len(losers) == 1 and len(directions) > 1:
        row = losers[0]
        flags.append(f"{row.get('direction')}方向贡献主要亏损，需要单独复核该方向的过滤条件。")
    if signal_count and conversion < 0.7:
        flags.append(f"信号成交转化率仅{conversion:.0%}，需检查拒单、冲突信号或持仓上限。")
    if exit_signals > 0 and diagnostics.get("strategy_exits", 0) == 0:
        flags.append("策略产生过exit_signal，但实际成交未记录为strategy_exit，需要复核信号时点和引擎执行顺序。")

    return {
        "avg_bars_held": avg_bars,
        "median_bars_held": median_bars,
        "max_bars_held": max_bars,
        "exit_reason": exit_rows,
        "signal_count": signal_count,
        "exit_signal_count": exit_signals,
        "entries_opened": entries_opened,
        "signal_to_entry_rate": conversion,
        "flags": flags,
    }


def _research_gate(summary: dict) -> dict:
    """Turn evidence into the next research action."""
    total_trades = int(summary.get("total_trades", 0) or 0)
    net_profit = float(summary.get("net_profit", 0) or 0)
    total_return = float(summary.get("total_return_pct", 0) or 0)
    sharpe = float(summary.get("sharpe_ratio", 0) or 0)
    behavior = summary.get("behavior_diagnostics") if isinstance(summary.get("behavior_diagnostics"), dict) else {}
    behavior_flags = behavior.get("flags") if isinstance(behavior.get("flags"), list) else []

    blockers = []
    warnings = []
    next_steps = []
    status = "继续研究"

    if total_trades == 0:
        blockers.append("回测没有产生交易，不能评价策略有效性。")
        next_steps.append("先检查阶段2入场信号、过滤条件和无交易诊断，再重新回测。")
        status = "先修信号"
    elif total_trades < 30:
        warnings.append(f"交易样本只有{total_trades}笔，统计显著性不足。")
        next_steps.append("先延长样本或补充更多品种，再判断是否值得优化。")

    if net_profit <= 0 and total_return <= 0:
        blockers.append("净利润和总收益率均未转正。")
        next_steps.append("先定位亏损方向、退出原因和参数敏感性，不要直接进入模拟盘。")
    if sharpe <= 0 and total_trades >= 10:
        blockers.append("夏普比率不为正，风险调整后收益不足。")

    for flag in behavior_flags:
        if "strategy_exit" in flag or "信号成交转化率" in flag:
            blockers.append(flag)
        else:
            warnings.append(flag)

    if total_trades >= 30 and net_profit > 0 and sharpe > 0.5 and not blockers:
        status = "可扩展验证"
        next_steps.append("进入多品种验证，检查不同市场状态下收益是否稳定。")
    elif blockers:
        status = "先修复再复测"
        if not next_steps:
            next_steps.append("先修复阻断项，再重新运行阶段3回测。")
    elif warnings:
        status = "谨慎补证据"
        if not next_steps:
            next_steps.append("补充参数敏感性和更多品种证据。")

    if total_trades >= 30:
        next_steps.append("补跑关键参数敏感性：入场周期、退出周期、止损倍数、过滤阈值。")
        next_steps.append("保留当前版本作为基线，创建迭代版本逐项修改。")

    return {
        "status": status,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "next_steps": list(dict.fromkeys(next_steps))[:5],
    }


def _no_trade_diagnosis(bars: pd.DataFrame, diagnostics: dict) -> list:
    """Explain why a backtest produced zero trades using observable evidence."""
    notes = []
    diagnostics = diagnostics or {}
    signals_seen = int(diagnostics.get("signals_seen", 0) or 0)
    exit_signals_seen = int(diagnostics.get("exit_signals_seen", 0) or 0)
    entries_opened = int(diagnostics.get("entries_opened", 0) or 0)

    if not bars.empty and "error" in bars.columns:
        errors = bars["error"].dropna().astype(str)
        errors = errors[errors.str.strip().ne("")]
        if not errors.empty:
            top_error = errors.value_counts().index[0]
            notes.append({
                "title": "策略代码运行异常",
                "detail": f"bars.csv 的 error 列显示：{top_error}。这通常意味着阶段2代码访问了不存在的字段，异常兜底后 signal_raw 被置为0。",
                "suggestion": "优先检查阶段2字段映射和数据频率。若策略原文要求小时线/多周期，而当前回测只提供日线，应先改成日线近似口径或阻断进入回测。",
            })

    if signals_seen == 0:
        notes.append({
            "title": "没有产生入场信号",
            "detail": "策略输出的 signal_raw 全部为0，回测引擎没有收到开多或开空指令。",
            "suggestion": "优先检查阶段2代码中的入场条件是否过严，或是否被冲突处理清零。",
        })
    elif entries_opened == 0:
        notes.append({
            "title": "有入场信号但没有成交",
            "detail": "策略产生过 signal_raw，但回测引擎没有实际开仓。",
            "suggestion": "检查资金不足、成交量过滤、涨跌停阻断、保证金和滑点设置。",
        })

    if exit_signals_seen > 0 and signals_seen == 0:
        notes.append({
            "title": "空仓时产生了离场信号",
            "detail": f"检测到 exit_signal={exit_signals_seen} 次，但 signal_raw=0 次。常见原因是离场条件没有按持仓状态限制。",
            "suggestion": "阶段2代码应使用 position 状态机：空仓只判断入场，持仓后才判断离场。",
        })

    if not bars.empty and {"close", "entry_high", "entry_low"}.issubset(bars.columns):
        close = pd.to_numeric(bars["close"], errors="coerce")
        entry_high = pd.to_numeric(bars["entry_high"], errors="coerce")
        entry_low = pd.to_numeric(bars["entry_low"], errors="coerce")
        simple_long = int((close > entry_high).fillna(False).sum())
        simple_short = int((close < entry_low).fillna(False).sum())
        if simple_long + simple_short > 0 and signals_seen == 0:
            notes.append({
                "title": "价格曾经突破通道，但最终信号为0",
                "detail": f"检测到基础通道突破：多头 {simple_long} 次，空头 {simple_short} 次；但最终没有入场信号。",
                "suggestion": "检查趋势过滤、OBV过滤或条件组合是否过严。",
            })

    if diagnostics.get("orders_rejected_insufficient_cash", 0):
        notes.append({
            "title": "资金不足导致拒单",
            "detail": f"资金不足拒单 {diagnostics.get('orders_rejected_insufficient_cash')} 次。",
            "suggestion": "降低合约手数、保证金比例或提高初始资金。",
        })
    if diagnostics.get("orders_blocked_price_limit", 0):
        notes.append({
            "title": "涨跌停阻断成交",
            "detail": f"涨跌停阻断 {diagnostics.get('orders_blocked_price_limit')} 次。",
            "suggestion": "检查成交价假设和涨跌停约束是否过于保守。",
        })
    return notes


def summarize_report(report_dir: str | Path, stem: str, initial_capital: float = 100000) -> Optional[dict]:
    """Read engine CSV/JSON outputs and return a frontend-friendly summary."""
    report_path = Path(report_dir)
    trades = _read_csv(report_path / f"{stem}_trades.csv", parse_dates=["entry_date", "exit_date"])
    equity = _read_csv(report_path / f"{stem}_equity.csv", parse_dates=["date"])
    bars = _read_csv(report_path / f"{stem}_bars.csv", parse_dates=["date"])
    stats = _read_json(report_path / f"{stem}_stats.json")
    diagnostics = stats.get("diagnostics") if isinstance(stats.get("diagnostics"), dict) else {}

    if trades.empty and equity.empty:
        return None

    if not trades.empty:
        for col in ["net_pnl", "gross_pnl", "fees", "slippage_cost", "impact_cost", "bars_held"]:
            if col in trades.columns:
                trades[col] = pd.to_numeric(trades[col], errors="coerce").fillna(0)

    if not equity.empty and "equity" in equity.columns:
        equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce").ffill()
        equity = equity.dropna(subset=["equity"])

    total_trades = int(len(trades))
    wins = trades[trades["net_pnl"] > 0] if "net_pnl" in trades.columns else pd.DataFrame()
    losses = trades[trades["net_pnl"] < 0] if "net_pnl" in trades.columns else pd.DataFrame()
    net_profit = float(trades["net_pnl"].sum()) if "net_pnl" in trades.columns else 0
    fee_col = "fee" if "fee" in trades.columns else "fees" if "fees" in trades.columns else ""
    total_fees = float(trades[fee_col].sum()) if fee_col else 0
    total_slippage = float(trades["slippage_cost"].sum()) if "slippage_cost" in trades.columns else 0
    total_impact = float(trades["impact_cost"].sum()) if "impact_cost" in trades.columns else 0
    avg_win = float(wins["net_pnl"].mean()) if len(wins) else 0
    avg_loss = abs(float(losses["net_pnl"].mean())) if len(losses) else 0
    payoff_ratio = avg_win / avg_loss if avg_loss else 0
    profit_factor = abs(float(wins["net_pnl"].sum()) / float(losses["net_pnl"].sum())) if len(losses) and float(losses["net_pnl"].sum()) else 0

    if equity.empty:
        start_equity = float(initial_capital)
        end_equity = float(initial_capital + net_profit)
        max_drawdown_pct = 0
        sharpe = 0
        annual_vol = 0
    else:
        start_equity = float(equity["equity"].iloc[0])
        end_equity = float(equity["equity"].iloc[-1])
        returns = equity["equity"].pct_change().dropna()
        peak = equity["equity"].cummax()
        drawdown = equity["equity"] / peak - 1
        max_drawdown_pct = float(drawdown.min() * 100) if len(drawdown) else 0
        annual_vol = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 1 else 0
        sharpe = 0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float((returns.mean() * 252) / (returns.std() * np.sqrt(252)))

    total_return_pct = (end_equity / start_equity - 1) * 100 if start_equity else 0
    max_drawdown_abs = abs(max_drawdown_pct)
    return_drawdown_ratio = total_return_pct / max_drawdown_abs if max_drawdown_abs else 0
    cost_total = total_fees + total_slippage + total_impact
    cost_profit_ratio = (cost_total / net_profit * 100) if net_profit > 0 else 0

    period = {}
    if not equity.empty and "date" in equity.columns:
        period = {
            "start": str(pd.to_datetime(equity["date"].iloc[0]).date()),
            "end": str(pd.to_datetime(equity["date"].iloc[-1]).date()),
        }
    elif not bars.empty and "date" in bars.columns:
        period = {
            "start": str(pd.to_datetime(bars["date"].iloc[0]).date()),
            "end": str(pd.to_datetime(bars["date"].iloc[-1]).date()),
        }

    summary = {
        "total_trades": total_trades,
        "win_count": int(len(wins)),
        "loss_count": int(len(losses)),
        "win_rate": _safe_round(len(wins) / total_trades, 4) if total_trades else 0,
        "net_profit": _safe_round(net_profit),
        "total_return_pct": _safe_round(total_return_pct),
        "annual_volatility_pct": _safe_round(annual_vol),
        "profit_factor": _safe_round(profit_factor),
        "max_drawdown": _safe_round(max_drawdown_pct),
        "max_drawdown_pct": _safe_round(max_drawdown_pct),
        "return_drawdown_ratio": _safe_round(return_drawdown_ratio),
        "avg_win": _safe_round(avg_win),
        "avg_loss": _safe_round(avg_loss),
        "payoff_ratio": _safe_round(payoff_ratio),
        "sharpe_ratio": _safe_round(sharpe),
        "total_fees": _safe_round(total_fees),
        "total_commission": _safe_round(total_fees),
        "total_slippage_cost": _safe_round(total_slippage),
        "total_impact_cost": _safe_round(total_impact),
        "cost_profit_ratio": _safe_round(cost_profit_ratio),
        "max_consecutive_losses": _max_consecutive_losses(trades),
        "best_trade": _safe_round(trades["net_pnl"].max()) if "net_pnl" in trades.columns and not trades.empty else 0,
        "worst_trade": _safe_round(trades["net_pnl"].min()) if "net_pnl" in trades.columns and not trades.empty else 0,
        "start_equity": _safe_round(start_equity),
        "end_equity": _safe_round(end_equity),
        "yearly": _yearly_summary(equity, trades),
        "direction": _direction_summary(trades),
        "data_rows": {"bars": int(len(bars)), "trades": total_trades, "equity": int(len(equity))},
        "period": period,
        "diagnostics": diagnostics,
        "engine_stats": stats,
    }
    if isinstance(stats.get("data_policy"), dict):
        summary["data_policy"] = stats["data_policy"]
    if isinstance(stats.get("execution_policy"), dict):
        summary["execution_policy"] = stats["execution_policy"]

    summary["behavior_diagnostics"] = _behavior_diagnostics(trades, summary, diagnostics)
    summary["research_gate"] = _research_gate(summary)
    if total_trades == 0:
        summary["no_trade_diagnosis"] = _no_trade_diagnosis(bars, diagnostics)
    summary["evaluation"] = evaluate_strategy(summary)
    return summary
