"""Strategy evaluation scorecard.

The scorecard is deliberately conservative. It does not decide whether a
strategy can trade live; it summarizes whether the current evidence is strong
enough to justify the next research step.
"""

from typing import Any, Dict, List


def _num(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = data.get(key, default)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _score_by_thresholds(value: float, thresholds: List[float]) -> int:
    score = 0
    for threshold in thresholds:
        if value >= threshold:
            score += 1
    return min(5, score)


def _dimension(name: str, score: int, comment: str, evidence: List[str]) -> Dict[str, Any]:
    score = max(0, min(5, int(score)))
    if score >= 4:
        status = "较强"
    elif score >= 3:
        status = "可接受"
    elif score >= 2:
        status = "偏弱"
    else:
        status = "不足"
    return {
        "name": name,
        "score": score,
        "status": status,
        "comment": comment,
        "evidence": evidence,
    }


def _yearly_stability(yearly: Any) -> Dict[str, Any]:
    if not isinstance(yearly, list) or not yearly:
        return {"score": 1, "positive_years": 0, "total_years": 0, "worst_year": 0.0}
    returns = [_num(row, "return_pct") for row in yearly if isinstance(row, dict)]
    if not returns:
        return {"score": 1, "positive_years": 0, "total_years": 0, "worst_year": 0.0}

    positive = len([r for r in returns if r > 0])
    ratio = positive / len(returns)
    worst = min(returns)
    score = 1
    if ratio >= 0.4:
        score += 1
    if ratio >= 0.6:
        score += 1
    if ratio >= 0.75:
        score += 1
    if worst > -10:
        score += 1
    return {"score": min(5, score), "positive_years": positive, "total_years": len(returns), "worst_year": worst}


def evaluate_strategy(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a repeatable strategy scorecard from backtest metrics."""
    if not isinstance(metrics, dict):
        metrics = {}

    trades = int(_num(metrics, "total_trades"))
    net_profit = _num(metrics, "net_profit")
    total_return = _num(metrics, "total_return_pct")
    sharpe = _num(metrics, "sharpe_ratio")
    max_dd = _num(metrics, "max_drawdown_pct")
    rdr = _num(metrics, "return_drawdown_ratio")
    win_rate = _num(metrics, "win_rate")
    payoff = _num(metrics, "payoff_ratio")
    profit_factor = _num(metrics, "profit_factor")
    cost_profit_ratio = _num(metrics, "cost_profit_ratio")
    max_losses = int(_num(metrics, "max_consecutive_losses"))
    diagnostics = metrics.get("diagnostics") if isinstance(metrics.get("diagnostics"), dict) else {}
    behavior = metrics.get("behavior_diagnostics") if isinstance(metrics.get("behavior_diagnostics"), dict) else {}
    behavior_flags = behavior.get("flags") if isinstance(behavior.get("flags"), list) else []
    yearly = _yearly_stability(metrics.get("yearly"))

    hard_flags: List[str] = []
    if trades < 10:
        hard_flags.append("交易样本少于10笔，不能支持策略有效性判断")
    elif trades < 30:
        hard_flags.append("交易样本少于30笔，只能视为初步证据")
    if net_profit <= 0 and total_return <= 0:
        hard_flags.append("净利润和总收益率均未转正")
    if sharpe <= 0 and trades >= 10:
        hard_flags.append("夏普比率不为正，风险调整后收益不足")
    if diagnostics.get("orders_rejected_insufficient_cash", 0):
        hard_flags.append("存在资金不足拒单，仓位或保证金设置需要复核")

    profit_score = 0
    if net_profit > 0:
        profit_score += 1
    profit_score += _score_by_thresholds(total_return, [1, 5, 15])
    if sharpe >= 0.5:
        profit_score += 1
    if sharpe >= 1.0:
        profit_score += 1
    profit_score = min(5, profit_score)

    risk_score = 1
    if max_dd > -30:
        risk_score += 1
    if max_dd > -15:
        risk_score += 1
    if max_dd > -8:
        risk_score += 1
    if max_losses <= 5:
        risk_score += 1
    if rdr >= 1.0:
        risk_score += 1
    risk_score = min(5, risk_score)

    quality_score = 0
    if trades >= 30:
        quality_score += 1
    if 0.25 <= win_rate <= 0.65:
        quality_score += 1
    if payoff >= 1.2:
        quality_score += 1
    if payoff >= 2.0:
        quality_score += 1
    if profit_factor >= 1.2:
        quality_score += 1
    if profit_factor >= 1.6:
        quality_score += 1
    quality_score = min(5, quality_score)
    if trades < 10:
        quality_score = min(quality_score, 1)
    elif trades < 30:
        quality_score = min(quality_score, 3)

    practical_score = 3
    if trades == 0:
        practical_score = 0
    elif trades > 500:
        practical_score -= 1
    if cost_profit_ratio > 30:
        practical_score -= 1
    if cost_profit_ratio > 80:
        practical_score -= 1
    if diagnostics.get("orders_blocked_price_limit", 0):
        practical_score -= 1
    if diagnostics.get("bars_skipped_low_volume", 0):
        practical_score -= 1
    if _num(metrics, "total_slippage_cost") == 0 and trades > 0:
        practical_score -= 1
    if metrics.get("data_rows"):
        practical_score += 1
    practical_score = max(0, min(5, practical_score))

    evidence_score = 1
    if trades >= 30:
        evidence_score += 1
    if metrics.get("yearly"):
        evidence_score += 1
    if metrics.get("direction"):
        evidence_score += 1
    if metrics.get("data_rows"):
        evidence_score += 1
    if diagnostics:
        evidence_score += 1
    evidence_score = min(5, evidence_score)
    if trades < 10:
        evidence_score = min(evidence_score, 2)
    elif trades < 30:
        evidence_score = min(evidence_score, 3)

    dimensions = [
        _dimension(
            "收益能力",
            profit_score,
            "看净利润、总收益率和夏普比率是否形成正向收益证据。",
            [f"净利润={net_profit:.2f}", f"总收益率={total_return:.2f}%", f"夏普={sharpe:.2f}"],
        ),
        _dimension(
            "风险控制",
            risk_score,
            "看最大回撤、收益回撤比和连续亏损是否在可承受范围内。",
            [f"最大回撤={max_dd:.2f}%", f"收益回撤比={rdr:.2f}", f"最长连续亏损={max_losses}"],
        ),
        _dimension(
            "交易质量",
            quality_score,
            "看交易样本、胜率、盈亏比和盈利因子是否匹配策略类型。",
            [f"交易次数={trades}", f"胜率={win_rate:.2%}", f"盈亏比={payoff:.2f}", f"盈利因子={profit_factor:.2f}"],
        ),
        _dimension(
            "稳定性",
            yearly["score"],
            "看收益是否分布在多个年份，而不是依赖单一行情。",
            [f"盈利年份={yearly['positive_years']}/{yearly['total_years']}", f"最差年度收益={yearly['worst_year']:.2f}%"],
        ),
        _dimension(
            "实盘可行性",
            practical_score,
            "看交易频率、成本、滑点、涨跌停和成交量约束对落地的影响。",
            [f"成本/净利润={cost_profit_ratio:.2f}%", f"滑点成本={_num(metrics, 'total_slippage_cost'):.2f}"],
        ),
        _dimension(
            "证据完整性",
            evidence_score,
            "看是否具备交易明细、年度拆分、方向拆分和执行诊断。",
            [f"交易次数={trades}", f"年度数据={'有' if metrics.get('yearly') else '无'}", f"执行诊断={'有' if diagnostics else '无'}"],
        ),
    ]

    raw_score = round(sum(item["score"] for item in dimensions) / (len(dimensions) * 5) * 100)
    score = raw_score
    if trades < 30:
        score = min(score, 59)
    if net_profit <= 0 and total_return <= 0:
        score = min(score, 49)
    if behavior_flags and trades >= 10:
        score = min(score, 74)

    if score >= 80:
        rating = "A"
        decision_state = "候选模拟盘观察"
    elif score >= 65:
        rating = "B"
        decision_state = "继续优化后复测"
    elif score >= 45:
        rating = "C"
        decision_state = "只作研究样本，需要迭代"
    else:
        rating = "D"
        decision_state = "放弃或重构"

    if trades < 10:
        rating = "D"
        decision_state = "样本不足，不能上线"
        score = min(score, 39)

    summary = f"评分{score}/100，评级{rating}。当前决策状态：{decision_state}。"
    if hard_flags:
        summary += " 主要约束：" + "；".join(hard_flags[:3]) + "。"

    return {
        "total_score": score,
        "raw_score": raw_score,
        "rating": rating,
        "decision_state": decision_state,
        "summary": summary,
        "dimensions": dimensions,
        "hard_flags": hard_flags,
    }
