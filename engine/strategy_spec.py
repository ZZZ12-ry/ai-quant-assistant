"""Build a structured strategy spec from stage-1 research output.

The spec is the contract between stage 1 and stage 2.  Stage 2 should compile
this contract into code instead of freely inventing missing trading rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

import yaml


SECTION_ALIASES = {
    "core_logic": ["策略核心思想", "核心思想", "策略概述", "一句话理解"],
    "parameters": ["参数定义", "参数定义表格", "参数表"],
    "indicators": ["指标计算公式", "指标计算", "公式说明"],
    "entry_rules": ["开仓逻辑", "入场规则", "开仓规则"],
    "exit_rules": ["出场与风控", "出场规则", "止损与出场"],
    "risk_controls": ["仓位与风控", "出场与风控", "风控规则"],
    "assumptions": ["平台 v1 默认复现口径", "关键待确认问题", "待确认问题", "待验证假设", "研究员需要重点观察"],
}


def _load_template_docs(root: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    base = root / "data" / "knowledge" / "strategy_templates"
    if not base.exists():
        return docs
    for path in base.glob("*.yaml"):
        if path.name.startswith("_"):
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        data["_template_id"] = path.stem
        docs.append(data)
    return docs


def _score_template(text: str, doc: dict[str, Any]) -> int:
    source = (text or "").lower()
    haystack = "\n".join([
        str(doc.get("strategy_name", "")),
        str(doc.get("strategy_type", "")),
        str(doc.get("core_logic", "")),
        json.dumps(doc.get("parameters", []), ensure_ascii=False),
    ]).lower()
    score = 0
    for token in re.split(r"[\s,，。；;、()（）\[\]{}<>《》\"'`]+", source):
        if len(token) >= 2 and token in haystack:
            score += 1
    boosts = {
        "expma": 10,
        "ema": 8,
        "均线": 5,
        "突破": 5,
        "atr": 4,
        "海龟": 8,
        "obv": 8,
        "vwma": 8,
        "锁定": 8,
        "斜率": 6,
    }
    for token, boost in boosts.items():
        if token in source and token in haystack:
            score += boost
    return score


def _best_template(text: str, root: Path) -> dict[str, Any] | None:
    source = (text or "").lower()

    # These direct recognizers avoid a broad "均线" match pulling the wrong
    # template for the locked VWMA/OBV sample strategy.
    if any(token in source for token in ["vwma", "obv", "maobv", "锁定均线", "能量潮"]):
        for doc in _load_template_docs(root):
            if "vwma" in str(doc).lower() or "obv" in str(doc).lower():
                return doc

    scored = []
    for doc in _load_template_docs(root):
        template_id = str(doc.get("_template_id", ""))
        if template_id == "turtle_trader_trend" and not any(
            token in source for token in ["海龟", "super turtle", "turtle", "唐奇安", "donchian", "atr", "n值"]
        ):
            continue
        if template_id == "expma_cross_trend" and not any(token in source for token in ["expma", "ema", "指数移动平均"]):
            continue
        if template_id == "dual_ma_trend":
            has_cross = any(token in source for token in ["金叉", "死叉", "上穿", "下穿", "交叉", "双均线"])
            if not has_cross or "斜率" in source:
                continue
        score = _score_template(text, doc)
        if score > 0:
            scored.append((score, doc))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored[0][0] >= 8 else None


def _extract_section(markdown: str, aliases: list[str]) -> str:
    lines = (markdown or "").splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("##"):
            continue
        heading_text = stripped.lstrip("#").strip()
        matched_alias = next((alias for alias in aliases if heading_text.startswith(alias)), None)
        if not matched_alias:
            continue
        first_line = heading_text[len(matched_alias):].strip()
        section_lines = [first_line] if first_line else []
        for later in lines[i + 1:]:
            if later.strip().startswith("##"):
                break
            section_lines.append(later)
        return "\n".join(section_lines).strip()
    return ""


def _extract_rules(markdown: str, key: str) -> list[str]:
    section = _extract_section(markdown, SECTION_ALIASES[key])
    if not section:
        return []
    rows: list[str] = []
    in_code = False
    for line in section.splitlines():
        text = line.strip()
        if text.startswith("```"):
            in_code = not in_code
            continue
        if not text or text.startswith("|"):
            continue
        if not in_code and text.startswith("#"):
            continue
        text = text.lstrip("-*0123456789.、 ").strip()
        if text:
            rows.append(text)
    return rows[:30]


def _extract_parameter_table(markdown: str) -> list[dict[str, Any]]:
    section = _extract_section(markdown, SECTION_ALIASES["parameters"])
    rows: list[dict[str, Any]] = []
    for line in section.splitlines():
        text = line.strip()
        if not text.startswith("|") or "---" in text:
            continue
        cells = [cell.strip().strip("`") for cell in text.strip("|").split("|")]
        if len(cells) < 3:
            continue
        name = cells[0]
        if name in {"参数", "符号", "名称"}:
            continue
        rows.append({
            "name": name,
            "description": cells[1] if len(cells) > 1 else "",
            "default": cells[2] if len(cells) > 2 else "",
            "range": cells[3] if len(cells) > 3 else "",
            "source": "stage_1",
        })
    return rows


def _template_parameters(template: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not template:
        return []
    rows: list[dict[str, Any]] = []
    for item in template.get("parameters") or []:
        if not isinstance(item, list) or len(item) < 4:
            continue
        rows.append({
            "name": item[0],
            "default": item[1],
            "range": item[2],
            "type": item[3],
            "description": item[4] if len(item) > 4 else "",
            "source": "rag_template",
        })
    return rows


def _extract_strategy_name(markdown: str) -> str:
    match = re.search(r"^#\s+(.+)$", markdown or "", flags=re.M)
    if match:
        return match.group(1).strip()[:60]
    return "未命名策略"


def _strategy_type(text: str, template: dict[str, Any] | None) -> str:
    if template and template.get("strategy_type"):
        return str(template["strategy_type"])
    source = (text or "").lower()
    if any(token in source for token in ["r-breaker", "rbreaker", "日内", "收盘前", "尾盘", "分钟", "5分钟", "intraday"]):
        return "intraday"
    if any(token in source for token in ["小时", "hour", "多时间", "多周期", "multi-timeframe", "multi timeframe"]):
        return "multi_timeframe"
    if any(token in source for token in ["突破", "通道", "donchian", "海龟"]):
        return "trend_breakout"
    if any(token in source for token in ["均线", "expma", "ema", "斜率", "obv", "vwma"]):
        return "trend_following"
    if any(token in source for token in ["套利", "价差"]):
        return "spread"
    return "unknown"


def infer_data_requirements(text: str) -> dict[str, Any]:
    """Infer the minimum data granularity needed to make a backtest meaningful."""
    source = (text or "").lower()
    reasons: list[str] = []
    required = "daily"
    supported_by_default = True

    intraday_terms = ["r-breaker", "rbreaker", "日内", "收盘前", "尾盘", "分钟", "5分钟", "intraday"]
    multiframe_terms = ["小时", "hour", "多时间", "多周期", "multi-timeframe", "multi timeframe"]
    if any(token in source for token in intraday_terms):
        required = "intraday"
        supported_by_default = False
        reasons.append("策略包含日内/尾盘/分钟级执行规则，日线OHLC无法还原盘中触发顺序。")
    if any(token in source for token in multiframe_terms):
        required = "multi_timeframe"
        supported_by_default = False
        reasons.append("策略包含多周期或小时线入场规则，当前默认日线数据不能验证完整逻辑。")
    if "前一日" in source and any(token in source for token in ["六个关键价位", "bbreak", "sbreak", "senter", "benter"]):
        required = "intraday"
        supported_by_default = False
        reasons.append("R-Breaker类规则需要以前一日价位指导当日盘中交易，不能只用日线收盘确认。")

    return {
        "required_frequency": required,
        "default_platform_frequency": "daily",
        "supported_by_default_backtest": supported_by_default,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _critical_pending_questions(rules: list[str]) -> list[str]:
    critical = []
    keywords = ["必须确认", "需要确认", "无法进入阶段2", "缺少", "缺失", "不明确", "未定义"]
    for item in rules:
        if any(key in item for key in keywords):
            critical.append(item)
    return critical[:5]


def _stage2_gate(markdown: str, spec: dict[str, Any]) -> dict[str, Any]:
    indicators = spec.get("indicators") or []
    entry = spec.get("entry_rules") or []
    exit_rules = spec.get("exit_rules") or []
    assumptions = spec.get("assumptions") or []
    blockers: list[str] = []

    if not indicators:
        blockers.append("缺少指标计算公式，阶段2无法确定要计算哪些字段。")
    if not entry:
        blockers.append("缺少入场规则，阶段2无法确定开仓条件。")
    if not exit_rules:
        blockers.append("缺少出场或风控规则，阶段2无法确定平仓/止损条件。")
    blockers.extend(_critical_pending_questions(assumptions))

    raw = markdown or ""
    has_pending_area = any(token in raw for token in ["关键待确认问题", "待确认问题"])
    has_pending_blocker = any(token in raw for token in ["缺少", "缺失", "不明确", "未定义", "必须确认", "需要确认"])
    if has_pending_area and has_pending_blocker:
        blockers.append("阶段1仍包含需要研究员显式确认的问题。")

    return {
        "stage2_allowed": not blockers,
        "rule_status": {
            "indicator_formula": {"status": "confirmed" if indicators else "missing"},
            "entry_rule": {"status": "confirmed" if entry else "missing"},
            "exit_rule": {"status": "confirmed" if exit_rules else "missing"},
            "position_sizing": {"status": "confirmed_or_default"},
        },
        "blocking_questions": list(dict.fromkeys(blockers)),
        "warnings": [] if not blockers else ["阶段2只能翻译已确认规则，不能自由补全关键交易逻辑。"],
        "policy": "阶段2只编译阶段1文档和 strategy_spec 中的规则；RAG 只作代码范式参考，不覆盖用户策略逻辑。",
    }


def build_strategy_spec(markdown: str, user_input: str = "", root: Path | None = None) -> dict[str, Any]:
    """Create a deterministic JSON spec used by stage 2 code generation."""
    root = root or Path(__file__).parent.parent
    source_text = "\n".join([user_input or "", markdown or ""])
    template = _best_template(source_text, root)
    parameters = _extract_parameter_table(markdown) or _template_parameters(template)

    spec = {
        "schema_version": "1.0",
        "strategy_name": _extract_strategy_name(markdown),
        "strategy_type": _strategy_type(source_text, template),
        "data_requirements": infer_data_requirements(source_text),
        "template_id": template.get("_template_id") if template else None,
        "template_source": "data/knowledge/strategy_templates" if template else None,
        "template_strategy_name": template.get("strategy_name") if template else None,
        "core_logic": _extract_section(markdown, SECTION_ALIASES["core_logic"]) or (template.get("core_logic") if template else ""),
        "parameters": parameters,
        "indicators": _extract_rules(markdown, "indicators"),
        "entry_rules": _extract_rules(markdown, "entry_rules"),
        "exit_rules": _extract_rules(markdown, "exit_rules"),
        "risk_controls": _extract_rules(markdown, "risk_controls"),
        "assumptions": _extract_rules(markdown, "assumptions"),
        "implementation_constraints": {
            "engine_interface": "compute_indicators -> generate_signals -> run",
            "signal_raw": "1=开多/加多/反手做多, -1=开空/加空/反手做空, 0=无新方向交易",
            "exit_signal": "1=下一根K线只平仓, 0=不平仓；没有独立平仓逻辑时也应输出该列并置0",
            "no_lookahead": True,
            "external_dependencies": ["pandas", "numpy"],
        },
        "source_policy": {
            "user_text_priority": True,
            "rag_is_reference_only": True,
            "do_not_import_history_strategy_params": True,
            "stage2_no_free_inference": True,
        },
    }
    spec["stage2_gate"] = _stage2_gate(markdown, spec)
    if template and template.get("implementation_guidance"):
        spec["implementation_guidance"] = template["implementation_guidance"]
    if template and template.get("signal_rules"):
        spec["rag_signal_rules"] = template["signal_rules"]
    return spec


def save_strategy_spec(strategy_dir: Path, markdown: str, user_input: str = "", root: Path | None = None) -> Path:
    spec = build_strategy_spec(markdown, user_input=user_input, root=root)
    path = strategy_dir / "strategy_spec.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
