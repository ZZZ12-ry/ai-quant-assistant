import sys; sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])
from pathlib import Path
import json
import yaml
import re
import html
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from fastapi.encoders import jsonable_encoder

from engine.reporting import summarize_report
from engine.evaluation import evaluate_strategy


AVAILABLE_SYMBOLS = {
    "RB0": "螺纹钢",
    "HC0": "热卷",
    "I0": "铁矿石",
    "J0": "焦炭",
    "JM0": "焦煤",
    "TA0": "PTA",
    "MA0": "甲醇",
    "PP0": "聚丙烯",
    "CU0": "沪铜",
    "AL0": "沪铝",
    "ZN0": "沪锌",
    "AU0": "黄金",
    "AG0": "白银",
    "M0": "豆粕",
    "Y0": "豆油",
    "P0": "棕榈油",
}


def default_backtest_window():
    """Default to daily data over the latest five years."""
    end = datetime.now()
    start = end - timedelta(days=365 * 5)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

def _infer_strategy_display_name(strategy_dir: Path, fallback: str) -> str:
    try:
        spec_path = strategy_dir / "strategy_spec.json"
        if spec_path.exists():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            name = str(spec.get("strategy_name") or "").strip()
            if name and name not in {"未命名策略", "海龟交易趋势突破", "双均线趋势跟踪"}:
                return name
    except Exception:
        pass
    try:
        readme = (strategy_dir / "README.md").read_text(encoding="utf-8")
        match = re.search(r"([A-Za-z][A-Za-z0-9_ -]{2,40})[（(]", readme)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    try:
        code = (strategy_dir / "model.py").read_text(encoding="utf-8")
        match = re.search(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", code)
        if match:
            return match.group(1)
    except Exception:
        pass
    return fallback

def _safe_report_prefix(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value or "").strip("_")

def _find_strategy_reports(report_dir: Path, sid: str, scope: str = "") -> list[Path]:
    if not report_dir.exists():
        return []
    patterns = [f"{sid}_*_backtest.html"]
    safe_sid = _safe_report_prefix(sid)
    if safe_sid and safe_sid != sid:
        patterns.append(f"{safe_sid}_*_backtest.html")
    seen_reports = {}
    for pattern in patterns:
        for rp in report_dir.glob(pattern):
            if scope == "complete" and rp.name.startswith(f"{sid}_iter_"):
                continue
            seen_reports[str(rp)] = rp
    return sorted(seen_reports.values(), key=lambda p: p.stat().st_mtime, reverse=True)

def get_strategy_info():
    from engine.strategy_manager import get_active, get_active_scope, get_strategy_dir, is_complete_strategy, list_strategies, load_params, set_active
    sid = get_active()
    scope = get_active_scope()
    root = Path(__file__).parent.parent
    ready_file = root / "data" / "stage2_ready.txt"
    ready_sid = ready_file.read_text(encoding="utf-8").strip() if ready_file.exists() else ""
    if scope != "draft" and not is_complete_strategy(sid) and ready_sid != sid:
        completed = list_strategies()
        if completed:
            sid = completed[0]["id"]
            set_active(sid, "complete")
            scope = "complete"
    sd = get_strategy_dir(sid, scope)
    params = load_params(sid)
    meta = {}
    meta_path = sd / "meta.yaml"
    if meta_path.exists():
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        except Exception:
            meta = {}
    code = (sd / "model.py").read_text(encoding="utf-8") if (sd / "model.py").exists() else ""
    readme = (sd / "README.md").read_text(encoding="utf-8") if (sd / "README.md").exists() else ""
    check_report = (sd / "check_report.md").read_text(encoding="utf-8") if (sd / "check_report.md").exists() else ""
    analysis_template = (sd / "analysis_template.md").read_text(encoding="utf-8") if (sd / "analysis_template.md").exists() else ""
    multi_symbol_validation = (sd / "multi_symbol_validation.md").read_text(encoding="utf-8") if (sd / "multi_symbol_validation.md").exists() else ""
    strategy_spec = {}
    spec_path = sd / "strategy_spec.json"
    readme_path = sd / "README.md"
    if readme_path.exists():
        try:
            needs_spec_refresh = not spec_path.exists()
            if spec_path.exists():
                current_spec = json.loads(spec_path.read_text(encoding="utf-8"))
                needs_spec_refresh = "stage2_gate" not in current_spec
            if needs_spec_refresh:
                from engine.strategy_spec import save_strategy_spec
                save_strategy_spec(sd, readme_path.read_text(encoding="utf-8"), root=root)
        except Exception:
            pass
    if spec_path.exists():
        try:
            strategy_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception:
            strategy_spec = {}
    iteration = {}
    iteration_path = sd / "iteration.yaml"
    if iteration_path.exists():
        try:
            iteration = yaml.safe_load(iteration_path.read_text(encoding="utf-8")) or {}
        except Exception:
            iteration = {}
    symbols = AVAILABLE_SYMBOLS
    report_dir = root / "reports" / "web"
    latest_report = None
    report_stats = None
    data_files = {}
    reports = _find_strategy_reports(report_dir, sid, scope)
    if reports:
        rp = reports[0]
        latest_report = f"/reports/web/{rp.name}"
        stem = rp.name.replace("_backtest.html", "")
        report_stats = summarize_report(report_dir, stem)
        for key, suffix in {"trades": "_trades.csv", "equity": "_equity.csv", "bars": "_bars.csv"}.items():
            fp = report_dir / f"{stem}{suffix}"
            if fp.exists():
                data_files[key] = f"/reports/web/{fp.name}"
    version_complete = bool(
        scope == "complete"
        or (
            scope == "draft"
            and iteration
            and latest_report
            and check_report
            and analysis_template
        )
    )
    iteration_suggestions = _iteration_suggestions(report_stats or {})
    iteration_compare = _iteration_compare(sid, scope, iteration, report_dir) if iteration else {}
    return {
        "name": meta.get("name") or sid,
        "active": sid,
        "scope": scope,
        "complete": is_complete_strategy(sid),
        "version_complete": version_complete,
        "can_iterate": version_complete,
        "meta": meta,
        "params": params,
        "code": code,
        "readme": readme,
        "check_report": check_report,
        "analysis_template": analysis_template,
        "multi_symbol_validation": multi_symbol_validation,
        "strategy_spec": strategy_spec,
        "iteration": iteration,
        "symbols": symbols,
        "report_url": latest_report,
        "report_stats": report_stats,
        "data_files": data_files,
        "iteration_suggestions": iteration_suggestions,
        "iteration_compare": iteration_compare,
    }


def _latest_strategy_stats(sid: str, report_dir: Path, scope: str = "complete") -> dict:
    reports = _find_strategy_reports(report_dir, sid, scope)
    if not reports:
        return {}
    stem = reports[0].name.replace("_backtest.html", "")
    return summarize_report(report_dir, stem) or {}


def _iteration_compare(sid: str, scope: str, iteration: dict, report_dir: Path) -> dict:
    if not isinstance(iteration, dict):
        return {}
    parent = iteration.get("parent_version_id") or iteration.get("parent_strategy")
    parent_scope = iteration.get("parent_version_scope") or "complete"
    if not parent:
        return {}
    child_stats = _latest_strategy_stats(sid, report_dir, scope)
    parent_stats = _latest_strategy_stats(parent, report_dir, parent_scope)
    if not child_stats or not parent_stats:
        return {"parent_strategy": parent, "status": "等待父子版本均完成回测"}
    keys = ["total_return_pct", "net_profit", "max_drawdown_pct", "sharpe_ratio", "total_trades", "payoff_ratio"]
    rows = []
    for key in keys:
        pv = parent_stats.get(key, 0)
        cv = child_stats.get(key, 0)
        try:
            delta = float(cv or 0) - float(pv or 0)
        except Exception:
            delta = 0
        rows.append({"metric": key, "parent": pv, "child": cv, "delta": round(delta, 4)})
    return {"parent_strategy": parent, "status": "ok", "rows": rows}


def _iteration_suggestions(stats: dict) -> list:
    if not isinstance(stats, dict) or not stats:
        return [
            {"id": "parameter", "type": "parameter", "title": "参数敏感性", "note": "扫描核心周期、止损倍数和过滤阈值，先建立基线参数稳定性。"},
            {"id": "multi_symbol", "type": "multi_symbol", "title": "多品种验证", "note": "保留当前逻辑，扩展到多个高流动性品种，判断是否依赖单一市场。"},
        ]
    suggestions = []
    behavior = stats.get("behavior_diagnostics") if isinstance(stats.get("behavior_diagnostics"), dict) else {}
    flags = "；".join(behavior.get("flags", []) if isinstance(behavior.get("flags"), list) else [])
    directions = stats.get("direction") if isinstance(stats.get("direction"), list) else []
    gate = stats.get("research_gate") if isinstance(stats.get("research_gate"), dict) else {}
    blockers = "；".join(gate.get("blockers", []) if isinstance(gate.get("blockers"), list) else [])

    if "平均持仓" in flags or "止损退出占比" in flags:
        suggestions.append({
            "id": "exit_loosen",
            "type": "entry_exit",
            "title": "出场逻辑迭代",
            "note": "当前行为显示持仓过短或止损退出占比过高，优先测试更宽止损、更长出场周期或移动止损规则。"
        })
    losing_dirs = [row for row in directions if float(row.get("net_profit", 0) or 0) < 0]
    if losing_dirs:
        worst = sorted(losing_dirs, key=lambda row: float(row.get("net_profit", 0) or 0))[0]
        suggestions.append({
            "id": "direction_filter",
            "type": "add_filter",
            "title": "方向过滤迭代",
            "note": f"{worst.get('direction')}方向贡献主要亏损，测试趋势过滤、成交量确认或只在强趋势环境启用该方向。"
        })
    if "净利润" in blockers or float(stats.get("sharpe_ratio", 0) or 0) <= 0:
        suggestions.append({
            "id": "parameter",
            "type": "parameter",
            "title": "参数敏感性",
            "note": "默认结果未形成正收益证据，先扫描入场周期、出场周期、止损倍数和过滤阈值，观察是否存在稳定区间。"
        })
    suggestions.append({
        "id": "multi_symbol",
        "type": "multi_symbol",
        "title": "多品种验证",
        "note": "保留当前版本作为基线，扩展到黑色、化工、有色、贵金属等品种，避免只根据单品种下结论。"
    })
    seen = set()
    out = []
    for item in suggestions:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        out.append(item)
    return out[:5]

def save_params(new_params: dict):
    from engine.strategy_manager import get_active, get_active_scope, get_strategy_dir
    sid = get_active()
    pp = get_strategy_dir(sid, get_active_scope()) / "params.yaml"
    with open(pp, "r", encoding="utf-8") as f:
        current = yaml.safe_load(f)
    current.update(new_params)
    with open(pp, "w", encoding="utf-8") as f:
        yaml.dump(current, f, default_flow_style=False, allow_unicode=True)
    return current

def run_backtest(symbol: str, params_override: dict = None):
    from engine.data import get_main_contract_data, describe_data_policy
    from engine.backtest import BacktestEngine
    from engine.strategy_manager import load_module, get_active, get_active_scope, get_strategy_dir, load_params
    import numpy as np
    import re
    sid = get_active()
    strategy_dir = get_strategy_dir(sid, get_active_scope())
    spec_path = strategy_dir / "strategy_spec.json"
    if spec_path.exists():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            data_req = spec.get("data_requirements") if isinstance(spec.get("data_requirements"), dict) else {}
            if data_req and not data_req.get("supported_by_default_backtest", True):
                required = data_req.get("required_frequency", "unknown")
                reasons = data_req.get("reasons") or []
                detail = "；".join(str(item) for item in reasons[:3])
                raise ValueError(
                    f"当前策略需要 {required} 数据，平台默认阶段三只提供日线回测。"
                    f"{detail} 请先接入对应频率数据，或在阶段一明确改成日线近似验证版本。"
                )
        except ValueError:
            raise
        except Exception:
            pass
    StrategyClass = load_module(sid)
    base = load_params(sid)
    if params_override: base.update(params_override)
    start_date, end_date = default_backtest_window()
    df = get_main_contract_data(symbol, start_date, end_date)
    data_policy = df.attrs.get("data_policy") if hasattr(df, "attrs") else None
    s = StrategyClass(base)
    bt = BacktestEngine(100000).run(s.run(df), base)
    stats = bt["stats"]; trades = bt["trades"]
    def js(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v
    root = Path(__file__).parent.parent
    report_dir = root / "reports" / "web"; report_dir.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{sid}_{symbol}").strip("_")
    trades.to_csv(report_dir / f"{label}_trades.csv", index=False)
    bt["equity_curve"].to_csv(report_dir / f"{label}_equity.csv", index=False)
    bt["bars"].to_csv(report_dir / f"{label}_bars.csv", index=False)
    stats_path = report_dir / f"{label}_stats.json"
    stats_path.write_text(
        json.dumps(jsonable_encoder(stats), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    enriched = summarize_report(report_dir, label) or stats
    if isinstance(enriched, dict) and enriched.get("error"):
        enriched = dict(stats)
        enriched["evaluation"] = evaluate_strategy(enriched)
    if isinstance(enriched, dict):
        enriched["symbol"] = symbol
        enriched["data_frequency"] = "日线"
        enriched["default_window"] = "近五年"
        enriched["requested_period"] = {"start": start_date, "end": end_date}
        enriched["data_policy"] = data_policy or describe_data_policy(symbol)
    bt["stats"] = enriched
    from engine.visualize import save_report
    rp = report_dir / f"{label}_backtest.html"
    save_report(bt, str(rp))
    stats_path.write_text(
        json.dumps(jsonable_encoder(enriched), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = {"symbol": symbol, "details": {k: js(v) for k, v in enriched.items()}, "trades_count": len(trades)}
    if len(trades) > 0: result["last_trades"] = trades.tail(5).to_dict("records")
    result["report_url"] = f"/reports/web/{label}_backtest.html"
    result["data_files"] = {
        "trades": f"/reports/web/{label}_trades.csv",
        "equity": f"/reports/web/{label}_equity.csv",
        "bars": f"/reports/web/{label}_bars.csv",
    }
    return result

def _fit_score(stats: dict) -> float:
    trades = float(stats.get("total_trades", 0) or 0)
    net_profit = float(stats.get("net_profit", 0) or 0)
    total_return = float(stats.get("total_return_pct", 0) or 0)
    sharpe = float(stats.get("sharpe_ratio", 0) or 0)
    max_dd = abs(float(stats.get("max_drawdown_pct", 0) or 0))
    payoff = float(stats.get("payoff_ratio", 0) or 0)

    # Screening score, not the final strategy rating. It ranks which symbol is
    # worth using as the main stage-4/5 sample by balancing reward, risk and
    # evidence sufficiency. High profit with an extreme drawdown should not win.
    return_score = min(30, max(0, total_return * 2.0))
    sharpe_score = min(22, max(0, sharpe * 14))
    payoff_score = min(12, max(0, payoff * 3))
    drawdown_score = max(0, 24 - max_dd * 1.8)
    sample_score = 12
    if trades < 10:
        sample_score = 2
    elif trades < 20:
        sample_score = 6
    elif trades < 30:
        sample_score = 9
    score = return_score + sharpe_score + payoff_score + drawdown_score + sample_score
    if net_profit <= 0:
        score -= 25
    if max_dd > 20:
        score -= min(25, (max_dd - 20) * 1.5)
    return round(max(0, min(100, score)), 2)

def _symbol_advice(row: dict, best: Optional[dict]) -> str:
    if row.get("error"):
        return "该品种本轮回测失败，暂不作为判断依据。"
    trades = int(row.get("total_trades", 0) or 0)
    net_profit = float(row.get("net_profit", 0) or 0)
    sharpe = float(row.get("sharpe_ratio", 0) or 0)
    max_dd = abs(float(row.get("max_drawdown_pct", 0) or 0))
    best_score = float(best.get("fit_score", 0) or 0) if best else 0
    score = float(row.get("fit_score", 0) or 0)
    if trades == 0:
        return "没有产生交易信号，可能是该品种趋势结构、成交量确认或突破条件不匹配。"
    if trades < 10:
        return "交易样本偏少，只能作为初步观察；需要更长周期或换品种验证。"
    if net_profit <= 0 and best_score - score >= 20:
        return "该品种表现弱，但其他品种明显更好，优先怀疑品种不适配，而不是直接否定策略。"
    if sharpe >= 0.8 and net_profit > 0 and max_dd <= 5:
        return "该品种与策略较匹配，可作为下一步检查和分析的主样本。"
    if net_profit > 0:
        return "该品种有正收益证据，但仍需检查样本量、回撤和交易成本。"
    return "当前表现偏弱，需要先换品种或调整参数后再进入深入分析。"

def run_multi_symbol_backtest(symbols: list[str], params_override: dict = None):
    clean_symbols = []
    for symbol in symbols or []:
        symbol = str(symbol or "").strip().upper()
        if symbol in AVAILABLE_SYMBOLS and symbol not in clean_symbols:
            clean_symbols.append(symbol)
    if not clean_symbols:
        clean_symbols = ["RB0", "HC0", "I0", "AU0", "AG0", "M0"]
    rows = []
    for symbol in clean_symbols:
        try:
            result = run_backtest(symbol, params_override)
            details = result.get("details", {}) or {}
            behavior = details.get("behavior_diagnostics") or {}
            rows.append({
                "symbol": symbol,
                "name": AVAILABLE_SYMBOLS.get(symbol, symbol),
                "report_url": result.get("report_url", ""),
                "data_files": result.get("data_files", {}),
                "total_trades": details.get("total_trades", 0),
                "net_profit": details.get("net_profit", 0),
                "total_return_pct": details.get("total_return_pct", 0),
                "max_drawdown_pct": details.get("max_drawdown_pct", 0),
                "sharpe_ratio": details.get("sharpe_ratio", 0),
                "payoff_ratio": details.get("payoff_ratio", 0),
                "win_rate": details.get("win_rate", 0),
                "avg_bars_held": behavior.get("avg_bars_held"),
            })
        except Exception as e:
            rows.append({"symbol": symbol, "name": AVAILABLE_SYMBOLS.get(symbol, symbol), "error": str(e)})
    for row in rows:
        row["fit_score"] = 0 if row.get("error") else _fit_score(row)
    valid = [row for row in rows if not row.get("error")]
    valid.sort(key=lambda row: row.get("fit_score", 0), reverse=True)
    best = valid[0] if valid else None
    for row in rows:
        row["advice"] = _symbol_advice(row, best)
    final_result = {}
    if best and best.get("symbol"):
        final_result = run_backtest(best["symbol"], params_override)
    return {
        "symbols": rows,
        "recommended_symbol": best.get("symbol", "") if best else "",
        "recommended_name": best.get("name", "") if best else "",
        "recommendation": (
            f"建议先以 {best.get('symbol')} · {best.get('name')} 作为阶段4/5主样本。"
            if best else "本轮没有找到可推荐品种。"
        ),
        "details": final_result.get("details", {}),
        "report_url": final_result.get("report_url", best.get("report_url", "") if best else ""),
        "data_files": final_result.get("data_files", best.get("data_files", {}) if best else {}),
    }

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

root = Path(__file__).parent.parent
app = FastAPI(title="AI量化研究平台")
app.mount("/reports", StaticFiles(directory=str(root / "reports")), name="reports")

class BacktestRequest(BaseModel): symbol: str = "RB0"; params: dict = {}
class BacktestBatchRequest(BaseModel): symbols: list[str] = []; params: dict = {}
class ParamUpdate(BaseModel): params: dict
class ChatRequest(BaseModel): messages: list = []
class ConfigUpdate(BaseModel): api_key: str = ""
class ArtifactSaveRequest(BaseModel):
    stage: int
    content: str
    strategy: str = ""
class CompleteStrategyRequest(BaseModel): name: str = ""
class IterationRequest(BaseModel):
    parent_strategy: str = ""
    parent_scope: str = "complete"
    iteration_type: str = "free_edit"
    note: str = ""
    name: str = ""

@app.get("/", response_class=HTMLResponse)
async def index():
    p = root / "app" / "static" / "index.html"
    return HTMLResponse(
        p.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
@app.get("/api/strategy")
async def api_strategy():
    return JSONResponse(
        get_strategy_info(),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def _export_md_to_html(markdown: str) -> str:
    """Small markdown renderer for print/export pages.

    The app already renders rich markdown in the browser. For PDF export we keep
    a dependency-free renderer that handles the strategy reports' common shapes:
    headings, paragraphs, lists, fenced code and markdown tables.
    """
    text = markdown or ""
    out = []
    paragraph = []
    in_code = False
    code_lines = []

    def flush_paragraph():
        if paragraph:
            body = " ".join(paragraph).strip()
            body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(body))
            body = re.sub(r"`([^`]+)`", lambda m: f"<code>{html.escape(m.group(1))}</code>", body)
            out.append(f"<p>{body}</p>")
            paragraph.clear()

    def render_table(rows: list[str]):
        parsed = [[html.escape(cell.strip()) for cell in row.strip().strip("|").split("|")] for row in rows]
        if not parsed:
            return
        out.append("<table>")
        out.append("<thead><tr>" + "".join(f"<th>{cell}</th>" for cell in parsed[0]) + "</tr></thead>")
        separator = len(parsed) > 1 and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in parsed[1])
        body_rows = parsed[2:] if separator else parsed[1:]
        out.append("<tbody>")
        for row in body_rows:
            out.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
        out.append("</tbody></table>")

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                code_lines = []
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(raw)
            i += 1
            continue
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            render_table(rows)
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = min(len(heading.group(1)) + 1, 4)
            out.append(f"<h{level}>{html.escape(heading.group(2).strip())}</h{level}>")
            i += 1
            continue
        if re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            items = []
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                item = re.sub(r"^[-*]\s+", "", lines[i].strip())
                items.append("<li>" + html.escape(item) + "</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        paragraph.append(stripped)
        i += 1
    flush_paragraph()
    if in_code:
        out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
    return "\n".join(out)


def _export_metric_table(stats: dict) -> str:
    if not isinstance(stats, dict) or not stats:
        return "<p>尚未生成阶段三回测摘要。</p>"
    rows = [
        ("交易次数", stats.get("total_trades", 0)),
        ("净利润", stats.get("net_profit", 0)),
        ("总收益率", f"{stats.get('total_return_pct', 0)}%"),
        ("最大回撤", f"{stats.get('max_drawdown_pct', 0)}%"),
        ("夏普比率", stats.get("sharpe_ratio", 0)),
        ("胜率", f"{round(float(stats.get('win_rate', 0) or 0) * 100, 2)}%"),
        ("盈亏比", stats.get("payoff_ratio", 0)),
    ]
    body = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


@app.get("/export/report", response_class=HTMLResponse)
async def export_current_report():
    current = get_strategy_info()
    title = current.get("name") or current.get("active") or "策略研究报告"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = [
        ("阶段一：策略构思", current.get("readme") or ""),
        ("阶段二：策略代码", "```python\n" + (current.get("code") or "尚未生成策略代码") + "\n```"),
        ("阶段三：回测摘要", ""),
        ("阶段四：策略检查", current.get("check_report") or ""),
        ("阶段五：策略分析", current.get("analysis_template") or ""),
    ]
    blocks = []
    for name, content in sections:
        blocks.append(f"<section><h1>{html.escape(name)}</h1>")
        if name.startswith("阶段三"):
            blocks.append(_export_metric_table(current.get("report_stats") or {}))
            if current.get("report_url"):
                blocks.append(f'<p>完整回测 HTML：<a href="{html.escape(current["report_url"])}">{html.escape(current["report_url"])}</a></p>')
        else:
            blocks.append(_export_md_to_html(content))
        blocks.append("</section>")
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(title)} - 五阶段研究报告</title>
<style>
body{{font-family:"Microsoft YaHei","Segoe UI",sans-serif;color:#1f2d3d;background:#fff;margin:0;padding:32px;line-height:1.75}}
.toolbar{{position:sticky;top:0;background:#fff;border-bottom:1px solid #d8e1eb;padding:0 0 14px;margin-bottom:24px;display:flex;gap:10px;align-items:center;justify-content:space-between}}
.toolbar button{{background:#1565c0;color:#fff;border:0;border-radius:6px;padding:9px 16px;cursor:pointer}}
.meta{{color:#65788b;font-size:13px}}
h1{{font-size:24px;color:#102a43;margin:28px 0 14px;border-bottom:1px solid #e6edf5;padding-bottom:8px}}
h2{{font-size:20px;color:#102a43;margin:22px 0 10px}}
h3{{font-size:17px;color:#1565c0;margin:18px 0 8px}}
p,li{{font-size:14px;color:#334e68}}
table{{width:100%;border-collapse:collapse;margin:14px 0 20px;page-break-inside:auto}}
th,td{{border:1px solid #d8e1eb;padding:8px 10px;font-size:13px;text-align:left;vertical-align:top}}
th{{background:#f3f8ff;color:#0f4fa8}}
pre{{background:#f8fafc;border:1px solid #d8e1eb;border-radius:8px;padding:14px;white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.6}}
code{{font-family:Consolas,"SFMono-Regular",monospace}}
section{{break-after:auto;margin-bottom:30px}}
@media print{{body{{padding:0}}.toolbar{{display:none}}a{{color:#1f2d3d;text-decoration:none}}section{{page-break-inside:auto}}}}
</style>
</head>
<body>
<div class="toolbar">
  <div><strong>{html.escape(title)}</strong><div class="meta">生成时间：{html.escape(now)} · 当前策略：{html.escape(current.get("active") or "")}</div></div>
  <button onclick="window.print()">导出/打印 PDF</button>
</div>
{''.join(blocks)}
<script>setTimeout(function(){{window.print()}}, 500);</script>
</body>
</html>"""
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})

@app.get("/api/strategies")
async def api_strategies():
    from engine.strategy_manager import list_strategies
    return JSONResponse(
        list_strategies(),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/api/strategy-drafts")
async def api_strategy_drafts():
    from engine.strategy_manager import list_draft_strategies
    return JSONResponse(
        list_draft_strategies(),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.post("/api/strategy/switch")
async def api_switch_strategy(id: str, scope: str = "complete"):
    from engine.strategy_manager import set_active
    from app.chat import set_stage
    set_active(id, scope if scope in {"complete", "draft"} else None)
    set_stage(1)
    return {"status": "ok", "active": id, "scope": scope, "stage": 1}

@app.delete("/api/strategy/{strategy_id}")
async def api_delete_strategy(strategy_id: str):
    try:
        from engine.strategy_manager import delete_complete_strategy, get_active, get_active_scope
        from app.chat import set_stage
        delete_complete_strategy(strategy_id)
        set_stage(1)
        return {"status": "ok", "deleted": strategy_id, "active": get_active(), "scope": get_active_scope()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/conversation/new")
async def api_new_conversation():
    from engine.strategy_manager import get_strategy_dir, set_active
    from app.chat import set_stage
    sid = "draft_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    sd = get_strategy_dir(sid, "draft")
    sd.mkdir(parents=True, exist_ok=True)
    with open(sd / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"complete": False, "name": "新策略草稿"}, f, default_flow_style=False, allow_unicode=True)
    set_active(sid, "draft")
    set_stage(1)
    return {"status": "ok", "active": sid, "scope": "draft", "stage": 1}

@app.post("/api/strategy/iterate")
async def api_create_iteration(req: IterationRequest):
    try:
        from engine.strategy_manager import create_strategy_iteration
        from app.chat import set_stage
        result = create_strategy_iteration(
            req.parent_strategy,
            iteration_type=req.iteration_type,
            note=req.note,
            name=req.name,
            parent_scope=req.parent_scope,
        )
        set_stage(1)
        result["stage"] = 1
        return result
    except Exception as e:
        return {"error": str(e)}

def _ensure_backtest_allowed():
    from app.chat import get_stage, is_stage2_ready, set_stage, validate_active_strategy
    from engine.strategy_manager import get_active, is_complete_strategy
    if is_complete_strategy(get_active()):
        return {"ok": True}
    stage = get_stage()
    if stage not in (2, 3):
        return {"error": f"必须按顺序推进：当前是阶段{stage}，只有完成阶段2后才能进入阶段3回测"}
    if stage == 2 and not is_stage2_ready():
        validation = validate_active_strategy()
        if not validation.get("ok"):
            return {"error": "阶段2策略校验失败：" + validation.get("error", "未知错误")}
    if stage == 2:
        set_stage(3)
    return {"ok": True}

@app.post("/api/backtest")
async def api_backtest(req: BacktestRequest):
    try:
        gate = _ensure_backtest_allowed()
        if not gate.get("ok"):
            return gate
        return run_backtest(req.symbol, req.params if req.params else None)
    except Exception as e: return {"error": str(e)}

@app.post("/api/backtest/batch")
async def api_backtest_batch(req: BacktestBatchRequest):
    try:
        gate = _ensure_backtest_allowed()
        if not gate.get("ok"):
            return gate
        return run_multi_symbol_backtest(req.symbols, req.params if req.params else None)
    except Exception as e: return {"error": str(e)}

@app.post("/api/params")
async def api_update_params(req: ParamUpdate):
    try: return {"status": "ok", "params": save_params(req.params)}
    except Exception as e: return {"error": str(e)}

@app.post("/api/stage2/validate")
async def api_validate_stage2():
    from app.chat import validate_active_strategy
    result = validate_active_strategy()
    if not result.get("ok"):
        return {"error": result.get("error", "阶段2策略校验失败"), "strategy": result.get("strategy")}
    return result

@app.get("/api/symbols")
async def api_symbols():
    return AVAILABLE_SYMBOLS

# Chat
from app.chat import chat_with_deepseek, load_config as load_api_config, save_config as save_api_config
@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    from engine.strategy_manager import get_active, get_active_scope
    result = chat_with_deepseek(req.messages)
    result["active"] = get_active()
    result["scope"] = get_active_scope()
    return result
@app.post("/api/config")
async def api_update_config(req: ConfigUpdate):
    config = save_api_config({"api_key": req.api_key})
    return {"status": "ok", "api_key_set": bool(req.api_key)}
@app.get("/api/config")
async def api_get_config():
    return {"api_key_set": bool(load_api_config().get("api_key", ""))}

@app.get("/api/stage")
async def api_get_stage():
    from app.chat import get_stage, infer_active_stage, stage1_gate_status
    from engine.strategy_manager import get_active, is_complete_strategy
    if not is_complete_strategy(get_active()) and not stage1_gate_status().get("allowed"):
        return {"stage": 1, "gate": stage1_gate_status()}
    return {"stage": max(get_stage(), infer_active_stage())}

@app.post("/api/stage/switch")
async def api_switch_stage(stage: int):
    from app.chat import advance_stage
    result = advance_stage(stage)
    if not result["ok"]:
        raise HTTPException(status_code=409, detail=result["error"])
    return {"status": "ok", "stage": result["stage"]}

@app.post("/api/stage1/confirm")
async def api_confirm_stage1():
    from app.chat import confirm_stage1_gate
    result = confirm_stage1_gate()
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("error", "阶段1确认失败"))
    return result

@app.post("/api/strategy/artifact")
async def api_save_artifact(req: ArtifactSaveRequest):
    from engine.strategy_manager import get_active, get_active_scope, get_strategy_dir
    sid = get_active()
    scope = get_active_scope()
    if scope != "draft":
        return {"error": "completed strategies are read-only", "strategy": sid, "scope": scope}
    if not req.strategy:
        return {"error": "strategy id is required", "strategy": sid, "scope": scope}
    if req.strategy != sid:
        return {"error": "active strategy changed, artifact not saved", "strategy": sid, "requested": req.strategy}
    sd = get_strategy_dir(sid, scope)
    sd.mkdir(parents=True, exist_ok=True)
    if int(req.stage) == 2:
        from app.chat import save_stage2_artifact_from_content
        result = save_stage2_artifact_from_content(req.content or "", sid=sid)
        if not result.get("ok"):
            return {"error": result.get("error", "stage 2 artifact save failed"), "strategy": sid, "stage": req.stage}
        return {
            "status": "ok",
            "strategy": result.get("strategy", sid),
            "stage": req.stage,
            "file": "model.py",
            "params": result.get("params", {}),
            "warnings": result.get("warnings", []),
        }
    if int(req.stage) in (4, 5):
        from app.chat import _latest_backtest_evidence
        from engine.stage_reports import build_stage4_check_report, build_stage5_analysis_report
        evidence = _latest_backtest_evidence(sid)
        if not evidence.get("summary"):
            return {"error": "缺少阶段3回测证据，无法保存阶段4/5报告", "strategy": sid, "stage": req.stage}
        content = (
            build_stage4_check_report(sd, evidence)
            if int(req.stage) == 4
            else build_stage5_analysis_report(sd, evidence)
        )
        filename = "check_report.md" if int(req.stage) == 4 else "analysis_template.md"
        (sd / filename).write_text(content, encoding="utf-8")
        return {"status": "ok", "strategy": sid, "stage": req.stage, "file": filename}
    stage_map = {
        1: "README.md",
        4: "check_report.md",
        5: "analysis_template.md",
    }
    filename = stage_map.get(int(req.stage))
    if not filename:
        return {"error": "unsupported stage"}
    (sd / filename).write_text(req.content or "", encoding="utf-8")
    if int(req.stage) == 1:
        from engine.strategy_spec import save_strategy_spec
        save_strategy_spec(sd, req.content or "", root=root)
    return {"status": "ok", "strategy": sid, "stage": req.stage, "file": filename}

@app.post("/api/strategy/complete")
async def api_complete_strategy(req: CompleteStrategyRequest):
    from engine.strategy_manager import get_active, get_active_scope, get_strategy_dir, mark_strategy_complete
    from app.chat import stage1_gate_status
    sid = get_active()
    sd = get_strategy_dir(sid, get_active_scope())
    gate = stage1_gate_status()
    if not gate.get("allowed"):
        questions = gate.get("blocking_questions") or []
        detail = "；".join(str(q) for q in questions[:5]) or gate.get("error", "阶段1仍有必须确认的问题")
        return {"error": "阶段1未确认，不能保存为完整策略：" + detail}
    required = [
        sd / "README.md",
        sd / "model.py",
        sd / "check_report.md",
        sd / "analysis_template.md",
    ]
    if not all(p.exists() for p in required):
        return {"error": "strategy artifacts are incomplete"}
    report_dir = root / "reports" / "web"
    has_report = report_dir.exists() and any(report_dir.glob(f"{sid}_*_backtest.html"))
    if not has_report:
        return {"error": "backtest report is missing"}
    requested_name = (req.name or "").strip()
    if not requested_name or requested_name == "新策略草稿" or requested_name.startswith("draft_"):
        requested_name = _infer_strategy_display_name(sd, sid)
    meta = mark_strategy_complete(sid, name=requested_name or sid, completed_stage=5, example=False)
    return {"status": "ok", "strategy": sid, "meta": meta}

def start():
    import webbrowser
    from app.chat import set_stage
    set_stage(1)
    webbrowser.open("http://localhost:8015")
    uvicorn.run(app, host="0.0.0.0", port=8015)

if __name__ == "__main__":
    start()

