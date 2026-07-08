"""AI聊天模块 — 自动代码保存"""
from pathlib import Path
import ast
from datetime import datetime, timedelta
from typing import Optional
import yaml, json, re
from engine.strategy_manager import list_strategies, get_active, get_active_scope, get_strategy_dir, set_active, load_module, load_params, save_strategy, is_complete_strategy
from engine.reporting import summarize_report
from engine.strategy_spec import save_strategy_spec
from engine.stage2_codegen import generate_stage2_code
from engine.stage_reports import build_stage4_check_report, build_stage5_analysis_report


def default_backtest_window():
    """Default to daily data over the latest five years."""
    end = datetime.now()
    start = end - timedelta(days=365 * 5)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _get_stage_file():
    return Path(__file__).parent.parent / "data" / "active_stage.txt"

def get_stage():
    sf = _get_stage_file()
    sf.parent.mkdir(exist_ok=True)
    if sf.exists():
        try: return int(sf.read_text(encoding="utf-8").strip())
        except: pass
    return 1

def set_stage(n):
    if int(n) <= 1:
        flag = _get_stage_file().parent / "stage2_ready.txt"
        if flag.exists():
            flag.unlink()
    _get_stage_file().write_text(str(n), encoding="utf-8")

def mark_stage2_ready(sid: str):
    (_get_stage_file().parent / "stage2_ready.txt").write_text(sid, encoding="utf-8")

def is_stage2_ready() -> bool:
    flag = _get_stage_file().parent / "stage2_ready.txt"
    return flag.exists() and flag.read_text(encoding="utf-8").strip() == get_active()

def stage1_gate_status() -> dict:
    """Return whether the current draft has confirmed enough stage-1 rules for stage 2."""
    sid = get_active()
    if is_complete_strategy(sid):
        return {"ok": True, "allowed": True, "strategy": sid, "complete": True, "blocking_questions": []}
    strategy_dir = get_strategy_dir(sid, get_active_scope())
    readme = strategy_dir / "README.md"
    if not readme.exists():
        return {"ok": False, "allowed": False, "strategy": sid, "error": "阶段1策略文档尚未生成", "blocking_questions": ["请先完成阶段1策略说明"]}
    spec_path = strategy_dir / "strategy_spec.json"
    if not spec_path.exists():
        try:
            save_strategy_spec(strategy_dir, readme.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"ok": False, "allowed": False, "strategy": sid, "error": f"阶段1规格生成失败: {exc}", "blocking_questions": []}
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if "stage2_gate" not in spec:
            save_strategy_spec(strategy_dir, readme.read_text(encoding="utf-8"))
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "allowed": False, "strategy": sid, "error": f"阶段1规格读取失败: {exc}", "blocking_questions": []}
    gate = spec.get("stage2_gate") or {}
    if gate.get("user_confirmed"):
        return {
            "ok": True,
            "allowed": True,
            "strategy": sid,
            "blocking_questions": [],
            "warnings": gate.get("warnings") or [],
            "confirmed": True,
            "confirmed_at": gate.get("confirmed_at", ""),
        }
    allowed = bool(gate.get("stage2_allowed"))
    questions = gate.get("blocking_questions") or []
    return {
        "ok": allowed,
        "allowed": allowed,
        "strategy": sid,
        "blocking_questions": questions,
        "warnings": gate.get("warnings") or [],
        "error": "" if allowed else "阶段1仍有必须确认的问题，不能进入阶段2",
    }

def confirm_stage1_gate() -> dict:
    """Persist an explicit user confirmation for stage-1 unresolved assumptions."""
    sid = get_active()
    if is_complete_strategy(sid):
        return {"ok": True, "strategy": sid, "complete": True}
    strategy_dir = get_strategy_dir(sid, get_active_scope())
    spec_path = strategy_dir / "strategy_spec.json"
    readme = strategy_dir / "README.md"
    if not spec_path.exists() and readme.exists():
        save_strategy_spec(strategy_dir, readme.read_text(encoding="utf-8"))
    if not spec_path.exists():
        return {"ok": False, "error": "阶段1规格不存在，无法确认"}
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    gate = spec.setdefault("stage2_gate", {})
    previous_questions = gate.get("blocking_questions") or []
    gate["user_confirmed"] = True
    gate["stage2_allowed"] = True
    gate["confirmed_at"] = datetime.now().isoformat(timespec="seconds")
    gate["confirmed_questions"] = previous_questions
    gate["blocking_questions"] = []
    gate["policy"] = "研究员已显式确认待确认项，可进入阶段2；阶段2仍不得新增未确认规则。"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    confirm_path = strategy_dir / "stage1_confirmations.json"
    confirm_path.write_text(
        json.dumps(
            {
                "strategy": sid,
                "confirmed_at": gate["confirmed_at"],
                "confirmed_questions": previous_questions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"ok": True, "strategy": sid, "confirmed_at": gate["confirmed_at"], "confirmed_questions": previous_questions}

def infer_active_stage() -> int:
    """Infer current strategy progress from saved artifacts."""
    sid = get_active()
    if not is_complete_strategy(sid) and not stage1_gate_status().get("allowed"):
        return 1
    strategy_dir = get_strategy_dir(sid, get_active_scope())
    stage = 1
    if (strategy_dir / "model.py").exists():
        stage = max(stage, 2)
    report_dir = Path(__file__).parent.parent / "reports" / "web"
    if report_dir.exists() and any(report_dir.glob(f"{sid}_*_backtest.html")):
        stage = max(stage, 3)
    if (strategy_dir / "check_report.md").exists():
        stage = max(stage, 4)
    if (strategy_dir / "analysis_template.md").exists():
        stage = max(stage, 5)
    return stage

def _stage4_allows_stage5() -> dict:
    """Block stage 5 when stage 4/backtest evidence says the result must be fixed first."""
    sid = get_active()
    strategy_dir = get_strategy_dir(sid, get_active_scope())
    if not (strategy_dir / "check_report.md").exists():
        return {"ok": False, "error": "阶段4检查报告尚未生成，不能进入阶段5分析"}

    evidence = _latest_backtest_evidence(sid)
    summary = evidence.get("summary", {}) if isinstance(evidence, dict) else {}
    gate = summary.get("research_gate") if isinstance(summary.get("research_gate"), dict) else {}
    blockers = gate.get("blockers") if isinstance(gate.get("blockers"), list) else []

    def is_technical_blocker(item) -> bool:
        text = str(item).lower()
        technical_terms = [
            "model.py", "代码", "接口", "signal", "exit", "future", "未来函数",
            "mismatch", "缺少", "无法", "无交易", "数据缺失", "回测失败",
            "strategy stop", "lookahead",
        ]
        research_only_terms = [
            "净利润", "总收益率", "夏普", "收益不足", "未转正", "回撤",
        ]
        if any(term.lower() in text for term in technical_terms):
            return True
        if any(term in text for term in research_only_terms):
            return False
        return False

    technical_blockers = [item for item in blockers if is_technical_blocker(item)]
    if technical_blockers:
        detail = "；".join(str(item) for item in technical_blockers[:3])
        return {
            "ok": False,
            "error": "阶段4已识别阻断项，当前不能进入阶段5。请先修复并重新回测：" + detail,
        }

    report_text = (strategy_dir / "check_report.md").read_text(encoding="utf-8", errors="ignore")
    blocking_terms = ["技术阻断项", "不能进入阶段5", "先修复再复测"]
    if any(term in report_text for term in blocking_terms):
        return {"ok": False, "error": "阶段4检查报告判定当前结果存在阻断项，不能进入阶段5。请先修复并重新回测。"}

    return {"ok": True}

def advance_stage(n: int) -> dict:
    """只允许停留当前阶段或进入下一阶段，防止误跳阶段。"""
    gate = stage1_gate_status()
    if not is_complete_strategy(get_active()) and not gate.get("allowed"):
        current = 1
    else:
        current = max(get_stage(), infer_active_stage())
    try:
        target = int(n)
    except Exception:
        return {"ok": False, "stage": current, "error": "阶段编号无效"}
    if target < 1 or target > 5:
        return {"ok": False, "stage": current, "error": "阶段编号必须在1到5之间"}
    if target == current:
        return {"ok": True, "stage": current}
    if target == current + 1:
        if target == 2:
            gate = stage1_gate_status()
            if not gate.get("allowed"):
                questions = gate.get("blocking_questions") or []
                detail = "；".join(str(q) for q in questions[:5]) or gate.get("error", "阶段1仍有必须确认的问题")
                return {"ok": False, "stage": current, "error": "阶段1未确认，不能进入阶段2：" + detail}
        if target == 3 and not is_stage2_ready():
            return {"ok": False, "stage": current, "error": "阶段2尚未生成并校验有效策略代码，不能进入阶段3回测"}
        if target == 5:
            stage5_gate = _stage4_allows_stage5()
            if not stage5_gate.get("ok"):
                return {"ok": False, "stage": current, "error": stage5_gate.get("error", "阶段4未通过，不能进入阶段5")}
        set_stage(target)
        return {"ok": True, "stage": target}
    return {
        "ok": False,
        "stage": current,
        "error": f"必须按顺序推进：当前是阶段{current}，不能直接进入阶段{target}"
    }


def load_config():
    return yaml.safe_load(open(Path(__file__).parent / "config.yaml", encoding="utf-8"))

def save_config(nc):
    p = Path(__file__).parent / "config.yaml"
    c = yaml.safe_load(open(p, encoding="utf-8")); c.update(nc)
    yaml.dump(c, open(p, "w", encoding="utf-8"), default_flow_style=False, allow_unicode=True)

def _latest_backtest_evidence(sid: str) -> dict:
    root = Path(__file__).parent.parent
    report_dir = root / "reports" / "web"
    reports = sorted(report_dir.glob(f"{sid}_*_backtest.html"), key=lambda p: p.stat().st_mtime, reverse=True) if report_dir.exists() else []
    if not reports:
        return {}
    report = reports[0]
    stem = report.name.replace("_backtest.html", "")
    summary = summarize_report(report_dir, stem) or {}
    return {
        "report_url": f"/reports/web/{report.name}",
        "summary": summary,
    }

def _line_numbered_text(text: str, max_lines: int = 260) -> str:
    """把代码转换成稳定行号证据，供阶段4引用。"""
    lines = (text or "").splitlines()
    numbered = [f"L{i:03d}: {line}" for i, line in enumerate(lines[:max_lines], start=1)]
    if len(lines) > max_lines:
        numbered.append(f"... 已截断，原文件共 {len(lines)} 行")
    return "\n".join(numbered)

def _strategy_artifact_context(sid: str) -> dict:
    sd = get_strategy_dir(sid, get_active_scope())
    items = {}
    for key, filename, limit in [
        ("策略说明", "README.md", 3500),
        ("策略代码摘要", "model.py", 4500),
        ("阶段4检查稿", "check_report.md", 2500),
        ("阶段5分析稿", "analysis_template.md", 2500),
    ]:
        path = sd / filename
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            items[key] = text[:limit] + ("\n..." if len(text) > limit else "")
    code_path = sd / "model.py"
    if code_path.exists():
        items["策略代码逐行证据"] = _line_numbered_text(code_path.read_text(encoding="utf-8"))
    spec_path = sd / "strategy_spec.json"
    if spec_path.exists():
        try:
            items["结构化策略规格"] = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception:
            items["结构化策略规格"] = spec_path.read_text(encoding="utf-8")[:4000]
    params_path = sd / "params.yaml"
    if params_path.exists():
        items["参数文件"] = params_path.read_text(encoding="utf-8")[:2000]
    return items

def _get_context(include_artifacts: bool = False):
    sid = get_active()
    ctx = {"活跃策略": sid, "参数": load_params(sid), "当前阶段": get_stage()}
    if include_artifacts:
        ctx.update(_strategy_artifact_context(sid))
        evidence = _latest_backtest_evidence(sid)
        if evidence:
            ctx["最新回测证据"] = evidence
    return ctx

def _strategy_qa_context(max_chars: int = 12000) -> str:
    """Build read-only current-strategy context for non-persistent Q&A."""
    sid = get_active()
    scope = get_active_scope()
    evidence = _latest_backtest_evidence(sid)
    context = {
        "active_strategy": sid,
        "scope": scope,
        "current_stage": get_stage(),
        "params": load_params(sid),
        "artifacts": _strategy_artifact_context(sid),
        "latest_backtest": evidence.get("summary", {}) if isinstance(evidence, dict) else {},
        "report_url": evidence.get("report_url", "") if isinstance(evidence, dict) else "",
        "qa_policy": (
            "只读上下文，仅用于回答研究员对当前策略的提问；"
            "不得推进阶段，不得生成或覆盖策略产物，不得把回答写入工作台。"
        ),
    }
    text = json.dumps(context, ensure_ascii=False, default=str)
    return text[:max_chars] + ("\n...<truncated>" if len(text) > max_chars else "")

def _strategy_qa_focus_note(user_text: str) -> str:
    """Add compact evidence for common current-strategy diagnostic questions."""
    text = user_text or ""
    diagnostic_terms = ["止损退出", "退出占比", "趋势奔跑", "充分让趋势", "stop_loss", "exit_signal"]
    if not any(term in text for term in diagnostic_terms):
        return ""
    sid = get_active()
    evidence = _latest_backtest_evidence(sid)
    summary = evidence.get("summary", {}) if isinstance(evidence, dict) else {}
    behavior = summary.get("behavior_diagnostics", {}) if isinstance(summary, dict) else {}
    diagnostics = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}
    strategy_dir = get_strategy_dir(sid, get_active_scope())
    code = ""
    code_path = strategy_dir / "model.py"
    if code_path.exists():
        code = code_path.read_text(encoding="utf-8", errors="ignore")
    has_exit_signal = "exit_signal" in code
    has_strategy_exit_col = "strategy_exit" in code
    has_strategy_stop_col = "strategy_stop_price" in code
    note = {
        "question_focus": "当前问题涉及止损退出占比和是否让趋势奔跑，必须先基于本策略证据回答。",
        "active_strategy": sid,
        "behavior_diagnostics": {
            "avg_bars_held": behavior.get("avg_bars_held"),
            "median_bars_held": behavior.get("median_bars_held"),
            "max_bars_held": behavior.get("max_bars_held"),
            "exit_reason": behavior.get("exit_reason"),
            "flags": behavior.get("flags"),
        },
        "engine_diagnostics": {
            "signals_seen": diagnostics.get("signals_seen"),
            "exit_signals_seen": diagnostics.get("exit_signals_seen"),
            "strategy_exits": diagnostics.get("strategy_exits"),
            "strategy_stop_exits": diagnostics.get("strategy_stop_exits"),
            "entries_opened": diagnostics.get("entries_opened"),
        },
        "code_evidence": {
            "model_py_contains_exit_signal": has_exit_signal,
            "model_py_contains_strategy_exit": has_strategy_exit_col,
            "model_py_contains_strategy_stop_price": has_strategy_stop_col,
            "interpretation_hint": (
                "如果退出原因为 strategy_trailing_stop 且 strategy_stop_exits 大于0，说明当前退出来自策略层输出的动态移动止损，"
                "不是通用回测引擎的普通 stop_loss。回答时应先说明这属于策略原文的时间衰减移动止损复现结果；"
                "如果平均持仓仍很短，再讨论参数过紧、入场过滤过严或品种行情不匹配。"
            ),
        },
    }
    return json.dumps(note, ensure_ascii=False, default=str)

def _get_stage_context(stage: int, last_user: str) -> dict:
    """按阶段隔离上下文，避免历史示例策略污染新策略生成。"""
    context_policy = {
        "context_policy": "只使用当前阶段结构化上下文和用户最近输入；不要引用未提供的聊天历史；非研究问题不写入策略产物。",
    }
    if stage == 1:
        return {
            **context_policy,
            "当前阶段": stage,
            "用户最近输入": last_user[:2000],
            "recent_user_input": last_user[:2000],
        }
    if stage == 2:
        ctx = {
            **context_policy,
            "当前阶段": stage,
            "用户最近输入": last_user[:2000],
            "recent_user_input": last_user[:2000],
        }
        sd = get_strategy_dir(get_active(), get_active_scope())
        readme = sd / "README.md"
        spec_path = sd / "strategy_spec.json"
        if readme.exists():
            text = readme.read_text(encoding="utf-8").strip()
            ctx["阶段1策略说明"] = text[:5000]
            if not spec_path.exists():
                try:
                    save_strategy_spec(sd, text)
                except Exception:
                    pass
        if spec_path.exists():
            try:
                ctx["strategy_spec"] = json.loads(spec_path.read_text(encoding="utf-8"))
            except Exception:
                ctx["strategy_spec"] = spec_path.read_text(encoding="utf-8")[:4000]
        return ctx
    ctx = _get_context(include_artifacts=stage in (4, 5))
    ctx.update(context_policy)
    ctx["当前阶段"] = stage
    ctx["用户最近输入"] = last_user[:2000]
    ctx["recent_user_input"] = last_user[:2000]
    return ctx

def _has_backtest_evidence(text: str) -> bool:
    """阶段5必须基于真实回测数据，不能只凭策略描述生成绩效报告。"""
    if not text:
        return False
    keys = [
        "total_trades", "trades_count", "equity_curve", "net_profit",
        "max_drawdown", "sharpe_ratio", "win_rate", "回测结果",
        "交易记录", "资金曲线", "最大回撤", "夏普", "胜率"
    ]
    return any(k in text for k in keys)

def _stage_command_target(user_text: str):
    """检测用户输入的阶段推进意图。"""
    m = re.search(r"进入阶段\s*([1-5])", user_text)
    if m:
        return int(m.group(1))
    text = user_text or ""
    if any(k in text for k in ["开始编写代码", "生成代码", "生成策略代码", "编写策略代码", "进入建模", "开始建模"]):
        return 2
    if any(k in text for k in ["开始回测", "执行回测", "运行回测"]):
        return 3
    if any(k in text for k in ["开始检查", "请检查", "检查策略", "策略检查", "一致性检查"]):
        return 4
    if any(k in text for k in ["开始分析", "生成分析", "分析报告"]):
        return 5
    return None

def _classify_intent(text: str) -> str:
    """轻量意图路由：决定是否进入五阶段策略产线。"""
    text = (text or "").strip()
    if not text:
        return "empty"
    if _stage_command_target(text) is not None:
        return "stage_command"
    if _looks_like_strategy_material(text):
        return "strategy_material"

    revision_words = ["修改", "调整", "改成", "换成", "增加", "加一个", "加入", "删除", "去掉", "参数", "默认值"]
    strategy_terms = ["止损", "止盈", "入场", "出场", "开仓", "平仓", "均线", "周期", "阈值", "加仓", "风控", "过滤", "ADX", "ATR", "OBV", "成交量", "量能"]
    if any(k in text for k in revision_words) and any(k in text for k in strategy_terms):
        return "strategy_revision"

    platform_words = [
        "怎么打开", "如何打开", "启动", "端口", "8015", "新建对话", "历史策略",
        "保存策略", "切换阶段", "按钮", "工作台", "对话框", "界面", "报错",
        "readme", "README", "api key", "API Key", "deepseek", "DeepSeek",
    ]
    if any(k in text for k in platform_words):
        return "platform_help"

    quant_words = [
        "量化", "CTA", "期货", "回测", "策略", "因子", "均线", "夏普", "回撤",
        "胜率", "盈亏比", "止损", "止盈", "开仓", "平仓", "RAG", "提示词",
        "滑点", "手续费", "保证金", "AKShare", "数据", "交易",
        "权益", "权益曲线", "资金曲线", "净值曲线", "净值", "equity", "equity_curve",
    ]
    if any(k.lower() in text.lower() for k in quant_words):
        return "quant_qa"

    return "off_topic"


def _should_persist_intent(intent: str) -> bool:
    return intent in {"stage_command", "strategy_material"}


def _build_non_persistent_system_prompt(intent: str) -> str:
    if intent == "strategy_revision":
        return (
            "你是AI量化研究平台的流程守门助手。用户提出了当前策略修改请求。"
            "请简洁说明：该修改会影响阶段1策略规格、阶段2代码和阶段3回测；"
            "为了避免污染已生成产物，建议用户回到阶段1修改策略说明，或新建对话作为新策略。"
            "不要直接生成新代码，不推进阶段，不覆盖当前策略。"
        )
    if intent == "platform_help":
        return (
            "你是AI量化研究平台的产品助手。只回答平台使用、流程、状态、按钮、"
            "数据/回测/策略产物相关问题。回答要简洁，不推进阶段，不生成策略文档。"
        )
    if intent == "quant_qa":
        return (
            "你是期货CTA量化研究顾问。回答用户的量化研究问题，但不要推进五阶段流程，"
            "不要生成或覆盖当前策略产物。需要区分通用建议和当前平台能力。"
        )
    return (
        "你是AI量化研究平台的助手。用户问题如果不是阶段产物生成、策略修改或流程推进请求，"
        "就作为普通对话自然回答。回答要简洁，不要推进阶段，不要生成或覆盖策略产物，"
        "不要声称已经写入工作台。"
    )


def _deterministic_non_persistent_reply(intent: str, text: str) -> Optional[str]:
    """对流程安全性要求高的问题直接本地回复，避免模型误推进阶段。"""
    if intent == "strategy_revision":
        return (
            "这是一个策略修改请求，我不会直接覆盖当前工作台产物。\n\n"
            "原因：修改策略逻辑或参数会影响阶段1策略规格、阶段2代码和阶段3回测结果，"
            "如果直接在当前阶段改，后面的检查和分析会失去一致性。\n\n"
            "建议二选一：\n"
            "1. 回到阶段1，把修改后的完整策略说明重新整理一遍，再重新生成代码和回测。\n"
            "2. 点击“新建对话”，把它作为一个新策略版本单独跑完整流程。\n\n"
            "这条回复不会写入策略工作台，也不会推进当前策略流程。"
        )
    if intent == "platform_help":
        lower = (text or "").lower()
        if any(k in text for k in ["怎么打开", "如何打开", "启动", "端口", "8015"]) or "localhost" in lower:
            return (
                "在项目根目录启动：\n\n"
                "```powershell\n"
                "cd C:\\Users\\Administrator\\Documents\\Codex\\2026-06-03\\vibecoding\n"
                "python -X utf8 -m uvicorn app.main:app --host 127.0.0.1 --port 8015\n"
                "```\n\n"
                "然后打开：`http://127.0.0.1:8015`\n\n"
                "这条回复不会写入策略工作台，也不会推进当前策略流程。"
            )
        if any(k in text for k in ["历史策略", "保存策略", "新建对话"]):
            return (
                "当前平台的策略入口分两类：\n\n"
                "- 新建对话：创建新的草稿策略，右侧工作台从阶段说明开始。\n"
                "- 历史策略：只展示已经完成五阶段并归档的策略。\n\n"
                "未完成草稿保存在 `strategies_drafts/`，不会出现在历史策略列表里；"
                "完成阶段5后点击保存，才会进入历史策略。\n\n"
                "这条回复不会写入策略工作台，也不会推进当前策略流程。"
            )
    return None


def _looks_like_strategy_material(text: str) -> bool:
    """识别研究员贴入的原始策略想法，避免被当前阶段状态误路由到阶段2。"""
    text = (text or "").strip()
    if len(text) < 20:
        return False
    command_words = [
        "开始编写代码", "生成代码", "编写策略代码", "开始建模", "进入建模",
        "开始回测", "执行回测", "运行回测", "开始检查", "开始分析",
    ]
    if any(k in text for k in command_words):
        return False
    material_words = [
        "策略思路", "策略逻辑", "交易思路", "研究笔记", "买入信号", "卖出信号",
        "入场", "出场", "做多", "做空", "上穿", "下穿", "均线", "止损", "止盈",
    ]
    return sum(1 for k in material_words if k in text) >= 2


def _create_draft_for_new_material() -> str:
    sid = "draft_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    sd = get_strategy_dir(sid, "draft")
    sd.mkdir(parents=True, exist_ok=True)
    with open(sd / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"complete": False, "name": "新策略草稿"}, f, default_flow_style=False, allow_unicode=True)
    set_active(sid, "draft")
    set_stage(1)
    return sid

def _literal_default(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None

def _extract_default_params(code: str) -> dict:
    """从 __init__ 里的 params.get("name", default) 提取默认参数。"""
    tree = ast.parse(code)
    params = {}
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]:
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Attribute) or node.func.attr != "get":
                    continue
                value = node.func.value
                if not isinstance(value, ast.Name) or value.id != "params":
                    continue
                if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    continue
                key = node.args[0].value
                default = _literal_default(node.args[1]) if len(node.args) > 1 else None
                params[key] = default
    return params

def _load_strategy_spec(strategy_dir: Optional[Path]) -> dict:
    if not strategy_dir:
        return {}
    spec_path = strategy_dir / "strategy_spec.json"
    if not spec_path.exists():
        return {}
    try:
        return json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _spec_text(spec: dict) -> str:
    if not spec:
        return ""
    return json.dumps(spec, ensure_ascii=False)

def _validate_spec_alignment(code: str, spec: dict):
    """Catch high-impact cases where code is runnable but does not implement the stage-1 spec."""
    text = _spec_text(spec)
    if not text:
        return
    gate = spec.get("stage2_gate") if isinstance(spec, dict) else {}
    if isinstance(gate, dict) and gate.get("stage2_allowed") is False:
        questions = gate.get("blocking_questions") or []
        detail = "；".join(str(item) for item in questions[:5]) or "阶段1存在未确认关键规则"
        raise ValueError("阶段2已被规格门禁拦截：关键规则尚未确认，不能自由补全。请先回到阶段1确认：" + detail)
    dynamic_terms = ["动态突破周期", "动态入场", "动态通道", "dynamic_entry", "long_entry_period", "short_entry_period"]
    requires_dynamic_channel = any(term in text for term in dynamic_terms)
    if requires_dynamic_channel:
        required_tokens = ["long_entry_period", "short_entry_period", "dynamic_entry_high", "dynamic_entry_low"]
        missing = [token for token in required_tokens if token not in code]
        if missing:
            raise ValueError("阶段2代码未实现阶段1要求的动态入场周期，缺少: " + ", ".join(missing))
        fixed_dynamic = re.search(
            r"dynamic_entry_(?:high|low)[^\n=]*=\s*[^\n]*rolling\s*\(\s*window\s*=\s*self\.entry_period_base",
            code,
        )
        if fixed_dynamic or "这里使用固定周期" in code:
            raise ValueError("阶段2代码声明了动态入场通道，但实际仍使用固定 entry_period_base 计算通道")
        if "_dynamic_rolling" not in code and "for i in range" not in code:
            raise ValueError("阶段2代码缺少逐K线动态窗口计算，无法真实实现动态突破周期")

def _validate_strategy_code(code: str, params: dict, strategy_dir: Optional[Path] = None) -> list:
    """校验阶段2生成代码：导入安全、接口完整、小样本可运行。"""
    import pandas as pd
    import numpy as np

    _validate_spec_alignment(code, _load_strategy_spec(strategy_dir))

    tree = ast.parse(code)
    allowed_imports = {"pandas", "numpy", "math", "typing", "datetime"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_imports:
                    raise ValueError(f"不允许导入模块: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root and root not in allowed_imports:
                raise ValueError(f"不允许导入模块: {node.module}")

    namespace = {}
    exec(compile(code, "<generated_strategy>", "exec"), namespace)
    strategy_classes = []
    fallback_classes = []
    for obj in namespace.values():
        if not isinstance(obj, type):
            continue
        has_contract = all(callable(getattr(obj, method, None)) for method in ["compute_indicators", "generate_signals", "run"])
        if not has_contract:
            continue
        if "Strategy" in obj.__name__:
            strategy_classes.append(obj)
        else:
            fallback_classes.append(obj)
    strategy_classes.extend(fallback_classes)
    if not strategy_classes:
        raise ValueError("未找到包含 compute_indicators、generate_signals、run 的策略类")
    StrategyClass = strategy_classes[0]

    for method in ["compute_indicators", "generate_signals", "run"]:
        if not callable(getattr(StrategyClass, method, None)):
            raise ValueError(f"缺少方法: {method}")

    n = 80
    close = np.linspace(100, 120, n) + np.sin(np.arange(n) / 3) * 2
    sample = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close - 0.5,
        "high": close + 1.5,
        "low": close - 1.5,
        "close": close,
        "volume": np.linspace(10000, 20000, n),
        "open_interest": np.linspace(50000, 60000, n),
    })
    result = StrategyClass(params).run(sample)
    if not isinstance(result, pd.DataFrame):
        raise ValueError("run() 必须返回 pandas.DataFrame")
    if "signal_raw" not in result.columns:
        raise ValueError("run() 返回结果缺少 signal_raw 列")
    signals = set(result["signal_raw"].dropna().unique().tolist())
    if not signals.issubset({-1, 0, 1}):
        raise ValueError("signal_raw 只能取值 -1、0、1")
    if "_error" in result.columns and result["_error"].dropna().astype(str).str.len().gt(0).any():
        raise ValueError(f"策略 dry-run 报错: {result['_error'].dropna().iloc[0]}")
    if re.search(r"df\.loc\[[^\]]*conflict[^\]]*,\s*['\"]exit_signal['\"]\]\s*=\s*0", code):
        raise ValueError("冲突信号处理错误：普通入场与只平仓冲突时应优先保留 exit_signal，并将 signal_raw 置0")
    risky_state = []
    risky_state_patterns = [
        r"self\.(?:_)?(?:current_)?position\b",
        r"self\.(?:_)?entry_price\b",
        r"self\.(?:_)?bars_(?:held|since_entry)\b",
        r"self\.(?:_)?stop_price\b",
        r"self\.(?:_)?has_added\b",
        r"self\.(?:_)?add_count\b",
    ]
    for pattern in risky_state_patterns:
        risky_state.extend(match.group(0) for match in re.finditer(pattern, code))
    risky_state = list(dict.fromkeys(risky_state))
    if risky_state:
        raise ValueError(
            "阶段2代码不得用 self 持久化交易状态，避免与回测引擎的下一根K线成交状态错位。"
            "请改用 generate_signals 内部局部状态并按 signal_raw 下一根K线成交对齐。命中: "
            + "、".join(risky_state)
        )

    warnings = []
    leaked_names = []
    current_sid = get_active()
    banned_template_names = ["locked_vwma_obv", "simple_ma", "expma_cross", "turtle_trader", "auto_strategy"]
    for name in banned_template_names:
        if name != current_sid and name.lower() in code.lower():
            leaked_names.append(name)
    if leaked_names:
        warnings.append("代码疑似混入历史模板名称：" + "、".join(leaked_names) + "；请确认不是套错策略模板")
    if "exit_signal" not in result.columns:
        warnings.append("未输出 exit_signal 列；新引擎会自动按0处理，但只平仓逻辑建议显式写入 exit_signal")
    elif "signal_raw" in result.columns:
        signal_count = int((result["signal_raw"].fillna(0).astype(int) != 0).sum())
        exit_count = int((result["exit_signal"].fillna(0).astype(int) != 0).sum())
        if signal_count == 0 and exit_count > 0:
            warnings.append("dry-run 检测到 exit_signal 但没有 signal_raw；请确认离场逻辑是否只在持仓状态下触发，避免空仓离场信号压制入场信号")
    if ".shift(" not in code:
        warnings.append("未检测到 .shift()，请人工确认没有未来函数")
    return warnings

def validate_active_strategy() -> dict:
    """校验当前 active 策略，供已有策略解锁阶段3。"""
    sid = get_active()
    strategy_dir = get_strategy_dir(sid, get_active_scope())
    model_path = strategy_dir / "model.py"
    if not model_path.exists():
        return {"ok": False, "strategy": sid, "error": "当前策略缺少 model.py"}
    code = model_path.read_text(encoding="utf-8").lstrip("\ufeff")
    params = load_params(sid) or {}
    try:
        extracted = _extract_default_params(code)
        merged = dict(extracted)
        merged.update(params)
        warnings = _validate_strategy_code(code, merged, strategy_dir)
        if extracted and not params:
            with open(strategy_dir / "params.yaml", "w", encoding="utf-8") as f:
                yaml.dump(extracted, f, default_flow_style=False, allow_unicode=True)
            merged = extracted
        mark_stage2_ready(sid)
        set_stage(2)
        return {"ok": True, "strategy": sid, "params": merged, "warnings": warnings}
    except Exception as e:
        return {"ok": False, "strategy": sid, "error": str(e)}

def _execute_function(name, args_str):
    try: args = json.loads(args_str) if args_str.strip() else {}
    except: return json.dumps({"error": "Invalid JSON"})
    if name == "run_backtest":
        stage = get_stage()
        if stage not in (2, 3):
            return json.dumps({"error": f"必须按顺序推进：当前是阶段{stage}，不能直接运行回测"})
        sid = args.get("strategy") or get_active()
        from engine.data import get_main_contract_data
        from engine.backtest import BacktestEngine
        StrategyClass = load_module(sid)
        params = load_params(sid); params.update(args.get("params", {}))
        symbol = args.get("symbol", "RB0")
        start_date, end_date = default_backtest_window()
        df = get_main_contract_data(symbol, start_date, end_date)
        bt = BacktestEngine(100000).run(StrategyClass(params).run(df), params)
        bt["stats"]["symbol"] = symbol
        bt["stats"]["data_frequency"] = "日线"
        bt["stats"]["default_window"] = "近五年"
        bt["stats"]["requested_period"] = {"start": start_date, "end": end_date}
        set_stage(3)
        return json.dumps({"symbol":symbol,"strategy":sid,"params":params,"stats":bt["stats"],"trades_count":len(bt["trades"])},ensure_ascii=False,default=str)
    elif name == "switch_strategy":
        sid = args.get("id","")
        strategy_dir = get_strategy_dir(sid)
        if not sid or not strategy_dir.exists(): return json.dumps({"error":f"不存在: {sid}"})
        scope = "complete" if "strategies_drafts" not in str(strategy_dir) else "draft"
        set_active(sid, scope); set_stage(1)
        return json.dumps({"status":"ok","active":sid})
    elif name == "create_strategy":
        sid = args.get("id", args.get("name","custom"))
        sid = re.sub(r'[^a-zA-Z0-9_]','_',sid.lower())[:30]
        code = args.get("code","")
        if not code: return json.dumps({"error":"缺少代码"})
        save_strategy(sid, code, args.get("params",{}))
        set_active(sid, "draft"); set_stage(2)
        return json.dumps({"status":"ok","strategy":sid})
    elif name == "update_params":
        sid = get_active(); cur = load_params(sid); cur.update(args)
        yaml.dump(cur, open(get_strategy_dir(sid, get_active_scope())/"params.yaml","w",encoding="utf-8"), default_flow_style=False, allow_unicode=True)
        return json.dumps({"status":"ok","updated":args})
    elif name == "list_strategies": return json.dumps(list_strategies(),ensure_ascii=False)
    elif name == "get_strategy_info":
        sid = get_active()
        return json.dumps({"active":sid,"params":load_params(sid),"all":[s["id"] for s in list_strategies()]},ensure_ascii=False)
    return json.dumps({"error":f"Unknown: {name}"})


def _extract_strategy_code(content: str) -> str:
    """Extract a strategy code artifact from an AI response or workspace content."""
    if not content:
        return ""
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', content, re.DOTALL | re.IGNORECASE)
    candidates = [block.strip() for block in blocks]
    candidates.append(content.strip())
    for candidate in candidates:
        if not candidate or "class " not in candidate or "def run" not in candidate:
            continue
        try:
            compile(candidate, "<generated_strategy>", "exec")
            return candidate
        except Exception:
            pass
    return ""


def _find_strategy_class_names(code: str) -> list:
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    required = {"__init__", "compute_indicators", "generate_signals", "run"}
    names = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
        if required.issubset(methods):
            names.append(node.name)
    return names


def _save_controlled_stage2_code(target_sid: str, target_dir: Optional[Path], reason: str) -> Optional[dict]:
    safe_reason = re.sub(r"self\.[A-Za-z_]+", "self_state", reason or "")
    generated_code, generated_params, family = generate_stage2_code(target_dir, reason=safe_reason)
    if not generated_code:
        return None
    generated_warnings = _validate_strategy_code(generated_code, generated_params, target_dir)
    save_strategy(target_sid, generated_code, generated_params)
    draft_dir = get_strategy_dir(target_sid, "draft")
    trace = {
        "strategy": target_sid,
        "family": family,
        "reason": reason,
        "source": "strategy_spec.json + README.md",
        "params": generated_params,
        "policy": "controlled_codegen_only_fills_template_from_stage1_spec",
    }
    (draft_dir / "stage2_codegen_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    set_active(target_sid, "draft")
    mark_stage2_ready(target_sid)
    set_stage(2)
    generated_warnings = list(generated_warnings or []) + [f"已使用平台受控模板生成阶段2代码: {family}"]
    return {
        "ok": True,
        "strategy": target_sid,
        "params": generated_params,
        "warnings": generated_warnings,
        "code": generated_code,
        "controlled_codegen": True,
        "family": family,
    }


def save_stage2_artifact_from_content(content: str, sid: Optional[str] = None) -> dict:
    """Validate and persist stage 2 code for the active draft strategy."""
    target_sid = sid or get_active()
    target_scope = get_active_scope()
    target_dir = get_strategy_dir(target_sid, target_scope) if target_sid else None
    code = _extract_strategy_code(content)
    if not code:
        try:
            controlled = _save_controlled_stage2_code(target_sid, target_dir, "AI response did not contain valid code")
            if controlled:
                controlled["warnings"].append("controlled fallback used because AI returned no code")
                return controlled
        except Exception as e:
            return {"ok": False, "error": f"no valid Python code; controlled codegen failed: {e}"}
        return {"ok": False, "error": "未找到可保存的 Python 策略代码"}
    if len(code) < 100:
        return {"ok": False, "error": "策略代码过短，未保存"}
    strategy_classes = _find_strategy_class_names(code)
    if not strategy_classes:
        return {"ok": False, "error": "策略代码未保存: 没有找到包含 __init__、compute_indicators、generate_signals、run 的策略类"}
    try:
        compile(code, "<generated_strategy>", "exec")
        params = _extract_default_params(code)
        target_sid = sid or get_active()
        target_scope = get_active_scope()
        target_dir = get_strategy_dir(target_sid, target_scope) if target_sid else None
        warnings = _validate_strategy_code(code, params, target_dir)
        if get_active_scope() != "draft" or not target_sid:
            class_name = strategy_classes[0] if strategy_classes else 'AutoStrategy'
            base_name = re.sub(r'(?i)strategy$', '', class_name).strip("_") or "auto"
            target_sid = re.sub(r'(?<!^)(?=[A-Z])', '_', base_name).lower()
            target_sid = re.sub(r'[^a-zA-Z0-9_]', '_', target_sid).strip("_")[:40] or "auto"
            target_sid = f"{target_sid}_strategy"
        save_strategy(target_sid, code, params)
        set_active(target_sid, "draft")
        mark_stage2_ready(target_sid)
        set_stage(2)
        return {"ok": True, "strategy": target_sid, "params": params, "warnings": warnings, "code": code}
    except Exception as e:
        try:
            controlled = _save_controlled_stage2_code(target_sid, target_dir, f"AI code failed validation: {e}")
            if controlled:
                controlled["warnings"].append(f"controlled fallback used after validation failure: {e}")
                controlled["original_error"] = str(e)
                return controlled
        except Exception as gen_error:
            return {"ok": False, "error": f"stage2 validation failed: {e}; controlled codegen failed: {gen_error}"}
        return {"ok": False, "error": f"保存失败: {e}"}


def _auto_save_code(content: str, stage: int) -> str:
    """自动从AI回复中提取Python代码并保存为策略"""
    if stage != 2 or not content:
        return content
    result = save_stage2_artifact_from_content(content)
    if not result.get("ok"):
        if "class " in content or "def run" in content or "```" in content:
            return content + f'\n\n[{result.get("error", "策略代码未保存")}]'
        return content
    notice = f'\n\n[策略代码已保存并通过基础校验: {result["strategy"]}]'
    if result.get("params"):
        notice += f'\n[参数已写入 params.yaml，共 {len(result["params"])} 项]'
    if result.get("warnings"):
        notice += "\n[校验提醒: " + "；".join(result["warnings"]) + "]"
    return content + notice


def chat_with_deepseek(messages: list) -> dict:
    from engine.ai.client import get_client
    from engine.ai.prompt_manager import PromptManager
    client = get_client(); pm = PromptManager()

    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user": last_user = m.get("content",""); break
    intent = _classify_intent(last_user)
    if intent == "empty":
        return {"content": "请输入策略想法、阶段指令或平台问题。", "error": None, "stage": get_stage(), "intent": intent, "persist": False}

    if not _should_persist_intent(intent):
        deterministic = _deterministic_non_persistent_reply(intent, last_user)
        if deterministic is not None:
            return {"content": deterministic, "error": None, "stage": get_stage(), "intent": intent, "persist": False}
        sys_prompt = _build_non_persistent_system_prompt(intent)
        if intent == "quant_qa":
            sys_prompt += (
                "\n\n你会收到当前策略的只读上下文。凡是用户问“这个策略/当前策略/样本策略/为什么会这样”，"
                "必须优先基于当前策略证据回答，先引用具体字段或代码证据，再给通用解释。"
                "不得直接给泛泛的量化知识回答。"
                "如果上下文没有证据，必须明确说缺少证据，不要编造。"
                "这类回答只用于对话解释，不推进阶段，也不写入工作台。"
            )
            focus_note = _strategy_qa_focus_note(last_user)
            safe_messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": (
                        ("当前问题相关证据：\n```json\n" + focus_note + "\n```\n\n" if focus_note else "")
                        + "当前策略只读上下文：\n```json\n"
                        + _strategy_qa_context()
                        + "\n```\n\n研究员问题：\n"
                        + last_user[:3000]
                    ),
                },
            ]
        else:
            safe_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": last_user[:3000]},
            ]
        result = client.chat(safe_messages)
        if result.get("error"):
            return {**result, "stage": get_stage(), "intent": intent, "persist": False}
        content = (result.get("content") or "").strip()
        return {"content": content, "error": None, "stage": get_stage(), "intent": intent, "persist": False}

    stage = get_stage()
    if intent == "strategy_material":
        if get_active_scope() == "complete" or stage != 1:
            _create_draft_for_new_material()
        else:
            set_stage(1)
        stage = 1
    target_stage = _stage_command_target(last_user) if last_user else None
    if target_stage is not None:
        adv = advance_stage(target_stage)
        if not adv["ok"]:
            return {"content": adv["error"], "error": None, "stage": stage, "intent": intent, "persist": False}
        stage = adv["stage"]
    set_stage(stage)

    local_evidence = _latest_backtest_evidence(get_active())
    if stage in (4, 5) and local_evidence.get("summary"):
        strategy_dir = get_strategy_dir(get_active(), get_active_scope())
        if stage == 4:
            content = build_stage4_check_report(strategy_dir, local_evidence)
        else:
            content = build_stage5_analysis_report(strategy_dir, local_evidence)
        return {"content": content, "error": None, "stage": stage, "intent": intent, "persist": True}

    if stage == 5 and not _has_backtest_evidence(last_user) and not local_evidence.get("summary"):
        return {
            "content": (
                "缺少真实回测数据，无法生成阶段5分析报告。\n\n"
                "请先完成阶段3回测，并提供或传入以下数据：\n"
                "- 绩效指标：收益率、净利润、最大回撤、夏普、胜率、盈亏比、交易次数\n"
                "- 交易明细：开平仓时间、方向、价格、盈亏、手续费\n"
                "- 资金曲线：每日/每根K线权益变化\n"
                "- 如需年度对比，还需要按年份拆分后的收益、回撤、交易次数"
            ),
            "error": None,
            "stage": stage,
            "intent": intent,
            "persist": False,
        }

    stage_templates = {1:"stage_1_conceive",2:"stage_2_model",3:"stage_2_model",4:"stage_4_check",5:"stage_5_analyze"}
    template_name = stage_templates.get(stage, "stage_1_conceive")
    ctx = _get_stage_context(stage, last_user)

    sys_prompt = pm.get_system_prompt(ctx)
    stage_prompt = pm.get_stage_prompt(template_name, ctx) if template_name else ""
    if stage_prompt:
        sys_prompt = sys_prompt + "\n\n" + stage_prompt

    # 后端只给模型当前阶段所需的结构化上下文和最近用户输入，避免长对话污染策略产物。
    working_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": last_user[:5000]},
    ]

    for _ in range(5):
        result = client.chat(working_messages)
        if result.get("error"):
            return {**result, "stage": stage, "intent": intent, "persist": False}
        content = result.get("content","").strip()

        if "FUNCTION_CALL:" in content:
            lines = content.split("\n"); fn = ""; fa = ""
            for i,line in enumerate(lines):
                if line.strip().startswith("FUNCTION_CALL:"):
                    fn = line.split(":",1)[1].strip()
                    for j in range(i+1,len(lines)): fa += lines[j].strip()
                    break
            if fn:
                r = _execute_function(fn, fa)
                working_messages.append({"role":"assistant","content":content})
                working_messages.append({"role":"user","content":f"执行结果:\n{r}\n\n用中文简洁总结。"})
                continue
        # Auto-progression: stage 1 doc -> trigger stage 2
        stage1_kw = ['策略核心思想', '参数定义表格', '入场规则', '待验证假设', '用户确认记录', '策略文档已完成']
        is_stage1_doc = any(kw in content for kw in stage1_kw)
        has_code = 'class ' in content and 'def compute_indicators' in content
        if stage == 1 and is_stage1_doc and not has_code and content.count('```python') == 0:
            return {"content": content, "error": None, "stage": stage, "intent": intent, "persist": True}
        content = _auto_save_code(content, stage)
        return {"content": content, "error": None, "stage": stage, "intent": intent, "persist": True}
    content = _auto_save_code(content, stage)
    return {"content": content, "error": None, "stage": stage, "intent": intent, "persist": True}
