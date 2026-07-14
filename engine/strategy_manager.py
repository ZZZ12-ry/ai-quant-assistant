"""Strategy storage and activation helpers."""
from pathlib import Path
import importlib.util
import shutil
import sys
import re
import json
from datetime import datetime
from typing import Optional, Tuple
import yaml

ROOT = Path(__file__).parent.parent
COMPLETED_DIR = ROOT / "strategies"
DRAFT_DIR = ROOT / "strategies_drafts"
ACTIVE_FILE = ROOT / "data" / "active_strategy.txt"


def _ensure_dirs():
    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_meta(strategy_dir: Path) -> dict:
    mp = strategy_dir / "meta.yaml"
    if not mp.exists():
        return {}
    with open(mp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _infer_display_name(strategy_dir: Path, fallback: str) -> str:
    spec_path = strategy_dir / "strategy_spec.json"
    if spec_path.exists():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            name = str(spec.get("strategy_name") or "").strip()
            if name and name not in {"未命名策略", "海龟交易趋势突破", "双均线趋势跟踪"}:
                return name
        except Exception:
            pass
    readme_path = strategy_dir / "README.md"
    if readme_path.exists():
        try:
            first_line = readme_path.read_text(encoding="utf-8").splitlines()[0].strip()
            if first_line.startswith("#"):
                title = first_line.lstrip("#").strip()
                if title:
                    return title
        except Exception:
            pass
    meta = _load_meta(strategy_dir)
    candidate = str(meta.get("name") or "").strip()
    if candidate and candidate != "新策略草稿" and not candidate.startswith("draft_"):
        return candidate
    return fallback


def _parse_active_ref(raw: str) -> Tuple[Optional[str], Optional[str]]:
    text = (raw or "").lstrip("\ufeff").strip()
    if not text:
        return None, None
    if ":" in text:
        scope, sid = text.split(":", 1)
        if scope in {"complete", "draft"} and sid:
            return scope, sid
    return None, text


def _strategy_dir_for_scope(sid: str, scope: str) -> Path:
    return (COMPLETED_DIR if scope == "complete" else DRAFT_DIR) / sid


def get_strategy_dir(sid: str, scope: Optional[str] = None) -> Path:
    _ensure_dirs()
    if scope in {"complete", "draft"}:
        return _strategy_dir_for_scope(sid, scope)
    complete_dir = COMPLETED_DIR / sid
    draft_dir = DRAFT_DIR / sid
    if complete_dir.exists():
        return complete_dir
    return draft_dir


def get_strategy_scope(sid: str) -> str:
    _ensure_dirs()
    if (COMPLETED_DIR / sid).exists():
        return "complete"
    return "draft"


def is_complete_strategy(sid: str) -> bool:
    return bool(_load_meta(COMPLETED_DIR / sid).get("complete", False))


def _safe_report_prefix(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value or "").strip("_")


def _has_report_artifact(sid: str) -> bool:
    report_dir = ROOT / "reports" / "web"
    if not report_dir.exists():
        return False
    prefixes = [sid]
    safe_sid = _safe_report_prefix(sid)
    if safe_sid and safe_sid != sid:
        prefixes.append(safe_sid)
    return any(any(report_dir.glob(f"{prefix}_*_backtest.html")) for prefix in prefixes if prefix)


def _is_complete_draft_version(sid: str) -> bool:
    draft_dir = DRAFT_DIR / sid
    if not draft_dir.exists() or not draft_dir.is_dir():
        return False
    return bool(
        (draft_dir / "model.py").exists()
        and (draft_dir / "check_report.md").exists()
        and (draft_dir / "analysis_template.md").exists()
        and _has_report_artifact(sid)
    )


def list_strategies():
    _ensure_dirs()
    result = []
    for d in COMPLETED_DIR.iterdir():
        if d.is_dir() and (d / "model.py").exists():
            meta = _load_meta(d)
            if not meta.get("complete", False):
                continue
            params = {}
            pp = d / "params.yaml"
            if pp.exists():
                with open(pp, "r", encoding="utf-8") as f:
                    params = yaml.safe_load(f) or {}
            display_name = _infer_display_name(d, d.name)
            versions = [{
                "id": d.name,
                "scope": "complete",
                "name": f"{display_name} v1",
                "label": "v1 基线",
                "params": params,
                "meta": meta,
            }]
            draft_dirs = sorted(
                (p for p in (DRAFT_DIR.iterdir() if DRAFT_DIR.exists() else []) if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
            )
            for draft in draft_dirs:
                if not draft.is_dir() or not (draft / "model.py").exists():
                    continue
                draft_meta = _load_meta(draft)
                if draft_meta.get("parent_strategy") != d.name:
                    continue
                if not _has_report_artifact(draft.name):
                    continue
                if not (draft / "check_report.md").exists() or not (draft / "analysis_template.md").exists():
                    continue
                draft_params = {}
                pp2 = draft / "params.yaml"
                if pp2.exists():
                    with open(pp2, "r", encoding="utf-8") as f:
                        draft_params = yaml.safe_load(f) or {}
                iteration = {}
                iteration_path = draft / "iteration.yaml"
                if iteration_path.exists():
                    with open(iteration_path, "r", encoding="utf-8") as f:
                        iteration = yaml.safe_load(f) or {}
                version_index = len(versions) + 1
                draft_name = _infer_display_name(draft, f"{display_name} v{version_index}")
                versions.append({
                    "id": draft.name,
                    "scope": "draft",
                    "name": draft_name,
                    "label": f"v{version_index} {draft_meta.get('iteration_type') or iteration.get('iteration_type') or '迭代'}",
                    "params": draft_params,
                    "meta": draft_meta,
                    "iteration": iteration,
                })
            result.append({"id": d.name, "name": display_name, "params": params, "meta": meta, "versions": versions})
    return result


def list_draft_strategies():
    _ensure_dirs()
    result = []
    if not DRAFT_DIR.exists():
        return result
    draft_dirs = sorted(
        (p for p in DRAFT_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in draft_dirs:
        meta = _load_meta(d)
        if meta.get("complete") or (COMPLETED_DIR / d.name).exists():
            continue
        if meta.get("parent_strategy"):
            continue
        readme = d / "README.md"
        model = d / "model.py"
        check_report = d / "check_report.md"
        analysis_template = d / "analysis_template.md"
        display_name = _infer_display_name(d, d.name)
        result.append({
            "id": d.name,
            "scope": "draft",
            "name": display_name,
            "meta": meta,
            "has_readme": readme.exists(),
            "has_model": model.exists(),
            "has_check_report": check_report.exists(),
            "has_analysis": analysis_template.exists(),
            "updated_at": d.stat().st_mtime,
        })
    return result


def _slugify(value: str, fallback: str = "iteration") -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or fallback)[:36]


def create_strategy_iteration(
    parent_sid: str,
    iteration_type: str = "free_edit",
    note: str = "",
    name: str = "",
    parent_scope: str = "complete",
) -> dict:
    """Create a draft strategy version from a completed parent version."""
    _ensure_dirs()
    parent_scope = parent_scope if parent_scope in {"complete", "draft"} else get_strategy_scope(parent_sid)
    parent_dir = _strategy_dir_for_scope(parent_sid, parent_scope)
    if parent_scope == "complete":
        if not parent_dir.exists() or not is_complete_strategy(parent_sid):
            raise ValueError(f"Parent strategy is not a completed strategy: {parent_sid}")
    elif not _is_complete_draft_version(parent_sid):
        raise ValueError(f"Parent version is not a completed version: {parent_sid}")

    parent_meta = _load_meta(parent_dir)
    parent_iteration = {}
    parent_iteration_path = parent_dir / "iteration.yaml"
    if parent_iteration_path.exists():
        with open(parent_iteration_path, "r", encoding="utf-8") as f:
            parent_iteration = yaml.safe_load(f) or {}
    root_sid = parent_meta.get("parent_strategy") or parent_iteration.get("parent_strategy") or parent_sid

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_source = note or iteration_type or "iteration"
    draft_sid = f"{root_sid}_iter_{_slugify(suffix_source)}_{timestamp}"
    draft_sid = draft_sid[:80]
    draft_dir = DRAFT_DIR / draft_sid
    counter = 1
    while draft_dir.exists():
        draft_sid = f"{root_sid}_iter_{_slugify(suffix_source)}_{timestamp}_{counter}"
        draft_sid = draft_sid[:80]
        draft_dir = DRAFT_DIR / draft_sid
        counter += 1

    shutil.copytree(parent_dir, draft_dir)
    display_parent = parent_meta.get("name") or parent_sid
    iteration_name = (name or "").strip() or f"{display_parent} 迭代版"
    changed_stages = ["stage_1", "stage_2", "stage_3", "stage_4", "stage_5"]
    if iteration_type == "parameter":
        changed_stages = ["stage_2", "stage_3", "stage_4", "stage_5"]

    iteration = {
        "parent_strategy": root_sid,
        "parent_version_id": parent_sid,
        "parent_version_scope": parent_scope,
        "parent_name": display_parent,
        "parent_version_name": display_parent,
        "iteration_type": iteration_type or "free_edit",
        "change_summary": note or "基于父版本创建迭代版本",
        "changed_stages": changed_stages,
        "hypothesis": note or "验证该改动是否改善父版本的回测证据",
        "success_metrics": [
            "净利润和总收益率较父版本改善",
            "夏普比率不低于父版本",
            "最大回撤不显著恶化",
            "交易次数保持可解释，不因样本过少产生虚假改善",
        ],
        "anti_overfit_checks": [
            "不要只看单品种最优结果",
            "参数改善需要观察稳定区间，而不是单点最优",
            "若改动减少交易次数，必须检查统计显著性",
        ],
        "parent_compare_status": "等待迭代版本完成回测后对比",
        "status": "draft",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(draft_dir / "iteration.yaml", "w", encoding="utf-8") as f:
        yaml.dump(iteration, f, default_flow_style=False, allow_unicode=True)

    meta = dict(parent_meta)
    meta.update({
        "complete": False,
        "example": False,
        "name": iteration_name,
        "parent_strategy": root_sid,
        "parent_version_id": parent_sid,
        "parent_version_scope": parent_scope,
        "iteration_type": iteration["iteration_type"],
        "iteration_note": iteration["change_summary"],
        "completed_stage": 0,
    })
    with open(draft_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, allow_unicode=True)

    # Keep copied artifacts as starting material, but remove downstream conclusions until rerun.
    for stale in ["check_report.md", "analysis_template.md"]:
        path = draft_dir / stale
        if path.exists():
            path.unlink()

    set_active(draft_sid, "draft")
    return {"status": "ok", "strategy": draft_sid, "scope": "draft", "meta": meta, "iteration": iteration}

def get_active_scope() -> str:
    _ensure_dirs()
    if ACTIVE_FILE.exists():
        scope, sid = _parse_active_ref(ACTIVE_FILE.read_text(encoding="utf-8"))
        if sid:
            if scope and _strategy_dir_for_scope(sid, scope).exists():
                return scope
            if (COMPLETED_DIR / sid).exists():
                return "complete"
            if (DRAFT_DIR / sid).exists():
                return "draft"
    return "complete"


def get_active() -> str:
    _ensure_dirs()
    if ACTIVE_FILE.exists():
        scope, sid = _parse_active_ref(ACTIVE_FILE.read_text(encoding="utf-8"))
        if sid:
            if scope and _strategy_dir_for_scope(sid, scope).exists():
                return sid
            if (COMPLETED_DIR / sid).exists() or (DRAFT_DIR / sid).exists():
                return sid
    listed = list_strategies()
    if listed:
        return listed[0]["id"]
    return "locked_vwma_obv"


def set_active(sid: str, scope: Optional[str] = None):
    _ensure_dirs()
    chosen_scope = scope
    if chosen_scope not in {"complete", "draft"}:
        if (COMPLETED_DIR / sid).exists():
            chosen_scope = "complete"
        else:
            chosen_scope = "draft"
    ACTIVE_FILE.write_text(f"{chosen_scope}:{sid}", encoding="utf-8")


def delete_complete_strategy(sid: str):
    _ensure_dirs()
    target = COMPLETED_DIR / sid
    if not sid or not target.exists() or not target.is_dir():
        raise ValueError(f"Completed strategy not found: {sid}")
    shutil.rmtree(target)
    report_dir = ROOT / "reports" / "web"
    if report_dir.exists():
        for path in report_dir.glob(f"{sid}_*"):
            if path.is_file():
                path.unlink()
    current_scope, current_sid = _parse_active_ref(ACTIVE_FILE.read_text(encoding="utf-8") if ACTIVE_FILE.exists() else "")
    if current_sid == sid and current_scope == "complete":
        listed = list_strategies()
        if listed:
            set_active(listed[0]["id"], "complete")
        else:
            ACTIVE_FILE.write_text("", encoding="utf-8")


def load_module(sid: str):
    strategy_dir = get_strategy_dir(sid, get_active_scope() if sid == get_active() else None)
    model_path = strategy_dir / "model.py"
    if not model_path.exists():
        raise ValueError(f"Strategy model not found: {model_path}")
    module_name = f"_strategy_{strategy_dir.parent.name}_{sid}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load strategy module: {model_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    fallback = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if not isinstance(obj, type):
            continue
        has_contract = all(hasattr(obj, method) for method in ("__init__", "compute_indicators", "generate_signals", "run"))
        if "Strategy" in name and has_contract:
            return obj
        if has_contract and fallback is None:
            fallback = obj
    if fallback is not None:
        return fallback
    raise ValueError(f"No strategy class with compute_indicators/generate_signals/run found in {model_path}")


def load_params(sid: str):
    pp = get_strategy_dir(sid, get_active_scope() if sid == get_active() else None) / "params.yaml"
    if pp.exists():
        with open(pp, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_strategy(sid: str, code: str, params: dict):
    _ensure_dirs()
    sd = DRAFT_DIR / sid
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "model.py").write_text(code, encoding="utf-8")
    with open(sd / "params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(params, f, default_flow_style=False, allow_unicode=True)
    return str(sd)


def mark_strategy_complete(sid: str, name: str = None, completed_stage: int = 5, example: bool = False):
    _ensure_dirs()
    source_dir = DRAFT_DIR / sid if (DRAFT_DIR / sid).exists() else COMPLETED_DIR / sid
    if not source_dir.exists():
        raise ValueError(f"Strategy not found: {sid}")
    target_dir = COMPLETED_DIR / sid
    if source_dir != target_dir:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
    existing = _load_meta(target_dir)
    candidate_name = (name or "").strip()
    if not candidate_name or set(candidate_name) == {"?"}:
        candidate_name = _infer_display_name(target_dir, existing.get("name") or sid)
    meta = {
        "complete": True,
        "example": bool(existing.get("example", False) or example),
        "name": candidate_name,
        "completed_stage": completed_stage,
    }
    if existing.get("description"):
        meta["description"] = existing["description"]
    with open(target_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, allow_unicode=True)
    set_active(sid, "complete")
    return meta

