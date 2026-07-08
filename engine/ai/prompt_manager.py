"""提示词管理器 — 加载模板、注入变量、组装Prompt"""
from pathlib import Path
import re
import yaml
from typing import Optional


class PromptManager:
    """管理所有阶段的提示词模板"""

    def __init__(self):
        self.templates_dir = Path(__file__).parent / "prompts"
        self.knowledge_dir = Path(__file__).parent.parent.parent / "data" / "knowledge"
        self._cache = {}
        self._knowledge_cache = None

    def _load_template(self, template_name: str) -> dict:
        if template_name in self._cache:
            return self._cache[template_name]
        path = self.templates_dir / f"{template_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            template = yaml.safe_load(f)
        self._cache[template_name] = template
        return template

    def get_system_prompt(self, variables: Optional[dict] = None) -> str:
        t = self._load_template("system")
        lines = [t.get("role", "")]
        context = t.get("context", {})
        if context:
            lines.append(f"\n平台：{context.get('platform', '')}")
            lines.append(f"引擎：{context.get('engine', '')}")
        reqs = t.get("requirements", [])
        if reqs:
            lines.append("\n要求：")
            for r in reqs:
                lines.append(f"- {r}")
        if variables:
            lines.append("\n当前上下文：")
            for k, v in variables.items():
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _format_value(self, value):
        """格式化模板值——字符串直接返回，结构化数据才dump"""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "\n".join([str(v) for v in value])
        return yaml.dump(value, default_flow_style=False, allow_unicode=True)

    def _flatten_text(self, value) -> str:
        if isinstance(value, dict):
            return "\n".join([f"{k}: {self._flatten_text(v)}" for k, v in value.items()])
        if isinstance(value, list):
            return "\n".join([self._flatten_text(v) for v in value])
        return str(value)

    def _load_knowledge_base(self) -> list:
        if self._knowledge_cache is not None:
            return self._knowledge_cache
        docs = []
        if not self.knowledge_dir.exists():
            self._knowledge_cache = docs
            return docs
        for path in self.knowledge_dir.rglob("*"):
            if not path.is_file() or path.name.startswith("_"):
                continue
            if path.suffix.lower() not in {".yaml", ".yml", ".md", ".txt"}:
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except Exception:
                continue
            parsed = None
            if path.suffix.lower() in {".yaml", ".yml"}:
                try:
                    parsed = yaml.safe_load(raw)
                except Exception:
                    parsed = None
            text = self._flatten_text(parsed) if parsed is not None else raw
            title = path.stem
            if isinstance(parsed, dict):
                title = (
                    parsed.get("strategy_name")
                    or parsed.get("factor_name")
                    or parsed.get("名称")
                    or title
                )
            docs.append({
                "path": str(path.relative_to(self.knowledge_dir.parent.parent)),
                "title": str(title),
                "text": text[:4000],
            })
        self._knowledge_cache = docs
        return docs

    def _query_terms(self, query: str) -> list:
        synonyms = {
            "螺纹钢": ["螺纹钢", "RB", "RB0"],
            "热卷": ["热卷", "HC", "HC0"],
            "铁矿石": ["铁矿石", "I", "I0"],
            "豆粕": ["豆粕", "M", "M0"],
            "均线": ["均线", "MA", "EMA", "EXPMA", "SMA"],
            "成交量": ["成交量", "volume", "OBV", "量能"],
            "趋势": ["趋势", "trend", "动量", "momentum"],
            "突破": ["突破", "breakout", "N周期高点", "N周期低点"],
            "止损": ["止损", "stop", "ATR", "移动止损"],
        }
        terms = set()
        text = query or ""
        for token in re.split(r"[\s,，。；;：:、/\\|()\[\]{}<>《》\"'`]+", text):
            token = token.strip()
            if len(token) >= 2:
                terms.add(token)
        for key, values in synonyms.items():
            group = [key] + values
            if any(v and v.lower() in text.lower() for v in group):
                terms.update(values)
        return list(terms)[:40]

    def search_knowledge(self, query: str, top_k: int = 4) -> str:
        terms = self._query_terms(query)
        if not terms:
            return ""
        scored = []
        for doc in self._load_knowledge_base():
            haystack = f"{doc['title']}\n{doc['text']}".lower()
            score = 0
            for term in terms:
                t = term.lower()
                if not t:
                    continue
                if t in haystack:
                    score += 3 if t in doc["title"].lower() else 1
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return ""
        blocks = []
        for score, doc in scored[:top_k]:
            text = doc["text"].strip().replace("\r\n", "\n")
            if len(text) > 1200:
                text = text[:1200].rstrip() + "\n..."
            blocks.append(f"来源：{doc['path']} | {doc['title']} | score={score}\n{text}")
        return "\n\n---\n\n".join(blocks)

    def get_stage_prompt(self, template_name: str, variables: Optional[dict] = None) -> str:
        t = self._load_template(template_name)
        lines = []

        # Task
        task = t.get("task", "")
        if task:
            lines.append(f"## {task.strip()}\n")

        # Context variables
        if variables:
            lines.append("上下文信息：")
            for k, v in variables.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        # Behavior
        behavior = t.get("behavior", [])
        if behavior:
            lines.append("行为准则：")
            for b in behavior:
                lines.append(f"- {b}")
            lines.append("")

        # Output instruction - use raw text, not YAML-dumped
        output_instruction = t.get("output_instruction", "")
        if output_instruction:
            lines.append(output_instruction.strip())
            lines.append("")

        # Other fields (not task, behavior, output_instruction)
        skip_keys = {"task", "behavior", "output_instruction"}
        for key, value in t.items():
            if key in skip_keys:
                continue
            formatted = self._format_value(value)
            lines.append(formatted)

        query = ""
        if variables:
            query = "\n".join([
                str(variables.get("recent_user_input", "")),
                str(variables.get("用户最近输入", "")),
                self._flatten_text(variables.get("strategy_spec", "")),
                str(variables.get("阶段1策略说明", "")),
            ])
        if template_name in {"stage_1_conceive", "stage_2_model"}:
            knowledge = self.search_knowledge(query)
            if knowledge:
                lines.append("\n## 本地知识库检索结果（RAG，只能作为参考，不能替代用户策略原文）")
                lines.append(knowledge)

        return "\n".join(lines)

    def build_messages(self, template_name: str, user_input: str, variables: Optional[dict] = None) -> list:
        system = self.get_system_prompt(variables)
        stage = self.get_stage_prompt(template_name, variables)
        final_user = f"{stage}\n\n用户输入：{user_input}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": final_user}
        ]
