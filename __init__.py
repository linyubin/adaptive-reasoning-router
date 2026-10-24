"""adaptive-reasoning-router — per-turn reasoning-effort routing by question difficulty.

Simple questions ("现在几点", "帮我记住…") don't need deep thinking; complex ones
do. This plugin rewrites the outbound LLM request's reasoning_effort per turn
using zero-latency heuristics (no extra LLM calls):

- **light** — greetings, acks, time/date/weather one-shots, "remember X", bare
  arithmetic → rewrite ``reasoning_effort`` to ``ARR_LIGHT_EFFORT`` (default low)
- **medium** — short factual questions ("还有多少空间", "X是多少") → ``ARR_MEDIUM_EFFORT``
  (default medium)
- **complex / unrecognized** — analysis/code/multi-step signals, long messages →
  untouched (keeps the configured effort)

Middleware contract: ``llm_request`` callbacks receive the assembled provider
kwargs and may return ``{"request": {...}}`` to replace them. This is the
official plugin surface (hermes_cli/middleware.py, applied in
agent/turn_api_request.py AFTER the provider profile built the wire fields), so
the rewrite lands on the exact keys the provider emits:

- chat_completions (zai/GLM): top-level ``reasoning_effort`` +
  ``extra_body.thinking``
- OpenAI Responses: ``reasoning.effort``
- relay shapes: ``extra_body.reasoning.effort`` / ``extra_body.reasoning_effort``

Core increments ``api_call_count`` at the START of every iteration
(agent/turn_iteration_prep.py::begin_iteration, line ~402) BEFORE the request is
built, so the turn's first LLM call arrives here with ``api_call_count == 1``
and tool-loop follow-ups carry >= 2. Only ``api_call_count == 1`` is rewritten:
tool-loop turns keep the session's configured effort, so mid-turn reasoning
continuity and prompt-cache keys are preserved.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PLUGIN_NAME = "adaptive-reasoning-router"


def _env_int(name: str, default: int) -> int:
    """Env override with a safe fallback: a bad value logs a warning instead of
    failing the plugin load (import-time crashes disable the whole plugin)."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("[%s] bad %s=%r, using default %d", PLUGIN_NAME, name, raw, default)
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return str(raw).strip() if raw and str(raw).strip() else default


# Effort levels substituted for light turns (clamped to the model's vocabulary at wire time
# by the provider profile — e.g. zai GLM-5.3 accepts low/medium/high/max).
LIGHT_EFFORT = _env_str("ARR_LIGHT_EFFORT", "low")
MEDIUM_EFFORT = _env_str("ARR_MEDIUM_EFFORT", "medium")

# Turn only rewrites when the turn's user message matches these size bounds;
# very long messages are assumed complex.
_LIGHT_MAX_CHARS = _env_int("ARR_LIGHT_MAX_CHARS", 120)
_MEDIUM_MAX_CHARS = _env_int("ARR_MEDIUM_MAX_CHARS", 300)

# Message shapes that are ALWAYS light regardless of content heuristics.
_SIMPLE_PATTERNS = [
    # greetings / thanks / acks (repeatable tokens: "好的收到", "ok ok", "thanks!")
    r"^((你好|您好|hi|hello|hey|yo|在吗|谢谢|多谢|感谢|辛苦了|好的|收到|明白|了解|got\s*it|roger|thanks|thank\s*you|thx|ty|ok|okay|kk|k|嗯|哦|哈哈|👍|🙏|👌|😅|smile|great|perfect|nice|cool|done|alright|sure|will\s*do|noted)[\s!！.。~～，,*]*)+$",
    # time / date / weather one-shots (Chinese)
    r"^(现在)?(几点|什么时间|今天(几号|日期|星期)|明天(几号|星期)|现在日期)",
    r"^(查|看)一下?(天气|今天天气)",
    r"^(今天|明天|现在|周末)?(天气|气温|温度)(怎么样|如何|怎样)?",
    # time / date / weather one-shots (English)
    r"^(what('?s| is)?\s+(the\s+)?(time|date|day|weather)|is\s+it\s+(raining|sunny)|how('?s| is)\s+the\s+weather)\b",
    # bare arithmetic (must contain an operator; a bare "2" is not arithmetic)
    r"^(?=[0-9+\-*/×÷=.()（）\s]*[+\-*/×÷=])[0-9+\-*/×÷=.()（）\s]+[=?？]?$",
    # "help me remember/note X" — storage, not reasoning (Chinese)
    r"^(帮我?|请)?(记住|记一下|存一下|记个|备注)[^?？]{0,60}$",
    # "remember/note X" — storage, not reasoning (English)
    r"^(please\s+)?(remember|note|save|jot\s+down|make\s+a\s+note(?!.*analy))\b[^?？]{0,80}$",
    # yes/no confirmations (Chinese)
    r"^(要不要|是不是|行不行|可不可以|对不对)[^?？]{0,40}[?？]?$",
    # yes/no confirmations (English)
    r"^(is|are|do|does|did|can|could|should|will|would)\s+(there\s+|it\s+|that\s+|we\s+|you\s+|this\s+)?\S{1,30}\s*\?\s*$",
]

# Strong signals of a complex turn (never downgraded even if short).
_COMPLEX_PATTERNS = [
    # analysis / engineering verbs
    r"(分析|对比|评估|规划|设计|实现|调试|排查|优化|重构|审阅|review|refactor|debug|analy[sz]e|evaluat|investigat|troubleshoot|optimi[sz]e|architect)",
    # causal / explanatory questions
    r"(为什么|为何|怎么才能|如何|原理|机制|架构|原因|导致|why\s|how\s+(do|does|did|can|could|should)|what\s+caus)",
    # build/write requests
    r"(写(一?个|个)|实现(一?个)?|开发|编写|write\s+(a|an|the|some)|implement|build\s+(a|an|the)|creat(e|ing)\s+(a|an)\s+(new\s+)?(function|class|script|tool|plugin|service))",
    # plans / strategies
    r"(步骤|方案|策略|计划|路线|roadmap|plan\s+(for|to)|strategy|step[- ]by[- ]step)",
    # multi-step instructions ("X; then Y")
    r"[;；]\s*(然后|再|接着|并且|then|after\s+that|next,)",
    # technical artifacts
    r"(代码|script|脚本|api|sql|正则|regex|code\b|function\s|class\s|```)",
]

_SIMPLE_RE = [re.compile(p, re.IGNORECASE) for p in _SIMPLE_PATTERNS]
_COMPLEX_RE = [re.compile(p, re.IGNORECASE) for p in _COMPLEX_PATTERNS]

# Factual-lookup question words → medium (never light).
_LOOKUP_ZH = re.compile(r"(多少|多大|多长|哪些|哪个|哪家|几号|几点|何时|在哪|是什么|是什么意思)")
_LOOKUP_EN = re.compile(r"\b(how\s+many|how\s+much|how\s+long|what\s+is|what('s| is)|when\s+is|where\s+is|which\s+)\b", re.IGNORECASE)
_QWORD_START_ZH = ("什么", "几", "哪", "是不是", "多少")


def _first_user_message(request: Dict[str, Any]) -> Optional[str]:
    """The last user-role message in the outbound payload."""
    msgs = request.get("messages")
    if not isinstance(msgs, list):
        return None
    for msg in reversed(msgs):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):  # multimodal parts
                texts = [p.get("text", "") for p in content
                         if isinstance(p, dict) and isinstance(p.get("text"), str)]
                joined = "\n".join(t for t in texts if t.strip())
                if joined.strip():
                    return joined
            return None
    return None


def _classify(text: str) -> Optional[str]:
    """Return 'light' | 'medium' | None (None = keep configured effort)."""
    if not text:
        return None
    stripped = text.strip()
    if len(stripped) > _MEDIUM_MAX_CHARS:
        return None  # long messages: keep configured effort
    if any(rx.search(stripped) for rx in _COMPLEX_RE):
        return None
    for rx in _SIMPLE_RE:
        if rx.match(stripped):
            return "light"
    if len(stripped) <= _LIGHT_MAX_CHARS and not any(
        ch in stripped for ch in ("{", "}", "```", "|", "=>")
    ):
        # Factual lookup phrasing (contains a question word) → medium; plain
        # conversational question → medium. Anything else stays at configured effort.
        if _LOOKUP_ZH.search(stripped) or _LOOKUP_EN.search(stripped) or \
                stripped.endswith(("?", "？")) or any(
            stripped.startswith(p) for p in _QWORD_START_ZH
        ):
            return "medium"
    return None


_EFFORT_MAP = {"light": LIGHT_EFFORT, "medium": MEDIUM_EFFORT}

#: api_call_count of the turn's FIRST LLM call. Core increments the counter in
#: begin_iteration() before the request is built, so the first call is 1 and
#: tool-loop follow-ups start at 2.
FIRST_CALL_COUNT = 1


def _on_llm_request(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """llm_request middleware: rewrite reasoning effort for light main-loop turns."""
    request = kwargs.get("request")
    if not isinstance(request, dict):
        return None
    # Turn's first LLM call only (== 1; tool-loop follow-ups carry >= 2 and must
    # keep continuity — see module docstring for the increment site).
    try:
        count = int(kwargs.get("api_call_count") or 0)
    except (TypeError, ValueError):
        count = 0
    if count != FIRST_CALL_COUNT:
        return None

    user_text = _first_user_message(request)
    verdict = _classify(user_text) if user_text else None
    if verdict is None:
        return None

    target = _EFFORT_MAP.get(verdict)
    if not target:
        return None

    rewritten = dict(request)
    changed = False

    # GLM / OpenAI-compatible chat_completions: top-level reasoning_effort.
    if "reasoning_effort" in rewritten and rewritten["reasoning_effort"] != target:
        rewritten["reasoning_effort"] = target
        changed = True
    # OpenAI Responses reasoning object form.
    if isinstance(rewritten.get("reasoning"), dict) and \
            isinstance(rewritten["reasoning"].get("effort"), str) and \
            rewritten["reasoning"]["effort"] != target:
        rewritten["reasoning"] = {**rewritten["reasoning"], "effort": target}
        changed = True
    # extra_body variants (zai relays, OpenRouter shapes).
    if isinstance(rewritten.get("extra_body"), dict):
        eb = dict(rewritten["extra_body"])
        eb_changed = False
        if isinstance(eb.get("reasoning"), dict) and \
                isinstance(eb["reasoning"].get("effort"), str) and \
                eb["reasoning"]["effort"] != target:
            eb["reasoning"] = {**eb["reasoning"], "effort": target}
            eb_changed = True
        if "reasoning_effort" in eb and eb["reasoning_effort"] != target:
            eb["reasoning_effort"] = target
            eb_changed = True
        if eb_changed:
            rewritten["extra_body"] = eb
            changed = True

    # GLM 5.x thinking stays enabled (it rejects disabled thinking); the effort
    # knob alone carries the depth signal. Nothing to change there.

    if not changed:
        return None
    logger.info("[%s] turn classified %s -> reasoning_effort=%s (was %r)",
                PLUGIN_NAME, verdict, target,
                request.get("reasoning_effort") or
                (request.get("reasoning") or {}).get("effort"))
    return {"request": rewritten, "source": PLUGIN_NAME, "reason": f"difficulty={verdict}"}


def register(ctx) -> None:
    ctx.register_middleware("llm_request", _on_llm_request)
    logger.info("[%s] registered llm_request middleware (light=%s medium=%s, first-call=%d)",
                PLUGIN_NAME, LIGHT_EFFORT, MEDIUM_EFFORT, FIRST_CALL_COUNT)
