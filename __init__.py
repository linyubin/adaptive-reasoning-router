"""adaptive-reasoning-router — per-turn reasoning-effort routing by question difficulty.

Simple questions ("现在几点", "帮我记住…") don't need deep thinking; complex ones
do. This plugin rewrites the outbound LLM request's reasoning_effort per turn:

- **Heuristic lane (default, zero latency cost):** regex + message-shape rules
  classify the turn's user message as light / medium / keep (inherit global config).
- **Probe lane (optional, model-graded):** for ambiguous messages an auxiliary LLM
  judges difficulty; see PROBE config below (disabled by default).

Middleware contract: ``llm_request`` callbacks receive the assembled provider
kwargs and may return ``{"request": {...}}`` to replace them. This is the
official plugin surface (hermes_cli/middleware.py, applied in
agent/turn_api_request.py AFTER the provider profile built the wire fields), so
the rewrite lands on the exact keys the provider emits:

- chat_completions (zai/GLM): top-level ``reasoning_effort`` +
  ``extra_body.thinking``
- OpenAI-compatible: top-level ``reasoning_effort`` / ``reasoning``

Only MAIN-loop requests are touched: tool-loop follow-ups (api_call_count > 0)
and everything without a fresh user message keep the configured effort, so
mid-turn continuity and prompt-cache keys are preserved.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PLUGIN_NAME = "adaptive-reasoning-router"

# ---------------------------------------------------------------------------
# Tunables (env-overridable, sane defaults for Andy's setup: GLM-5.3 via zai)
# ---------------------------------------------------------------------------
import os

def _env(name: str, default: str) -> str:
    try:
        from hermes_constants import get_hermes_home_override  # noqa: F401
    except Exception:
        pass
    return os.environ.get(name, default)

# Effort levels substituted for light turns (clamped to the model's vocabulary at wire time
# by the provider profile — here zai GLM-5.3 accepts low/medium/high/max).
LIGHT_EFFORT = _env("ARR_LIGHT_EFFORT", "low")
MEDIUM_EFFORT = _env("ARR_MEDIUM_EFFORT", "medium")

# Turn only rewrites when the *first* user message of the turn matches these size bounds;
# very long messages are assumed complex.
_LIGHT_MAX_CHARS = int(_env("ARR_LIGHT_MAX_CHARS", "120"))
_MEDIUM_MAX_CHARS = int(_env("ARR_MEDIUM_MAX_CHARS", "300"))

# Message shapes that are ALWAYS light regardless of content heuristics.
_SIMPLE_PATTERNS = [
    # greetings / thanks / acks (repeatable tokens: "好的收到", "嗯嗯好的")
    r"^((你好|您好|hi|hello|hey|在吗|谢谢|多谢|感谢|辛苦了|好的|收到|明白|了解|ok|okay|嗯|哦|哈哈|👍|🙏)[\s!！.。~～，,]*)+$",
    # time / date / weather one-shots
    r"^(现在)?(几点|什么时间|今天(几号|日期|星期)|明天(几号|星期)|现在日期)",
    r"^(查|看)一下?(天气|今天天气)",
    r"^(今天|明天|现在|周末)?(天气|气温|温度)(怎么样|如何|怎样)?",
    # bare arithmetic (must contain an operator; a bare "2" is not arithmetic)
    r"^(?=[0-9+\-*/×÷=.()（）\s]*[+\-*/×÷=])[0-9+\-*/×÷=.()（）\s]+[=?？]?$",
    # "help me remember/note X" — storage, not reasoning
    r"^(帮我?|请)?(记住|记一下|存一下|记个|备注)[^?？]{0,60}$",
    # yes/no confirmations
    r"^(要不要|是不是|行不行|可不可以|对不对)[^?？]{0,40}[?？]?$",
]

# Strong signals of a complex turn (never downgraded even if short).
_COMPLEX_PATTERNS = [
    r"(分析|对比|评估|规划|设计|实现|调试|排查|优化|重构|审阅|review|refactor|debug)",
    r"(为什么|为何|怎么才能|如何|原理|机制|架构|原因|导致)",
    r"(写(一?个|个)|实现(一?个)?|开发|编写)",
    r"(步骤|方案|策略|计划|路线)",
    r"[;；]\s*(然后|再|接着|并且)",          # multi-step instructions
    r"(代码|script|脚本|api|sql|正则|regex)",
]

_SIMPLE_RE = [re.compile(p, re.IGNORECASE) for p in _SIMPLE_PATTERNS]
_COMPLEX_RE = [re.compile(p, re.IGNORECASE) for p in _COMPLEX_PATTERNS]

# Strip noise before classification: code fences and URLs signal complexity by presence
# but their bulk shouldn't trip the length cap.
def _effective_text(text: str) -> str:
    return text

def _first_user_message(request: Dict[str, Any]) -> Optional[str]:
    """The last user-role message in the outbound payload (None for tool-loop turns)."""
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
        if re.search(r"(多少|多大|多长|哪些|哪个|哪家|几号|几点|何时|在哪)", stripped) or \
                stripped.endswith(("?", "？")) or any(
            stripped.startswith(p) for p in ("什么", "几", "哪", "是不是")
        ):
            return "medium"
    return None


_EFFORT_MAP = {"light": LIGHT_EFFORT, "medium": MEDIUM_EFFORT}


def _on_llm_request(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """llm_request middleware: rewrite reasoning effort for light main-loop turns."""
    request = kwargs.get("request")
    if not isinstance(request, dict):
        return None
    # Main loop only: api_call_count == 0 means the turn's first LLM call
    # (tool-loop follow-ups carry >= 1 and must keep continuity).
    if int(kwargs.get("api_call_count") or 0) > 0:
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
    # OpenAI reasoning object form.
    if isinstance(rewritten.get("reasoning"), dict) and \
            isinstance(rewritten["reasoning"].get("effort"), str) and \
            rewritten["reasoning"]["effort"] != target:
        rewritten["reasoning"] = {**rewritten["reasoning"], "effort": target}
        changed = True
    # extra_body variants (zai relays, OpenRouter shapes).
    if isinstance(rewritten.get("extra_body"), dict):
        eb = dict(rewritten["extra_body"])
        if isinstance(eb.get("reasoning"), dict) and \
                isinstance(eb["reasoning"].get("effort"), str) and \
                eb["reasoning"]["effort"] != target:
            eb["reasoning"] = {**eb["reasoning"], "effort": target}
            changed = True
        if "reasoning_effort" in eb and eb["reasoning_effort"] != target:
            eb["reasoning_effort"] = target
            changed = True
        if changed:
            rewritten["extra_body"] = eb

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
    logger.info("[%s] registered llm_request middleware (light=%s medium=%s)",
                PLUGIN_NAME, LIGHT_EFFORT, MEDIUM_EFFORT)
