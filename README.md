# adaptive-reasoning-router

A Hermes Agent plugin that routes reasoning depth per turn: simple questions
get fast, low-effort responses; complex ones keep your configured reasoning
effort. Zero added latency (pure heuristic classification, no extra LLM calls).

## The problem

Hermes has a single global `agent.reasoning_effort`. Set it high and every
"现在几点" (what time is it?) burns deep-thinking tokens and latency; set it
low and hard problems get shallow answers. There is no built-in
difficulty-based routing (`smart_model_routing` in the source is an unused
placeholder; `fallback_providers` only reacts to errors, not difficulty).

## How it works

The plugin registers an `llm_request` middleware — the official
behavior-changing surface that rewrites the outbound provider payload after
Hermes assembles it and before it hits the wire. Per turn:

1. Extracts the turn's latest user message.
2. Classifies it with regex heuristics (Chinese + English patterns):
   - **light** — greetings, acks, time/date/weather lookups, "remember X",
     bare arithmetic → rewrite `reasoning_effort` to `low`
   - **medium** — short factual questions ("还有多少空间", "X是多少") → `medium`
   - **complex** (or unrecognized) — analysis/code/multi-step signals,
     long messages → leave untouched (keeps your configured effort)
3. Rewrites every effort field the route actually emits: top-level
   `reasoning_effort` (GLM / OpenAI-compatible chat_completions),
   `reasoning.effort` (OpenAI Responses), and `extra_body.reasoning.effort`
   (relay shapes).

### Safety boundaries

- **Main-loop first call only** (`api_call_count == 0`): tool-loop follow-ups
  keep the session's configured effort, so mid-turn reasoning continuity and
  prompt-cache keys are preserved.
- **The original request dict is never mutated** — the middleware returns a
  rewritten copy, matching the Hermes middleware contract.
- **GLM-5.x thinking stays enabled** — GLM rejects disabled thinking; the
  plugin only moves the effort knob (levels clamp at the provider anyway).
- **Fail-open**: any classification error leaves the request untouched.

## Install

```bash
hermes plugins install https://github.com/linyubin/adaptive-reasoning-router
hermes plugins enable adaptive-reasoning-router
```

Then restart the gateway (or start a new CLI session).

Requires `agent.reasoning_effort` to be set in your config (the plugin only
downgrades from whatever you configured; it never upgrades).

## Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `ARR_LIGHT_EFFORT` | `low` | Effort substituted for light turns |
| `ARR_MEDIUM_EFFORT` | `medium` | Effort substituted for medium turns |
| `ARR_LIGHT_MAX_CHARS` | `120` | Max message length eligible for light |
| `ARR_MEDIUM_MAX_CHARS` | `300` | Messages longer than this are never downgraded |

To tune the patterns, edit `_SIMPLE_PATTERNS` / `_COMPLEX_PATTERNS` in
`__init__.py` (delete `__pycache__` and restart afterwards).

## Verification

The plugin logs every rewrite at INFO:

```
grep adaptive-reasoning-router ~/.hermes/logs/gateway.log
```

## Disclosure

- No network calls, no telemetry, no background processes, no stored
  credentials. Pure in-process string matching + payload rewrite.
- Only public Hermes surfaces: `ctx.register_middleware("llm_request", ...)`.
- Limitation: heuristic classification is conservative — short ambiguous
  messages ("帮我看看这个") stay at the configured effort rather than being
  downgraded.

## License

MIT
