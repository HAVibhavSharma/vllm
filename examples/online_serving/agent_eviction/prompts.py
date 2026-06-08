# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long, stable per-agent system prompts for the eviction demo.

The prompts are intentionally long (~4K tokens each at default
``filler_lines=220``, scale up via ``--filler-lines`` on the runner) so a
prefix-cache miss costs a noticeable prefill and shows up clearly in
TTFT.

Each agent's prompt starts with a unique persona sentinel so APC/LMCache
treat them as distinct cache entries. Filler text is deterministic and
parametrized by ``(agent_id, filler_lines)`` so the *same* agent always
produces the *same* prompt across turns (the whole point -- we want APC
keys to match on re-visit).

The first four agents (``agent_a..agent_d``) have hand-written personas.
Beyond that the rotation is synthesized programmatically so the demo
scales to whatever ``--num-agents`` the runner is asked to use. This is
the knob that matters on a large GPU (e.g. H100 80GB): with only 4 short
prompts the KV cache cannot be pressured, so the demo needs to scale up
the number of distinct prompts in flight before LRU has anything real to
evict.
"""

from __future__ import annotations

# Hand-written personas for the first four agents. New agents beyond
# these reuse synthesized variants of the same shape.
_HAND_WRITTEN_PERSONAS: dict[str, str] = {
    "agent_a": (
        "You are AGENT-A, a careful research assistant who summarizes "
        "complex technical material into short, precise bullets. You "
        "never speculate, always cite the section of the source you "
        "drew from, and prefer plain language over jargon."
    ),
    "agent_b": (
        "You are AGENT-B, a software architect. You critique designs "
        "for correctness, scalability, and operational simplicity. You "
        "respond with a short verdict followed by a numbered list of "
        "risks and trade-offs. You never propose code, only design."
    ),
    "agent_c": (
        "You are AGENT-C, a planner. You translate high-level goals "
        "into ordered, testable steps. Each step you produce includes "
        "a clear owner, a deliverable, and a measurable success "
        "criterion. You never write prose paragraphs."
    ),
    "agent_d": (
        "You are AGENT-D, a customer-facing technical writer. You "
        "explain features and changes in plain language for "
        "non-technical readers, while preserving every load-bearing "
        "detail. You always lead with the user-visible impact."
    ),
}

# Pool of role descriptors used to synthesize personas for agents beyond
# the hand-written four. Kept distinct from the hand-written set so each
# (agent_id, role) pair is stable across runs.
_SYNTH_ROLES: list[str] = [
    "an incident responder who triages alerts and proposes mitigations",
    "a database engineer who reasons about query plans and indexes",
    "a security reviewer who finds and explains vulnerability classes",
    "a release manager who tracks risk against ship dates",
    "a performance engineer who profiles hot paths and proposes fixes",
    "a documentation editor who tightens prose without losing nuance",
    "a test author who writes deterministic, fast unit tests",
    "an API designer who weighs ergonomics against backwards compatibility",
    "a data scientist who turns raw metrics into decision-grade summaries",
    "a build systems engineer who keeps CI fast and deterministic",
    "a UX writer who phrases error states for non-technical readers",
    "a deployment specialist who designs zero-downtime rollouts",
    "a code reviewer who flags subtle correctness bugs",
    "a metrics owner who picks SLIs and SLOs for a service",
    "a cost engineer who attributes cloud spend to features",
    "a privacy reviewer who maps data flows against retention rules",
]


def _agent_id(idx: int) -> str:
    """Return a stable id for the ``idx``-th agent (0-indexed).

    Indices 0-3 map to the hand-written ``agent_a..agent_d``. Higher
    indices produce ``agent_e``, ``agent_f``, ... and wrap past ``z``
    into ``agent_aa``, ``agent_ab``, etc. so we can scale arbitrarily.
    """
    if idx < 0:
        raise ValueError(f"agent index must be >= 0, got {idx}")
    # Build a base-26 letter suffix: 0 -> "a", 25 -> "z", 26 -> "aa".
    letters: list[str] = []
    n = idx
    while True:
        letters.append(chr(ord("a") + (n % 26)))
        n = n // 26 - 1
        if n < 0:
            break
    return "agent_" + "".join(reversed(letters))


def _persona_for(agent_id: str, idx: int) -> str:
    """Return the persona sentence for ``agent_id``."""
    if agent_id in _HAND_WRITTEN_PERSONAS:
        return _HAND_WRITTEN_PERSONAS[agent_id]
    role = _SYNTH_ROLES[(idx - len(_HAND_WRITTEN_PERSONAS)) % len(_SYNTH_ROLES)]
    upper = agent_id.upper().replace("_", "-")
    return (
        f"You are {upper}, {role}. You answer in short, precise "
        "bullets. You never speculate, you always cite the section "
        "of the source you drew from, and you prefer plain language "
        "over jargon."
    )


def build_rotation(num_agents: int) -> list[str]:
    """Return a deterministic list of ``num_agents`` agent ids.

    The first ``min(num_agents, 4)`` are the hand-written
    ``agent_a..agent_d``; the rest are synthesized.
    """
    if num_agents < 2:
        raise ValueError(f"num_agents must be >= 2, got {num_agents}")
    return [_agent_id(i) for i in range(num_agents)]


# A small bank of short user queries. We deliberately rotate through a
# tiny set so the cached *prefix* (system prompt) is the only thing
# growing -- the user message is small and changes each turn.
_GENERIC_USER_QUERIES: list[str] = [
    "Summarize the most important point in one sentence.",
    "List the top two risks worth flagging.",
    "What is the first concrete action to take?",
    "Name one assumption that should be double-checked.",
    "Give a single-sentence headline for this.",
    "What is the smallest measurable milestone?",
    "Which component is the single point of failure?",
    "Suggest one observability metric to add.",
]

# Hand-tuned queries for the four hand-written agents. Synthesized
# agents fall back to ``_GENERIC_USER_QUERIES``.
_HAND_WRITTEN_USER_QUERIES: dict[str, list[str]] = {
    "agent_a": [
        "Summarize the most important point in one sentence.",
        "Give two follow-up questions a reviewer would ask.",
        "Name one assumption worth double-checking.",
    ],
    "agent_b": [
        "List the top two operational risks.",
        "What would you change first if latency doubled?",
        "Which component is the single point of failure?",
    ],
    "agent_c": [
        "Break this down into three steps with owners.",
        "What is the first measurable milestone?",
        "Which step has the most schedule risk?",
    ],
    "agent_d": [
        "Write a one-line release note for end users.",
        "Explain the change in one sentence for a non-engineer.",
        "Suggest a headline for the announcement.",
    ],
}


_FILLER_TEMPLATE = (
    "(agent={agent} filler line {i}: maintain prior conventions, "
    "observe the established style, avoid speculation, defer to the "
    "system designer's choices, keep responses tightly scoped, and "
    "respect the project's operational constraints at all times.)"
)


def build_system_prompt(agent_id: str, filler_lines: int = 220) -> str:
    """Return the long, stable system prompt for ``agent_id``.

    Each filler line tokenizes to ~33 tokens on Qwen tokenizers, so
    ``filler_lines=220`` lands at roughly 7K tokens, and ``500`` at
    ~16K tokens. Bump it for larger GPUs that can absorb more pressure;
    decrease it on small models or tight ``--max-model-len`` settings.
    Make sure the server's ``--max-model-len`` leaves room for the
    prompt + ``--max-output-tokens`` (e.g. 32K for ``--filler-lines 500``).
    """
    # Recover the index so synthesized personas stay stable.
    idx = _index_for(agent_id)
    header = _persona_for(agent_id, idx)
    filler = " ".join(
        _FILLER_TEMPLATE.format(agent=agent_id, i=i) for i in range(filler_lines)
    )
    return f"{header} {filler}"


def pick_user_query(agent_id: str, turn_idx: int) -> str:
    """Deterministically rotate through the agent's user queries."""
    queries = _HAND_WRITTEN_USER_QUERIES.get(agent_id, _GENERIC_USER_QUERIES)
    return queries[turn_idx % len(queries)]


def _index_for(agent_id: str) -> int:
    """Inverse of ``_agent_id``: parse the letter suffix back to an int."""
    if not agent_id.startswith("agent_"):
        raise KeyError(f"unknown agent_id={agent_id!r}")
    suffix = agent_id[len("agent_"):]
    if not suffix or not suffix.isalpha() or not suffix.islower():
        raise KeyError(f"unknown agent_id={agent_id!r}")
    n = 0
    for i, ch in enumerate(suffix):
        digit = ord(ch) - ord("a")
        if i == 0:
            n = digit
        else:
            n = (n + 1) * 26 + digit
    return n


# Backwards-compatible default for code that imported the old constant.
# Equivalent to ``build_rotation(4)``.
AGENT_IDS: list[str] = build_rotation(4)
