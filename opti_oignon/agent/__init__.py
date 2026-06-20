#!/usr/bin/env python3
"""opti_oignon.agent -- the sandboxed agent loop (S175, Theme 3 / Odysseus Core).

A thin facade over the agent package built in S175:

- ``loop``             -- the multi-turn streaming loop with a bounded verifier.
- ``dispatch``         -- dual tool dispatch (native vs parser) and the sandbox
                          dispatch invariant (S73/S74 bwrap, never the host).
- ``tool_parsing``     -- fenced / bracketed / XML tool-block parsing.
- ``allowlists``       -- per-mode (Daily / Bulbe) gating and the Bulbe approval
                          seam.
- ``untrusted_context`` -- the Odysseus prompt-security pattern: external
                          content wrapped as untrusted user-role data.
- ``tools``            -- the concrete sandboxed tool set plus the web-search
                          and memory handlers, with per-mode schemas (S176).
- ``teacher``          -- student-to-teacher escalation with an authoritative
                          SKILL.md draft and the human-approval gate hook (S176).
- ``config_loader``    -- the guarded loader for ``config.yaml`` (S176).
- ``skills``           -- the on-disk SKILL.md registry plus the approval-gated
                          ``manage_skills`` tool and the teacher-draft publish
                          path (S177).
"""

from __future__ import annotations

# Module conventions (Theme 3).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

from .allowlists import (
    BULBE_ALLOWLIST,
    DAILY_ALLOWLIST,
    SANDBOX_TOOL_NAMES,
    SESSION_STATE_TOOLS,
    SUBAGENT_TOOLS,
    GateDecision,
    current_mode,
    evaluate,
    is_sandbox_tool,
    is_tool_allowed,
    request_approval,
    requires_approval,
)
from .dispatch import (
    DispatchResult,
    ToolCall,
    dispatch_round,
    dispatch_tool_call,
    extract_native_calls,
    extract_text,
    resolve_tool_calls,
    sandbox_ready,
)
from .loop import (
    AGENT_OBS_MAX_BYTES,
    AGENT_OBS_MAX_LINES,
    AGENT_ROUND_OBS_BUDGET,
    DOOM_LOOP_THRESHOLD,
    HARDENING_DEFAULTS,
    MAX_AGENT_ROUNDS,
    MAX_STEPS_REMINDER,
    PRUNE_PROTECT_ROUNDS,
    PRUNE_TARGET_CHARS,
    PRUNE_TRIGGER_CHARS,
    STOP_DOOM_LOOP,
    TASK_CHILD_CAP,
    AgentEvent,
    AgentRunResult,
    VerifierResult,
    load_hardening_config,
    run,
)
from .tool_parsing import (
    ParsedToolCall,
    parse_tool_blocks,
)
from .untrusted_context import (
    UNTRUSTED_POLICY,
    memory_untrusted_message,
    untrusted_message,
    untrusted_message_many,
    wrap,
)
from .tools import (
    ALL_SCHEMAS,
    HANDLER_TOOL_NAMES,
    ToolRegistry,
    ToolSchema,
    ToolSet,
    build_tool_set,
    get_tool_registry,
    make_todo_handler,
    native_tools_for,
    reset_tool_registry,
    system_prompt_section_for,
)
from .teacher import (
    SOURCE_TEACHER,
    EscalationDecision,
    EscalationPolicy,
    EscalationResult,
    TeacherEscalator,
    TeacherSkillDraft,
    escalate,
    request_skill_approval,
    should_escalate,
)
from .config_loader import (
    AgentConfig,
    available_presets,
    get_agent_config,
    load_config,
    reset_agent_config,
)
from .skills import (
    SkillConsultation,
    SkillPublishResult,
    SkillRegistry,
    Skill,
    VerificationResult,
    consult_skills,
    get_skill_registry,
    make_manage_skills_handler,
    publish_teacher_draft,
    reset_skill_registry,
    sandbox_test_verification,
    set_skill_registry,
)

__all__ = [
    "checkpoint_before_apply",
    "FEATURE_AVAILABLE",
    # loop
    "run",
    "MAX_AGENT_ROUNDS",
    "TASK_CHILD_CAP",
    "AgentRunResult",
    "AgentEvent",
    "VerifierResult",
    # loop hardening (S229, AGT_SPEC Section 6)
    "STOP_DOOM_LOOP",
    "MAX_STEPS_REMINDER",
    "AGENT_OBS_MAX_BYTES",
    "AGENT_OBS_MAX_LINES",
    "AGENT_ROUND_OBS_BUDGET",
    "PRUNE_TRIGGER_CHARS",
    "PRUNE_TARGET_CHARS",
    "PRUNE_PROTECT_ROUNDS",
    "DOOM_LOOP_THRESHOLD",
    "HARDENING_DEFAULTS",
    "load_hardening_config",
    # dispatch
    "ToolCall",
    "DispatchResult",
    "resolve_tool_calls",
    "dispatch_tool_call",
    "dispatch_round",
    "sandbox_ready",
    "extract_native_calls",
    "extract_text",
    # tool parsing
    "parse_tool_blocks",
    "ParsedToolCall",
    # allowlists
    "DAILY_ALLOWLIST",
    "BULBE_ALLOWLIST",
    "SANDBOX_TOOL_NAMES",
    "SESSION_STATE_TOOLS",
    "SUBAGENT_TOOLS",
    "GateDecision",
    "evaluate",
    "is_tool_allowed",
    "is_sandbox_tool",
    "requires_approval",
    "request_approval",
    "current_mode",
    # untrusted context
    "untrusted_message",
    "untrusted_message_many",
    "wrap",
    "memory_untrusted_message",
    "UNTRUSTED_POLICY",
    # tools (S176)
    "ToolSchema",
    "ToolSet",
    "ToolRegistry",
    "ALL_SCHEMAS",
    "HANDLER_TOOL_NAMES",
    "build_tool_set",
    "make_todo_handler",
    "native_tools_for",
    "system_prompt_section_for",
    "get_tool_registry",
    "reset_tool_registry",
    # teacher (S176)
    "EscalationPolicy",
    "EscalationDecision",
    "EscalationResult",
    "TeacherSkillDraft",
    "TeacherEscalator",
    "should_escalate",
    "escalate",
    "request_skill_approval",
    "SOURCE_TEACHER",
    # config (S176)
    "AgentConfig",
    "load_config",
    "available_presets",
    "get_agent_config",
    "reset_agent_config",
    # skills (S177)
    "SkillRegistry",
    "Skill",
    "VerificationResult",
    "SkillPublishResult",
    "SkillConsultation",
    "get_skill_registry",
    "set_skill_registry",
    "reset_skill_registry",
    "make_manage_skills_handler",
    "publish_teacher_draft",
    "sandbox_test_verification",
    "consult_skills",
]
