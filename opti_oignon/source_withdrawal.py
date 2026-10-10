#!/usr/bin/env python3
"""The user's withdrawal of a source: every turn it reached is lowered, wherever it is kept.

A source is named as a lineage entry -- ``kind:identifier``: the digest of a
document's text, the digest of a page's address, the id of a fact of memory.
Withdrawing it adds the withdrawn kind to the context of every turn whose
lineage holds it, in the conversations and in their branches, and a fact of
memory withdrawn is set aside as well (soft-deleted, so the user can restore
it). A withdrawal only ever lowers: nothing it does raises a turn back.
"""

from __future__ import annotations

import hashlib
from typing import Any

# The kinds a value can be named by: a text or an address is digested, an id
# is taken as it is.
_DIGESTED = ("document", "file", "retrieved", "tool", "web")
_BY_ID = ("memory",)


def source_for(kind: str, value: str) -> str:
    """The lineage entry that names ``value`` as a source of ``kind``.

    A document, a file, a page, a tool's output or a retrieved snippet is
    named by the digest of its text (a page by the digest of its address); a
    fact of memory by its id. Raises ValueError for a kind no lineage names
    by value.
    """
    if kind in _DIGESTED:
        return f"{kind}:{hashlib.sha256(str(value).encode('utf-8')).hexdigest()}"
    if kind in _BY_ID:
        return f"{kind}:{str(value).strip()}"
    raise ValueError(f"no source of kind {str(kind)[:24]!r} is named by a value")


def _default_conversations() -> Any:
    from opti_oignon.conversation import conversation_manager

    return conversation_manager


def _default_branches() -> Any:
    try:
        from opti_oignon.conversation_branches import branch_manager
    except Exception:  # noqa: BLE001 - no branch store: nothing there to lower
        return None
    return branch_manager


def _default_memory() -> Any:
    try:
        from opti_oignon.memory.dedup import get_memory_store
    except Exception:  # noqa: BLE001 - no memory store: no fact to set aside
        return None
    return get_memory_store()


def withdraw(source: str, *, conversations: Any = None, branches: Any = None, memory: Any = None) -> dict[str, Any]:
    """Withdraw ``source``: lower every turn and branch message it reached, and set aside a fact it names.

    The conversation store judges the entry first and raises its OriginError
    for one outside the grammar, before anything is written anywhere.
    Returns how many turns and branch messages were lowered, and whether a
    fact was set aside.
    """
    conversations = conversations if conversations is not None else _default_conversations()
    turns = conversations.withdraw_source(source)
    branches = branches if branches is not None else _default_branches()
    branch_messages = branches.withdraw_source(source) if branches is not None else 0
    fact_set_aside = False
    if source.startswith("memory:"):
        memory = memory if memory is not None else _default_memory()
        if memory is not None:
            fact_set_aside = bool(memory.soft_delete(source.split(":", 1)[1]))
    return {"source": source, "turns": turns, "branch_messages": branch_messages, "fact_set_aside": fact_set_aside}
