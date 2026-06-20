#!/usr/bin/env python3
"""
S184 / UD-01: per-user RAG collection ownership must not collide across users
whose IDs share a prefix.

``user_collection_name`` produces ``user_{id}_{collection}``. The sibling
helpers (``get_user_collections``, ``strip_user_prefix``) match on
``user_{id}_`` with a trailing separator, but ``is_user_collection`` matched on
``user_{id}`` without it, so user "1" was judged to own "10"/"11"/"1abc"
collections. ``is_user_collection`` is used by both the GDPR export
(``UserDataExporter._export_rag``) and the cascade delete
(``UserDataDeleter._delete_rag``), so the bug leaked one user's RAG collections
into another's export and let a delete remove another user's collections.

These tests load the module in isolation (no heavy import chain) and check the
ownership predicate directly.
"""

import importlib.util
import os
import sys
import types

sys.modules.setdefault("ollama", types.ModuleType("ollama"))

_path = os.path.join(
    os.path.dirname(__file__), os.pardir, "opti_oignon", "user_data_manager.py"
)
_spec = importlib.util.spec_from_file_location("user_data_manager_s184", _path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["user_data_manager_s184"] = _mod
_spec.loader.exec_module(_mod)

user_collection_name = _mod.user_collection_name
is_user_collection = _mod.is_user_collection
get_user_collections = _mod.get_user_collections


class TestIsUserCollectionNoPrefixCollision:
    def test_owns_own_collection(self):
        name = user_collection_name("1", "docs")
        assert is_user_collection("1", name)

    def test_does_not_own_prefix_sibling(self):
        # user "10" owns this; user "1" must NOT.
        name = user_collection_name("10", "docs")
        assert not is_user_collection("1", name)

    def test_does_not_own_alpha_prefix_sibling(self):
        name = user_collection_name("1abc", "docs")
        assert not is_user_collection("1", name)

    def test_consistent_with_get_user_collections(self):
        # is_user_collection and get_user_collections must agree on ownership.
        names = [
            user_collection_name("1", "a"),
            user_collection_name("10", "b"),
            user_collection_name("1abc", "c"),
        ]
        via_predicate = [n for n in names if is_user_collection("1", n)]
        via_filter = get_user_collections("1", names)
        assert via_predicate == via_filter
        assert len(via_predicate) == 1  # only user "1"'s own collection

    def test_uuid_ids_isolated(self):
        a = "11111111-1111-1111-1111-111111111111"
        b = "11111111-1111-1111-1111-111111111112"
        name_b = user_collection_name(b, "docs")
        assert not is_user_collection(a, name_b)
        assert is_user_collection(b, name_b)
