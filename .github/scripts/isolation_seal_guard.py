#!/usr/bin/env python3
"""Isolation-seal guard: every test window must go through the shared one.

A contract suite that loads a module from its file has to manufacture a package
window, and hand-rolling that window is how an estate ends up green for the
wrong reason. Two independent routes let a real project module resolve behind a
test's back -- the module cache, which Python reads before it consults any
finder, and a finder that answers on the module NAME and ignores the parent
package's path, which is what an editable install registers. A window that
closes one route and not the other silently blocks nothing, and the suite then
draws its conclusion from an absence it never created.

The shared window in ``tests/_isolation.py`` closes both routes and PROVES on
every call that it did. This guard exists so no suite can quietly stop using
it: a file that builds a window of its own is a violation unless the ledger
below owes for it.

RATCHET. The ledger is the debt that predates the shared window, and it is a
SEAL, not a list of names. A ratchet that only counts is a ratchet on the count:
skipping an owed file whole -- the old ``if name in LEDGER: continue`` -- let a
suite with an unsound window absorb new contracts for as long as anyone cared to
write them, while the debt still read the same figure. "May only shrink" was
true of the number and said nothing whatever about the estate.

So every owed name carries the digest of the suite as it stood when the debt was
enumerated. An owed suite that changes by one line no longer matches its seal:
touch it, and you migrate it. The debt is frozen exactly as it was found, it can
be paid, and it cannot grow -- not in names, and no longer in lines.

Most of the debt closes only ONE route: the cache key is evicted, the finder is
never guarded, and under an editable install the real module loads anyway. The
census printed at the end reports that, because a count that hides what it is
counting is the comfort this guard was written against.

Three questions, three answers, and their domains do not overlap, so no one of
them can quietly cover for another:

  * ``find_violations``           -- a window hand-rolled by a suite nobody owes for.
  * ``find_broken_seals``         -- an owed suite whose bytes moved.
  * ``find_stale_ledger_entries`` -- an owed suite that migrated, or vanished.

The helpers are pure and import-safe, so they are unit-tested without touching a
filesystem; ``main`` scans the test tree and exits non-zero on any finding.
Usage: ``isolation_seal_guard.py [TESTS_DIR]`` (default: tests/).
"""

import hashlib
import sys
from pathlib import Path

# A file builds an isolation window when it loads a module from a path or puts
# a finder ahead of the import machinery.
_NAME_ROUTE_MARK = "sys.meta_path"
_WINDOW_MARKS = ("spec_from_file_location", _NAME_ROUTE_MARK)

# A file uses the shared window when it imports it.
_SHARED_MARK = "from _isolation import"

# Debt that predates the shared window: name -> sha256 of the suite as the debt
# was enumerated. MAY ONLY SHRINK, and no entry may move -- see RATCHET above.
LEDGER = {
    "test_adaptive_routing_bounds_contracts.py": "941d0d93f15359877c44b6c83e31090c2fa0cb98082ea055b764a91d205da127",
    "test_agent_run_model_capability_gate_contracts.py": "0dad2100fa2f9d1513aa82844deee3f86d0fdb50d9239bc5492de56e90758d95",
    "test_agent_run_teacher_wiring_contracts.py": "d9e474d6989dc69f5ef34e71304fcea1840db3e2069b8df4eaa71850024c09d3",
    "test_agentic_classifier_contracts.py": "778c879f8b2f6e9e859030d2ff4fcb5fe29a0d5723b660d7356f1be237ec2b3e",
    "test_agentic_summary_alignment_contracts.py": "286f9c6ce564378a8b31b16fb0f3f222f2245df0fd8ac534a17d0809ce18fa18",
    "test_agentic_thinking_guard_contracts.py": "9a40e39968bad0449ab97d774860e6969db269648df5582727123f6a17902e6a",
    "test_agents_exec_surface_contracts.py": "fb3939fc44d81160c516d6c6fa8301dafb0927fafe27d8acc8358cd53c98c6f9",
    "test_auth_2fa_secret_at_rest_contracts.py": "1eb3ad229db134bbda4ad1bf8acb39812831fc2f63fd2d66f09a81332882ec3e",
    "test_auth_2fa_verification_bounds_contracts.py": "5d28b964742ba03ad7448fe4289771bc42c367f33776dcfff9c26007885454c2",
    "test_auto_capture.py": "cf84d5916bf518fb148afb4e28d8206254bbfef06096a1926c8d186c2750a98e",
    "test_auto_tuner_sweep_bounds_contracts.py": "891993c81186f77718736c48b4b7c14e08913b79c73cdc45039b264f9c9f6313",
    "test_backend_routing.py": "58effc7706be9fa50d47af9e05841d8ec33b9b16d9af4fc3064ecbc273f91eb7",
    "test_backend_routing_cache.py": "d9bd7906b11cee2309c45a27611559fbf7b9e3ca88b0391c17cbc77438b0b1cf",
    "test_backup_encrypted_format_contracts.py": "11a7cc96ad45728eeea38fe0fdf364257317a2170d07d4fce3015d94565fb570",
    "test_backup_export_content_contracts.py": "5027ce23197e40f21b993e1688c62ec4661f55321e05361ae3d8375a0a7f5385",
    "test_backup_import_signature_rollback_contracts.py": "bbb08937db5307093e56e7c95c0680c2b788f0b2c2232dc31839ffaa6bfcea19",
    "test_benchmark_evaluator_scoring_bounds_contracts.py": "c20aa9d098b2a319d9725a6f3177d51170bc199fee6b70abd7d129da6850947a",
    "test_benchmark_recommendation_selection_contracts.py": "ace500852c114e41d99adf5755703bb736b9cb0ce74b6c4c39d7d5fc2f79608a",
    "test_benchmark_trigger_optin_bounds_contracts.py": "04c2bd3e68bce25de64170a1b156ed9c0084e2f90a03e7e222af6961e61e643f",
    "test_blob_framed.py": "6bb6421973052c51b2803414e38e074a1a95fdf83b2c0416681c3ece786f8b5f",
    "test_blob_transfer.py": "d2504c5fb7b696732e6209480cc3a2951fd490009680034a9c9475b88b5298dd",
    "test_capability_manifest_contracts.py": "354a4bb8138be925a92a53078263e7dc7059d5ccce5a9602b18aa7c73e4d4b59",
    "test_capability_wiring_contracts.py": "63fedb961ddaae97c69b0cc186f1e2ec064ecd9f3ed2af4fa24fc8250e8c9698",
    "test_chat_eval_harness_contracts.py": "d6600959981a5c81b930347e57a709c02bfad80ba2d8802e429bccdb5f0d6cf2",
    "test_chat_exec_failure_hint_contracts.py": "e1cd7beaf5cce64753ad99f11fdbd1d2abaeaf8e8127b6e1b0b937dd5db6fab8",
    "test_chat_fallback_decision_contracts.py": "6b13892f407b33f0ea3551fe103f5fad2781f6d355bbcee8e60bb63e294fc48a",
    "test_chat_fr_hint_curve_contracts.py": "22d00fc29649bc715c7cb0ab38da35dfd2c7db2289591188f12142531def0f66",
    "test_chat_retry_model_contracts.py": "da9db5d913dfd91e4dbf9a328db51924c17ab6242af4d79f8f182afe495e8ad9",
    "test_coding_agent_apply_boundary_contracts.py": "70432ad3fe84dca1defad32265627ef3db1d97b224d60984840e2d795c7c6d4a",
    "test_context_manifest_pinning_contracts.py": "4568f1eda1c4cb9b140905660375a6125a09d4d2349a74b877584c41307fc7fd",
    "test_context_optimizer_preservation_contracts.py": "097cdcc11d15246b9f44ef1dd46e28be92ef14f815ef22b8083581d2d25066ad",
    "test_conversation_apply.py": "c7c8a334bade508df7315b430879d63bad189d735e42a85efd559b6b7fd8b79a",
    "test_direct_answer_reuse_contracts.py": "2999a103993b4a22b0ce8882cad6909eb8b6834b3c57d9b7e5afb0196f88514f",
    "test_fine_tune_export_escaping_contracts.py": "4e517c07b9668613e2ff9bc6350e6cf4c4f34b4f3b2914580c15612545870ca1",
    "test_fine_tune_tracker_persistence_contracts.py": "c0bfa1b28942941885e41a54ba2bcf237c49f3f75b96333ca2073ed3b2be0620",
    "test_humanizer_preservation_contracts.py": "519fb96e53bf0378c413f238443d2b7a898c91a2e626329372f076c914f6e54d",
    "test_learned_router_integrity_contracts.py": "f2b9d5c6f854c164e97befcfc562c654925e79fa453eda9e1cc55aff4fb06650",
    "test_luks_detector_contracts.py": "0fc4c17ff2452edec9b3b6b6298174b8874a359a3629e24729285e31d6fbb9f0",
    "test_luks_no_false_positive_contracts.py": "441c0089d95371b86d134bcad6bbd08fe03d2f1228e791850a9abc701c917966",
    "test_memory_agent_untrusted_wrap_contracts.py": "18fa60701a0a79458dd560ca5c27fa10e17722c1528c280e0599d374ebca814b",
    "test_memory_block.py": "daa6ccf15a391ce005c7795d687267f0bf358116ea3425e7c614ff1de8426ddf",
    "test_memory_canonical_apply.py": "a2d943954b9fb9a85bc092c78a4c4b8d6ceccb2bd3fa59ca17c6246f1c79b855",
    "test_memory_canonical_sql_hygiene_contracts.py": "97966f6669008514899f3fafaa78bb0b66453d70d17e8eb6f9a6639841febed8",
    "test_memory_curation_conservatism_contracts.py": "8ad5bb9bd6850e7bdd5723d7d0793381c6258a2df69f8af8d89af2c4866eb20d",
    "test_memory_dedup_coordination_contracts.py": "b16e34e24fe635238977c1b93a7af1a11f1fb0512f82389114d494b30b61c599",
    "test_memory_dual_layer_invariant_contracts.py": "ff272130c0c0735e82d7c23287fa196b37aa34cc51cc09d628185e067ebc7766",
    "test_memory_extraction_bounds_contracts.py": "c9c5f8413d1a6979a5fcbb50c2916903a54e906fcfaeda3ced127f14491f5afa",
    "test_memory_migration.py": "ea82e89f2bbd829a5a3fab45701ed5ba113141f27868c56c76818735315bfd5d",
    "test_memory_routes.py": "a36b22f66ab267a0823cc4170e88b9a4fff76e73453479aca2ed25eb11f56179",
    "test_model_download_digest_pin_contracts.py": "4de3c0f479820d8a0fad1cc0396c12dc1c75f35d112325538077245314064606",
    "test_model_download_ssrf_defense_contracts.py": "22023f571b55f89ff85eacd50fb572eb661fa5336d3dc010624af81f823cfb44",
    "test_model_load_provenance_gate_contracts.py": "c4e75ec32157b841560d410be48396c7a90314eaacf441d089cb0a3fbfe029af",
    "test_model_provenance_contracts.py": "c6322f55667dc27828b6fef85e183e8896893c009658666deadaa262bb6c12c3",
    "test_note_update_store_contracts.py": "96120eb74ba29182a3fc6facb89a294b5d691befd7a2dd0f2250f20ba7e604e6",
    "test_notes_apply.py": "251d946f234944e1bbd650258ed47ebf486ae6f933ca7d6a30ca4caf9bf5ecc3",
    "test_notes_mobile_optin_contracts.py": "899f4e78efa5d54d14328b3377b6ecab33b7cf1c5a42bb0f130457334d7367b2",
    "test_notes_send_half_contracts.py": "9d7d64109e70af093d8d7cd3ad7c92e8abac5c603f4780046d72413e060eda30",
    "test_pipeline_persistence.py": "8e3dcbea2f716195f6a9e550ce683337a0375dbf0e4ac99ff7778ca9542f2c30",
    "test_plugin_allowlist_contracts.py": "d5b5bb21b9b197e368db89f269959b99b88802cc2a14569b916563d4e8a14554",
    "test_plugin_discovery_paths_contracts.py": "f6f0dd7a2535aade4105d868e649105f09be3c63c81cb50d939ad5d3f8ebf293",
    "test_plugin_tool_network_capability_contracts.py": "1a0126c68eefd9214438275116a71e81827817e2e40cae7a39448ec1881f924a",
    "test_plugin_worker_context_contracts.py": "41aa0c191f477ea3b5a0a5c28d53af6c556a26102c0986c7cea5cd69211f4bf2",
    "test_plugin_worker_isolation_contracts.py": "5de16d5955eb2fc086705da3c81265853cb46294b1256b6b8f6033080631d2a2",
    "test_prompt_budget_bounds_contracts.py": "a2bfb827dab6786601414dbcaeea8cc2b401fedf3b684f9e4bc518f283e43a17",
    "test_quick_sandbox_binding_contracts.py": "33a2d9784d0a7ab667cc36db53b1519b1a4457b2153c59c7eb78f12aac4856ca",
    "test_quick_sandbox_effective_id_contracts.py": "9ce6b3a478551cf8f64f29a468958164393031c9f1da05e6d651873401054a01",
    "test_redteam_egress_guard_contracts.py": "101509bcc54a0d16e1ca3330b8876e36fd438146a623eefaaaf70a4d2b0872c5",
    "test_redteam_feedback_apply_gate_contracts.py": "71239d6de6f51745719a92dd0f4a3afaca7eb7daf65640b999a513f5b762ac57",
    "test_redteam_feedback_sanitization_shape_contracts.py": "eded8af14b2ebf99bede10142897a8faff096fb124dcf3000fefbb0c9d84c7ff",
    "test_redteam_report_permissions_contracts.py": "2a5a12878291adbbaee29f4d121dde53c80c80f2542e9fa5a28f683f36fa9a6a",
    "test_redteam_runner_no_egress_bypass_contracts.py": "1e9c7cdc52adf27511f915519b577af3864525ec1b29dffbdd7920b5eea6892f",
    "test_redteam_scoring_classification_bounds_contracts.py": "337e53e7656fa8853904253aeea8726fd5e9ec90d6a5ec703af83020a1ebc689",
    "test_redteam_scoring_faithfulness_contracts.py": "c537807df67dd6aad1485cd1991ccef68fa6f0c839d37f840bd133824ccf592c",
    "test_remote_inference_continuation_contracts.py": "4d9911f89360275c5a15bdf2e6bde86c2931018a991e1da7499aa6b45fe77beb",
    "test_remote_inference_contracts.py": "443beb85b4367fcbd38630815c99344cfd5eadf4184d813c8241f927e33da6a0",
    "test_remote_inference_reauth_contracts.py": "f7bd6bd1dd5d4fd47144d59685027b7ad07ae9212db19ae67bc2d693a29c0dc3",
    "test_remote_streaming_channel_contracts.py": "25d73e2df2afbb450ae986e4fbe58bd58fc57b887fdfa8f073fe8be0b105768e",
    "test_resource_governor_contracts.py": "e85b546ed4114739e21e86a2995262a353312ae2200aa68daf11663210da1320",
    "test_resource_governor_gate_contracts.py": "0c54371e4879c91b2f1ae62636c85fc4338792b8aefdf26896584d23ff29dc19",
    "test_resource_governor_queue_contracts.py": "723153ac29dd96e113f504403babda19292d99b81cd6fadba7989c980d5e3d52",
    "test_response_cache_contracts.py": "1712028cd501dd2dcc2ae247033a25a0eb3a715057f4447b8d15b7b8da339956",
    "test_response_cache_invalidation_contracts.py": "73d7fbe45ab9c34310f4827118223f536f60387f8305d3cd5cd08ed612a80bd4",
    "test_response_cache_warm_contracts.py": "57563f8253eef1e4e9312dfe8ecb97004252572b58ccf1dd18ece3beebf73494",
    "test_robust_toolcalling.py": "0da716e79d9a481c52a3f1d95b67f1fdf5f1fa5e69126f4d66f4c2341c994643",
    "test_router_requirement_propagation_contracts.py": "7e7305b5ccdb83b5aa1c372b9129c2c76fb5c7f29fd1fd3992256c5beb4e5990",
    "test_router_tool_calling_enforcement_contracts.py": "4355196c2a660963b6cb3901b2268e55c04abc7d17610807057f40378e23076e",
    "test_sandbox_egress_gate_contracts.py": "29f1cf35a8450f5e45181afc22b1b69781594d0cf487a1ce6ff885e64489960b",
    "test_sandbox_rest_confinement_contracts.py": "3b118c8165fc5a24d442db33cfddc934d0f68e456c48a9c8145dc76b9313253b",
    "test_sandbox_tools_confinement_contracts.py": "eb1f20ecb603f1fed0c3a00f22ef5cec7961b463dab8da8ff7e127dadd55f858",
    "test_security_auth_failclosed_contracts.py": "d22b0b0720abe269dd3ba5a83a818d3ae6220148d9a3cc9c1ae08b2dff7bca94",
    "test_self_correction_loop_contracts.py": "1407841bdf17b7788512aa8ceac14a43b3bfab7c8b798076862384bb4032c0d5",
    "test_semantic_cache_contracts.py": "aa79e329c00216137942eebeb6e79e7c54c885f57b5eb2dbb88164ecd53bf673",
    "test_semantic_cache_fallback_contracts.py": "5c3229b9aa301c49ba1f359c05e15bcba403b2a0a1f54b3c13fa535f2492dfef",
    "test_semantic_cache_management_contracts.py": "e06438a28c7b0b472f514e23d1f768a45627156d3f648167798c52d9d66bdf04",
    "test_semantic_cache_no_model_lookup_contracts.py": "9d640a4eb959aaedc5b0b9af2b0af972903fd214844cd77556b33acc64b6473a",
    "test_semantic_cache_semantic_tier_contracts.py": "a4c64afb1dc6f8286dacc4f57f17d31399737d690782888e29bec8c72c66c691",
    "test_session_fingerprint_bounds_at_rest_contracts.py": "20e48826a7a0ec596028ea94b0e558c2eb44e62637935f01fd1a41f56a6d2c20",
    "test_skill_apply.py": "34dd0af8030aaa17612b3885af1fc5958101c2483ff5f2967a01ea7a87cf4341",
    "test_skill_consultation_untrusted_wrap_contracts.py": "73ccb9c608297c45e8eb1eea6874e711a0a0ab8ab790e903904f155129912e01",
    "test_skill_lifecycle_versioning_contracts.py": "70a0ab3bfa87a70b12848aa80fa4107eef0458193cb88a8f887c67f67d6953d6",
    "test_skill_registry_path_confinement_contracts.py": "aac595346387418c8b654af7348a83465f1bb670b6151e855dd0a0566b869971",
    "test_skill_teacher_publish_gate_contracts.py": "f989d35f7c3b9ced77050e1e7558892a5753308e38a7521e5573b9aec576d71d",
    "test_skill_write_gate_failsecure_contracts.py": "d9bbc6bbe0008249036596c4207220c852dd2ab7edc59f9666188d625122d645",
    "test_speculative_argv_materialisation_contracts.py": "8eb979f268a75594cf308aecd2097d149020d339a83d9f2658c155931d6b0835",
    "test_stage_manifest_routing_contracts.py": "0ec7a7cf38867187fa2508bedd1ffb01e27c8bcdceea4c0b6dccc1bcccbd2d8c",
    "test_startup_swap_no_false_positive_contracts.py": "b25ccbac9523bf4f85e4b7182b40f0c40c07d6675fa58d7682562bc1fe7f8e65",
    "test_syn01_blob_vault_contracts.py": "b97346e72b660d167757a82ac7f10226ae8ad6c8c9e2083071d690d5cd4fd2e1",
    "test_syn01_receive_contracts.py": "3362edf9f9eb75980fb2cb5a8afa73dca2327c0644c17d1d1a47c4905d24faf0",
    "test_sync_deferred_remote_channel_contracts.py": "59c1681cc8ecce6c124046f75c522cb57b4194138a62f7f4000b7d2c4ac3243a",
    "test_sync_pairing_trust_contracts.py": "b1121e5f37c1475981eddd358a861d127d04075c1a994196c09ee567585982e2",
    "test_sync_run_wire_gate_contracts.py": "3d2709229c98d2c8f974924f6734f0c20c4e8c8d9fc7d96c85c432d2ec8c6b93",
    "test_sync_service.py": "7afbdd9d498971bb824a6aacd4479a0cacc91275ecf0ad2fafd93885bbb0d3f6",
    "test_telemetry_profiler_consumer_contracts.py": "2febae71f5498bc12566873847bb1b60daae655985da23e7ba6a61b1300b8b00",
    "test_tool_loop_attribution_contracts.py": "0f77c4484ee0fcce359961db79952fda5e5bd1fec3d20e98e941af6d0d678973",
    "test_tool_transcript_contracts.py": "a5ce2a4084ad738a3fbfe0b648691f20fea8d22fda5368e669b364ee477807ca",
    "test_tool_transcript_gate_contracts.py": "545f1a401454ee73c9b597d81d1e8ab5ed36ec6ef5eda8082b555014143e8f77",
    "test_vault_manifest.py": "d32aaa13d4313afa3d79adb9d7336f8b698fd1822cdd2d5fb024c445efe89880",
    "test_vector_store_health.py": "2b30bb4f74103058e6bd5c7d3342e0e70de3ef519194631548b6599423e58f69",
}


def digest(text):
    """The seal of a suite, taken on the text this guard reads."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def builds_a_window(text):
    """True when the file manufactures a package window of its own."""
    return any(mark in text for mark in _WINDOW_MARKS)


def uses_shared_window(text):
    """True when the file goes through the shared isolation window."""
    return _SHARED_MARK in text


def guards_the_name_route(text):
    """True when the file closes the SECOND route: a finder answering on a name.

    Evicting the cache key closes the first route and nothing else. An editable
    install registers a finder that resolves a submodule from a name table and
    never looks at the parent package's path, so a stand-in parent whose path is
    empty stops no one and the real module loads. A window without this closes
    only what was easy to close.
    """
    return _NAME_ROUTE_MARK in text


def find_violations(files):
    """Files that hand-roll a window and that the ledger does not owe for.

    ``files`` is an iterable of (name, text) pairs. Pure and import-safe so it
    can be unit-tested without touching a filesystem.

    An owed name is passed over HERE and answered for by the seal below. The two
    domains are disjoint by construction, so neither can hide a failure of the
    other -- a guard whose failure another guard conceals is not a guard.
    """
    violations = []
    for name, text in files:
        if not builds_a_window(text):
            continue
        if uses_shared_window(text):
            continue
        if name in LEDGER:
            continue
        violations.append(name)
    return sorted(violations)


def find_broken_seals(files):
    """Owed suites whose bytes no longer match their seal: touch it, migrate it.

    This is the ratchet's tooth. A suite the ledger owes for carries a window
    that closes at most one of the two routes, so a new contract written into it
    draws its verdict from an absence nobody manufactured. The debt may be
    carried; it may not be added to.

    A suite that migrated is NOT broken -- migrating is exactly what this asks
    for, and charging it anyway would leave no way to pay. It becomes stale
    instead, and comes off the ledger.
    """
    seen = {name: text for name, text in files}
    broken = []
    for name, sealed in LEDGER.items():
        text = seen.get(name)
        if text is None:
            continue
        if uses_shared_window(text):
            continue
        if digest(text) != sealed:
            broken.append(name)
    return sorted(broken)


def find_stale_ledger_entries(files):
    """Ledger names that no longer hand-roll a window: the ratchet has slipped.

    An entry that has migrated must come OFF the list, or the debt count stops
    meaning anything and a later regression could hide behind it.
    """
    seen = {name: text for name, text in files}
    stale = []
    for name in LEDGER:
        text = seen.get(name)
        if text is None:
            stale.append(name)
        elif uses_shared_window(text) or not builds_a_window(text):
            stale.append(name)
    return sorted(stale)


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path("tests")
    files = [
        (p.name, p.read_text(encoding="utf-8", errors="ignore"))
        for p in sorted(root.glob("test_*.py"))
    ]
    violations = find_violations(files)
    broken = find_broken_seals(files)
    stale = find_stale_ledger_entries(files)

    if violations:
        print("Isolation-seal violations -- these suites build a window of")
        print("their own instead of the shared one in tests/_isolation.py:")
        for name in violations:
            print(f"  {name}")
        print()
        print("A hand-rolled window does not close both routes into the")
        print("package, and a suite that reasons about an absence it never")
        print("created proves nothing. Use the shared window.")
    if broken:
        print("Broken seals -- the ledger owes for these suites and their bytes")
        print("have moved. Touch an owed suite and you migrate it:")
        for name in broken:
            print(f"  {name}")
        print()
        print("The window in an owed suite closes at most one of the two routes")
        print("into the package, so a contract written into it draws its verdict")
        print("from an absence nobody manufactured. Move the suite onto the")
        print("shared window and take it off the ledger. The debt may be")
        print("carried; it may not be added to.")
    if stale:
        print("Stale ledger entries -- these have migrated and must be removed")
        print("from LEDGER so the debt count stays honest:")
        for name in stale:
            print(f"  {name}")

    if violations or broken or stale:
        return 1

    shared = sum(1 for _, text in files if uses_shared_window(text))
    cache_only = sum(
        1
        for name, text in files
        if name in LEDGER and not guards_the_name_route(text)
    )
    print(
        f"Isolation seal OK: {shared} suites on the shared window; "
        f"{len(LEDGER)} owed, {cache_only} of them carrying a window that "
        f"closes the cache route and leaves the finder open. The ledger is "
        f"sealed: it may only shrink, and an owed suite that changes must "
        f"migrate."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
