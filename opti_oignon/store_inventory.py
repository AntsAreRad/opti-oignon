"""Every store the package keeps, and what the account wipe does with each.

The one table the wipe's lists are derived from (``user_data_manager``: ``WIPE_NOT_COVERED``,
``WIPE_SINGLE_USER``, ``WIPE_RETAINED_BY_DESIGN``), kept complete against the code by a census: every database
name the package spells, and every name it joins to a data folder, is classified here -- as a store, or as a name
that is not one, with why. A store holds one of the ``HOLDS`` and the wipe does one of the ``WIPES`` with it; one
file may be listed twice when its parts fare differently (the sandbox audit's words and its record of what ran).

Data only: nothing of the package is imported to be listed.
"""

from __future__ import annotations

from typing import NamedTuple

# What a store holds, closed.
WORDS = "words"  # the user's own: turns, notes, commands, feedback, queries
DRAWN = "drawn words"  # words drawn from the user's: answers, facts, summaries, chunks of their documents
IDS = "ids and counts"
SECRETS = "secrets"
NOTHING = "nothing of the user's"
HOLDS = (WORDS, DRAWN, IDS, SECRETS, NOTHING)

# What the wipe does with it, closed.
REMOVED = "removed"  # for the user asked, in any mode
ONE_USER = "one user"  # for a single-user install's one user; named as not covered otherwise
NOT_COVERED = "not covered"  # named in the answer, the label saying why
KEPT = "kept"  # retained by design, named in the answer, the label saying why
NOT_THE_USERS = "not the user's"  # never named: nothing of anyone's
WIPES = (REMOVED, ONE_USER, NOT_COVERED, KEPT, NOT_THE_USERS)


class Entry(NamedTuple):
    """One store: the label the wipe's answer uses, the files or folders as the code spells them (``*`` for one
    joined to a data folder under a name built at run time), the modules that spell or build them (relative to the
    package), what it holds, what the wipe does with it, and the labels a failure to remove or empty it is named by
    in the answer."""

    label: str
    names: tuple[str, ...]
    modules: tuple[str, ...]
    holds: str
    wipe: str
    failures: tuple[str, ...] = ()


STORES: tuple[Entry, ...] = (
    # -- Conversations and what they leave ------------------------------------------------------------------------
    Entry("conversations (store not user-scoped)", ("conversations.db",),
          ("conversation.py", "conversation_branches.py", "user_isolation.py"), WORDS, ONE_USER,
          failures=("conversations",)),
    Entry("artifacts (stored in conversation metadata)", ("conversations.db",), ("conversation.py",), DRAWN,
          ONE_USER),
    Entry("conversation branches store (single-user)", ("branches.db",), ("conversation_branches.py",), WORDS,
          ONE_USER),
    Entry("the onion's store (keyed by conversation)", ("*",), ("memory/librarian.py", "memory/onion_store.py"),
          DRAWN, ONE_USER),
    Entry("response/semantic caches (conversation-scoped)", ("response_cache.db", "semantic_cache.db"),
          ("response_cache.py", "semantic_cache.py"), DRAWN, ONE_USER,
          failures=("response cache", "semantic cache")),
    Entry("offline queue (single-user)", ("sync_queue.db",), ("sync_queue.py",), WORDS, ONE_USER,
          failures=("offline queue",)),
    Entry("sandbox audit's words (single-user; the record of what ran is retained)", ("sandbox_audit.db",),
          ("sandbox_manager.py", "db_encryption.py"), WORDS, ONE_USER,
          failures=("sandbox audit's words",)),
    Entry("sandbox audit: the record of what ran, without its words", ("sandbox_audit.db",),
          ("sandbox_manager.py",), IDS, KEPT),
    Entry("run journal (opt-in record of agent runs)", ("run_journal",), ("run_journal.py",), WORDS, REMOVED,
          failures=("run journal",)),
    Entry("feedback store (unscoped, FBK-01)", ("feedback.db",), ("feedback.py", "user_isolation.py"), WORDS,
          ONE_USER,
          failures=("feedback store",)),
    # -- Memory ---------------------------------------------------------------------------------------------------
    Entry("facts and their vectors (the vector store's namespace of the facts)", ("memory_facts.db", "vectors.db"),
          ("memory/canonical_store.py", "chroma_migration.py", "memory/vector_store.py", "vector_db.py"), DRAWN,
          REMOVED,
          failures=("facts", "facts' log", "facts' vectors")),
    Entry("old memory store (single-user, emptied at boot once migrated)", ("memories.db",),
          ("memory/legacy.py", "memory/migration.py"), DRAWN, ONE_USER,
          failures=("old memory store",)),
    Entry("pending writes (proposals waiting for the user)", ("pending_writes.db",), ("pending_writes.py",), DRAWN,
          REMOVED,
          failures=("pending writes",)),
    # -- The stores that only keep copies of the user's words: emptied for the one user ---------------------------
    Entry("the coding agent's decision preferences (not user-scoped)", ("fingerprint.db", "*"),
          ("session_fingerprint.py", "user_isolation.py"), WORDS, ONE_USER,
          failures=("decision preferences",)),
    Entry("humanizer feedback store", ("humanizer_feedback.db", "*"), ("humanizer.py",), DRAWN, ONE_USER,
          failures=("humanizer feedback",)),
    Entry("learned routing store (the queries, word for word, and the model learned from them)",
          ("learned_router.db", "learned_router.pkl"), ("learned_router.py",), WORDS, ONE_USER,
          failures=("learned routing store",)),
    Entry("the command line's history (questions and answers, not user-scoped)", ("history",),
          ("history.py", "conversation.py"), WORDS, ONE_USER,
          failures=("command line history",)),
    Entry("the knowledge base's search log (the queries, not user-scoped)", ("rag_documents.db",),
          ("rag_store.py", "rag_dashboard.py"), WORDS, ONE_USER,
          failures=("knowledge base search log",)),
    # -- The content the user manages: named --------------------------------------------------------------------
    Entry("notes and their updates", ("notes.db", "note_updates.db"),
          ("notes/notes_store.py", "notes/note_updates_store.py"), WORDS, NOT_COVERED),
    Entry("note attachments (voice, images)", ("notes_blobs",), ("notes/blob_store.py",), WORDS, NOT_COVERED),
    Entry("projects store (single-user)", ("projects.db",), ("projects.py", "user_isolation.py"), WORDS,
          NOT_COVERED),
    Entry("project files and saved outputs", ("projects",), ("projects.py",), WORDS, NOT_COVERED),
    Entry("the knowledge base: documents, their chunks and vectors (not user-scoped)",
          ("rag_documents.db", "vectors.db", "*"), ("rag_store.py", "rag/config.py"), WORDS, NOT_COVERED,
          failures=("knowledge base collections named for the user",)),
    Entry("the knowledge base's index cache (the documents' paths and hashes)", ("cache", "index_cache.json"),
          ("rag/config.py", "rag/indexer.py"), IDS, NOT_COVERED),
    Entry("project indexes (chunks of the project files)", ("vectors.db",), ("project_context.py",), WORDS,
          NOT_COVERED),
    Entry("the knowledge base's injection audit (the flagged chunks' text)", ("rag_injection_audit.db", "*"),
          ("rag_sanitizer.py",), DRAWN, NOT_COVERED),
    Entry("knowledge-base ingest jobs", ("rag_ingest_jobs.db",), ("rag/batch_ingest.py",), IDS, NOT_COVERED),
    Entry("legacy vector folders (the documents' chunks in plain text until `db_encryption migrate`)",
          ("chroma_v2", "project_chroma", "chroma_db", "chroma.sqlite3"),
          ("rag_store.py", "project_context.py", "rag/config.py", "vector_db.py", "db_encryption.py",
           "chroma_migration.py"), WORDS, NOT_COVERED),
    Entry("coding history store", ("coding_history.db",), ("coding_history.py",), WORDS, NOT_COVERED),
    Entry("code execution outputs (the files a conversation's code wrote, kept by no conversation)",
          ("exec_outputs",), ("code_executor.py",), DRAWN, NOT_COVERED),
    Entry("sandbox exports (the files copied out of a sandbox)", ("sandbox_exports",), ("api/routes_sandbox.py",),
          DRAWN, NOT_COVERED),
    Entry("skills (written by the user or accepted from the agent)", ("skills",), ("agent/skills.py",), WORDS,
          NOT_COVERED),
    Entry("the garden (allium; no wipe of its own yet)",
          (".autobio.db", ".autobio.glass.db", ".glass.db", ".sowing.db", ".sowing.glass.db"),
          ("allium/store.py",), DRAWN, NOT_COVERED),
    Entry("plugin-owned data stores (plugin-local DBs)", ("plugin_data", "github_auth.db", "scratchpad.db",
                                                          "tasks.db"),
          ("plugin_isolation.py", "plugins/github-connector/entry_point.py", "plugins/scratchpad/entry_point.py",
           "plugins/task-extractor/entry_point.py"), WORDS, NOT_COVERED),
    Entry("plugin logs (not checked for words)", ("plugin_logs",), ("plugin_subprocess.py",), DRAWN, NOT_COVERED),
    Entry("settings files the user writes (presets, custom pipelines, benchmark profiles, their backups)",
          ("user_config.yaml", "user_presets.yaml", "pipelines_custom.yaml", "execution_pipelines.yaml",
           "custom_benchmark_profiles.yaml", "config_backups", "cli.yaml"),
          ("config.py", "presets.py", "pipeline_manager.py", "pipelines.py", "benchmark_custom_profiles.py",
           "system_presets.py", "cli/config.py"),
          WORDS, NOT_COVERED),
    # -- Sync -----------------------------------------------------------------------------------------------------
    Entry("sync: paired devices, skills a peer offered, and every record's latest version in the change feed",
          ("veilid_change_feed.db", "veilid_deferred.db", "veilid_peers.db"),
          ("veilid/change_feed.py", "veilid/deferred_ledger.py", "veilid/peers.py"), DRAWN, NOT_COVERED),
    # -- Counts, measures and models ------------------------------------------------------------------------------
    Entry("the onion's counters (counts only)", ("*",), ("memory/librarian.py",), IDS, NOT_COVERED),
    Entry("context ledger (what each window held, counted)", ("context_ledger.db",), ("context_ledger.py",), IDS,
          NOT_COVERED),
    Entry("performance metrics store", ("performance_metrics.db",), ("performance_monitor.py",), IDS, NOT_COVERED),
    Entry("telemetry/analytics stores", ("analytics.db", "telemetry_history.db"),
          ("analytics.py", "telemetry_history.py"), IDS, NOT_COVERED),
    Entry("benchmark stores (history, results)", ("benchmark_history.db", "benchmark_results.db"),
          ("benchmark_history.py", "benchmark_runner.py", "benchmark_judge.py", "benchmark_recommendations.py"),
          DRAWN, NOT_COVERED),
    Entry("fine-tune variants registry", ("fine_tune_variants.db",), ("fine_tune_tracker.py",), DRAWN,
          NOT_COVERED),
    Entry("agent evaluation stores", ("agent_eval_results.db", "chat_eval_traces", "redteam_seeds.json"),
          ("agent_eval/store.py", "agent_eval/chat_runner.py", "redteam/generator.py"), IDS, NOT_COVERED),
    # -- The account ----------------------------------------------------------------------------------------------
    Entry("settings and encryption key salts", ("auth.db",),
          ("user_isolation.py", "user_key_manager.py"), WORDS, REMOVED,
          failures=("settings", "encryption key salts")),
    Entry("plugin settings", ("plugin_user_config.db",), ("plugin_user_config.py",), WORDS, REMOVED,
          failures=("plugin settings",)),
    Entry("plugin reviews", ("plugin_reviews.db",), ("plugin_reviews.py",), WORDS, REMOVED,
          failures=("plugin reviews",)),
    Entry("the account itself and its second factor (deleting it is its own act)",
          ("data/auth.db", "auth_2fa.db"), ("auth.py", "allium/service.py", "auth_2fa.py"), SECRETS,
          NOT_COVERED),
    # -- Trails kept by design ------------------------------------------------------------------------------------
    Entry("admin audit log (accountability trail)", ("admin_audit.db",), ("admin_audit.py",), IDS, KEPT),
    Entry("audit chain (tamper-evident security trail)", ("audit_chain.db", "data/audit_chain.db"),
          ("signed_audit_log.py",), IDS, KEPT),
    # -- Nothing of the user's ------------------------------------------------------------------------------------
    Entry("plugin catalogue and manifests", ("plugin_index.db", "plugins.db", "plugins"),
          ("plugin_index.py", "plugin_manifest.py", "plugin_installer.py", "plugin_loader.py", "plugin_template.py"),
          NOTHING, NOT_THE_USERS),
    Entry("shipped definitions (the system presets)", ("system_presets.yaml",), ("system_presets.py",), NOTHING,
          NOT_THE_USERS),
    Entry("resource governor (machine measurements)",
          ("resource_governor.db", "opti_oignon/data/resource_governor.db"), ("resource_governor.py",), NOTHING,
          NOT_THE_USERS),
    Entry("install state (model provenance and aliases, plugin allow-list, security mode lock, search kill switch, "
          "tuning and speculative decoding results)",
          ("model_provenance.json", "model_aliases.json", "plugin_allowlist.json", ".security_mode_lock",
           ".search_killswitch.json", "tuner_results.json", "speculative_stats.json", ".legacy_memory_migrated"),
          ("model_provenance.py", "model_lifecycle.py", "plugin_allowlist.py", "security_mode.py",
           "search_killswitch.py", "auto_tuner.py", "speculative_decoding.py", "memory/migration.py"), NOTHING,
          NOT_THE_USERS),
    # The install's key file is the loader's, the mode's and the old recovery codes' to name, and nobody else's (ky7
    # of the key's contracts): it is not spelled here.
    Entry("the install's keys (the stores that remain are sealed or signed with them, the server's certificates)",
          (".pqc_keypair", ".veilid_signing_key", "tls", "ca.crt", "ca.key", "ca.key.enc", "client.crt", "clients",
           "crl.pem", "metadata.json", "server.crt", "server.key", "*"),
          ("pqc_signatures.py", "veilid/signing.py", "tls_manager.py", "network_bind_guard.py", "encryption.py"),
          SECRETS, NOT_THE_USERS),
)

# Names the package spells that are not stores: with why, and the modules that spell them.
NOT_STORES: dict[str, tuple[str, tuple[str, ...]]] = {
    "data/app.db": ("an example in a docstring", ("connection_pool.py",)),
    "data/my.db": ("an example in a docstring", ("db_utils.py",)),
    "data/rag/jobs.db": ("an example in a docstring", ("rag/pool_integration.py",)),
    "/path/to/rag_ingest_jobs.db": ("an example in a docstring", ("rag/pool_integration.py",)),
    "bench_cache.db": ("a throwaway file in a temporary folder", ("performance_benchmark.py",)),
    "bench_conv.db": ("a throwaway file in a temporary folder", ("performance_benchmark.py",)),
    "bench_mem.db": ("a throwaway file in a temporary folder", ("performance_benchmark.py",)),
    "bench_sem.db": ("a throwaway file in a temporary folder", ("performance_benchmark.py",)),
    "test_conversations.db": ("the command line's self-test, in a temporary folder", ("conversation.py",)),
    "needle_store.db": ("the fidelity evaluation's planted conversations, in the caller's folder",
                        ("agent_eval/fidelity.py",)),
    "Thumbs.db": ("a pattern the ingestion skips", ("rag/config.py",)),
    "memory.db": ("a stale name in the user id migration's list: no module creates it", ("user_isolation.py",)),
    "onion.db": ("an example name: the onion's store is the file the local configuration names",
                 ("memory/librarian.py",)),
    "logs": ("the knowledge base's log folder, which nothing writes", ("rag/config.py",)),
    "rag": ("the knowledge base's folder: its stores are listed by name",
            ("rag_store.py", "project_context.py", "rag/batch_ingest.py", "rag/config.py", "rag_dashboard.py")),
    ".db": ("a suffix: the garden's beings' files end in it, and the encryption tool finds every database by it",
            ("allium/store.py", "db_encryption.py")),
}
