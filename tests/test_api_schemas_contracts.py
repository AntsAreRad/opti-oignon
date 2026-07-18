#!/usr/bin/env python3
"""Contracts for the REST schema layer (``opti_oignon/api/schemas.py``).

The module under contract is the request/response boundary of the whole API:
322 flat pydantic v2 models and nothing else -- two imports (``typing.Any``,
``pydantic.BaseModel``/``Field``), no validators, no ``model_config``, no
module state, no I/O, no threads. What the file DOES is therefore almost
entirely what pydantic does with its declarations, and that is exactly what
these contracts pin: the observable validation, defaults, coercion and
serialization behaviour under the exact pydantic in the reference
environment -- never the shape of the code.

Three census contracts sweep every model programmatically and compare the
result against a literal snapshot: the full map of required fields (the
API's 422 gates), the full map of numeric/length/pattern constraints (there
are exactly three constrained classes), and the named security-relevant
defaults (approval, degraded-mode, network, signature and PII postures whose
safe values are load-bearing for the platform's fail-secure design). A quiet
edit anywhere in the 3248 lines lands in at least one census.

Isolation: the shared window loads the module from its file with ``ollama``
proven unreachable and ZERO seeded project modules -- the module is
import-pure, and q1 proves that purity rather than assuming it. The module
itself is left byte-identical by this suite.

Two recorded observations are pinned as they stand rather than judged here:
the ``bound_conversation_id`` annotation appears twice in
``SandboxSessionInfo`` (the second one is dead -- pydantic collapses them
into a single field at its first position), and ``HealthDashboard.version``
still defaults to the historical ``"1.6.6"`` string while the package
register says otherwise. Both are pinned exactly so any change surfaces.
"""

import sys
import warnings

import pytest
from _isolation import isolate, source
from pydantic import BaseModel, ValidationError

_SCHEMAS = "opti_oignon.api.schemas"


def _load():
    """Open the window and load the schema module from its source file."""
    loaded, restore = isolate(
        targets={_SCHEMAS: source("api", "schemas.py")},
        blocked=("ollama",),
        packages=("opti_oignon.api",),
    )
    return loaded[_SCHEMAS], restore


def _models(mod):
    """Every pydantic model class defined by the module, in definition order."""
    return {
        k: v
        for k, v in vars(mod).items()
        if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel
    }


def _rejects(cls, **kwargs):
    with pytest.raises(ValidationError):
        cls(**kwargs)


# --- Census snapshots -------------------------------------------------------
# Generated from the module under the reference pydantic and frozen here.
# Keys are model names in definition order; values are the tuples of required
# field names, in field order. A model with no required fields has no entry.

REQUIRED = {
    'ConversationSummary': ('id', 'title'),
    'ConversationDetail': ('id', 'title'),
    'ConversationRename': ('title',),
    'MessageItem': ('role', 'content'),
    'ModelInfo': ('name',),
    'EffectiveModelResponse': ('model', 'source'),
    'ErrorResponse': ('detail',),
    'ChatRequest': ('message',),
    'ChatToken': ('type',),
    'ChatResponse': ('conversation_id', 'content', 'model'),
    'ChatCancelRequest': ('conversation_id',),
    'ChatRetryRequest': ('conversation_id',),
    'ArtifactInfo': ('id', 'artifact_type', 'title', 'language', 'created_at'),
    'ArtifactContent': ('id', 'artifact_type', 'title', 'content', 'language', 'created_at'),
    'ArtifactExport': ('filename', 'content'),
    'CodeExecuteRequest': ('code',),
    'CodeExecuteResponse': ('success',),
    'CodeBlocksRequest': ('text',),
    'CodeBlockInfo': ('code', 'language', 'start_pos', 'end_pos'),
    'CodeBlocksResponse': ('blocks',),
    'MemoryFactSchema': ('id', 'fact', 'category'),
    'MemoryAddRequest': ('fact',),
    'MemoryExtractResponse': ('conversation_id',),
    'MemoryRecordSchema': ('id', 'text', 'category'),
    'NoteSchema': ('id', 'title'),
    'NoteCreateRequest': ('title',),
    'NoteUpdateAppendRequest': ('update_blob_b64',),
    'NoteUpdateRecordSchema': ('id', 'note_id', 'seq', 'update_blob_b64'),
    'NoteActionRequest': ('action',),
    'NoteActionResultSchema': ('action', 'ok'),
    'AttachmentSchema': ('id', 'note_id', 'kind'),
    'TranscriptionResultSchema': ('attachment_id', 'ok'),
    'CaptionResultSchema': ('attachment_id', 'ok'),
    'BenchmarkResultSchema': ('name',),
    'FileUploadResponse': ('filename', 'size_bytes', 'content', 'extension'),
    'ImageUploadResponse': ('filename', 'size_bytes', 'base64_data', 'mime_type'),
    'ExportResponse': ('conversation_id', 'format', 'content'),
    'PresetInfo': ('id', 'name'),
    'PresetCreate': ('id', 'name'),
    'PresetMatchResult': ('preset',),
    'PresetDuplicateRequest': ('new_id', 'new_name'),
    'PipelineStepSchema': ('name', 'agent'),
    'PipelineInfo': ('id', 'name'),
    'PipelineCreate': ('id', 'name'),
    'PipelineDuplicateRequest': ('new_id',),
    'ModelProfileInfo': ('name',),
    'RoutingReasonInfo': ('model',),
    'SettingValue': ('key',),
    'SettingSetRequest': ('value',),
    'ThemeConfigRequest': ('accent_hue',),
    'ThemeConfigResponse': ('accent_hue', 'accent_saturation', 'secondary_hue', 'secondary_saturation', 'mode'),
    'ThemePresetResponse': ('id', 'name', 'accent_hue', 'accent_saturation', 'secondary_hue', 'secondary_saturation'),
    'CustomPresetCreateRequest': ('name', 'accent_hue'),
    'CustomPresetImportRequest': ('presets',),
    'CustomPresetsExportResponse': ('presets_json',),
    'ConsensusRequest': ('message',),
    'ConsensusModelResponseSchema': ('model', 'content'),
    'ConsensusResponse': ('strategy',),
    'CascadeTierSchema': ('name', 'model'),
    'CascadeTierResultSchema': ('tier_name', 'model'),
    'CascadeResultSchema': ('final_response', 'model_used', 'tier_index', 'tier_name', 'score'),
    'CascadeTestRequest': ('query',),
    'CascadeTestResponse': ('result',),
    'SpeculativeResultSchema': ('final_response',),
    'SpeculativeTestRequest': ('query',),
    'SpeculativeTestResponse': ('result',),
    'DriftEntry': ('model', 'metric', 'baseline_value', 'recent_value', 'change_ratio', 'is_drifted', 'direction'),
    'RecommendationEntry': ('model', 'metric', 'message', 'severity'),
    'SandboxCreateResponse': ('session_id', 'workspace_path', 'isolation_backend'),
    'SandboxInjectRequest': ('session_id',),
    'SandboxInjectResponse': ('session_id',),
    'SandboxFileEntry': ('path',),
    'SandboxFilesResponse': ('session_id',),
    'SandboxExecuteRequest': ('session_id', 'tool_name'),
    'SandboxExecuteResponse': ('session_id', 'tool_name'),
    'SandboxSessionInfo': ('session_id', 'workspace_path'),
    'SandboxDestroyResponse': ('session_id',),
    'SandboxStopResponse': ('session_id',),
    'SandboxBindRequest': ('conversation_id', 'session_id'),
    'SandboxBindingResponse': ('conversation_id',),
    'SandboxUploadRefused': ('name', 'reason'),
    'SandboxUploadResponse': ('session_id',),
    'HostBrowseEntry': ('name', 'type'),
    'HostBrowseResponse': ('path',),
    'SandboxCloneRequest': ('src_path',),
    'SandboxCloneResponse': ('session_id', 'dest', 'cloned_root'),
    'SandboxDiffEntry': ('path', 'kind'),
    'SandboxDiffResponse': ('session_id',),
    'SandboxConfirmDeletionsRefused': ('path', 'reason'),
    'SandboxConfirmDeletionsResponse': ('session_id',),
    'SandboxApplyRequest': ('diff_hash',),
    'SandboxApplyEntry': ('path', 'action'),
    'SandboxApplyRefusedEntry': ('path', 'error'),
    'SandboxApplyResponse': ('session_id', 'target'),
    'SandboxNetworkToggleRequest': ('enabled',),
    'SandboxNetworkToggleResponse': ('session_id',),
    'SandboxProvisionRequest': ('requirements_path',),
    'SandboxProvisionResponse': ('session_id',),
    'SandboxPreviewResponse': ('session_id', 'path'),
    'SandboxApproveRequest': ('paths',),
    'SandboxApproveResponse': ('session_id',),
    'SandboxCopyOutEntry': ('src_path', 'dest_path'),
    'SandboxCopyOutResponse': ('session_id',),
    'SandboxRejectResponse': ('session_id',),
    'SandboxApprovalInfoResponse': ('session_id',),
    'QuickSandboxToggleRequest': ('enabled',),
    'QuickSandboxSessionInfo': ('session_id',),
    'QuickSandboxTTLRequest': ('auto_destroy_minutes',),
    'ChatCodingToggleRequest': ('enabled',),
    'ChatCodingSessionInfo': ('session_id', 'conversation_id'),
    'CodingTaskRequest': ('task',),
    'CodingCheckpointRequest': ('decision',),
    'SystemPresetInfo': ('id', 'name'),
    'HumanizerRewriteRequest': ('text',),
    'HumanizerFeedbackRequest': ('comparison_id', 'winner'),
    'BenchmarkV2CustomProfileCreate': ('name',),
    'BackendActivateRequest': ('name',),
    'GGUFDownloadRequest': ('url',),
    'TunerRunRequest': ('model_name',),
    'ModelPullRequest': ('model_name',),
    'ModelDeleteRequest': ('model_name',),
    'ModelAliasRequest': ('alias', 'model_name'),
}

# Every constrained field in the module, with its exact constraint set.
# Anything appearing here that is not in the module, or in the module and not
# here, is a drift.

CONSTRAINED = {
    ('ThemeConfigRequest', 'accent_hue'): ('ge=0', 'le=359'),
    ('ThemeConfigRequest', 'accent_saturation'): ('ge=0', 'le=100'),
    ('ThemeConfigRequest', 'secondary_hue'): ('ge=-1', 'le=359'),
    ('ThemeConfigRequest', 'secondary_saturation'): ('ge=0', 'le=100'),
    ('ThemeConfigRequest', 'accent_lightness_offset'): ('ge=-50', 'le=50'),
    ('ThemeConfigRequest', 'secondary_lightness_offset'): ('ge=-50', 'le=50'),
    ('ThemeConfigRequest', 'accent_warmth'): ('ge=-30', 'le=30'),
    ('ThemeConfigRequest', 'secondary_warmth'): ('ge=-30', 'le=30'),
    ('ThemeConfigRequest', 'mode'): ("pattern='^(dark|light)$'",),
    ('CustomPresetCreateRequest', 'name'): ('max_length=50',),
    ('CustomPresetCreateRequest', 'description'): ('max_length=200',),
    ('CustomPresetCreateRequest', 'accent_hue'): ('ge=0', 'le=359'),
    ('CustomPresetCreateRequest', 'accent_saturation'): ('ge=0', 'le=100'),
    ('CustomPresetCreateRequest', 'secondary_hue'): ('ge=-1', 'le=359'),
    ('CustomPresetCreateRequest', 'secondary_saturation'): ('ge=0', 'le=100'),
    ('CustomPresetCreateRequest', 'accent_lightness_offset'): ('ge=-50', 'le=50'),
    ('CustomPresetCreateRequest', 'secondary_lightness_offset'): ('ge=-50', 'le=50'),
    ('CustomPresetCreateRequest', 'accent_warmth'): ('ge=-30', 'le=30'),
    ('CustomPresetCreateRequest', 'secondary_warmth'): ('ge=-30', 'le=30'),
    ('QuickSandboxTTLRequest', 'auto_destroy_minutes'): ('gt=0', 'le=1440'),
}

# Named security-relevant declaration defaults. These values are the safe
# side of each toggle: human approval before persistence, signature policy
# honoured, degraded isolation refused, network closed, PII scrubbing on,
# failure codes that read as failure.
SAFE_DEFAULTS = {
    ("TranscriptionRequest", "approve"): False,
    ("CaptionRequest", "approve"): False,
    ("NoteSchema", "mobile_allowed"): False,
    ("NoteUpdateRequest", "mobile_allowed"): None,
    ("BackupImportRequest", "allow_unsigned"): False,
    ("BackupImportRequest", "strategy"): "merge",
    ("BackupPreviewRequest", "strategy"): "merge",
    ("SandboxCreateRequest", "allow_degraded"): False,
    ("CodingTaskRequest", "allow_degraded"): False,
    ("SandboxStatusResponse", "network_allowed"): False,
    ("SandboxSessionInfo", "network_enabled"): False,
    ("SandboxSessionInfo", "owner_user_id"): "local",
    ("SandboxNetworkToggleResponse", "network_enabled"): False,
    ("ModelPullRequest", "insecure"): False,
    ("SandboxProvisionResponse", "return_code"): -1,
    ("CodingTestResultResponse", "return_code"): -1,
    ("ProxyConfigRequest", "mode"): "off",
    ("ProxyConfigResponse", "mode"): "off",
    ("ProxyConfigResponse", "pii_sanitization_enabled"): True,
    ("SandboxExecuteResponse", "blocked"): False,
}


# --- Load posture -----------------------------------------------------------


def test_q1_window_holds_and_the_module_is_import_pure():
    """The file loads with zero seeded project modules and pulls nothing.

    Inside the window, ``ollama`` is neutralised (proven unreachable by the
    window before the module ran), the only project entries are the two
    stand-in packages and the target, and the module's public namespace is
    exactly its 322 models plus ``Any`` and ``Field`` (``BaseModel`` is
    counted with the model classes). Nothing else resolves, so the module
    demonstrably needs nothing else.
    """
    S, restore = _load()
    try:
        assert sys.modules.get("ollama", "absent") is None
        project = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "opti_oignon"}
        for name, mod in project.items():
            assert mod is None or name in (
                "opti_oignon",
                "opti_oignon.api",
                _SCHEMAS,
            ), f"unexpected live project module inside the window: {name}"
        classes = _models(S)
        assert len(classes) == 322
        residue = {
            k
            for k, v in vars(S).items()
            if not k.startswith("_")
            and not (isinstance(v, type) and issubclass(v, BaseModel))
        }
        assert residue == {"Any", "Field"}
    finally:
        restore()


def test_q2_the_module_loads_silently():
    """Building all 322 models emits no warning under the reference pydantic.

    The many ``model_*`` field names stay quiet; a future pydantic or a new
    declaration that starts warning at import would surface here.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, restore = _load()
        restore()
    assert [str(w.message) for w in caught] == []


# --- Pydantic v2 posture (the module opts into every default) ---------------


def test_q3_unknown_request_fields_are_ignored_not_rejected():
    S, restore = _load()
    try:
        m = S.ChatRequest(message="x", totally_unknown_field=1)
        assert "totally_unknown_field" not in m.model_dump()
    finally:
        restore()


def test_q4_lax_coercion_arms():
    """v2 lax mode as the module actually exposes it.

    A str field REJECTS an int (no v1-style stringification); a float field
    accepts an int; a bool field accepts "true"; an int field accepts "5".
    """
    S, restore = _load()
    try:
        _rejects(S.ChatRequest, message=123)
        assert S.ChatRequest(message="x", temperature=1).temperature == 1.0
        assert S.QuickSandboxToggleRequest(enabled="true").enabled is True
        assert S.CodeExecuteRequest(code="c", timeout="5").timeout == 5
    finally:
        restore()


def test_q5_mutable_defaults_are_per_instance():
    """Bare ``[]`` / ``{}`` defaults and lambda factories never share state.

    pydantic deep-copies declaration defaults per instance, so the module's
    mixed style (bare literals next to ``default_factory``) is behaviourally
    uniform. Machinery posture: pinned, with no way to break it from the
    module's own text.
    """
    S, restore = _load()
    try:
        a = S.SandboxDiffResponse(session_id="a")
        b = S.SandboxDiffResponse(session_id="b")
        a.entries.append("poke")
        assert b.entries == []
        c, d = S.BackupImportRequest(), S.BackupImportRequest()
        c.backup["k"] = 1
        assert d.backup == {}
        e, f = S.ProxyConfigResponse(), S.ProxyConfigResponse()
        assert e.retry_backoff is not f.retry_backoff
    finally:
        restore()


def test_q6_assignment_is_not_validated():
    """No model opts into validate_assignment: a bound field takes any value."""
    S, restore = _load()
    try:
        m = S.ThemeConfigRequest(accent_hue=1)
        m.accent_hue = 999
        assert m.accent_hue == 999
    finally:
        restore()


def test_q7_any_without_default_is_required_yet_nullable():
    S, restore = _load()
    try:
        _rejects(S.SettingSetRequest)
        assert S.SettingSetRequest(value=None).value is None
    finally:
        restore()


def test_q8_str_or_list_union_keeps_the_input_shape():
    S, restore = _load()
    try:
        u = S.BenchmarkV2AutoTriggerConfigUpdate
        assert u(trigger_models="all_new").trigger_models == "all_new"
        assert u(trigger_models=["a", "b"]).trigger_models == ["a", "b"]
        assert u().trigger_models is None
    finally:
        restore()


def test_q9_round_trip_and_nested_dict_coercion():
    """dump/validate is lossless and a dict payload builds the nested model."""
    S, restore = _load()
    try:
        m = S.ChatRequest(message="hello", exec_pipeline="p1")
        again = S.ChatRequest.model_validate(m.model_dump())
        assert again.exec_pipeline == "p1" and again.use_presets is True
        pm = S.PresetMatchResult(preset={"id": "p", "name": "n"})
        assert isinstance(pm.preset, S.PresetInfo)
        assert pm.preset.temperature == 0.5
        _rejects(S.CascadeTestResponse)  # the nested result is required
    finally:
        restore()


# --- The real constraint sites ----------------------------------------------


def test_q10_theme_configuration_bounds():
    S, restore = _load()
    try:
        S.ThemeConfigRequest(accent_hue=359)
        _rejects(S.ThemeConfigRequest, accent_hue=360)
        _rejects(S.ThemeConfigRequest, accent_hue=-1)
        _rejects(S.ThemeConfigRequest, accent_hue=0, accent_saturation=101)
        _rejects(S.ThemeConfigRequest, accent_hue=0, mode="blue")
        _rejects(S.ThemeConfigRequest)  # accent_hue is the one required knob
        assert S.ThemeConfigRequest(accent_hue=0).mode == "dark"
    finally:
        restore()


def test_q11_custom_preset_length_caps():
    S, restore = _load()
    try:
        S.CustomPresetCreateRequest(name="n" * 50, accent_hue=1)
        _rejects(S.CustomPresetCreateRequest, name="n" * 51, accent_hue=1)
        _rejects(
            S.CustomPresetCreateRequest, name="n", description="d" * 201, accent_hue=1
        )
        ok = S.CustomPresetCreateRequest(name="n", description="d" * 200, accent_hue=1)
        assert ok.description == "d" * 200
    finally:
        restore()


def test_q12_quick_sandbox_ttl_window():
    S, restore = _load()
    try:
        assert S.QuickSandboxTTLRequest(auto_destroy_minutes=1440).auto_destroy_minutes == 1440
        assert S.QuickSandboxTTLRequest(auto_destroy_minutes="60").auto_destroy_minutes == 60
        _rejects(S.QuickSandboxTTLRequest, auto_destroy_minutes=0)
        _rejects(S.QuickSandboxTTLRequest, auto_destroy_minutes=1441)
        _rejects(S.QuickSandboxTTLRequest)
    finally:
        restore()


def test_q13_coding_gates_are_explicit():
    """The coding task text and the checkpoint decision cannot be omitted."""
    S, restore = _load()
    try:
        _rejects(S.CodingTaskRequest)
        t = S.CodingTaskRequest(task="t")
        assert t.project_path is None and t.model is None
        _rejects(S.CodingCheckpointRequest)
        c = S.CodingCheckpointRequest(decision="approve")
        assert c.modified_plan is None
    finally:
        restore()


# --- Named security defaults ------------------------------------------------


def test_q14_media_and_notes_default_to_review_not_persistence():
    S, restore = _load()
    try:
        assert S.TranscriptionRequest().approve is False
        assert S.CaptionRequest().approve is False
        assert S.NoteSchema(id="i", title="t").mobile_allowed is False
        u = S.NoteUpdateRequest()
        assert (
            u.title is None
            and u.body_crdt_b64 is None
            and u.tags is None
            and u.pinned is None
            and u.mobile_allowed is None
            and u.checkpoint_watermark is None
        )
    finally:
        restore()


def test_q15_backup_import_defaults_keep_the_signature_policy():
    S, restore = _load()
    try:
        bi = S.BackupImportRequest()
        assert bi.allow_unsigned is False
        assert bi.strategy == "merge"
        assert bi.backup == {}
        assert S.BackupPreviewRequest().strategy == "merge"
    finally:
        restore()


def test_q16_degraded_isolation_is_opt_in():
    S, restore = _load()
    try:
        cr = S.SandboxCreateRequest()
        assert cr.allow_degraded is False
        assert cr.session_id is None and cr.label == "" and cr.timeout is None
        assert S.CodingTaskRequest(task="t").allow_degraded is False
    finally:
        restore()


def test_q17_network_posture_defaults_closed():
    S, restore = _load()
    try:
        st = S.SandboxStatusResponse()
        assert st.network_allowed is False
        assert st.available is False and st.enabled is False
        assert st.max_sessions == 5
        si = S.SandboxSessionInfo(session_id="s", workspace_path="/w")
        assert si.network_enabled is False
        assert si.owner_user_id == "local"
        assert si.approval_state == "pending"
        _rejects(S.SandboxNetworkToggleRequest)  # flipping requires an explicit bool
        assert S.SandboxNetworkToggleResponse(session_id="s").network_enabled is False
    finally:
        restore()


def test_q18_failure_codes_default_to_failure():
    S, restore = _load()
    try:
        assert S.SandboxProvisionResponse(session_id="s").return_code == -1
        tr = S.CodingTestResultResponse()
        assert tr.return_code == -1 and tr.passed is False
        _rejects(S.SandboxProvisionRequest)  # requirements_path cannot be omitted
        assert S.SandboxProvisionRequest(requirements_path="r.txt").venv_dir == ".venv"
    finally:
        restore()


def test_q19_model_pull_and_gguf_download_postures():
    S, restore = _load()
    try:
        assert S.ModelPullRequest(model_name="m").insecure is False
        _rejects(S.GGUFDownloadRequest)  # the url is required
        g = S.GGUFDownloadRequest(url="https://example.invalid/m.gguf")
        assert g.expected_sha256 is None
        assert g.filename is None and g.target_dir is None
    finally:
        restore()


def test_q20_proxy_defaults_off_with_pii_scrubbing_on():
    S, restore = _load()
    try:
        req = S.ProxyConfigRequest()
        assert req.mode == "off" and req.pii_sanitization_enabled is None
        cfg = S.ProxyConfigResponse()
        assert cfg.mode == "off"
        assert cfg.pii_sanitization_enabled is True
        assert cfg.proxy_timeout == 15 and cfg.max_retries == 3
    finally:
        restore()


def test_q21_execution_reports_default_unblocked_and_honest():
    S, restore = _load()
    try:
        ex = S.SandboxExecuteResponse(session_id="s", tool_name="t")
        assert ex.blocked is False and ex.block_reason == ""
        assert ex.timed_out is False and ex.result == ""
        _rejects(S.SandboxExecuteRequest, session_id="s")  # tool_name required
        assert S.SandboxExecuteRequest(session_id="s", tool_name="t").arguments == {}
        assert S.CodeExecuteRequest(code="c").language == "python"
    finally:
        restore()


# --- Named 422 gates ---------------------------------------------------------


def test_q22_apply_demands_the_reviewed_diff_hash():
    S, restore = _load()
    try:
        _rejects(S.SandboxApplyRequest, target_dir="x")
        ap = S.SandboxApplyRequest(diff_hash="h")
        assert ap.target_dir is None
        d = S.SandboxDiffResponse(session_id="s")
        assert d.baseline_present is False and d.entries == []
        assert d.diff_hash == "" and d.approved_paths == []
        assert d.confirmed_deletions == []
        assert S.SandboxConfirmDeletionsRequest().paths == []
    finally:
        restore()


def test_q23_copy_out_approval_names_its_paths():
    S, restore = _load()
    try:
        _rejects(S.SandboxApproveRequest, dest_dir="d")
        a = S.SandboxApproveRequest(paths=["out.txt"])
        assert a.dest_dir is None
        assert S.SandboxApproveResponse(session_id="s").approval_state == "pending"
        assert S.SandboxRejectResponse(session_id="s").approval_state == "rejected"
    finally:
        restore()


def test_q24_note_update_appends_carry_their_blob():
    S, restore = _load()
    try:
        _rejects(S.NoteUpdateAppendRequest)
        rec_required = tuple(
            n for n, f in S.NoteUpdateRecordSchema.model_fields.items() if f.is_required()
        )
        assert rec_required == ("id", "note_id", "seq", "update_blob_b64")
        r = S.NoteUpdateRecordSchema(id=1, note_id="n", seq=1, update_blob_b64="b")
        assert r.author_device is None and r.created_at == ""
    finally:
        restore()


def test_q25_chat_request_surface_is_exactly_its_seventeen_fields():
    S, restore = _load()
    try:
        fields = list(S.ChatRequest.model_fields)
        assert len(fields) == 17
        assert set(fields) == {
            "conversation_id",
            "message",
            "model",
            "preset",
            "temperature",
            "use_presets",
            "think",
            "web_search",
            "images",
            "consensus",
            "consensus_models",
            "consensus_strategy",
            "self_correct",
            "optimize",
            "quick_sandbox",
            "chat_coding",
            "exec_pipeline",
        }
        _rejects(S.ChatRequest)  # message is the single required field
        m = S.ChatRequest(message="x")
        assert m.use_presets is True and m.exec_pipeline is None
        _rejects(S.ChatRetryRequest)
        assert S.ChatRetryRequest(conversation_id="c").model is None
        _rejects(S.ChatCancelRequest)
    finally:
        restore()


# --- Censuses ----------------------------------------------------------------


def test_q26_required_field_census_matches_the_snapshot():
    """The complete 422 map: which fields each model refuses to go without."""
    S, restore = _load()
    try:
        actual = {}
        for name, cls in _models(S).items():
            req = tuple(n for n, f in cls.model_fields.items() if f.is_required())
            if req:
                actual[name] = req
        assert actual == REQUIRED
    finally:
        restore()


def test_q27_constraint_census_matches_the_snapshot():
    """Every ge/le/gt/lt/length/pattern site in the module, and no other."""
    S, restore = _load()
    try:
        actual = {}
        for name, cls in _models(S).items():
            for fname, f in cls.model_fields.items():
                found = []
                for m in f.metadata:
                    for attr in ("ge", "le", "gt", "lt", "max_length", "min_length", "pattern"):
                        v = getattr(m, attr, None)
                        if v is not None:
                            found.append(f"{attr}={v!r}")
                if found:
                    actual[(name, fname)] = tuple(found)
        assert actual == CONSTRAINED
    finally:
        restore()


def test_q28_security_default_census_matches_the_snapshot():
    S, restore = _load()
    try:
        classes = _models(S)
        actual = {
            key: classes[key[0]].model_fields[key[1]].default for key in SAFE_DEFAULTS
        }
        assert actual == SAFE_DEFAULTS
    finally:
        restore()


def test_q29_no_model_carries_validators_or_config():
    """Structural sweep: 322 plain declarations, nothing decorated, no config.

    Every behaviour this suite pins therefore comes from the declarations
    themselves plus stock pydantic -- a validator or a ``model_config``
    appearing anywhere would change what the other contracts mean.
    """
    S, restore = _load()
    try:
        slots = (
            "validators",
            "field_validators",
            "root_validators",
            "field_serializers",
            "model_serializers",
            "model_validators",
            "computed_fields",
        )
        for name, cls in _models(S).items():
            deco = cls.__pydantic_decorators__
            for slot in slots:
                assert not getattr(deco, slot), f"{name} carries {slot}"
            assert cls.model_config == {}, f"{name} carries model_config"
    finally:
        restore()


# --- Recorded observations, pinned as they stand -----------------------------


def test_q30_duplicated_annotation_collapses_to_one_early_field():
    """``SandboxSessionInfo`` declares ``bound_conversation_id`` twice.

    Python's annotation dict keeps the FIRST position and the later duplicate
    is dead text: pydantic builds exactly one field, sitting right after
    ``session_id``, defaulting to None.
    """
    S, restore = _load()
    try:
        names = list(S.SandboxSessionInfo.model_fields)
        assert names.count("bound_conversation_id") == 1
        assert names.index("bound_conversation_id") == 1
        m = S.SandboxSessionInfo(session_id="s", workspace_path="/w")
        assert m.bound_conversation_id is None
    finally:
        restore()


def test_q31_health_dashboard_still_defaults_to_the_historical_version():
    """Pinned as found: the declaration default lags the package register.

    The route is expected to overwrite it; the pin exists so the day the
    default moves (or starts leaking through) is a visible day.
    """
    S, restore = _load()
    try:
        assert S.HealthDashboard().version == "1.6.6"
    finally:
        restore()


def test_q32_factory_defaults_carry_their_exact_values():
    S, restore = _load()
    try:
        assert S.ProxyConfigResponse().retry_backoff == [2, 5, 10]
        p = S.BenchmarkV2CustomProfileCreate(name="p")
        assert p.expected_length_range == [10, 600]
        assert p.timeout == 45 and p.max_response_tokens == 800
        space = S.ParameterSpaceSchema()
        assert space.batch_size == [512, 1024, 2048, 4096]
        assert space.ubatch_size == [256, 512, 1024]
        assert space.threads == [2, 4, 6, 8]
        assert space.flash_attention == [True, False]
    finally:
        restore()
