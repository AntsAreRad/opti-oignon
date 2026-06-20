#!/usr/bin/env python3
"""
Pydantic v2 schemas for the Opti-Oignon REST API.

Defines request/response models for all API endpoints.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

# -- Conversations --

class ConversationSummary(BaseModel):
    """Resume of ae conversation (sans messages)."""
    id: str
    title: str
    created_at: str | None = None
    updated_at: str | None = None
    message_count: int = 0
    model: str | None = None
    task_type: str | None = None
    preset: str | None = None


class ConversationDetail(BaseModel):
    """Detail complet of ae conversation avec messages."""
    id: str
    title: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    model: str | None = None
    task_type: str | None = None
    preset: str | None = None
    message_count: int = 0
    total_tokens: int = 0


class ConversationCreate(BaseModel):
    """Corps de request pour creer une conversation."""
    title: str | None = None
    model: str | None = None
    preset: str | None = None


class ConversationRename(BaseModel):
    """Corps de request pour renommer une conversation."""
    title: str


# -- Messages --

class MessageItem(BaseModel):
    """Un message individuel."""
    id: int | None = None
    role: str
    content: str
    timestamp: str | None = None
    model: str | None = None
    token_estimate: int = 0


# -- Models --

class ModelInfo(BaseModel):
    """Informations sur un model Ollama."""
    name: str
    size: str | None = None
    modified_at: str | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None
    mtp_capable: bool = False


class ModelListResponse(BaseModel):
    """Reponse de la liste des models."""
    models: list[ModelInfo] = Field(default_factory=list)
    count: int = 0


class EffectiveModelResponse(BaseModel):
    """Response of the resolved effective model."""
    model: str
    source: str


# -- Errors --

class ErrorResponse(BaseModel):
    """Reponse d'erreur standard."""
    detail: str


# -- Chat --

class ChatRequest(BaseModel):
    """Request de chat via WebSocket."""
    conversation_id: str | None = None  # None = creer new conv
    message: str
    model: str | None = None  # Force model
    preset: str | None = None
    temperature: float | None = None
    use_presets: bool = True
    # S42: Controles de chat
    think: bool | None = None        # None = auto, True = force, False = disable
    web_search: bool | None = None   # None = auto, True = force, False = disable
    # S48: Images pour vision multimodale
    images: list[str] | None = None  # Liste de base64 (sans prefixe data:...)
    # S50: Consensus multi-model
    consensus: bool | None = None             # None = auto, True = force
    consensus_models: list[str] | None = None  # Models specifiques
    consensus_strategy: str | None = None      # best_of_n, weighted_vote, llm_merge
    # S51: Auto-correction
    self_correct: bool | None = None          # None = auto, True = force
    # S117: Quick sandbox mode for chat code execution
    quick_sandbox: bool | None = None         # None = use config, True = force on
    # S118: Chat coding agent mode (conversational coding with persistent sandbox)
    chat_coding: bool | None = None           # None = use config, True = force on
    # S216 (PIP-06): execution pipeline selection. The frontend has sent this
    # field since S53; the model never carried it, so it was silently dropped.
    exec_pipeline: str | None = None          # None = plain chat, id = run that pipeline

class ChatToken(BaseModel):
    """Token de streaming sent via WebSocket."""
    type: str  # "token", "thinking", "done", "error", "metadata"
    content: str = ""
    metadata: dict[str, Any] | None = None

class ChatResponse(BaseModel):
    """Reponse complete (apres streaming termine)."""
    conversation_id: str
    message_id: int | None = None
    content: str
    model: str
    tokens: int = 0
    duration_ms: int = 0

class ChatCancelRequest(BaseModel):
    """Request d'annulation de generation."""
    conversation_id: str

class ChatRetryRequest(BaseModel):
    """Request de retry via WebSocket."""
    conversation_id: str


# -- Artifacts --

class ArtifactInfo(BaseModel):
    """Informations sur un artifact."""
    id: str
    artifact_type: str
    title: str
    language: str
    created_at: str
    conversation_id: str = ""
    display_mode: str = "code"
    line_count: int = 0
    version: int = 1
    parent_id: str = ""

class ArtifactContent(BaseModel):
    """Contenu complet of a artifact."""
    id: str
    artifact_type: str
    title: str
    content: str
    language: str
    created_at: str
    display_mode: str = "code"
    line_count: int = 0
    version: int = 1
    parent_id: str = ""
    filename: str = ""

class ArtifactExport(BaseModel):
    """An exported artifact (file)."""
    filename: str
    content: str


# -- Code Execution --

class CodeExecuteRequest(BaseModel):
    """Request d'execution de code."""
    code: str
    language: str = "python"
    timeout: int | None = None
    conv_id: str | None = None

class CodeExecuteResponse(BaseModel):
    """Result d'execution de code."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    language: str = ""
    truncated: bool = False
    error_message: str = ""
    output_files: list[str] = Field(default_factory=list)

class CodeBlocksRequest(BaseModel):
    """Request d'extraction de blocs de code."""
    text: str

class CodeBlockInfo(BaseModel):
    """Information sur un bloc de code extrait."""
    code: str
    language: str
    start_pos: int
    end_pos: int

class CodeBlocksResponse(BaseModel):
    """Response wrapper for extracted code blocks (BUG-07 S108)."""
    blocks: list[CodeBlockInfo]


# -- Memory --

class MemoryFactSchema(BaseModel):
    """A fact in memory."""
    id: str
    fact: str
    category: str
    source_conversation_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    confidence: float = 1.0
    active: bool = True

class MemoryAddRequest(BaseModel):
    """Request d'ajout de fait."""
    fact: str
    category: str = "context"
    source_conversation_id: str = ""
    confidence: float = 1.0

class MemoryExtractResponse(BaseModel):
    """Result d'extraction de faits."""
    conversation_id: str
    facts_added: int = 0


# -- Memory store (S174, two-tier MemoryStore) --

class MemoryRecordSchema(BaseModel):
    """A fact in the two-tier memory store (S174)."""
    id: str
    text: str
    category: str
    source: str = ""
    created_at: str = ""
    updated_at: str = ""
    active: bool = True
    use_count: int = 0

class MemoryEditRequest(BaseModel):
    """Edit a stored memory fact (S174). Omitted fields are left unchanged."""
    text: str | None = None
    category: str | None = None


# -- Notes (N.2) --

class NoteSchema(BaseModel):
    """A note's metadata and opaque CRDT body (the N.2 read surface).

    The body is an opaque, client-owned CRDT, carried base64-encoded; the
    backend never interprets it. Tags are an OR-Set carried as a JSON array.
    """
    id: str
    title: str
    body_crdt_b64: str = ""
    tags: list[str] = Field(default_factory=list)
    pinned: bool = False
    created_at: str = ""
    updated_at: str = ""
    deleted: bool = False
    # N.9 (S256): the per-item phone-sync opt-in (MOBILE_THREAT_MODEL.md
    # section 3). False is the secure default; the PATCH leg flips it
    # through the store's dedicated setter only.
    mobile_allowed: bool = False

class NoteCreateRequest(BaseModel):
    """Create a note. The body is optional (an empty note is valid)."""
    title: str
    body_crdt_b64: str = ""
    tags: list[str] = Field(default_factory=list)
    pinned: bool = False

class NoteUpdateRequest(BaseModel):
    """Update a note. Omitted fields are left unchanged; an empty tags list is a
    deliberate clear."""
    title: str | None = None
    body_crdt_b64: str | None = None
    tags: list[str] | None = None
    pinned: bool | None = None
    # N.9 (S256): the per-item phone-sync opt-in rides the existing PATCH
    # leg (no new route). Omitted means unchanged; the route flips it
    # through the store's dedicated setter only (decision N9-D3), never the
    # generic update path.
    mobile_allowed: bool | None = None
    # N.8 (S265): the section-4 compaction watermark -- the highest local seq
    # the client folded into this whole-blob PATCH. When present the route
    # records it through the update store and prunes the folded tail lazily;
    # omitted records nothing (fail-secure). It never decreases: the store
    # setter rejects a regression.
    checkpoint_watermark: int | None = None


class NoteUpdateAppendRequest(BaseModel):
    """Append one opaque Yjs update to a note's log (N.8 editor seam).

    The body is the single opaque update blob the client's Y.Doc produced,
    carried base64-encoded; the backend never interprets it. No author or
    device identity rides this payload (decision N9-D3): the local author's
    signature is attached by the sync engine at publish, and the append seam
    mints the per-(user, note) ``seq`` itself.
    """

    update_blob_b64: str


class NoteUpdateRecordSchema(BaseModel):
    """One appended update as served by the append / tail-read legs.

    The opaque blob is carried base64-encoded; ``seq`` is the per-(user, note)
    append order (the platform's only ordering duty, NOTES_CRDT_SPEC.md
    section 4). ``author_device`` is informational local metadata and may be
    absent on a locally appended row.
    """

    id: int
    note_id: str
    seq: int
    update_blob_b64: str
    author_device: str | None = None
    created_at: str = ""

class NoteActionRequest(BaseModel):
    """Run a selection action over a note's selected text (the N.3 surface).

    The selection is wrapped as untrusted context by note_actions; the model is
    the user's selected model (wired into a one-shot client by the route). An
    empty selection is a clean structured failure, not a 422."""
    action: str
    selection: str = ""
    model: str = ""

class NoteActionResultSchema(BaseModel):
    """The structured outcome of a selection action (mirrors NoteActionResult).

    ``ok`` True carries the model ``text``; ``refused`` True marks the structured
    Daily-only web-egress refusal; any other failure is ``ok`` False with a
    ``reason`` and ``refused`` False."""
    action: str
    ok: bool
    text: str = ""
    refused: bool = False
    reason: str = ""

class AttachmentSchema(BaseModel):
    """A note attachment's manifest (the N.5 / N.6 / N.7 media read surface).

    One row per media blob; the encrypted bytes live in the two-layer
    ``NotesBlobStore``, never in this row. ``transcript_text`` (audio),
    ``caption_text`` / ``ocr_text`` (image) are populated by the later opt-in,
    sandboxed post-processing blocs and are None until then.
    """
    id: str
    note_id: str
    kind: str
    mime: str = ""
    byte_size: int = 0
    nonce: str = ""
    created_at: str = ""
    transcript_text: str | None = None
    caption_text: str | None = None
    ocr_text: str | None = None

class AttachmentDeleteResponse(BaseModel):
    """The outcome of deleting one attachment (blob plus manifest)."""
    deleted: bool = False
    id: str = ""

class TranscriptionRequest(BaseModel):
    """Trigger the opt-in, sandboxed transcription of an audio attachment (N.5).

    ``approve`` is the human approval for the durable write-back. The default is
    the safe one (False): the transcript is returned for review but NOT persisted
    until the user approves it."""
    approve: bool = False

class TranscriptionResultSchema(BaseModel):
    """The structured outcome of a transcription request (mirrors
    transcription.TranscriptionResult).

    ``ok`` True carries the ``transcript_text``; ``written_back`` records whether
    the transcript was persisted (only on approval). ``refused`` True marks a
    structured refusal (the fail-secure sandbox gate, a missing or non-audio
    attachment, an unavailable blob); any other failure is ``ok`` False with a
    ``reason`` and ``refused`` False."""
    attachment_id: str
    ok: bool
    transcript_text: str | None = None
    written_back: bool = False
    refused: bool = False
    reason: str = ""

class CaptionRequest(BaseModel):
    """Trigger the opt-in, sandboxed caption / OCR of an image attachment (N.6).

    ``approve`` is the human approval for the durable write-back. The default is
    the safe one (False): the caption / OCR text is returned for review but NOT
    persisted until the user approves it."""
    approve: bool = False

class CaptionResultSchema(BaseModel):
    """The structured outcome of a caption / OCR request (mirrors
    caption.CaptionResult).

    ``ok`` True carries ``caption_text`` and/or ``ocr_text`` (whichever the tool
    produced); ``written_back`` records whether any produced leg was persisted
    (only on approval). ``refused`` True marks a structured refusal (the
    fail-secure sandbox gate, a missing or non-image attachment, an unavailable
    blob, an absent captioner); any other failure is ``ok`` False with a
    ``reason`` and ``refused`` False."""
    attachment_id: str
    ok: bool
    caption_text: str | None = None
    ocr_text: str | None = None
    written_back: bool = False
    refused: bool = False
    reason: str = ""


# -- Cache --

class CacheStatsSchema(BaseModel):
    """Response cache statistics."""
    total_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    entries_by_model: dict[str, int] = Field(default_factory=dict)
    oldest_entry: float = 0.0
    total_size_bytes: int = 0

class SemanticCacheStatsSchema(BaseModel):
    """Semantic cache statistics."""
    total_embeddings: int = 0
    semantic_hits: int = 0
    semantic_misses: int = 0
    avg_similarity: float = 0.0
    embedding_model: str = ""
    threshold: float = 0.0

class CacheCombinedStats(BaseModel):
    """Statistiques combinees des deux caches."""
    response_cache: CacheStatsSchema | None = None
    semantic_cache: SemanticCacheStatsSchema | None = None

class CacheClearResponse(BaseModel):
    """Cache flush result."""
    entries_removed: int = 0
    source: str = ""


# -- Health Dashboard --

class HealthDashboard(BaseModel):
    """Data completes du tableau de bord."""
    status: str = "ok"
    version: str = "1.6.6"
    modules: dict[str, bool] = Field(default_factory=dict)
    conversation_count: int = 0
    memory_fact_count: int = 0
    cache_stats: CacheStatsSchema | None = None
    warmup_status: dict[str, Any] | None = None
    context_health: dict[str, Any] | None = None

class BenchmarkResultSchema(BaseModel):
    """Result of a benchmark."""
    name: str
    iterations: int = 0
    total_time_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    stddev_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput_ops: float = 0.0
    error: str | None = None


# -- Files --

class FileUploadResponse(BaseModel):
    """Result d'upload de fichier."""
    filename: str
    size_bytes: int
    content: str
    extension: str


class ImageUploadResponse(BaseModel):
    """Result d'upload d'image (S48)."""
    filename: str
    size_bytes: int
    base64_data: str  # Base64 sans prefixe data:...
    mime_type: str    # image/png, image/jpeg, etc.
    width: int | None = None
    height: int | None = None


# -- Export --

class ExportResponse(BaseModel):
    """Result d'export de conversation."""
    conversation_id: str
    format: str
    content: str


# -- Presets (S29) --

class PresetInfo(BaseModel):
    """Informations sur un preset."""
    id: str
    name: str
    description: str = ""
    task: str = ""
    model: str = ""
    temperature: float = 0.5
    prompt_variant: str = "standard"
    icon: str = ""
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    detection_weight: float = 0.5
    custom_prompt: str | None = None


class PresetCreate(BaseModel):
    """Corps de request pour creer un preset."""
    id: str
    name: str
    task: str = "simple_question"
    model: str = "qwen3-coder:30b"
    temperature: float = 0.5
    prompt_variant: str = "standard"
    description: str = ""
    icon: str = ""
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    detection_weight: float = 0.5
    custom_prompt: str | None = None


class PresetUpdate(BaseModel):
    """Corps de request pour modifier un preset."""
    name: str | None = None
    task: str | None = None
    model: str | None = None
    temperature: float | None = None
    prompt_variant: str | None = None
    description: str | None = None
    icon: str | None = None
    tags: list[str] | None = None
    keywords: list[str] | None = None
    detection_weight: float | None = None
    custom_prompt: str | None = None


class PresetMatchResult(BaseModel):
    """Result de matching de preset par keywords."""
    preset: PresetInfo
    score: float = 0.0
    matches: int = 0


class PresetDuplicateRequest(BaseModel):
    """Corps de request pour dupliquer un preset."""
    new_id: str
    new_name: str


# -- Pipelines (S29) --

class PipelineStepSchema(BaseModel):
    """Schema of ae etape de pipeline."""
    name: str
    agent: str
    prompt_template: str | None = None
    description: str = ""
    system_prompt: str | None = None
    model: str | None = None


class PipelineInfo(BaseModel):
    """Pipeline information."""
    id: str
    name: str
    description: str = ""
    pattern: str | None = "chain"
    emoji: str = ""
    steps: list[PipelineStepSchema] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    detection_weight: float = 0.5
    created_at: str | None = None
    is_builtin: bool = False
    step_count: int = 0


class PipelineCreate(BaseModel):
    """Request body for creating a pipeline."""
    id: str
    name: str
    description: str = ""
    pattern: str = "chain"
    emoji: str = ""
    steps: list[PipelineStepSchema] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    detection_weight: float = 0.5


class PipelineUpdate(BaseModel):
    """Request body for modifying a pipeline."""
    name: str | None = None
    description: str | None = None
    pattern: str | None = None
    emoji: str | None = None
    steps: list[PipelineStepSchema] | None = None
    keywords: list[str] | None = None
    detection_weight: float | None = None


class PipelineDuplicateRequest(BaseModel):
    """Request body for duplicating a pipeline."""
    new_id: str


class PipelineStats(BaseModel):
    """Statistiques des pipelines."""
    total: int = 0
    builtin: int = 0
    custom: int = 0
    total_steps: int = 0
    total_keywords: int = 0
    by_pattern: dict[str, int] = Field(default_factory=dict)
    available_agents: int = 0
    available_templates: int = 0


class PipelineExportRequest(BaseModel):
    """Corps de request pour exporter les pipelines."""
    custom_only: bool = False


# -- Model Profiles (S46) --

class ModelProfileInfo(BaseModel):
    """Informations sur le profil of a model."""
    name: str
    display_name: str = ""
    capabilities: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    context_window: int = 32768
    speed_tier: str = "medium"
    quality_tier: str = "medium"
    recommended_for: list[str] = Field(default_factory=list)
    not_recommended_for: list[str] = Field(default_factory=list)


class ModelProfilesResponse(BaseModel):
    """Reponse liste des profils de models."""
    profiles: dict[str, ModelProfileInfo] = Field(default_factory=dict)
    count: int = 0


class RoutingReasonInfo(BaseModel):
    """Information de routage transparent."""
    model: str
    display_name: str = ""
    task_type: str = ""
    pipeline: str = ""
    reason: str = ""
    score: float = 0.0
    alternatives: list[str] = Field(default_factory=list)
    profile_used: bool = False


# -- Settings (S29) --

class SettingsResponse(BaseModel):
    """Configuration globale."""
    models: dict[str, Any] = Field(default_factory=dict)
    presets: dict[str, Any] = Field(default_factory=dict)
    user: dict[str, Any] = Field(default_factory=dict)


class SettingValue(BaseModel):
    """Valeur of a parameter individuel."""
    key: str
    value: Any = None


class SettingSetRequest(BaseModel):
    """Corps de request pour definir un parameter."""
    value: Any


# -- Theme (S152) --

class ThemeConfigRequest(BaseModel):
    """Request body for saving a user theme configuration."""
    accent_hue: int = Field(ge=0, le=359, description="Primary accent hue (0-359)")
    accent_saturation: int = Field(default=70, ge=0, le=100)
    secondary_hue: int = Field(default=-1, ge=-1, le=359)
    secondary_saturation: int = Field(default=30, ge=0, le=100)
    accent_lightness_offset: int = Field(default=0, ge=-50, le=50)
    secondary_lightness_offset: int = Field(default=0, ge=-50, le=50)
    accent_warmth: int = Field(default=0, ge=-30, le=30)
    secondary_warmth: int = Field(default=0, ge=-30, le=30)
    mode: str = Field(default="dark", pattern="^(dark|light)$")
    preset_id: str | None = Field(default=None, description="Preset id if using a preset")


class ThemeConfigResponse(BaseModel):
    """Response containing the saved theme configuration."""
    accent_hue: int
    accent_saturation: int
    secondary_hue: int
    secondary_saturation: int
    accent_lightness_offset: int = 0
    secondary_lightness_offset: int = 0
    accent_warmth: int = 0
    secondary_warmth: int = 0
    mode: str
    preset_id: str | None = None
    variables: dict[str, str] = Field(default_factory=dict)


class ThemePresetResponse(BaseModel):
    """A single theme preset (built-in or custom)."""
    id: str
    name: str
    description: str = ""
    accent_hue: int
    accent_saturation: int
    secondary_hue: int
    secondary_saturation: int
    accent_lightness_offset: int = 0
    secondary_lightness_offset: int = 0
    accent_warmth: int = 0
    secondary_warmth: int = 0
    builtin: bool = False


class ThemePresetsListResponse(BaseModel):
    """List of available theme presets."""
    presets: list[ThemePresetResponse] = Field(default_factory=list)


class CustomPresetCreateRequest(BaseModel):
    """Request body for creating a custom user preset."""
    name: str = Field(max_length=50)
    description: str = Field(default="", max_length=200)
    accent_hue: int = Field(ge=0, le=359)
    accent_saturation: int = Field(default=70, ge=0, le=100)
    secondary_hue: int = Field(default=-1, ge=-1, le=359)
    secondary_saturation: int = Field(default=30, ge=0, le=100)
    accent_lightness_offset: int = Field(default=0, ge=-50, le=50)
    secondary_lightness_offset: int = Field(default=0, ge=-50, le=50)
    accent_warmth: int = Field(default=0, ge=-30, le=30)
    secondary_warmth: int = Field(default=0, ge=-30, le=30)


class CustomPresetImportRequest(BaseModel):
    """Request body for importing custom presets from JSON."""
    presets: list[dict[str, Any]] = Field(
        description="Array of preset objects to import"
    )


class CustomPresetsExportResponse(BaseModel):
    """Response for exporting custom presets."""
    presets_json: str = Field(description="JSON string of custom presets")


# -- Consensus (S50) --

class ConsensusRequest(BaseModel):
    """Request de consensus multi-model."""
    message: str
    models: list[str] | None = None
    strategy: str | None = None  # best_of_n, weighted_vote, llm_merge
    system_prompt: str | None = None
    temperature: float | None = None
    conversation_id: str | None = None


class ConsensusModelResponseSchema(BaseModel):
    """Reponse individuelle of a model dans le consensus."""
    model: str
    content: str
    duration_ms: int = 0
    success: bool = True
    error: str = ""
    quality_tier: str = "medium"


class ConsensusComparisonSchema(BaseModel):
    """Comparaison entre les reponses des models."""
    agreement_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    average_agreement: float = 0.0
    areas_of_agreement: list[str] = Field(default_factory=list)
    areas_of_disagreement: list[str] = Field(default_factory=list)


class ConsensusResponse(BaseModel):
    """Reponse complete du consensus."""
    strategy: str
    selected_response: str = ""
    selected_model: str = ""
    confidence: float = 0.0
    individual_responses: list[ConsensusModelResponseSchema] = Field(default_factory=list)
    comparison: ConsensusComparisonSchema | None = None
    total_duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsensusConfigResponse(BaseModel):
    """Configuration actuelle du consensus."""
    default_models: list[str] = Field(default_factory=list)
    strategy: str = "best_of_n"
    max_models: int = 3
    timeout_per_model: int = 60
    min_agreement_threshold: float = 0.3
    available: bool = False


# -- Self-Correction (S51) --

class CorrectionIterationSchema(BaseModel):
    """Une iteration de correction."""
    iteration: int = 0
    compliance_score: float = 0.0
    quality_score: float = 0.0
    improvements: list[str] = Field(default_factory=list)
    duration_ms: int = 0


class CorrectionResultSchema(BaseModel):
    """Result complet d'auto-correction."""
    was_corrected: bool = False
    iterations_performed: int = 0
    compliance_before: float = 1.0
    compliance_after: float = 1.0
    quality_before: float = 1.0
    quality_after: float = 1.0
    total_duration_ms: int = 0
    model_used: str = ""


class CorrectionConfigResponse(BaseModel):
    """Configuration actuelle de l'auto-correction."""
    enable_auto: bool = False
    max_iterations: int = 2
    compliance_threshold: float = 0.7
    quality_threshold: float = 0.6
    check_instructions: bool = True
    check_facts: bool = True
    check_quality: bool = True
    available: bool = False


# -- S68: Semantic Cache (enhanced) --

class S68CacheStatsSchema(BaseModel):
    """S68 enhanced semantic cache statistics."""
    total_entries: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    exact_hit_rate: float = 0.0
    semantic_hit_rate: float = 0.0
    tokens_saved: int = 0
    size_bytes: int = 0
    max_entries: int = 1000
    ttl_seconds: int = 3600
    similarity_threshold: float = 0.92
    embedding_model: str = ""
    scope: str = "global"
    enabled: bool = False
    embeddings_available: bool = False


class S68CacheStatusResponse(BaseModel):
    """S68 cache status for ChatControlBar and panels."""
    enabled: bool = False
    available: bool = False
    stats: S68CacheStatsSchema | None = None
    config: dict = Field(default_factory=dict)


class S68CacheConfigUpdate(BaseModel):
    """S68 cache configuration update request."""
    enabled: bool | None = None
    similarity_threshold: float | None = None
    ttl_seconds: int | None = None
    max_entries: int | None = None
    scope: str | None = None
    exact_match_enabled: bool | None = None
    semantic_match_enabled: bool | None = None


class S68CacheClearRequest(BaseModel):
    """S68 cache clear request."""
    conversation_id: str | None = None


# -- S69: Cascading Inference --

class CascadeTierSchema(BaseModel):
    """Schema for a single cascade tier configuration."""
    name: str
    model: str
    threshold: float = 0.0
    max_tokens: int = 4096
    temperature: float = 0.5


class CascadeTierResultSchema(BaseModel):
    """Schema for a single tier attempt result."""
    tier_name: str
    model: str
    response: str = ""
    score: float = 0.0
    latency_ms: float = 0.0
    escalation_reason: str | None = None


class CascadeResultSchema(BaseModel):
    """Schema for a full cascade execution result."""
    final_response: str
    model_used: str
    tier_index: int
    tier_name: str
    score: float
    attempts: list[CascadeTierResultSchema] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    escalation_reasons: list[str] = Field(default_factory=list)


class CascadeStatusResponse(BaseModel):
    """Cascading inference status response."""
    enabled: bool = False
    available: bool = False
    tier_count: int = 0
    tiers: list[CascadeTierSchema] = Field(default_factory=list)
    last_result: dict | None = None
    config: dict = Field(default_factory=dict)


class CascadeConfigUpdate(BaseModel):
    """Cascading inference configuration update request."""
    enabled: bool | None = None
    tiers: list[dict] | None = None
    max_retries_per_tier: int | None = None
    timeout_per_tier_seconds: int | None = None
    score_weights: dict[str, float] | None = None


class CascadeTestRequest(BaseModel):
    """Request to run a test cascade on a sample query."""
    query: str
    task_type: str | None = None


class CascadeTestResponse(BaseModel):
    """Response from a test cascade run."""
    result: CascadeResultSchema
    config: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Speculative Generation (S70)
# ---------------------------------------------------------------------------


class SpeculativeResultSchema(BaseModel):
    """Schema for a speculative generation result."""
    final_response: str
    draft_response: str = ""
    verify_response: str = ""
    draft_model: str = ""
    verify_model: str = ""
    draft_accepted: bool = False
    iterations: int = 0
    total_latency_ms: float = 0.0
    draft_latency_ms: float = 0.0
    verify_latency_ms: float = 0.0
    convergence_score: float = 0.0


class SpeculativeStatusResponse(BaseModel):
    """Speculative generation status response."""
    enabled: bool = False
    available: bool = False
    draft_model: str = ""
    verify_model: str = ""
    max_iterations: int = 2
    convergence_threshold: float = 0.85
    last_result: dict | None = None
    config: dict = Field(default_factory=dict)


class SpeculativeConfigUpdate(BaseModel):
    """Speculative generation configuration update request."""
    enabled: bool | None = None
    draft_model: str | None = None
    verify_model: str | None = None
    max_iterations: int | None = None
    convergence_threshold: float | None = None
    draft_max_tokens: int | None = None
    verify_max_tokens: int | None = None
    draft_temperature: float | None = None
    verify_temperature: float | None = None


class SpeculativeTestRequest(BaseModel):
    """Request to run a test speculative generation."""
    query: str
    task_type: str | None = None


class SpeculativeTestResponse(BaseModel):
    """Response from a test speculative generation."""
    result: SpeculativeResultSchema
    config: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Network Manager / Offline-First Intelligence (S71)
# ---------------------------------------------------------------------------


class NetworkStatusResponse(BaseModel):
    """Network connectivity status response."""
    available: bool = False
    online: bool = False
    ollama_reachable: bool = False
    embedding_reachable: bool = False
    last_check: float = 0.0
    last_error: str = ""
    latency_ms: float = 0.0
    consecutive_failures: int = 0
    polling_active: bool = False
    queue_size: int = 0
    config: dict = Field(default_factory=dict)


class QueueEntrySchema(BaseModel):
    """Schema for a single queue entry."""
    id: str = ""
    query: str = ""
    task_type: str = "general"
    priority: int = 5
    created_at: float = 0.0
    status: str = "pending"
    error: str = ""
    model: str = ""


class QueueListResponse(BaseModel):
    """Response listing queue entries."""
    available: bool = False
    entries: list[QueueEntrySchema] = Field(default_factory=list)
    total: int = 0
    pending: int = 0


class QueueProcessResponse(BaseModel):
    """Response from queue processing."""
    processed: int = 0
    results: list[dict] = Field(default_factory=list)


class PreCacheResponse(BaseModel):
    """Response from a pre-cache warming run."""
    total: int = 0
    cached: int = 0
    skipped: int = 0
    failed: int = 0
    duration_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)


# -- Performance Monitor (S72) --

class PerformanceSummaryResponse(BaseModel):
    """Full performance summary."""
    available: bool = False
    enabled: bool = False
    throughput: dict = Field(default_factory=dict)
    latency: dict = Field(default_factory=dict)
    utilization: dict = Field(default_factory=dict)


class LatencyStatsResponse(BaseModel):
    """Latency statistics for a model."""
    available: bool = False
    model: str | None = None
    window_seconds: int = 300
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    count: int = 0


class DriftEntry(BaseModel):
    """Single drift detection result."""
    model: str
    metric: str
    baseline_value: float
    recent_value: float
    change_ratio: float
    is_drifted: bool
    direction: str


class DriftResponse(BaseModel):
    """Response with all drift detections."""
    available: bool = False
    drifts: list[DriftEntry] = Field(default_factory=list)


class RecommendationEntry(BaseModel):
    """Single optimization recommendation."""
    model: str
    metric: str
    message: str
    severity: str
    value: float = 0.0


class RecommendationsResponse(BaseModel):
    """Response with all recommendations."""
    available: bool = False
    recommendations: list[RecommendationEntry] = Field(default_factory=list)


class PerformanceHistoryResponse(BaseModel):
    """Response with raw metric records."""
    available: bool = False
    model: str | None = None
    hours: int = 24
    count: int = 0
    records: list[dict] = Field(default_factory=list)


class ThroughputResponse(BaseModel):
    """Token throughput over a window."""
    available: bool = False
    tokens_in_per_sec: float = 0.0
    tokens_out_per_sec: float = 0.0
    total_tokens: int = 0
    request_count: int = 0
    window_seconds: int = 300


class UtilizationResponse(BaseModel):
    """Model utilization distribution."""
    available: bool = False
    window_seconds: int = 3600
    models: dict[str, float] = Field(default_factory=dict)


# -- Sandbox (S73) --

class SandboxCreateRequest(BaseModel):
    """Request to create a new sandbox session."""
    session_id: str | None = None
    allow_degraded: bool = False
    # S210 (Bloc 1): optional human label and per-sandbox command timeout.
    label: str = ""
    timeout: int | None = None


class SandboxCreateResponse(BaseModel):
    """Response after creating a sandbox session."""
    session_id: str
    workspace_path: str
    isolation_backend: str
    degraded: bool = False
    label: str = ""


class SandboxInjectRequest(BaseModel):
    """Request to inject files into a sandbox."""
    session_id: str
    file_paths: list[str] = Field(default_factory=list)


class SandboxInjectResponse(BaseModel):
    """Response after injecting files."""
    session_id: str
    injected_count: int = 0
    injected_paths: list[str] = Field(default_factory=list)


class SandboxFileEntry(BaseModel):
    """A file in the sandbox workspace."""
    path: str
    size: int = 0
    modified: float = 0.0
    approved: bool = False


class SandboxFilesResponse(BaseModel):
    """Response listing sandbox files."""
    session_id: str
    files: list[SandboxFileEntry] = Field(default_factory=list)
    count: int = 0
    approval_state: str = "pending"


class SandboxExecuteRequest(BaseModel):
    """Request to execute a tool in the sandbox."""
    session_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class SandboxExecuteResponse(BaseModel):
    """Response after executing a sandbox tool."""
    session_id: str
    tool_name: str
    result: str = ""
    blocked: bool = False
    block_reason: str = ""
    timed_out: bool = False
    isolation_backend: str = ""


class SandboxStatusResponse(BaseModel):
    """Overall sandbox system status."""
    available: bool = False
    enabled: bool = False
    isolation_backend: str = ""
    bwrap_available: bool = False
    degraded_mode: bool = False
    degraded_confirmed: bool = False
    active_sessions: int = 0
    max_sessions: int = 5
    # S213 (Bloc 4): the live egress-gate answer (True only in Daily;
    # fail-secure False when the gate is unavailable) -- the SyncPanel
    # bulbe_disabled precedent -- plus the configured caps, read-only, for
    # the per-workspace settings strip. All additive with safe defaults.
    network_allowed: bool = False
    command_timeout_default: int | None = None
    limit_memory_bytes: int | None = None
    limit_nproc: int | None = None
    limit_cpu_seconds: int | None = None
    disk_soft_limit_bytes: int | None = None


class SandboxSessionInfo(BaseModel):
    """Info about a sandbox session."""
    session_id: str
    workspace_path: str
    isolation_backend: str = ""
    created_at: float = 0.0
    active: bool = True
    command_count: int = 0
    approval_state: str = "pending"
    approved_paths: list[str] = Field(default_factory=list)
    approved_at: float | None = None
    # S210 (Bloc 1): the workspace-manager view. network_enabled is the
    # per-workspace flag (S213, Bloc 4: user-set only, Daily-only, default
    # False); disk_use_bytes is approximate (bounded walk); running reflects
    # the per-session process registry. has_cloned_baseline (S213) is True
    # when a host clone recorded a baseline root -- the settings strip
    # sharpens the exfiltration warning with it.
    label: str = ""
    owner_user_id: str = "local"
    bound_conversation_id: str | None = None
    network_enabled: bool = False
    last_activity: float = 0.0
    timeout_override: int | None = None
    age_seconds: float = 0.0
    running: bool = False
    disk_use_bytes: int = 0
    has_cloned_baseline: bool = False


class SandboxAuditEntry(BaseModel):
    """A single audit log entry."""
    id: int = 0
    session_id: str = ""
    timestamp: float = 0.0
    command: str = ""
    return_code: int | None = None
    blocked: bool = False
    block_reason: str = ""
    timed_out: bool = False
    stdout_len: int = 0
    stderr_len: int = 0
    isolation_backend: str = ""


class SandboxAuditResponse(BaseModel):
    """Response with audit log entries."""
    entries: list[SandboxAuditEntry] = Field(default_factory=list)
    count: int = 0


class SandboxDestroyResponse(BaseModel):
    """Response after destroying a sandbox."""
    session_id: str
    destroyed: bool = False


class SandboxStopResponse(BaseModel):
    """Response after the stop path (S210).

    stopped is False when nothing was running (honest no-op, not an error);
    the workspace itself persists either way.
    """
    session_id: str
    stopped: bool = False


class SandboxBindRequest(BaseModel):
    """Bind a conversation to a workspace (S210)."""
    conversation_id: str
    session_id: str


class SandboxBindingResponse(BaseModel):
    """Current binding of a conversation (S210)."""
    conversation_id: str
    session_id: str | None = None
    bound: bool = False


class SandboxUploadRefused(BaseModel):
    """A per-file upload refusal (S211): invalid name or collision."""
    name: str
    reason: str


class SandboxUploadResponse(BaseModel):
    """Result of a multipart drag-and-drop upload (S211, Bloc 2)."""
    session_id: str
    uploaded_paths: list[str] = Field(default_factory=list)
    refused: list[SandboxUploadRefused] = Field(default_factory=list)
    uploaded_bytes: int = 0
    manifest_files: int = 0


class HostBrowseEntry(BaseModel):
    """One immediate entry of an allowlisted host directory (S211)."""
    name: str
    type: str  # "dir" | "file" | "symlink" | "special"
    size: int = 0
    hidden: bool = False


class HostBrowseResponse(BaseModel):
    """Allowlisted host directory listing (S211, Bloc 2)."""
    path: str
    roots: list[str] = Field(default_factory=list)
    entries: list[HostBrowseEntry] = Field(default_factory=list)


class SandboxCloneRequest(BaseModel):
    """Request to clone an allowlisted host directory (S211, Bloc 2)."""
    src_path: str
    dest_subdir: str = ""


class SandboxCloneResponse(BaseModel):
    """Result of a symlink-safe host clone (S211, Bloc 2)."""
    session_id: str
    dest: str
    cloned_root: str
    copied_files: int = 0
    copied_bytes: int = 0
    skipped_symlinks: int = 0
    skipped_special: int = 0
    manifest_files: int = 0


# -- Sandbox diff-gated write-back (S212, Bloc 3) --

class SandboxDiffEntry(BaseModel):
    """One classified change against the baseline manifest (S212)."""
    path: str
    kind: str  # "added" | "modified" | "deleted"
    size: int = 0
    baseline_hash: str = ""
    current_hash: str = ""


class SandboxDiffResponse(BaseModel):
    """The workspace diff against the recorded baseline (S212).

    diff_hash is the review-integrity digest the apply request must echo;
    baseline_present is False when no baseline exists (everything is
    "added", and there is no implicit write-back target).
    """
    session_id: str
    baseline_present: bool = False
    cloned_root: str | None = None
    cloned_mount: str | None = None
    entries: list[SandboxDiffEntry] = []
    unchanged: int = 0
    skipped_symlinks: int = 0
    skipped_special: int = 0
    diff_hash: str = ""
    approved_paths: list[str] = []
    confirmed_deletions: list[str] = []


class SandboxConfirmDeletionsRequest(BaseModel):
    """Explicit confirmation of deletions for apply (S212).

    Distinct from approval by design: removing a host file is never
    bundled into a blanket approve-all.
    """
    paths: list[str] = []


class SandboxConfirmDeletionsRefused(BaseModel):
    """A per-path deletion-confirmation refusal (S212)."""
    path: str
    reason: str


class SandboxConfirmDeletionsResponse(BaseModel):
    """Result of a deletion-confirmation request (S212)."""
    session_id: str
    confirmed: list[str] = []
    refused: list[SandboxConfirmDeletionsRefused] = []


class SandboxApplyRequest(BaseModel):
    """Apply approved changes back to the host (S212).

    diff_hash MUST be the digest received with the reviewed diff; apply
    recomputes the live diff and refuses (409) on any drift. target_dir
    is required only for upload-only workspaces (no cloned root) and must
    resolve under the host share-root allowlist.
    """
    diff_hash: str
    target_dir: str | None = None


class SandboxApplyEntry(BaseModel):
    """One applied change (S212)."""
    path: str
    action: str  # "created" | "modified" | "deleted" | "already_absent"
    bytes: int = 0


class SandboxApplyRefusedEntry(BaseModel):
    """One refused apply path with its honest reason (S212)."""
    path: str
    error: str


class SandboxApplyResponse(BaseModel):
    """Result of an apply-to-host run (S212)."""
    session_id: str
    target: str
    applied: list[SandboxApplyEntry] = []
    deleted: list[SandboxApplyEntry] = []
    refused: list[SandboxApplyRefusedEntry] = []
    skipped_unapproved: int = 0
    skipped_unconfirmed: int = 0
    diff_hash: str = ""


# -- Sandbox network gate + provision phase (S213, Bloc 4) --

class SandboxNetworkToggleRequest(BaseModel):
    """Flip the per-workspace network flag (user action only)."""
    enabled: bool


class SandboxNetworkToggleResponse(BaseModel):
    """The flag state after a toggle."""
    session_id: str
    network_enabled: bool = False


class SandboxProvisionRequest(BaseModel):
    """Run the provision phase: install a hash-pinned set into a venv."""
    requirements_path: str
    venv_dir: str = ".venv"


class SandboxProvisionRefusedLine(BaseModel):
    """One refused requirements line (honest, per line)."""
    line: int = 0
    text: str = ""
    reason: str = ""


class SandboxProvisionResponse(BaseModel):
    """Result of a provision run (mirrors the execute posture)."""
    session_id: str
    command: str = ""
    return_code: int = -1
    blocked: bool = False
    block_reason: str = ""
    timed_out: bool = False
    isolation_backend: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    accepted_requirements: list[str] = Field(default_factory=list)


class SandboxConfirmDegradedResponse(BaseModel):
    """Response after confirming degraded mode."""
    confirmed: bool = False
    warning: str = ""


# -- Sandbox Copy-Out (S116) --

class SandboxPreviewResponse(BaseModel):
    """Response with file content preview from a sandbox."""
    session_id: str
    path: str
    content: str = ""
    size: int = 0
    truncated: bool = False
    is_binary: bool = False


class SandboxApproveRequest(BaseModel):
    """Request to approve specific files for copy-out."""
    paths: list[str] = Field(..., description="Relative paths to approve")
    dest_dir: str | None = Field(
        default=None,
        description="Destination directory (default: data/sandbox_exports/)",
    )


class SandboxApproveResponse(BaseModel):
    """Response after approving files for copy-out."""
    session_id: str
    approved_paths: list[str] = Field(default_factory=list)
    approved_count: int = 0
    approval_state: str = "pending"


class SandboxCopyOutEntry(BaseModel):
    """A single file copied out of the sandbox."""
    src_path: str
    dest_path: str
    size: int = 0


class SandboxCopyOutResponse(BaseModel):
    """Response after copying approved files out of the sandbox."""
    session_id: str
    copied: list[SandboxCopyOutEntry] = Field(default_factory=list)
    copied_count: int = 0
    dest_dir: str = ""


class SandboxRejectResponse(BaseModel):
    """Response after rejecting all files in a sandbox."""
    session_id: str
    rejected: bool = False
    approval_state: str = "rejected"


class SandboxApprovalInfoResponse(BaseModel):
    """Approval state summary for a sandbox session."""
    session_id: str
    approval_state: str = "pending"
    approved_paths: list[str] = Field(default_factory=list)
    approved_at: float | None = None


class SandboxApprovalAuditEntry(BaseModel):
    """A single approval audit log entry."""
    id: int = 0
    session_id: str = ""
    timestamp: float = 0.0
    action: str = ""
    paths: str = ""
    dest_dir: str = ""
    detail: str = ""


class SandboxApprovalAuditResponse(BaseModel):
    """Response with approval audit entries."""
    entries: list[SandboxApprovalAuditEntry] = Field(default_factory=list)
    count: int = 0


# -- Quick Sandbox (S117) --

class QuickSandboxStatusResponse(BaseModel):
    """Status of the quick sandbox system."""
    enabled: bool = False
    available: bool = False
    auto_destroy_minutes: int = 30
    max_concurrent_sessions: int = 3
    active_sessions: int = 0


class QuickSandboxToggleRequest(BaseModel):
    """Request to enable/disable quick sandbox."""
    enabled: bool


class QuickSandboxSessionInfo(BaseModel):
    """Info about a quick sandbox session."""
    session_id: str
    active: bool = False
    expired: bool = False
    created_at: float = 0.0
    files_created: list[str] = Field(default_factory=list)


# -- Chat Coding Agent (S118) --

class ChatCodingStatusResponse(BaseModel):
    """Status of the chat coding agent system."""
    enabled: bool = False
    available: bool = False
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 3
    active_sessions: int = 0
    auto_test: bool = True
    max_fix_retries: int = 3


class ChatCodingToggleRequest(BaseModel):
    """Request to enable/disable chat coding agent."""
    enabled: bool


class ChatCodingSessionInfo(BaseModel):
    """Info about a chat coding session."""
    session_id: str
    conversation_id: str
    active: bool = False
    expired: bool = False
    created_at: float = 0.0
    last_activity: float = 0.0
    turn_count: int = 0
    files: list[str] = Field(default_factory=list)
    last_test_passed: bool | None = None
    compression_active: bool = False


# -- Coding Agent (S74) --

class CodingTaskRequest(BaseModel):
    """Request to start a coding task."""
    task: str = Field(..., description="Natural language task description")
    project_path: str | None = Field(
        None, description="Path to project directory to inject into sandbox"
    )
    model: str | None = Field(
        None, description="Model override for this task"
    )
    allow_degraded: bool = Field(
        False, description="Allow tempdir sandbox without confirmation"
    )


class CodingPlanStepResponse(BaseModel):
    """A single step in the coding plan."""
    step_number: int = 0
    step_type: str = ""
    description: str = ""
    file_path: str = ""
    command: str = ""
    completed: bool = False
    result: str = ""
    error: str = ""


class CodingPlanResponse(BaseModel):
    """Coding plan generated by the LLM."""
    task: str = ""
    summary: str = ""
    estimated_files: int = 0
    total_steps: int = 0
    completed_steps: int = 0
    steps: list[CodingPlanStepResponse] = Field(default_factory=list)


class CodingCheckpointRequest(BaseModel):
    """Human checkpoint decision."""
    decision: str = Field(
        ..., description="One of: approve, modify, abort"
    )
    modified_plan: dict[str, Any] | None = Field(
        None, description="Modified plan data (when decision=modify)"
    )


class CodingStepResponse(BaseModel):
    """Response after executing a plan step."""
    step_number: int = 0
    step_type: str = ""
    description: str = ""
    completed: bool = False
    result: str = ""
    error: str = ""


class CodingTestResultResponse(BaseModel):
    """Result of running tests."""
    passed: bool = False
    output: str = ""
    error: str = ""
    return_code: int = -1


class CodingDiffEntry(BaseModel):
    """Diff for a single file."""
    path: str = ""
    is_new: bool = False
    is_deleted: bool = False
    diff: str = ""


class CodingDiffResponse(BaseModel):
    """All diffs for review."""
    count: int = 0
    diffs: list[CodingDiffEntry] = Field(default_factory=list)


class CodingApplyRequest(BaseModel):
    """Request to apply sandbox changes to real filesystem."""
    target_path: str | None = Field(
        None, description="Override target directory"
    )


class CodingApplyResponse(BaseModel):
    """Response after applying changes."""
    applied: int = 0
    files: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class CodingHistoryEntryResponse(BaseModel):
    """A single entry in the coding history."""
    timestamp: float = 0.0
    phase: str = ""
    action: str = ""
    detail: str = ""
    success: bool = True


class CodingStatusResponse(BaseModel):
    """Full coding agent status."""
    task_id: str = ""
    task: str = ""
    phase: str = "idle"
    session_active: bool = False
    plan: CodingPlanResponse | None = None
    current_step: int = 0
    total_steps: int = 0
    iteration: int = 0
    max_iterations: int = 10
    fix_count: int = 0
    max_fix_retries: int = 3
    test_results: list[CodingTestResultResponse] = Field(default_factory=list)
    diffs: list[CodingDiffEntry] = Field(default_factory=list)
    history_count: int = 0
    history: list[CodingHistoryEntryResponse] = Field(default_factory=list)
    working_memory: dict[str, Any] | None = None
    cascading: dict[str, Any] | None = None


# ============================================================================
# Coding History schemas (S76)
# ============================================================================


class CodingTaskSummaryResponse(BaseModel):
    """Summary of a persisted coding agent task."""
    task_id: str = ""
    task_text: str = ""
    project_path: str = ""
    model: str = ""
    status: str = ""
    step_count: int = 0
    completed_steps: int = 0
    test_runs: int = 0
    last_passed: bool | None = None
    created_at: float = 0.0
    completed_at: float | None = None


class CodingTaskDetailResponse(BaseModel):
    """Full detail of a persisted coding agent task."""
    task_id: str = ""
    task_text: str = ""
    project_path: str = ""
    model: str = ""
    status: str = ""
    plan_json: dict[str, Any] | None = None
    created_at: float = 0.0
    completed_at: float | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    tests: list[dict[str, Any]] = Field(default_factory=list)
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)


class CodingHistoryListResponse(BaseModel):
    """Paginated list of persisted coding tasks."""
    tasks: list[CodingTaskSummaryResponse] = Field(default_factory=list)
    total: int = 0


class CodingHistoryStatsResponse(BaseModel):
    """Aggregate statistics for coding history."""
    total_tasks: int = 0
    by_status: dict[str, int] = Field(default_factory=dict)
    total_steps: int = 0
    total_tests: int = 0
    passed_tests: int = 0
    total_checkpoints: int = 0


# -- Coding Analytics (S78 SQ-08) --

class CodingModelSuccessRate(BaseModel):
    """Success rate entry for a single model."""
    model: str = ""
    total: int = 0
    completed: int = 0
    success_rate: float = 0.0


class CodingModelAvgSteps(BaseModel):
    """Average step count entry for a single model."""
    model: str = ""
    avg_steps: float = 0.0
    min_steps: int = 0
    max_steps: int = 0
    task_count: int = 0


class CodingAvgStepsOverall(BaseModel):
    """Overall average step count across all tasks."""
    avg_steps: float = 0.0
    min_steps: int = 0
    max_steps: int = 0
    task_count: int = 0


class CodingFailureReason(BaseModel):
    """Failure reason distribution entry."""
    failure_phase: str = "unknown"
    count: int = 0


class CodingTimeTrend(BaseModel):
    """Time-to-completion entry for a single task."""
    task_id: str = ""
    model: str = ""
    created_at: float = 0.0
    completed_at: float = 0.0
    duration_seconds: float = 0.0


class CodingTestPassRate(BaseModel):
    """Test pass rate entry for a single task."""
    task_id: str = ""
    model: str = ""
    total_runs: int = 0
    passed_runs: int = 0
    pass_rate: float = 0.0


class CodingStepsDistribution(BaseModel):
    """Steps distribution entry (how many tasks had N steps)."""
    step_count: int = 0
    task_count: int = 0


class CodingAnalyticsResponse(BaseModel):
    """Full analytics payload for coding history (S78 SQ-08)."""
    total_tasks: int = 0
    completed_tasks: int = 0
    overall_success_rate: float = 0.0
    success_rate_by_model: list[CodingModelSuccessRate] = Field(
        default_factory=list
    )
    avg_steps_by_model: list[CodingModelAvgSteps] = Field(
        default_factory=list
    )
    avg_steps_overall: CodingAvgStepsOverall = Field(
        default_factory=CodingAvgStepsOverall
    )
    failure_reasons: list[CodingFailureReason] = Field(
        default_factory=list
    )
    time_trends: list[CodingTimeTrend] = Field(default_factory=list)
    test_pass_rate_per_task: list[CodingTestPassRate] = Field(
        default_factory=list
    )
    steps_distribution: list[CodingStepsDistribution] = Field(
        default_factory=list
    )


class CodingResumeRequest(BaseModel):
    """Request to resume a previously interrupted task."""
    model: str | None = None


# -- Export & Batch Delete (S79) --

class CodingBatchDeleteRequest(BaseModel):
    """Request to batch delete coding tasks.

    Provide either task_ids (list of IDs) or before_date (ISO date string).
    If both are provided, task_ids takes precedence.
    """
    task_ids: list[str] | None = None
    before_date: str | None = None


class CodingBatchDeleteResponse(BaseModel):
    """Response from batch delete operation."""
    deleted: int = 0


class CodingExportRow(BaseModel):
    """Flat export row for CSV format."""
    task_id: str = ""
    task_text: str = ""
    model: str = ""
    status: str = ""
    step_count: int = 0
    test_runs: int = 0
    pass_rate: float = 0.0
    created_at: float | None = None
    completed_at: float | None = None
    duration_seconds: float | None = None


# -- Working Memory (S80) --

class WorkingMemoryResponse(BaseModel):
    """Working memory state for a coding agent task."""
    task_id: str = ""
    decisions: list[str] = Field(default_factory=list)
    modified_files: dict[str, str] = Field(default_factory=dict)
    errors_encountered: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    progress_notes: list[str] = Field(default_factory=list)


class WorkingMemoryCompactResponse(BaseModel):
    """Compact working memory for context injection."""
    task_id: str = ""
    compact: str = ""


# ============================================================================
# Session Fingerprint schemas (S75)
# ============================================================================


class FingerprintDimensionResponse(BaseModel):
    """A single fingerprint dimension state."""
    name: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class FingerprintCompactResponse(BaseModel):
    """Compact fingerprint for context injection."""
    compact: str = ""
    token_estimate: int = 0


class FingerprintFullResponse(BaseModel):
    """Full fingerprint state for debugging/API."""
    d1_task_type: dict[str, Any] = Field(default_factory=dict)
    d2_stack: dict[str, Any] = Field(default_factory=dict)
    d3_hot_files: dict[str, Any] = Field(default_factory=dict)
    d4_recent_bugs: dict[str, Any] = Field(default_factory=dict)
    d5_test_health: dict[str, Any] = Field(default_factory=dict)
    d6_momentum: dict[str, Any] = Field(default_factory=dict)
    d7_domain_terms: dict[str, Any] = Field(default_factory=dict)
    d8_dep_clusters: dict[str, Any] = Field(default_factory=dict)
    d9_user_preferences: dict[str, Any] = Field(default_factory=dict)
    d10_context_anchors: dict[str, Any] = Field(default_factory=dict)
    step_count: int = 0
    config: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Web Search / Proxy schemas (S82)
# ============================================================================


class ProxyStatusResponse(BaseModel):
    """Proxy health check result."""
    configured: bool = False
    proxy_url: str | None = None
    reachable: bool = False
    latency_ms: float | None = None
    exit_ip: str | None = None
    error: str | None = None


class ProxyConfigRequest(BaseModel):
    """Request to update proxy configuration."""
    mode: str = "off"  # "off", "tor", "custom"
    proxy_url: str | None = None  # required when mode is "custom"
    proxy_timeout: int | None = None
    max_retries: int | None = None
    pii_sanitization_enabled: bool | None = None


class ProxyConfigResponse(BaseModel):
    """Current proxy configuration."""
    mode: str = "off"
    proxy_url: str | None = None
    proxy_timeout: int = 15
    max_retries: int = 3
    retry_backoff: list[int] = Field(default_factory=lambda: [2, 5, 10])
    pii_sanitization_enabled: bool = True


class PIISanitizePreviewItem(BaseModel):
    """A single PII item found in a query."""
    original: str = ""
    replacement: str = ""
    category: str = ""


class PIISanitizePreviewRequest(BaseModel):
    """Request to preview PII sanitization."""
    query: str = ""


class PIISanitizePreviewResponse(BaseModel):
    """PII sanitization preview result."""
    original: str = ""
    sanitized: str = ""
    items: list[PIISanitizePreviewItem] = Field(default_factory=list)
    was_modified: bool = False


class SearchConfigResponse(BaseModel):
    """Web search configuration and stats overview."""
    ddgs_available: bool = False
    pii_available: bool = False
    proxy_configured: bool = False
    cache_size: int = 0
    total_searches: int = 0
    cache_hits: int = 0
    errors: int = 0
    retries: int = 0
    pii_sanitizations: int = 0
    proxy_searches: int = 0


# -- System Presets (S84) --

class SystemPresetModelInfo(BaseModel):
    """Detected Ollama model information."""
    name: str = ""
    size_bytes: int = 0
    parameter_count_b: float = 0.0
    quantization: str = ""
    family: str = ""
    size_category: str = ""


class SystemPresetInfo(BaseModel):
    """System preset description (infrastructure-level)."""
    id: str
    name: str
    description: str = ""
    icon: str = ""
    recommended_vram_gb: int = 0
    recommended_ram_gb: int = 0
    model_strategy: str = "smallest"
    pipelines: list[str] = Field(default_factory=list)


class SystemPresetListResponse(BaseModel):
    """List of available system presets."""
    presets: list[SystemPresetInfo] = Field(default_factory=list)


class SystemPresetDetectResponse(BaseModel):
    """Result of auto-detection and recommendation."""
    models: list[SystemPresetModelInfo] = Field(default_factory=list)
    recommended_preset: str = "balanced"
    reason: str = ""
    model_counts: dict[str, int] = Field(default_factory=dict)
    total_estimated_vram_gb: float = 0.0


class SystemPresetApplyResponse(BaseModel):
    """Result of applying a system preset."""
    applied: bool = False
    preset_id: str = ""
    preset_name: str = ""
    selected_model: str | None = None
    applied_configs: dict[str, list[str]] = Field(default_factory=dict)
    pipelines: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class OnboardingStateResponse(BaseModel):
    """Current onboarding state."""
    user_initialized: bool = False
    applied_preset: str | None = None
    applied_at: str | None = None


# ---------------------------------------------------------------------------
# Humanizer (S86)
# ---------------------------------------------------------------------------

class HumanizerRewriteRequest(BaseModel):
    """Request to humanize a text passage."""
    text: str
    model: str | None = None
    mode: str | None = None
    intensity: str | None = None
    formality: str | None = None


class HumanizerRewriteResponse(BaseModel):
    """Result of a humanization pass."""
    original: str = ""
    humanized: str = ""
    strategies_applied: list[str] = Field(default_factory=list)
    replacements_count: int = 0
    rewrite_model: str | None = None
    latency_ms: float = 0.0
    mode: str = "rewrite"
    intensity: str = "moderate"
    comparison_id: str = ""


class HumanizerConfigResponse(BaseModel):
    """Humanizer configuration status."""
    enabled: bool = False
    available: bool = False
    mode: str = "rewrite"
    intensity: str = "moderate"
    formality: str = "neutral"
    rewrite_model: str | None = None
    max_input_length: int = 8000
    banned_phrases: list[str] = Field(default_factory=list)
    vocabulary_replacements: dict[str, str] = Field(default_factory=dict)


class HumanizerConfigUpdate(BaseModel):
    """Partial update for humanizer configuration."""
    enabled: bool | None = None
    mode: str | None = None
    intensity: str | None = None
    formality: str | None = None
    rewrite_model: str | None = None
    max_input_length: int | None = None
    banned_phrases: list[str] | None = None
    vocabulary_replacements: dict[str, str] | None = None


class HumanizerFeedbackRequest(BaseModel):
    """A/B comparison rating submission."""
    comparison_id: str
    winner: str  # "humanized", "original", or "tie"


class HumanizerFeedbackResponse(BaseModel):
    """Feedback submission result."""
    success: bool = False
    comparison_id: str = ""
    winner: str = ""


class HumanizerStrategyStats(BaseModel):
    """Win/loss/tie breakdown for a single category."""
    humanized: int = 0
    original: int = 0
    tie: int = 0


class HumanizerStatsResponse(BaseModel):
    """Aggregated humanizer feedback statistics."""
    total_ratings: int = 0
    humanized_wins: int = 0
    original_wins: int = 0
    ties: int = 0
    win_rate: float = 0.0
    by_strategy: dict[str, HumanizerStrategyStats] = Field(default_factory=dict)
    by_model: dict[str, HumanizerStrategyStats] = Field(default_factory=dict)
    by_intensity: dict[str, HumanizerStrategyStats] = Field(default_factory=dict)


# -- Benchmark V2 (S88) --

class BenchmarkV2ProfileSchema(BaseModel):
    """Description of a benchmark profile."""
    id: str = ""
    name: str = ""
    description: str = ""
    categories: list[str] = Field(default_factory=list)
    weight_preset: str = "balanced"
    custom: bool = False


class BenchmarkV2ProfilesResponse(BaseModel):
    """List of available benchmark profiles."""
    profiles: list[BenchmarkV2ProfileSchema] = Field(default_factory=list)
    available_categories: list[str] = Field(default_factory=list)
    total_questions: int = 0


class BenchmarkV2RunRequest(BaseModel):
    """Request to start a benchmark run."""
    profile: str = "all_round"
    models: list[str] = Field(default_factory=list)
    use_judge: bool = False
    judge_model: str = ""
    custom_weights: dict[str, float] | None = None


class BenchmarkV2RunStarted(BaseModel):
    """Confirmation that a benchmark run has started."""
    run_id: str = ""
    profile: str = ""
    models: list[str] = Field(default_factory=list)
    status: str = "pending"


class BenchmarkV2ProgressResponse(BaseModel):
    """Progress state of a running benchmark."""
    run_id: str = ""
    status: str = "pending"
    total_questions: int = 0
    completed_questions: int = 0
    current_model: str = ""
    current_question: str = ""
    elapsed_ms: float = 0.0
    error: str = ""


class BenchmarkV2ModelScore(BaseModel):
    """Aggregated scores for a single model in a run."""
    model: str = ""
    accuracy_avg: float = 0.0
    code_avg: float = 0.0
    structure_avg: float = 0.0
    speed_avg: float = 0.0
    composite: float = 0.0
    questions_evaluated: int = 0


class BenchmarkV2QuestionResult(BaseModel):
    """Per-question evaluation result."""
    question_id: str = ""
    category: str = ""
    prompt: str = ""
    response: str = ""
    accuracy_score: float = 0.0
    code_score: float = 0.0
    structure_score: float = 0.0
    speed_score: float = 0.0
    composite_score: float = 0.0
    details: dict = Field(default_factory=dict)


class BenchmarkV2ResultsResponse(BaseModel):
    """Detailed results for a completed benchmark run."""
    run_id: str = ""
    profile: str = ""
    models: list[str] = Field(default_factory=list)
    status: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    weight_preset: str = "balanced"
    custom_weights: dict[str, float] | None = None
    model_scores: dict[str, BenchmarkV2ModelScore] = Field(default_factory=dict)
    question_results: dict[str, list[BenchmarkV2QuestionResult]] = Field(default_factory=dict)
    judge_scores: list[dict] = Field(default_factory=list)
    judge_summary: dict = Field(default_factory=dict)
    error: str = ""


class BenchmarkV2CompareResponse(BaseModel):
    """Cross-model comparison results."""
    models: list[dict] = Field(default_factory=list)
    profile_filter: str | None = None
    model_filter: list[str] | None = None


class BenchmarkV2HistoryEntry(BaseModel):
    """Summary of a historical benchmark run."""
    run_id: str = ""
    profile: str = ""
    models: list[str] = Field(default_factory=list)
    status: str = ""
    started_at: float = 0.0
    duration_ms: float = 0.0
    weight_preset: str = "balanced"
    custom_weights: dict[str, float] | None = None
    model_scores: dict[str, BenchmarkV2ModelScore] = Field(default_factory=dict)


class BenchmarkV2HistoryResponse(BaseModel):
    """Historical benchmark results."""
    runs: list[BenchmarkV2HistoryEntry] = Field(default_factory=list)
    total: int = 0


# ---------------------------------------------------------------------------
# S89 — LLM-as-Judge, Leaderboard, Head-to-Head, Trends, Recommendations
# ---------------------------------------------------------------------------

class BenchmarkV2JudgeScore(BaseModel):
    """Single judge evaluation score for a question."""
    question_id: str = ""
    model: str = ""
    judge_model: str = ""
    accuracy: int = 0
    relevance: int = 0
    completeness: int = 0
    conciseness: int = 0
    reasoning: int = 0
    justification: str = ""
    weighted_score: float = 0.0
    tokens_used: int = 0
    eval_time_ms: float = 0.0
    error: str = ""


class BenchmarkV2JudgeSummary(BaseModel):
    """Aggregate judge stats for a run."""
    run_id: str = ""
    judge_model: str = ""
    total_tokens: int = 0
    models: dict = Field(default_factory=dict)


class BenchmarkV2LeaderboardEntry(BaseModel):
    """Single model entry in the leaderboard."""
    rank: int = 0
    model: str = ""
    composite: float = 0.0
    accuracy_avg: float = 0.0
    code_avg: float = 0.0
    structure_avg: float = 0.0
    speed_avg: float = 0.0
    judge_avg: float = 0.0
    run_count: int = 0
    last_run: float = 0.0


class BenchmarkV2LeaderboardResponse(BaseModel):
    """Ranked model leaderboard."""
    profile: str = ""
    entries: list[BenchmarkV2LeaderboardEntry] = Field(default_factory=list)
    total: int = 0


class BenchmarkV2HeadToHeadMetric(BaseModel):
    """Single metric comparison between two models."""
    metric: str = ""
    model_a_value: float = 0.0
    model_b_value: float = 0.0
    winner: str = ""


class BenchmarkV2HeadToHeadResponse(BaseModel):
    """Side-by-side comparison of two models."""
    model_a: str = ""
    model_b: str = ""
    metrics: list[BenchmarkV2HeadToHeadMetric] = Field(default_factory=list)
    overall_winner: str = ""
    model_a_wins: int = 0
    model_b_wins: int = 0
    ties: int = 0


class BenchmarkV2TrendPoint(BaseModel):
    """Single data point in a performance trend."""
    run_id: str = ""
    timestamp: float = 0.0
    composite: float = 0.0
    accuracy: float = 0.0
    code: float = 0.0
    structure: float = 0.0
    speed: float = 0.0
    profile: str = ""


class BenchmarkV2TrendResponse(BaseModel):
    """Temporal performance data for a model."""
    model: str = ""
    points: list[BenchmarkV2TrendPoint] = Field(default_factory=list)
    trend_direction: str = ""
    regression_detected: bool = False


class BenchmarkV2RecommendationEntry(BaseModel):
    """Single model recommendation for a role."""
    role: str = ""
    model: str = ""
    composite_score: float = 0.0
    speed_score: float = 0.0
    accuracy_score: float = 0.0
    code_score: float = 0.0
    structure_score: float = 0.0
    tokens_per_second: float = 0.0
    reason: str = ""


class BenchmarkV2RecommendationsResponse(BaseModel):
    """Current best-model suggestions."""
    snapshot_id: str = ""
    created_at: float = 0.0
    profile: str = ""
    recommendations: list[BenchmarkV2RecommendationEntry] = Field(default_factory=list)
    applied: bool = False
    applied_at: float = 0.0


class BenchmarkV2ApplyResponse(BaseModel):
    """Result of applying recommendations to smart router."""
    applied: bool = False
    snapshot_id: str = ""
    changes: dict = Field(default_factory=dict)
    error: str = ""


class BenchmarkV2ExportResponse(BaseModel):
    """Export metadata (actual data returned as JSON or CSV content)."""
    run_id: str = ""
    format: str = "json"
    model_count: int = 0
    question_count: int = 0


# ---------------------------------------------------------------------------
# S90 — Custom Profiles
# ---------------------------------------------------------------------------

class BenchmarkV2CustomProfileCreate(BaseModel):
    """Request to create a custom benchmark profile."""
    name: str
    description: str = ""
    categories: list[str] = Field(default_factory=list)
    weight_preset: str = "balanced"
    custom_weights: dict[str, float] | None = None
    timeout: int = 45
    max_response_tokens: int = 800
    expected_length_range: list[int] = Field(default_factory=lambda: [10, 600])


class BenchmarkV2CustomProfileUpdate(BaseModel):
    """Request to update a custom benchmark profile."""
    name: str | None = None
    description: str | None = None
    categories: list[str] | None = None
    weight_preset: str | None = None
    custom_weights: dict[str, float] | None = None
    timeout: int | None = None
    max_response_tokens: int | None = None
    expected_length_range: list[int] | None = None


class BenchmarkV2CustomProfileResponse(BaseModel):
    """A single custom benchmark profile."""
    profile_id: str = ""
    name: str = ""
    description: str = ""
    categories: list[str] = Field(default_factory=list)
    weight_preset: str = "balanced"
    custom_weights: dict[str, float] | None = None
    timeout: int = 45
    max_response_tokens: int = 800
    expected_length_range: list[int] = Field(default_factory=lambda: [10, 600])
    created_at: float = 0.0
    updated_at: float = 0.0


class BenchmarkV2CustomProfilesListResponse(BaseModel):
    """List of custom benchmark profiles."""
    profiles: list[BenchmarkV2CustomProfileResponse] = Field(default_factory=list)
    count: int = 0


class BenchmarkV2QuestionPreviewResponse(BaseModel):
    """Preview of questions matching given categories."""
    category_counts: dict[str, int] = Field(default_factory=dict)
    total: int = 0


# ---------------------------------------------------------------------------
# S90 — Auto-Trigger
# ---------------------------------------------------------------------------

class BenchmarkV2AutoTriggerStatusResponse(BaseModel):
    """Current status of the auto-trigger system."""
    enabled: bool = False
    running: bool = False
    poll_interval_seconds: float = 120.0
    cooldown_seconds: float = 1800.0
    cooldown_remaining: float = 0.0
    trigger_profile: str = "all_round"
    last_trigger_time: float = 0.0
    known_models: int = 0
    recent_events: int = 0
    resource_guard_active: bool = False
    resource_guard_load_max: float = 0.0


class BenchmarkV2AutoTriggerConfigUpdate(BaseModel):
    """Request to update auto-trigger configuration."""
    enabled: bool | None = None
    poll_interval_seconds: float | None = None
    cooldown_seconds: float | None = None
    trigger_profile: str | None = None
    trigger_models: str | list[str] | None = None
    resource_guard_load_max: float | None = None
    use_judge: bool | None = None
    judge_model: str | None = None


class BenchmarkV2AutoTriggerConfigResponse(BaseModel):
    """Current auto-trigger configuration."""
    enabled: bool = False
    poll_interval_seconds: float = 120.0
    cooldown_seconds: float = 1800.0
    trigger_profile: str = "all_round"
    trigger_models: str | list[str] = "all_new"
    resource_guard_load_max: float = 0.0
    use_judge: bool = False
    judge_model: str = ""


class BenchmarkV2AutoTriggerEventResponse(BaseModel):
    """A single auto-trigger event."""
    event_id: str = ""
    timestamp: float = 0.0
    trigger_type: str = ""
    models: list[str] = Field(default_factory=list)
    run_id: str = ""
    profile: str = ""
    skipped: bool = False
    skip_reason: str = ""


class BenchmarkV2AutoTriggerEventsResponse(BaseModel):
    """List of recent auto-trigger events."""
    events: list[BenchmarkV2AutoTriggerEventResponse] = Field(default_factory=list)
    count: int = 0


class BenchmarkV2AutoTriggerTestPollResponse(BaseModel):
    """Result of a test poll (single poll without triggering)."""
    ok: bool = False
    error: str = ""
    snapshot_models: int = 0
    model_names: list[str] = Field(default_factory=list)
    diff: dict | None = None


# =========================================================================
# S105 — Inference backends
# =========================================================================

class BackendStatusResponse(BaseModel):
    """Status of a single inference backend."""
    name: str = ""
    display_name: str = ""
    healthy: bool = False
    active: bool = False
    model_count: int = 0


class BackendListResponse(BaseModel):
    """List of all registered backends."""
    backends: list[BackendStatusResponse] = Field(default_factory=list)
    active_backend: str | None = None


class BackendActivateRequest(BaseModel):
    """Request to activate a backend."""
    name: str


class BackendActivateResponse(BaseModel):
    """Result of backend activation."""
    success: bool = False
    active_backend: str = ""
    message: str = ""


class GGUFModelInfoResponse(BaseModel):
    """Metadata for a single GGUF model file."""
    filename: str = ""
    path: str = ""
    file_size: int = 0
    file_size_human: str = ""
    gguf_version: int = 0
    tensor_count: int = 0
    architecture: str | None = None
    model_name: str | None = None
    author: str | None = None
    context_length: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    head_count: int | None = None
    vocab_size: int | None = None
    file_type: int | None = None
    quantization_name: str | None = None
    parameter_count: int | None = None
    parameter_count_human: str | None = None


class GGUFModelListResponse(BaseModel):
    """List of GGUF models."""
    models: list[GGUFModelInfoResponse] = Field(default_factory=list)
    count: int = 0


class GGUFDownloadRequest(BaseModel):
    """Request to download a GGUF model."""
    url: str
    filename: str | None = None
    target_dir: str | None = None


class GGUFDownloadResponse(BaseModel):
    """Result of a GGUF download."""
    status: str = ""
    path: str = ""
    filename: str = ""
    size: int = 0
    size_human: str = ""
    message: str = ""


class GGUFStorageResponse(BaseModel):
    """Storage usage for GGUF models."""
    total_size: int = 0
    total_size_human: str = ""
    model_count: int = 0
    directories: list[dict] = Field(default_factory=list)


class BackendModelsResponse(BaseModel):
    """Models from all backends."""
    models: list[dict] = Field(default_factory=list)
    count: int = 0


# ---------------------------------------------------------------------------
# Speculative Decoding (S110) — llama.cpp native speculative decoding
# ---------------------------------------------------------------------------

class SpeculativeDecodingConfigSchema(BaseModel):
    """Current speculative decoding configuration."""
    enabled: bool = False
    draft_model: str = ""
    draft_max: int = 16
    draft_min: int = 5
    draft_gpu_layers: int = 99
    auto_select_draft: bool = True


class SpeculativeDecodingConfigUpdate(BaseModel):
    """Partial update for speculative decoding config."""
    enabled: bool | None = None
    draft_model: str | None = None
    draft_max: int | None = None
    draft_min: int | None = None
    draft_gpu_layers: int | None = None
    auto_select_draft: bool | None = None


class SpeculativeDecodingStatsSchema(BaseModel):
    """Acceptance rate statistics."""
    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    total_runs: int = 0
    overall_acceptance_rate: float = 0.0
    last_acceptance_rate: float = 0.0
    last_speedup_factor: float = 1.0
    last_updated: float = 0.0
    history_size: int = 0
    rolling_acceptance_rate: float = 0.0


class SpeculativeDecodingStatusResponse(BaseModel):
    """Full speculative decoding status."""
    config: SpeculativeDecodingConfigSchema = Field(
        default_factory=SpeculativeDecodingConfigSchema
    )
    stats: SpeculativeDecodingStatsSchema = Field(
        default_factory=SpeculativeDecodingStatsSchema
    )
    available: bool = False
    backend_required: str = "llama_cpp"


class DraftCandidateSchema(BaseModel):
    """A candidate draft model."""
    name: str = ""
    path: str = ""
    family: str = ""
    parameter_size_b: float = 0.0
    quantization: str = ""
    estimated_vram_gb: float = 0.0
    compatibility_score: float = 0.0


class CompatibleDraftsResponse(BaseModel):
    """List of compatible draft models for the current main model."""
    main_model: str = ""
    drafts: list[DraftCandidateSchema] = Field(default_factory=list)
    count: int = 0


class VRAMBudgetResponse(BaseModel):
    """VRAM budget check result."""
    fits: bool = False
    main_vram_gb: float = 0.0
    draft_vram_gb: float = 0.0
    total_vram_gb: float = 0.0
    available_vram_gb: float = 0.0
    headroom_gb: float = 0.0


# ---------------------------------------------------------------------------
# Auto-Tuner (S110) — inference parameter optimization
# ---------------------------------------------------------------------------

class TunerConfigSchema(BaseModel):
    """Auto-tuner configuration."""
    enabled: bool = True
    warmup_runs: int = 3
    benchmark_tokens: int = 128
    benchmark_prompt_tokens: int = 128
    trials_per_param: int = 3
    auto_apply: bool = False


class ParameterSpaceSchema(BaseModel):
    """Parameter search space."""
    batch_size: list[int] = Field(default_factory=lambda: [512, 1024, 2048, 4096])
    ubatch_size: list[int] = Field(default_factory=lambda: [256, 512, 1024])
    threads: list[int] = Field(default_factory=lambda: [2, 4, 6, 8])
    flash_attention: list[bool] = Field(default_factory=lambda: [True, False])


class TunerStatusResponse(BaseModel):
    """Full auto-tuner status."""
    config: TunerConfigSchema = Field(default_factory=TunerConfigSchema)
    param_space: ParameterSpaceSchema = Field(default_factory=ParameterSpaceSchema)
    active_jobs: dict = Field(default_factory=dict)
    saved_profiles: list[str] = Field(default_factory=list)
    available: bool = False


class TunerRunRequest(BaseModel):
    """Request to start a tuning session."""
    model_name: str


class TunerJobSchema(BaseModel):
    """Tuner job status."""
    job_id: str = ""
    model_name: str = ""
    status: str = "pending"
    progress: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    result: dict | None = None
    error: str = ""


class TunerProfileSchema(BaseModel):
    """Best parameters for a model."""
    model_name: str = ""
    best_params: dict = Field(default_factory=dict)
    best_tg_speed: float = 0.0
    best_pp_speed: float = 0.0
    baseline_tg_speed: float = 0.0
    baseline_pp_speed: float = 0.0
    speedup_factor: float = 1.0
    hardware_fingerprint: str = ""
    timestamp: float = 0.0
    all_results: list[dict] = Field(default_factory=list)


class TunerResultsResponse(BaseModel):
    """All tuning results."""
    results: dict = Field(default_factory=dict)
    count: int = 0


# ---------------------------------------------------------------------------
# Tuner Recommendations (S112)
# ---------------------------------------------------------------------------

class TunerRecommendationSchema(BaseModel):
    """A single tuning recommendation."""
    title: str = ""
    description: str = ""
    parameter: str = ""
    current_value: Any = None
    recommended_value: Any = None
    estimated_speedup: float = 1.0
    confidence: str = "medium"
    category: str = "performance"
    applied: bool = False

class TunerRecommendationsResponse(BaseModel):
    """Recommendations for a tuned model."""
    model_name: str = ""
    recommendations: list[TunerRecommendationSchema] = Field(default_factory=list)
    count: int = 0


# ---------------------------------------------------------------------------
# Live Metrics (S111)
# ---------------------------------------------------------------------------

class LiveMetricsSampleSchema(BaseModel):
    """Single point-in-time metrics snapshot."""
    timestamp: float = 0.0
    tokens_per_second: float = 0.0
    prompt_eval_time_ms: float = 0.0
    eval_time_ms: float = 0.0
    total_tokens: int = 0
    pending_tokens: int = 0
    gpu_utilization_pct: float = -1.0
    gpu_memory_used_mb: float = -1.0
    gpu_memory_total_mb: float = -1.0
    gpu_temperature_c: float = -1.0
    system_memory_used_mb: float = 0.0
    system_memory_total_mb: float = 0.0
    is_generating: bool = False
    active_model: str = ""


class LiveMetricsConfigSchema(BaseModel):
    """Live metrics collector configuration."""
    enabled: bool = True
    sample_interval_ms: int = 500
    window_seconds: int = 60
    rolling_speed_window_s: float = 5.0
    gpu_monitoring: bool = True


class LiveMetricsStatusResponse(BaseModel):
    """Live metrics collector status."""
    running: bool = False
    config: LiveMetricsConfigSchema = Field(default_factory=LiveMetricsConfigSchema)
    gpu_available: bool = False
    history_size: int = 0
    total_tokens_all_time: int = 0
    is_generating: bool = False
    active_model: str = ""
    available: bool = True


class LiveMetricsHistoryResponse(BaseModel):
    """Historical metrics samples."""
    samples: list[dict] = Field(default_factory=list)
    count: int = 0


# ---------------------------------------------------------------------------
# Model Lifecycle (S112)
# ---------------------------------------------------------------------------

class ModelPullRequest(BaseModel):
    """Request to pull a model."""
    model_name: str
    insecure: bool = False

class ModelPullJobSchema(BaseModel):
    """Pull job status."""
    job_id: str = ""
    model_name: str = ""
    status: str = ""
    progress: dict = Field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0
    error: str = ""

class ModelDeleteRequest(BaseModel):
    """Request to delete a model."""
    model_name: str

class ModelDeleteResponse(BaseModel):
    """Delete operation result."""
    success: bool = False
    model: str = ""
    error: str = ""

class ModelUpdateCheckRequest(BaseModel):
    """Request to check model updates."""
    model_names: list[str] = Field(default_factory=list)

class ModelUpdateInfoSchema(BaseModel):
    """Update check result for one model."""
    model_name: str = ""
    current_digest: str = ""
    latest_digest: str = ""
    has_update: bool = False
    checked_at: float = 0.0
    error: str = ""

class ModelUpdatesResponse(BaseModel):
    """Batch update check results."""
    results: list[ModelUpdateInfoSchema] = Field(default_factory=list)

class ModelAliasRequest(BaseModel):
    """Request to set a model alias."""
    alias: str
    model_name: str

class ModelAliasesResponse(BaseModel):
    """All model aliases."""
    aliases: dict[str, str] = Field(default_factory=dict)

class ModelLifecycleStatusResponse(BaseModel):
    """Lifecycle manager status."""
    available: bool = True
    enabled: bool = True
    ollama_base_url: str = ""
    max_concurrent_pulls: int = 2
    active_pulls: int = 0
    alias_count: int = 0
    stale_threshold_days: int = 30

class StaleModelsResponse(BaseModel):
    """Stale model detection results."""
    models: list[dict] = Field(default_factory=list)
    threshold_days: int = 30


# ---------------------------------------------------------------------------
# Telemetry Dashboard (S113)
# ---------------------------------------------------------------------------


class TelemetryConsumerInfoSchema(BaseModel):
    """Information about a registered telemetry consumer."""
    name: str = ""
    healthy: bool = True


class TelemetryStatsResponse(BaseModel):
    """Telemetry collector statistics."""
    enabled: bool = True
    total_events: int = 0
    total_requests: int = 0
    total_tokens: int = 0
    active_requests: int = 0
    buffer_size: int = 0
    buffer_max_size: int = 64
    consumer_count: int = 0


class TelemetryConsumersResponse(BaseModel):
    """Registered telemetry consumers and health."""
    consumers: list[TelemetryConsumerInfoSchema] = Field(default_factory=list)
    count: int = 0


class TelemetryFlushResponse(BaseModel):
    """Result of a manual telemetry flush."""
    flushed_events: int = 0


# ---------------------------------------------------------------------------
# Inference Profiler (S113)
# ---------------------------------------------------------------------------


class InferenceProfileSchema(BaseModel):
    """Detailed time breakdown of a single inference request."""
    request_id: str = ""
    model: str = ""
    timestamp: float = 0.0
    total_ms: float = 0.0
    prompt_eval_ms: float = 0.0
    token_gen_ms: float = 0.0
    overhead_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tok_per_sec: float = 0.0


class ProfilerSummarySchema(BaseModel):
    """Aggregated profiling statistics for a model."""
    model: str = ""
    request_count: int = 0
    avg_total_ms: float = 0.0
    p50_total_ms: float = 0.0
    p95_total_ms: float = 0.0
    p99_total_ms: float = 0.0
    avg_prompt_eval_ms: float = 0.0
    avg_token_gen_ms: float = 0.0
    avg_overhead_ms: float = 0.0
    avg_tok_per_sec: float = 0.0


class ProfilerSummaryResponse(BaseModel):
    """Aggregated profiling stats across all models."""
    models: list[ProfilerSummarySchema] = Field(default_factory=list)
    total_profiled_requests: int = 0


class ProfilerRecentResponse(BaseModel):
    """Most recent request profiles."""
    profiles: list[InferenceProfileSchema] = Field(default_factory=list)
    count: int = 0


# =========================================================================
# S114 — Telemetry History
# =========================================================================


class HistoryEventSchema(BaseModel):
    """A persisted inference event from telemetry history."""
    id: int = 0
    request_id: str = ""
    model: str = ""
    timestamp: float = 0.0
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tok_per_sec: float = 0.0
    prompt_eval_ms: float = 0.0
    token_gen_ms: float = 0.0


class TelemetryHistoryResponse(BaseModel):
    """Paginated event history response."""
    events: list[HistoryEventSchema] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


class TrendBucketSchema(BaseModel):
    """Aggregated data for a time bucket."""
    bucket_start: float = 0.0
    bucket_label: str = ""
    event_count: int = 0
    avg_latency_ms: float = 0.0
    avg_tok_per_sec: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0


class TelemetryTrendsResponse(BaseModel):
    """Aggregated latency/throughput trends over time."""
    buckets: list[TrendBucketSchema] = Field(default_factory=list)
    hours: int = 24
    model: str = ""


class ModelBreakdownSchema(BaseModel):
    """Per-model event count and average stats."""
    model: str = ""
    event_count: int = 0
    avg_latency_ms: float = 0.0
    avg_tok_per_sec: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0


class TelemetryHistoryPurgeResponse(BaseModel):
    """Result of purging old events."""
    purged_count: int = 0


class TelemetryHistoryStatsResponse(BaseModel):
    """Quick overview stats for history store."""
    available: bool = False
    total_stored: int = 0
    retention_days: int = 7
    oldest_event_ts: float = 0.0
    max_events: int = 50000
    auto_purge_enabled: bool = False


class TelemetryHistorySettingsRequest(BaseModel):
    """Request body for updating history retention settings (S115)."""
    retention_days: int | None = None
    auto_purge_enabled: bool | None = None


class TelemetryHistorySettingsResponse(BaseModel):
    """Response after updating history settings (S115)."""
    retention_days: int = 7
    auto_purge_enabled: bool = False


# =========================================================================
# S121 -- Backup / Restore
# =========================================================================


class BackupSectionInfo(BaseModel):
    """Information about a single backup section."""
    name: str = ""
    description: str = ""
    item_count: int = 0
    available: bool = False


class BackupSectionsResponse(BaseModel):
    """Response listing available backup sections."""
    sections: list[BackupSectionInfo] = []


class BackupImportRequest(BaseModel):
    """Request body for backup import."""
    backup: dict = {}
    strategy: str = "merge"
    # BK-03 (S194): explicit user override for the BK-01 signature policy.
    # Never relaxes a FAILED verification (manager-level invariant).
    allow_unsigned: bool = False


class BackupPreviewRequest(BaseModel):
    """Request body for backup import preview."""
    backup: dict = {}
    strategy: str = "merge"


class BackupDiffItemResponse(BaseModel):
    """A single diff item in the preview."""
    section: str = ""
    key: str = ""
    action: str = ""
    current_value: Optional[Any] = None
    incoming_value: Optional[Any] = None


class BackupPreviewResponse(BaseModel):
    """Response for import preview."""
    valid: bool = True
    strategy: str = "merge"
    sections: list[str] = []
    diff: list[BackupDiffItemResponse] = []
    errors: list[str] = []
    summary: dict[str, int] = {}


class BackupImportResponse(BaseModel):
    """Response for import operation."""
    success: bool = False
    sections_imported: list[str] = []
    sections_failed: list[str] = []
    errors: list[str] = []
    rolled_back: bool = False


# -- S153: Keyboard Shortcuts --

class ShortcutBindingSchema(BaseModel):
    """A single keyboard shortcut binding."""
    action: str = ""
    key: str = ""
    ctrl: bool = False
    shift: bool = False
    alt: bool = False
    meta: bool = False
    description: str = ""
    category: str = "general"


class KeyboardShortcutsResponse(BaseModel):
    """Response for GET /api/settings/keyboard_shortcuts."""
    shortcuts: dict[str, Any] = Field(default_factory=dict)
    custom_overrides: dict[str, Any] = Field(default_factory=dict)
    browser_conflicts: list[dict[str, Any]] = Field(default_factory=list)


class KeyboardShortcutsUpdateRequest(BaseModel):
    """Request for PUT /api/settings/keyboard_shortcuts."""
    custom_bindings: dict[str, Any] = Field(default_factory=dict)


class KeyboardShortcutsUpdateResponse(BaseModel):
    """Response for PUT /api/settings/keyboard_shortcuts."""
    success: bool = False
    shortcuts: dict[str, Any] = Field(default_factory=dict)
    custom_overrides: dict[str, Any] = Field(default_factory=dict)
    browser_conflicts: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
