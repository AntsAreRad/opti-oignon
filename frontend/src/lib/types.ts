/**
 * TypeScript interfaces matching Pydantic schemas from opti_oignon/api/schemas.py.
 *
 * Each interface mirrors its Python counterpart for type-safe API communication.
 */

// -- Conversations --

export interface ConversationSummary {
	id: string;
	title: string;
	created_at: string | null;
	updated_at: string | null;
	message_count: number;
	model: string | null;
	task_type: string | null;
	preset: string | null;
	total_tokens?: number;
}

export interface ConversationDetail {
	id: string;
	title: string;
	messages: MessageItem[];
	created_at: string | null;
	updated_at: string | null;
	model: string | null;
	task_type: string | null;
	preset: string | null;
	message_count: number;
	total_tokens: number;
}

export interface ConversationCreate {
	title?: string;
	model?: string;
	preset?: string;
}

export interface ConversationRename {
	title: string;
}

// -- Messages --

export interface MessageItem {
	id: number | null;
	role: string;
	content: string;
	timestamp: string | null;
	model: string | null;
	token_estimate: number;
	thinking?: string;  // Contenu de reflexion (chain-of-thought)
	verification?: VerificationInfo[];  // Resultats de verification de code
	tool_calls?: ToolCallInfo[];  // Appels d'outils
	reasoning_steps?: ReasoningStepInfo[];  // Etapes de raisonnement
	reasoning_meta?: ReasoningMetaInfo;  // Metadonnees de raisonnement
	correction?: CorrectionInfo;  // Info d'auto-correction
	vision_delegation?: VisionDelegationInfo;  // Vision delegation info
	sandbox_meta?: SandboxMeta;  // Quick sandbox metadata
}

// Quick sandbox metadata attached to assistant messages
export interface SandboxMeta {
	sandbox_active: boolean;
	sandbox_session_id: string;
	sandbox_files: unknown[];
	sandbox_files_created: string[];
}

// Information de verification de code
export interface VerificationInfo {
	status: 'passed' | 'fixed' | 'failed';
	iterations: number;
	language: string;
	errors?: string[];
	fixes?: string[];
	execution_output?: string;
}

// Information d'appel d'outil
export interface ToolCallInfo {
	tool_name: string;
	arguments: Record<string, unknown>;
	status: 'executing' | 'complete' | 'error';
	result_preview?: string;
	execution_time?: number;
	success: boolean;
	reasoning?: string;
	turn?: number;  // Turn number for multi-turn tool history
}

// Information d'etape de raisonnement
export interface ReasoningStepInfo {
	step_number: number;
	title: string;
	content: string;
	duration_ms: number;
}

// Metadonnees de raisonnement
export interface ReasoningMetaInfo {
	strategy: string;
	steps_count: number;
	confidence: number;
	total_duration_ms: number;
}

// -- Models --

export interface ModelInfo {
	name: string;
	size: string | null;
	modified_at: string | null;
	family: string | null;
	parameter_size: string | null;
	quantization_level: string | null;
	mtp_capable: boolean;
}

export interface ModelListResponse {
	models: ModelInfo[];
	count: number;
}

export interface EffectiveModelResponse {
	model: string;
	source: string;
}

// Model profiles and routing transparency
export interface ModelProfileInfo {
	name: string;
	display_name: string;
	capabilities: string[];
	strengths: string[];
	weaknesses: string[];
	context_window: number;
	speed_tier: 'fast' | 'medium' | 'slow';
	quality_tier: 'high' | 'medium' | 'low';
	recommended_for: string[];
	not_recommended_for: string[];
	// Numeric task scores for smart routing
	task_scores?: Record<string, number>;
	parameter_count?: string;
	quantization?: string;
	family?: string;
	auto_detected?: boolean;
}

export interface RoutingReasonInfo {
	model: string;
	display_name: string;
	task_type: string;
	pipeline: string;
	reason: string;
	score: number;
	alternatives: string[];
	profile_used: boolean;
}

export interface ModelProfilesResponse {
	profiles: Record<string, ModelProfileInfo>;
	count: number;
}

// Smart routing types
export interface SmartRoutingResult {
	model: string;
	score: number;
	task_score: number;
	speed_weight: number;
	context_fit: number;
	reason: string;
	alternatives: { model: string; display_name: string; score: number }[];
	profile_used: boolean;
	fallback: boolean;
	feedback_adjusted: boolean; //
	failover: boolean; //
	original_model: string; //
}

export interface SmartRouterConfig {
	enabled: boolean;
	profiles_available: boolean;
	operational: boolean;
	default_model: string;
	speed_preference: 'fast' | 'balanced' | 'quality';
	speed_weights: Record<string, number>;
	profile_count: number;
}

// -- Chat --

export interface ChatRequest {
	conversation_id: string | null;
	message: string;
	model?: string;
	preset?: string;
	temperature?: number;
	use_presets?: boolean;
	think?: boolean;        // Mode reflexion
	web_search?: boolean;   // Recherche web
	images?: string[];      // Images base64 pour vision
	prompt_enhance?: boolean; // Prompt optimization/enhancement
}

export interface ChatToken {
	type: 'token' | 'thinking' | 'done' | 'error' | 'metadata' | 'verification' | 'tool_call' | 'reasoning_step' | 'reasoning_done' | 'consensus_model_done' | 'consensus_done' | 'correction_step' | 'correction_done' | 'vision_delegation' | 'status' | 'coding_plan' | 'coding_step' | 'coding_test' | 'coding_fix' | 'coding_done' | 'coding_status' | 'coding_error';
	content: string;
	metadata?: Record<string, unknown>;
}

export interface ChatResponse {
	conversation_id: string;
	message_id: number | null;
	content: string;
	model: string;
	tokens: number;
	duration_ms: number;
	// Quick sandbox metadata (present when sandbox was used)
	sandbox_active?: boolean;
	sandbox_session_id?: string;
	sandbox_files?: unknown[];
	sandbox_files_created?: string[];
	// Chat coding agent metadata (present when coding agent was used)
	chat_coding?: boolean;
	coding_result?: Record<string, unknown>;
	turn_count?: number;
}

export interface ChatRetryRequest {
	conversation_id: string;
}

export interface ChatCancelRequest {
	conversation_id: string;
}

export interface ChatStreamCallbacks {
	onToken: (content: string) => void;
	onThinking?: (content: string) => void;  // Tokens de reflexion
	onVerification?: (info: VerificationInfo) => void;  // Verification de code
	onToolCall?: (info: ToolCallInfo) => void;  // Appels d'outils
	onReasoningStep?: (info: ReasoningStepInfo) => void;  // Etapes de raisonnement
	onReasoningDone?: (info: ReasoningMetaInfo) => void;  // Fin du raisonnement
	onVisionDelegation?: (info: Record<string, unknown>) => void;  // Vision delegation
	onStatus?: (message: string) => void;  // Intermediate status feedback
	onCodingEvent?: (eventType: string, data: Record<string, unknown>) => void;  // Coding agent events
	onDone: (response: ChatResponse) => void;
	onError: (error: string) => void;
	onMetadata?: (metadata: Record<string, unknown>) => void;
}

// -- Presets --

export interface PresetInfo {
	id: string;
	name: string;
	description: string;
	task: string;
	model: string;
	temperature: number;
	prompt_variant: string;
	icon: string;
	tags: string[];
	keywords: string[];
	detection_weight: number;
	custom_prompt: string | null;
}

// -- Settings --

export interface SettingsResponse {
	models: Record<string, unknown>;
	presets: Record<string, unknown>;
	user: Record<string, unknown>;
}

export interface SettingValue {
	key: string;
	value: unknown;
}

// -- Health --

export interface HealthResponse {
	status: string;
	version: string;
	modules: Record<string, boolean>;
}

export interface HealthDashboard {
	status: string;
	version: string;
	modules: Record<string, boolean>;
	conversation_count: number;
	memory_fact_count: number;
	cache_stats: CacheStatsSchema | null;
	warmup_status: Record<string, unknown> | null;
}

export interface BenchmarkResultSchema {
	name: string;
	iterations: number;
	total_time_ms: number;
	mean_ms: number;
	median_ms: number;
	min_ms: number;
	max_ms: number;
	stddev_ms: number;
	p95_ms: number;
	p99_ms: number;
	throughput_ops: number;
	error: string | null;
}

// -- Cache --

export interface CacheStatsSchema {
	total_entries: number;
	total_hits: number;
	total_misses: number;
	hit_rate: number;
	entries_by_model: Record<string, number>;
	oldest_entry: number;
	total_size_bytes: number;
}

export interface SemanticCacheStatsSchema {
	total_embeddings: number;
	semantic_hits: number;
	semantic_misses: number;
	avg_similarity: number;
	embedding_model: string;
	threshold: number;
}

export interface CacheCombinedStats {
	response_cache: CacheStatsSchema | null;
	semantic_cache: SemanticCacheStatsSchema | null;
}

export interface CacheClearResponse {
	entries_removed: number;
	source: string;
}

// --: Semantic Cache (enhanced) --

export interface S68CacheStats {
	total_entries: number;
	exact_hits: number;
	semantic_hits: number;
	total_misses: number;
	hit_rate: number;
	exact_hit_rate: number;
	semantic_hit_rate: number;
	tokens_saved: number;
	size_bytes: number;
	max_entries: number;
	ttl_seconds: number;
	similarity_threshold: number;
	embedding_model: string;
	scope: string;
	enabled: boolean;
	embeddings_available: boolean;
}

export interface S68CacheStatus {
	enabled: boolean;
	available: boolean;
	stats: S68CacheStats | null;
	config: Record<string, unknown>;
}

export interface S68CacheConfigUpdate {
	enabled?: boolean;
	similarity_threshold?: number;
	ttl_seconds?: number;
	max_entries?: number;
	scope?: string;
	exact_match_enabled?: boolean;
	semantic_match_enabled?: boolean;
}

// --: Cascading Inference --

export interface CascadeTier {
	name: string;
	model: string;
	threshold: number;
	max_tokens: number;
	temperature: number;
}

export interface CascadeTierResult {
	tier_name: string;
	model: string;
	response: string;
	score: number;
	latency_ms: number;
	escalation_reason: string | null;
}

export interface CascadeResult {
	final_response: string;
	model_used: string;
	tier_index: number;
	tier_name: string;
	score: number;
	attempts: CascadeTierResult[];
	total_latency_ms: number;
	escalation_reasons: string[];
}

export interface CascadeStatus {
	enabled: boolean;
	available: boolean;
	tier_count: number;
	tiers: CascadeTier[];
	last_result: Record<string, unknown> | null;
	config: Record<string, unknown>;
}

export interface CascadeConfigUpdate {
	enabled?: boolean;
	tiers?: Record<string, unknown>[];
	max_retries_per_tier?: number;
	timeout_per_tier_seconds?: number;
	score_weights?: Record<string, number>;
}

export interface CascadeTestResult {
	result: CascadeResult;
	config: Record<string, unknown>;
}

// -- Speculative Generation --

export interface SpeculativeResult {
	final_response: string;
	draft_response: string;
	verify_response: string;
	draft_model: string;
	verify_model: string;
	draft_accepted: boolean;
	iterations: number;
	total_latency_ms: number;
	draft_latency_ms: number;
	verify_latency_ms: number;
	convergence_score: number;
}

export interface SpeculativeStatus {
	enabled: boolean;
	available: boolean;
	draft_model: string;
	verify_model: string;
	max_iterations: number;
	convergence_threshold: number;
	last_result: Record<string, unknown> | null;
	config: Record<string, unknown>;
}

export interface SpeculativeConfigUpdate {
	enabled?: boolean;
	draft_model?: string;
	verify_model?: string;
	max_iterations?: number;
	convergence_threshold?: number;
	draft_max_tokens?: number;
	verify_max_tokens?: number;
	draft_temperature?: number;
	verify_temperature?: number;
}

export interface SpeculativeTestResult {
	result: SpeculativeResult;
	config: Record<string, unknown>;
}

// -- Artifacts --

export interface ArtifactInfo {
	id: string;
	artifact_type: string;
	title: string;
	language: string;
	created_at: string;
	conversation_id: string;
	display_mode: string;
	line_count: number;
	version: number;
	parent_id: string;
}

export interface ArtifactContent extends ArtifactInfo {
	content: string;
	filename: string;
}

export interface ArtifactExport {
	filename: string;
	content: string;
}

// -- Code Execution --

export interface CodeExecuteRequest {
	code: string;
	language: string;
	timeout?: number;
	conv_id?: string;
}

export interface CodeExecuteResponse {
	success: boolean;
	stdout: string;
	stderr: string;
	return_code: number;
	execution_time: number;
	language: string;
	truncated: boolean;
	error_message: string;
	output_files: string[];
}

export interface CodeBlockInfo {
	code: string;
	language: string;
	start_pos: number;
	end_pos: number;
}

// -- File Upload --

export interface FileUploadResponse {
	filename: string;
	size_bytes: number;
	content: string;
	extension: string;
}

export interface AttachedFile {
	filename: string;
	content: string;
	size_bytes: number;
	extension: string;
}

// Image upload types
export interface ImageUploadResponse {
	filename: string;
	size_bytes: number;
	base64_data: string;
	mime_type: string;
	width?: number;
	height?: number;
}

export interface AttachedImage {
	filename: string;
	base64_data: string;
	mime_type: string;
	size_bytes: number;
	preview_url: string;  // Object URL pour la miniature
}

// -- Memory --

export interface MemoryFact {
	id: string;
	fact: string;
	category: string;
	source_conversation_id: string;
	created_at: string;
	updated_at: string;
	confidence: number;
	active: boolean;
}

export interface MemoryAddRequest {
	fact: string;
	category?: string;
	source_conversation_id?: string;
	confidence?: number;
}

export interface MemoryExtractResponse {
	conversation_id: string;
	facts_added: number;
}

// -- Search --

export interface SearchResult {
	title: string;
	url: string;
	snippet: string;
	source: string;
	relevance_score: number;
}

export interface SearchResponse {
	results: SearchResult[];
	query: string;
	engine: string;
	citations: string[];
}

export interface SearchEngine {
	id: string;
	name: string;
	available: boolean;
}

export interface SearchHistoryEntry {
	query: string;
	timestamp: string;
	result_count: number;
}

// -- Proxy & PII Sanitization --

export interface ProxyStatusResponse {
	configured: boolean;
	proxy_url: string | null;
	reachable: boolean;
	latency_ms: number | null;
	exit_ip: string | null;
	error: string | null;
}

export interface ProxyConfigRequest {
	mode: string; // "off" | "tor" | "custom"
	proxy_url?: string | null;
	proxy_timeout?: number | null;
	max_retries?: number | null;
	pii_sanitization_enabled?: boolean | null;
}

export interface ProxyConfigResponse {
	mode: string;
	proxy_url: string | null;
	proxy_timeout: number;
	max_retries: number;
	retry_backoff: number[];
	pii_sanitization_enabled: boolean;
}

export interface PIISanitizePreviewItem {
	original: string;
	replacement: string;
	category: string;
}

export interface PIISanitizePreviewResponse {
	original: string;
	sanitized: string;
	items: PIISanitizePreviewItem[];
	was_modified: boolean;
}

export interface SearchConfigResponse {
	ddgs_available: boolean;
	pii_available: boolean;
	proxy_configured: boolean;
	cache_size: number;
	total_searches: number;
	cache_hits: number;
	errors: number;
	retries: number;
	pii_sanitizations: number;
	proxy_searches: number;
}

// -- Pipelines (frontend types matching API schemas) --

export interface PipelineStepInfo {
	name: string;
	agent: string;
	prompt_template: string | null;
	description: string;
	system_prompt: string | null;
	model: string | null;
}

export interface PipelineInfo {
	id: string;
	name: string;
	description: string;
	pattern: string | null;
	emoji: string;
	steps: PipelineStepInfo[];
	keywords: string[];
	detection_weight: number;
	created_at: string | null;
	is_builtin: boolean;
	step_count: number;
}

export interface PipelineCreate {
	id: string;
	name: string;
	description?: string;
	pattern?: string;
	emoji?: string;
	steps?: PipelineStepInfo[];
	keywords?: string[];
	detection_weight?: number;
}

export interface PipelineUpdate {
	name?: string;
	description?: string;
	pattern?: string;
	emoji?: string;
	steps?: PipelineStepInfo[];
	keywords?: string[];
	detection_weight?: number;
}

export interface PipelineStats {
	total: number;
	builtin: number;
	custom: number;
	total_steps: number;
	total_keywords: number;
	by_pattern: Record<string, number>;
	available_agents: number;
	available_templates: number;
}

export interface PipelineExportData {
	yaml_content: string;
}


// -- Execution Pipelines --

export interface ExecStepInfo {
	step_type: string;
	label: string;
	model_override: string | null;
	parameters: Record<string, any>;
	condition: string | null;
	pass_previous_output: boolean;
}

export interface ExecPipelineInfo {
	id: string;
	name: string;
	description: string;
	steps: ExecStepInfo[];
	created_at: string;
	updated_at: string;
	is_builtin: boolean;
	step_count: number;
	step_types_summary: string;
}

export interface ExecPipelineCreate {
	id: string;
	name: string;
	description?: string;
	steps: ExecStepInfo[];
}

export interface ExecPipelineUpdate {
	name?: string;
	description?: string;
	steps?: ExecStepInfo[];
}

export interface ExecStepTypeInfo {
	type: string;
	description: string;
}
// -- Export --

export interface ExportResponse {
	conversation_id: string;
	format: string;
	content: string;
	filename?: string;
}

// -- Keyboard Shortcuts --

export interface KeyboardShortcut {
	key: string;
	ctrl?: boolean;
	shift?: boolean;
	alt?: boolean;
	description: string;
	action: string;
}

// -- Panels --

export type PanelType = 'none' | 'artifacts' | 'code' | 'memory' | 'pipelines' | 'context' | 'exec-pipelines' | 'plugins' | 'agent' | 'sandbox';

// --: Feedback & Analytics types --

export interface FeedbackSubmitRequest {
	conversation_id?: string;
	message_id?: string;
	rating_type: 'thumbs' | 'stars';
	rating_value: number;
	feedback_text?: string;
	model_used?: string;
	pipeline_used?: string;
	task_type?: string;
}

export interface FeedbackEntryInfo {
	feedback_id: string;
	conversation_id: string;
	message_id: string;
	rating_type: string;
	rating_value: number;
	feedback_text: string;
	model_used: string;
	pipeline_used: string;
	task_type: string;
	timestamp: number;
}

export interface FeedbackStatsInfo {
	total_count: number;
	positive_count: number;
	negative_count: number;
	average_score: number;
	thumbs_up: number;
	thumbs_down: number;
	star_distribution: Record<number, number>;
	by_model: Record<string, Record<string, any>>;
	by_pipeline: Record<string, Record<string, any>>;
	by_task_type: Record<string, Record<string, any>>;
}

export interface AnalyticsOverviewInfo {
	total_requests: number;
	success_count: number;
	error_count: number;
	success_rate: number;
	avg_response_time_ms: number;
	avg_tokens_per_second: number;
	total_tokens_processed: number;
	pipeline_distribution: Record<string, number>;
	model_distribution: Record<string, number>;
	task_type_distribution: Record<string, number>;
	model_performance: Record<string, Record<string, any>>;
	pipeline_performance: Record<string, Record<string, any>>;
}

export interface TrendPointInfo {
	window_start: number;
	window_end: number;
	count: number;
	avg_response_time_ms: number;
	avg_tokens_per_second: number;
	total_tokens: number;
	success_rate: number;
}

export interface TrendsInfo {
	window: string;
	buckets: number;
	model: string | null;
	pipeline: string | null;
	data: TrendPointInfo[];
}

export interface RoutingAccuracyInfo {
	routed: Record<string, any>;
	unrouted: Record<string, any>;
}

// -- Context Health --

export interface ContextConversationInfo {
	conversation_id: string | null;
	model: string | null;
	model_context_window: number;
	messages_count: number;
	estimated_tokens: number;
	usage_percent: number;
	trimming_active: boolean;
	last_window_stats: Record<string, any>;
}

export interface ContextBudgetAllocation {
	system_prompt: number;
	history: number;
	reserved_for_response: number;
	total_allocated: number;
	context_window: number;
	system_ratio: number;
	history_ratio: number;
	generation_ratio: number;
}

export interface ContextHealthResponse {
	status: string;
	context_window_available: boolean;
	executor_available: boolean;
	conversation_available: boolean;
	current_conversation: ContextConversationInfo;
	budget_allocation: ContextBudgetAllocation;
}

export interface ContextBudgetResponse {
	available: boolean;
	model: string;
	budget: ContextBudgetAllocation | null;
}

export interface ContextStatsResponse {
	available: boolean;
	window_stats: Record<string, any>;
	trimming_history: any[];
}

// -- Errors --

export interface ApiErrorDetail {
	detail: string;
}

// -- Consensus --

export interface ConsensusModelResponse {
	model: string;
	content: string;
	duration_ms: number;
	success: boolean;
	error: string;
	quality_tier: string;
}

export interface ConsensusComparison {
	agreement_matrix: Record<string, Record<string, number>>;
	average_agreement: number;
	areas_of_agreement: string[];
	areas_of_disagreement: string[];
}

export interface ConsensusResult {
	strategy: string;
	selected_response: string;
	selected_model: string;
	confidence: number;
	individual_responses: ConsensusModelResponse[];
	comparison: ConsensusComparison | null;
	total_duration_ms: number;
	metadata: Record<string, any>;
}

export interface ConsensusConfig {
	default_models: string[];
	strategy: string;
	max_models: number;
	timeout_per_model: number;
	min_agreement_threshold: number;
	available: boolean;
}

// -- Self-Correction --

export interface CorrectionInfo {
	was_corrected: boolean;
	iterations_performed: number;
	compliance_before: number;
	compliance_after: number;
	quality_before: number;
	quality_after: number;
	total_duration_ms: number;
}

export interface CorrectionStepInfo {
	iteration: number;
	compliance_score: number;
	quality_score: number;
	improvements: string[];
	duration_ms: number;
}

// Vision delegation info
export interface VisionDelegationInfo {
	vision_model: string;
	description_length: number;
	duration_ms: number;
}

export interface CorrectionConfig {
	enable_auto: boolean;
	max_iterations: number;
	compliance_threshold: number;
	quality_threshold: number;
	check_instructions: boolean;
	check_facts: boolean;
	check_quality: boolean;
	available: boolean;
}

// -- Projects --

export interface ProjectInfo {
	id: string;
	name: string;
	description: string;
	system_instructions: string;
	settings: Record<string, unknown>;
	created_at: string;
	updated_at: string;
}

export interface ProjectDetailInfo extends ProjectInfo {
	files: ProjectFileInfo[];
	outputs: ProjectOutputInfo[];
	conversations: { conversation_id: string; linked_at: string }[];
	stats: ProjectStats;
}

export interface ProjectStats {
	file_count: number;
	output_count: number;
	conversation_count: number;
	total_file_size_bytes: number;
	indexed_file_count: number;
	total_chunk_count: number;
}

export interface ProjectFileInfo {
	id: string;
	project_id: string;
	filename: string;
	file_path: string;
	file_type: string;
	file_size_bytes: number;
	indexed: boolean;
	chunk_count: number;
	summary: string;
	key_terms: string[];
	uploaded_at: string;
	updated_at: string;
}

export interface ProjectOutputInfo {
	id: string;
	project_id: string;
	source_conversation_id: string;
	filename: string;
	file_path: string;
	output_type: string;
	description: string;
	created_at: string;
}

export interface ProjectContextPreview {
	trigger: {
		relevant: boolean;
		level: string;
		confidence: number;
		reason: string;
		duration_ms: number;
	};
	context: {
		chunks_retrieved: number;
		total_tokens: number;
		source_files: string[];
		content_preview: string;
	} | null;
	system_instructions: string;
}

export interface ProjectFileSummary {
	file_id: string;
	filename: string;
	summary: string;
	key_terms: string[];
	indexed: boolean;
	chunk_count: number;
}

// -- Benchmark Dashboard --

export interface BenchmarkSuiteInfo {
	id: string;
	name: string;
	description: string;
	task_count: number;
	tasks: string[];
	categories: string[];
}

export interface BenchmarkSuiteDetail {
	id: string;
	name: string;
	description: string;
	tasks: BenchmarkTaskInfo[];
}

export interface BenchmarkTaskInfo {
	id: string;
	name: string;
	description: string;
	category: string;
	prompt?: string;
	expected_keywords?: string[];
	max_expected_time?: number;
	scoring_method?: string;
}

export interface BenchmarkRunRequest {
	models?: string[];
	tasks?: string[];
	suite_id?: string;
	temperature?: number;
	timeout?: number;
	max_tokens?: number;
}

export interface BenchmarkRunSummaryInfo {
	id: string;
	run_type: string;
	started_at: string;
	completed_at: string;
	status: string;
	models_tested: string[];
	tasks_tested: string[];
	total_tests: number;
	avg_score: number | null;
	best_model: string | null;
	duration_sec: number | null;
}

export interface BenchmarkResultItem {
	model: string;
	task: string;
	task_name: string;
	category: string;
	score: number;
	auto_score: number;
	user_score: number | null;
	time_seconds: number;
	status: string;
	response_preview: string;
	keywords_found: string[];
	keywords_missing: string[];
	error_message: string | null;
}

export interface BenchmarkRunDetailInfo {
	id: string;
	run_type: string;
	started_at: string;
	completed_at: string;
	status: string;
	models: string[];
	tasks: string[];
	total_tests: number;
	avg_score: number | null;
	best_model: string | null;
	duration_sec: number | null;
	results: BenchmarkResultItem[];
	global_ranking: Array<{ model: string; avg_score: number; avg_time: number; tests: number; rank: number }>;
	best_by_category: Record<string, string>;
	config_snapshot: Record<string, unknown>;
	error: string | null;
}

export interface BenchmarkComparisonInfo {
	runs: BenchmarkRunSummaryInfo[];
	matrix: Record<string, Record<string, Array<number | null>>>;
	deltas: Array<{ model: string; task: string; delta: number; direction: string }>;
	regressions: Array<{ model: string; task: string; delta: number; direction: string }>;
}

export interface BenchmarkModelTrendInfo {
	model: string;
	run_ids: string[];
	run_dates: string[];
	avg_scores: number[];
	avg_times: number[];
}

export interface BenchmarkProgressEvent {
	total_tests: number;
	completed_tests: number;
	current_model: string;
	current_task: string;
	current_task_name: string;
	percent: number;
	elapsed_sec: number;
	estimated_remaining_sec: number;
}

export interface ModelConfigInfo {
	config: Record<string, unknown>;
	installed_models: string[];
}

export interface ModelRoleInfo {
	role: string;
	primary: string;
	fast: string;
	quality: string;
}

// ---------------------------------------------------------------------------
// Network Manager / Offline-First Intelligence
// ---------------------------------------------------------------------------

export interface NetworkStatusInfo {
	available: boolean;
	online: boolean;
	ollama_reachable: boolean;
	embedding_reachable: boolean;
	last_check: number;
	last_error: string;
	latency_ms: number;
	consecutive_failures: number;
	polling_active: boolean;
	queue_size: number;
	config: Record<string, unknown>;
}

export interface QueueEntryInfo {
	id: string;
	query: string;
	task_type: string;
	priority: number;
	created_at: number;
	status: string;
	error: string;
	model: string;
}

export interface QueueListInfo {
	available: boolean;
	entries: QueueEntryInfo[];
	total: number;
	pending: number;
}

export interface PreCacheInfo {
	total: number;
	cached: number;
	skipped: number;
	failed: number;
	duration_ms: number;
	errors: string[];
}

// -- Performance Monitor --

export interface PerformanceSummary {
	available: boolean;
	enabled: boolean;
	throughput: {
		tokens_in_per_sec: number;
		tokens_out_per_sec: number;
		total_tokens: number;
		request_count: number;
		window_seconds: number;
	};
	latency: {
		p50: number;
		p95: number;
		p99: number;
		mean: number;
		count: number;
	};
	utilization: Record<string, number>;
}

export interface PerformanceLatencyStats {
	available: boolean;
	model: string | null;
	window_seconds: number;
	p50: number;
	p95: number;
	p99: number;
	mean: number;
	count: number;
}

export interface PerformanceDriftEntry {
	model: string;
	metric: string;
	baseline_value: number;
	recent_value: number;
	change_ratio: number;
	is_drifted: boolean;
	direction: string;
}

export interface PerformanceDriftResponse {
	available: boolean;
	drifts: PerformanceDriftEntry[];
}

export interface PerformanceRecommendation {
	model: string;
	metric: string;
	message: string;
	severity: string;
	value: number;
}

export interface PerformanceRecommendationsResponse {
	available: boolean;
	recommendations: PerformanceRecommendation[];
}

export interface PerformanceHistoryRecord {
	model: string;
	task_type: string;
	latency_ms: number;
	tokens_in: number;
	tokens_out: number;
	quality_score: number;
	timestamp: number;
}

export interface PerformanceHistoryResponse {
	available: boolean;
	model: string | null;
	hours: number;
	count: number;
	records: PerformanceHistoryRecord[];
}

export interface PerformanceThroughput {
	available: boolean;
	tokens_in_per_sec: number;
	tokens_out_per_sec: number;
	total_tokens: number;
	request_count: number;
	window_seconds: number;
}

export interface PerformanceUtilization {
	available: boolean;
	window_seconds: number;
	models: Record<string, number>;
}

// -- Sandbox --

export interface SandboxStatusResponse {
	available: boolean;
	enabled: boolean;
	isolation_backend: string;
	bwrap_available: boolean;
	degraded_mode: boolean;
	degraded_confirmed: boolean;
	active_sessions: number;
	max_sessions: number;
	/** (Bloc 4): live egress-gate answer (true only in Daily; the
	 * SyncPanel bulbe_disabled precedent, fail-secure false) plus the
	 * configured caps, read-only, for the settings strip. */
	network_allowed?: boolean;
	command_timeout_default?: number | null;
	limit_memory_bytes?: number | null;
	limit_nproc?: number | null;
	limit_cpu_seconds?: number | null;
	disk_soft_limit_bytes?: number | null;
}

export interface SandboxCreateRequest {
	session_id?: string;
	allow_degraded?: boolean;
	/** (Bloc 1): optional human label and per-sandbox command timeout. */
	label?: string;
	timeout?: number | null;
}

export interface SandboxCreateResponse {
	session_id: string;
	workspace_path: string;
	isolation_backend: string;
	degraded: boolean;
	label?: string;
}

export interface SandboxInjectRequest {
	session_id: string;
	file_paths: string[];
}

export interface SandboxInjectResponse {
	session_id: string;
	injected_count: number;
	injected_paths: string[];
}

export interface SandboxFileEntry {
	path: string;
	size: number;
	modified: number;
	approved: boolean;
}

export interface SandboxFilesResponse {
	session_id: string;
	files: SandboxFileEntry[];
	count: number;
	approval_state: string;
}

export interface SandboxExecuteRequest {
	session_id: string;
	tool_name: string;
	arguments: Record<string, unknown>;
}

export interface SandboxExecuteResponse {
	session_id: string;
	tool_name: string;
	result: string;
	blocked: boolean;
	block_reason: string;
	timed_out: boolean;
	isolation_backend: string;
}

export interface SandboxDestroyResponse {
	session_id: string;
	destroyed: boolean;
}

export interface SandboxSessionInfo {
	session_id: string;
	workspace_path: string;
	isolation_backend: string;
	created_at: number;
	active: boolean;
	command_count: number;
	approval_state: string;
	approved_paths: string[];
	approved_at: number | null;
	/** (Bloc 1): the workspace-manager view. network_enabled is the
	 * per-workspace flag (Bloc 4: user-set only, Daily-only, default
	 * false); disk_use_bytes is approximate. has_cloned_baseline is
	 * true when a host clone recorded a baseline root -- the settings strip
	 * sharpens the exfiltration warning with it. */
	label: string;
	owner_user_id: string;
	bound_conversation_id: string | null;
	network_enabled: boolean;
	last_activity: number;
	timeout_override: number | null;
	age_seconds: number;
	running: boolean;
	disk_use_bytes: number;
	has_cloned_baseline?: boolean;
}

/** (Bloc 1): stop-path response. stopped=false means nothing was
 * running (honest no-op); the workspace persists either way. */
export interface SandboxStopResponse {
	session_id: string;
	stopped: boolean;
}

/** (Bloc 1): bind a conversation to a workspace. */
export interface SandboxBindRequest {
	conversation_id: string;
	session_id: string;
}

/** (Bloc 1): the current binding of a conversation. */
export interface SandboxBindingResponse {
	conversation_id: string;
	session_id: string | null;
	bound: boolean;
}

/** Per-file upload refusal: invalid name or destination collision. */
export interface SandboxUploadRefused {
	name: string;
	reason: string;
}

/** Result of a multipart drag-and-drop upload (Bloc 2). */
export interface SandboxUploadResponse {
	session_id: string;
	uploaded_paths: string[];
	refused: SandboxUploadRefused[];
	uploaded_bytes: number;
	manifest_files: number;
}

/** One immediate entry of an allowlisted host directory. */
export interface HostBrowseEntry {
	name: string;
	type: 'dir' | 'file' | 'symlink' | 'special';
	size: number;
	hidden: boolean;
}

/** Allowlisted host directory listing (Bloc 2). */
export interface HostBrowseResponse {
	path: string;
	roots: string[];
	entries: HostBrowseEntry[];
}

/** Request to clone an allowlisted host directory (Bloc 2). */
export interface SandboxCloneRequest {
	src_path: string;
	dest_subdir?: string;
}

/** Result of a symlink-safe host clone (Bloc 2). */
export interface SandboxCloneResponse {
	session_id: string;
	dest: string;
	cloned_root: string;
	copied_files: number;
	copied_bytes: number;
	skipped_symlinks: number;
	skipped_special: number;
	manifest_files: number;
}

/** One classified change against the baseline manifest (Bloc 3). */
export interface SandboxDiffEntry {
	path: string;
	kind: 'added' | 'modified' | 'deleted';
	size: number;
	baseline_hash: string;
	current_hash: string;
}

/** The workspace diff against the recorded baseline (Bloc 3).
 * diff_hash is the review-integrity digest the apply request must echo;
 * baseline_present false means no baseline (everything "added", no
 * implicit write-back target). */
export interface SandboxDiffResponse {
	session_id: string;
	baseline_present: boolean;
	cloned_root: string | null;
	cloned_mount: string | null;
	entries: SandboxDiffEntry[];
	unchanged: number;
	skipped_symlinks: number;
	skipped_special: number;
	diff_hash: string;
	approved_paths: string[];
	confirmed_deletions: string[];
}

/** Explicit deletion confirmation for apply: never bundled into a
 * blanket approve-all. */
export interface SandboxConfirmDeletionsRequest {
	paths: string[];
}

/** A per-path deletion-confirmation refusal. */
export interface SandboxConfirmDeletionsRefused {
	path: string;
	reason: string;
}

/** Result of a deletion-confirmation request. */
export interface SandboxConfirmDeletionsResponse {
	session_id: string;
	confirmed: string[];
	refused: SandboxConfirmDeletionsRefused[];
}

/** Apply approved changes back to the host (Bloc 3). */
export interface SandboxApplyRequest {
	diff_hash: string;
	target_dir?: string;
}

/** One applied change. */
export interface SandboxApplyEntry {
	path: string;
	action: 'created' | 'modified' | 'deleted' | 'already_absent';
	bytes: number;
}

/** One refused apply path with its honest reason. */
export interface SandboxApplyRefusedEntry {
	path: string;
	error: string;
}

/** Result of an apply-to-host run (Bloc 3). */
export interface SandboxApplyResponse {
	session_id: string;
	target: string;
	applied: SandboxApplyEntry[];
	deleted: SandboxApplyEntry[];
	refused: SandboxApplyRefusedEntry[];
	skipped_unapproved: number;
	skipped_unconfirmed: number;
	diff_hash: string;
}

/** (Bloc 4): the per-workspace network toggle. Enabling is Daily-only
 * (403 under Bulbe at the binding-layer gate; an unset or unknown mode is
 * treated as Bulbe); disabling works in any mode. User action only. */
export interface SandboxNetworkToggleRequest {
	enabled: boolean;
}

export interface SandboxNetworkToggleResponse {
	session_id: string;
	network_enabled: boolean;
}

/** (Bloc 4): the provision run -- the one scoped egress. The
 * requirements set must be exact name==version pins carrying
 * --hash=sha256: hashes; option lines are refused per line and nothing
 * installs on a partial validation. */
export interface SandboxProvisionRequest {
	requirements_path: string;
	venv_dir?: string;
}

export interface SandboxProvisionRefusedLine {
	line: number;
	text: string;
	reason: string;
}

export interface SandboxProvisionResponse {
	session_id: string;
	command: string;
	return_code: number;
	blocked: boolean;
	block_reason: string;
	timed_out: boolean;
	isolation_backend: string;
	stdout_tail: string;
	stderr_tail: string;
	accepted_requirements: string[];
}

export interface SandboxAuditEntry {
	id: number;
	session_id: string;
	timestamp: number;
	command: string;
	return_code: number | null;
	blocked: boolean;
	block_reason: string;
	timed_out: boolean;
	stdout_len: number;
	stderr_len: number;
	isolation_backend: string;
}

export interface SandboxAuditResponse {
	entries: SandboxAuditEntry[];
	count: number;
}

export interface SandboxConfirmDegradedResponse {
	confirmed: boolean;
	warning: string;
}

// -- Sandbox Copy-Out --

export interface SandboxPreviewResponse {
	session_id: string;
	path: string;
	content: string;
	size: number;
	truncated: boolean;
	is_binary: boolean;
}

export interface SandboxApproveRequest {
	paths: string[];
	dest_dir?: string | null;
}

export interface SandboxApproveResponse {
	session_id: string;
	approved_paths: string[];
	approved_count: number;
	approval_state: string;
}

export interface SandboxCopyOutEntry {
	src_path: string;
	dest_path: string;
	size: number;
}

export interface SandboxCopyOutResponse {
	session_id: string;
	copied: SandboxCopyOutEntry[];
	copied_count: number;
	dest_dir: string;
}

export interface SandboxRejectResponse {
	session_id: string;
	rejected: boolean;
	approval_state: string;
}

export interface SandboxApprovalInfoResponse {
	session_id: string;
	approval_state: string;
	approved_paths: string[];
	approved_at: number | null;
}

export interface SandboxApprovalAuditEntry {
	id: number;
	session_id: string;
	timestamp: number;
	action: string;
	paths: string;
	dest_dir: string;
	detail: string;
}

export interface SandboxApprovalAuditResponse {
	entries: SandboxApprovalAuditEntry[];
	count: number;
}

// -- Coding Agent --

export interface CodingTaskRequest {
	task: string;
	project_path?: string | null;
	model?: string | null;
	allow_degraded?: boolean;
}

export interface CodingPlanStepResponse {
	step_number: number;
	step_type: string;
	description: string;
	file_path: string;
	command: string;
	completed: boolean;
	result: string;
	error: string;
}

export interface CodingPlanResponse {
	task: string;
	summary: string;
	estimated_files: number;
	total_steps: number;
	completed_steps: number;
	steps: CodingPlanStepResponse[];
}

export interface CodingCheckpointRequest {
	decision: string;
	modified_plan?: Record<string, unknown> | null;
}

export interface CodingStepResponse {
	step_number: number;
	step_type: string;
	description: string;
	completed: boolean;
	result: string;
	error: string;
}

export interface CodingTestResultResponse {
	passed: boolean;
	output: string;
	error: string;
	return_code: number;
}

export interface CodingDiffEntry {
	path: string;
	is_new: boolean;
	is_deleted: boolean;
	diff: string;
}

export interface CodingDiffResponse {
	count: number;
	diffs: CodingDiffEntry[];
}

export interface CodingApplyRequest {
	target_path?: string | null;
}

export interface CodingApplyResponse {
	applied: number;
	files: Record<string, unknown>[];
	errors: Record<string, unknown>[];
}

export interface CodingHistoryEntryResponse {
	timestamp: number;
	phase: string;
	action: string;
	detail: string;
	success: boolean;
}

export interface CodingStatusResponse {
	task_id: string;
	task: string;
	phase: string;
	session_active: boolean;
	plan: CodingPlanResponse | null;
	current_step: number;
	total_steps: number;
	iteration: number;
	max_iterations: number;
	fix_count: number;
	max_fix_retries: number;
	test_results: CodingTestResultResponse[];
	diffs: CodingDiffEntry[];
	history_count: number;
	history: CodingHistoryEntryResponse[];
	working_memory: WorkingMemoryResponse | null;
	cascading: CodingCascadingStatus | null;
}

// -- Cascading Status --

export interface CodingCascadingStatus {
	enabled: boolean;
	available: boolean;
	escalated_model: string | null;
	consecutive_fix_failures: number;
	escalate_after_failures: number;
	per_step_routing: boolean;
}

// -- Working Memory --

export interface WorkingMemoryResponse {
	task_id: string;
	decisions: string[];
	modified_files: Record<string, string>;
	errors_encountered: string[];
	open_questions: string[];
	progress_notes: string[];
}

export interface WorkingMemoryCompactResponse {
	task_id: string;
	compact: string;
}

// -- Coding History --

export interface CodingTaskSummaryResponse {
	task_id: string;
	task_text: string;
	project_path: string;
	model: string;
	status: string;
	step_count: number;
	completed_steps: number;
	test_runs: number;
	last_passed: boolean | null;
	created_at: number;
	completed_at: number | null;
}

export interface CodingTaskDetailResponse {
	task_id: string;
	task_text: string;
	project_path: string;
	model: string;
	status: string;
	plan_json: Record<string, unknown> | null;
	created_at: number;
	completed_at: number | null;
	steps: Record<string, unknown>[];
	tests: Record<string, unknown>[];
	checkpoints: Record<string, unknown>[];
}

export interface CodingHistoryListResponse {
	tasks: CodingTaskSummaryResponse[];
	total: number;
}

export interface CodingHistoryStatsResponse {
	total_tasks: number;
	by_status: Record<string, number>;
	total_steps: number;
	total_tests: number;
	passed_tests: number;
	total_checkpoints: number;
}

export interface CodingResumeRequest {
	model?: string;
}

export interface CodingResumeResponse {
	task_id: string;
	task_text: string;
	project_path: string;
	model: string;
	plan_json: Record<string, unknown> | null;
	current_step: number;
	phase: string;
	originals_hash: string;
}

// -- Coding Analytics (SQ-08) --

export interface CodingModelSuccessRate {
	model: string;
	total: number;
	completed: number;
	success_rate: number;
}

export interface CodingModelAvgSteps {
	model: string;
	avg_steps: number;
	min_steps: number;
	max_steps: number;
	task_count: number;
}

export interface CodingAvgStepsOverall {
	avg_steps: number;
	min_steps: number;
	max_steps: number;
	task_count: number;
}

export interface CodingFailureReason {
	failure_phase: string;
	count: number;
}

export interface CodingTimeTrend {
	task_id: string;
	model: string;
	created_at: number;
	completed_at: number;
	duration_seconds: number;
}

export interface CodingTestPassRate {
	task_id: string;
	model: string;
	total_runs: number;
	passed_runs: number;
	pass_rate: number;
}

export interface CodingStepsDistribution {
	step_count: number;
	task_count: number;
}

export interface CodingAnalyticsResponse {
	total_tasks: number;
	completed_tasks: number;
	overall_success_rate: number;
	success_rate_by_model: CodingModelSuccessRate[];
	avg_steps_by_model: CodingModelAvgSteps[];
	avg_steps_overall: CodingAvgStepsOverall;
	failure_reasons: CodingFailureReason[];
	time_trends: CodingTimeTrend[];
	test_pass_rate_per_task: CodingTestPassRate[];
	steps_distribution: CodingStepsDistribution[];
}

export interface CodingExecuteAllStatus {
	is_running: boolean;
	should_stop: boolean;
	error: string;
	executed_count: number;
	task_id: string;
}

// -- Export & Batch Delete --

export interface CodingBatchDeleteRequest {
	task_ids?: string[];
	before_date?: string;
}

export interface CodingBatchDeleteResponse {
	deleted: number;
}

export interface CodingExportRow {
	task_id: string;
	task_text: string;
	model: string;
	status: string;
	step_count: number;
	test_runs: number;
	pass_rate: number;
	created_at: number | null;
	completed_at: number | null;
	duration_seconds: number | null;
}

// Session Fingerprint

export interface FingerprintDimensionResponse {
	name: string;
	data: Record<string, unknown>;
}

export interface FingerprintCompactResponse {
	compact: string;
	token_estimate: number;
}

export interface FingerprintFullResponse {
	d1_task_type: Record<string, unknown>;
	d2_stack: Record<string, unknown>;
	d3_hot_files: Record<string, unknown>;
	d4_recent_bugs: Record<string, unknown>;
	d5_test_health: Record<string, unknown>;
	d6_momentum: Record<string, unknown>;
	d7_domain_terms: Record<string, unknown>;
	d8_dep_clusters: Record<string, unknown>;
	d9_user_preferences: Record<string, unknown>;
	d10_context_anchors: Record<string, unknown>;
	step_count: number;
	config: Record<string, unknown>;
}

// System Presets & Onboarding

export interface SystemPresetModelInfo {
	name: string;
	size_bytes: number;
	parameter_count_b: number;
	quantization: string;
	family: string;
	size_category: string;
}

export interface SystemPresetInfo {
	id: string;
	name: string;
	description: string;
	icon: string;
	recommended_vram_gb: number;
	recommended_ram_gb: number;
	model_strategy: string;
	pipelines: string[];
}

export interface SystemPresetListResponse {
	presets: SystemPresetInfo[];
}

export interface SystemPresetDetectResponse {
	models: SystemPresetModelInfo[];
	recommended_preset: string;
	reason: string;
	model_counts: Record<string, number>;
	total_estimated_vram_gb: number;
}

export interface SystemPresetApplyResponse {
	applied: boolean;
	preset_id: string;
	preset_name: string;
	selected_model: string | null;
	applied_configs: Record<string, string[]>;
	pipelines: string[];
	warnings: string[];
	error: string | null;
}

export interface OnboardingStateResponse {
	user_initialized: boolean;
	applied_preset: string | null;
	applied_at: string | null;
}

// -- Humanizer --

export interface HumanizerRewriteRequest {
	text: string;
	model?: string | null;
	mode?: string | null;
	intensity?: string | null;
	formality?: string | null;
}

export interface HumanizerRewriteResponse {
	original: string;
	humanized: string;
	strategies_applied: string[];
	replacements_count: number;
	rewrite_model: string | null;
	latency_ms: number;
	mode: string;
	intensity: string;
	comparison_id: string;
}

export interface HumanizerConfigResponse {
	enabled: boolean;
	available: boolean;
	mode: string;
	intensity: string;
	formality: string;
	rewrite_model: string | null;
	max_input_length: number;
	banned_phrases: string[];
	vocabulary_replacements: Record<string, string>;
}

export interface HumanizerConfigUpdate {
	enabled?: boolean | null;
	mode?: string | null;
	intensity?: string | null;
	formality?: string | null;
	rewrite_model?: string | null;
	max_input_length?: number | null;
	banned_phrases?: string[] | null;
	vocabulary_replacements?: Record<string, string> | null;
}

export interface HumanizerFeedbackRequest {
	comparison_id: string;
	winner: string;
}

export interface HumanizerFeedbackResponse {
	success: boolean;
	comparison_id: string;
	winner: string;
}

export interface HumanizerStrategyStats {
	humanized: number;
	original: number;
	tie: number;
}

export interface HumanizerStatsResponse {
	total_ratings: number;
	humanized_wins: number;
	original_wins: number;
	ties: number;
	win_rate: number;
	by_strategy: Record<string, HumanizerStrategyStats>;
	by_model: Record<string, HumanizerStrategyStats>;
	by_intensity: Record<string, HumanizerStrategyStats>;
}

// -- Benchmark V2 --

export interface BenchmarkV2Profile {
	id: string;
	name: string;
	description: string;
	categories: string[];
	weight_preset: string;
	custom?: boolean;
}

export interface BenchmarkV2ProfilesResponse {
	profiles: BenchmarkV2Profile[];
	available_categories: string[];
	total_questions: number;
}

export interface BenchmarkV2RunRequest {
	profile: string;
	models: string[];
	use_judge?: boolean;
	judge_model?: string;
	custom_weights?: Record<string, number> | null;
}

export interface BenchmarkV2RunStarted {
	run_id: string;
	profile: string;
	models: string[];
	status: string;
}

export interface BenchmarkV2Progress {
	run_id: string;
	status: string;
	total_questions: number;
	completed_questions: number;
	current_model: string;
	current_question: string;
	elapsed_ms: number;
	error: string;
}

export interface BenchmarkV2ModelScore {
	model: string;
	accuracy_avg: number;
	code_avg: number;
	structure_avg: number;
	speed_avg: number;
	composite: number;
	questions_evaluated: number;
}

export interface BenchmarkV2QuestionResult {
	question_id: string;
	category: string;
	prompt: string;
	response: string;
	accuracy_score: number;
	code_score: number;
	structure_score: number;
	speed_score: number;
	composite_score: number;
	details: Record<string, unknown>;
}

export interface BenchmarkV2Results {
	run_id: string;
	profile: string;
	models: string[];
	status: string;
	started_at: number;
	finished_at: number;
	duration_ms: number;
	weight_preset: string;
	custom_weights: Record<string, number> | null;
	model_scores: Record<string, BenchmarkV2ModelScore>;
	question_results: Record<string, BenchmarkV2QuestionResult[]>;
	judge_scores: BenchmarkV2JudgeScore[];
	judge_summary: Record<string, unknown>;
	error: string;
}

export interface BenchmarkV2CompareResponse {
	models: Record<string, unknown>[];
	profile_filter: string | null;
	model_filter: string[] | null;
}

export interface BenchmarkV2HistoryEntry {
	run_id: string;
	profile: string;
	models: string[];
	status: string;
	started_at: number;
	duration_ms: number;
	weight_preset: string;
	custom_weights: Record<string, number> | null;
	model_scores: Record<string, BenchmarkV2ModelScore>;
}

export interface BenchmarkV2HistoryResponse {
	runs: BenchmarkV2HistoryEntry[];
	total: number;
}

// — LLM-as-Judge, Leaderboard, Head-to-Head, Trends, Recommendations

export interface BenchmarkV2JudgeScore {
	question_id: string;
	model: string;
	judge_model: string;
	accuracy: number;
	relevance: number;
	completeness: number;
	conciseness: number;
	reasoning: number;
	justification: string;
	weighted_score: number;
	tokens_used: number;
	eval_time_ms: number;
	error: string;
}

export interface BenchmarkV2LeaderboardEntry {
	rank: number;
	model: string;
	composite: number;
	accuracy_avg: number;
	code_avg: number;
	structure_avg: number;
	speed_avg: number;
	judge_avg: number;
	run_count: number;
	last_run: number;
}

export interface BenchmarkV2LeaderboardResponse {
	profile: string;
	entries: BenchmarkV2LeaderboardEntry[];
	total: number;
}

export interface BenchmarkV2HeadToHeadMetric {
	metric: string;
	model_a_value: number;
	model_b_value: number;
	winner: string;
}

export interface BenchmarkV2HeadToHeadResponse {
	model_a: string;
	model_b: string;
	metrics: BenchmarkV2HeadToHeadMetric[];
	overall_winner: string;
	model_a_wins: number;
	model_b_wins: number;
	ties: number;
}

export interface BenchmarkV2TrendPoint {
	run_id: string;
	timestamp: number;
	composite: number;
	accuracy: number;
	code: number;
	structure: number;
	speed: number;
	profile: string;
}

export interface BenchmarkV2TrendResponse {
	model: string;
	points: BenchmarkV2TrendPoint[];
	trend_direction: string;
	regression_detected: boolean;
}

export interface BenchmarkV2RecommendationEntry {
	role: string;
	model: string;
	composite_score: number;
	speed_score: number;
	accuracy_score: number;
	code_score: number;
	structure_score: number;
	tokens_per_second: number;
	reason: string;
}

export interface BenchmarkV2RecommendationsResponse {
	snapshot_id: string;
	created_at: number;
	profile: string;
	recommendations: BenchmarkV2RecommendationEntry[];
	applied: boolean;
	applied_at: number;
}

export interface BenchmarkV2ApplyResponse {
	applied: boolean;
	snapshot_id: string;
	changes: Record<string, unknown>;
	error: string;
}

// — Custom Profiles

export interface BenchmarkV2CustomProfile {
	profile_id: string;
	name: string;
	description: string;
	categories: string[];
	weight_preset: string;
	custom_weights?: Record<string, number> | null;
	timeout: number;
	max_response_tokens: number;
	expected_length_range: number[];
	created_at: number;
	updated_at: number;
}

export interface BenchmarkV2CustomProfileCreate {
	name: string;
	description?: string;
	categories: string[];
	weight_preset?: string;
	custom_weights?: Record<string, number> | null;
	timeout?: number;
	max_response_tokens?: number;
	expected_length_range?: number[];
}

export interface BenchmarkV2CustomProfileUpdate {
	name?: string;
	description?: string;
	categories?: string[];
	weight_preset?: string;
	custom_weights?: Record<string, number> | null;
	timeout?: number;
	max_response_tokens?: number;
	expected_length_range?: number[];
}

export interface BenchmarkV2CustomProfilesListResponse {
	profiles: BenchmarkV2CustomProfile[];
	count: number;
}

export interface BenchmarkV2QuestionPreview {
	category_counts: Record<string, number>;
	total: number;
}

// — Auto-Trigger

export interface BenchmarkV2AutoTriggerStatus {
	enabled: boolean;
	running: boolean;
	poll_interval_seconds: number;
	cooldown_seconds: number;
	cooldown_remaining: number;
	trigger_profile: string;
	last_trigger_time: number;
	known_models: number;
	recent_events: number;
	resource_guard_active: boolean;
	resource_guard_load_max: number;
}

export interface BenchmarkV2AutoTriggerConfig {
	enabled: boolean;
	poll_interval_seconds: number;
	cooldown_seconds: number;
	trigger_profile: string;
	trigger_models: string | string[];
	resource_guard_load_max: number;
	use_judge: boolean;
	judge_model: string;
}

export interface BenchmarkV2AutoTriggerConfigUpdate {
	enabled?: boolean;
	poll_interval_seconds?: number;
	cooldown_seconds?: number;
	trigger_profile?: string;
	trigger_models?: string | string[];
	resource_guard_load_max?: number;
	use_judge?: boolean;
	judge_model?: string;
}

export interface BenchmarkV2AutoTriggerEvent {
	event_id: string;
	timestamp: number;
	trigger_type: string;
	models: string[];
	run_id: string;
	profile: string;
	skipped: boolean;
	skip_reason: string;
}

export interface BenchmarkV2AutoTriggerEventsResponse {
	events: BenchmarkV2AutoTriggerEvent[];
	count: number;
}

export interface BenchmarkV2AutoTriggerTestPollResponse {
	ok: boolean;
	error: string;
	snapshot_models: number;
	model_names: string[];
	diff: {
		added: string[];
		removed: string[];
		updated: string[];
		has_changes: boolean;
	} | null;
}

// -- Fine-Tune --

export interface FineTuneExportRequest {
	format: string;
	conversation_ids?: string[] | null;
	date_from?: string | null;
	date_to?: string | null;
	model?: string | null;
	min_quality?: number;
	min_turns?: number;
}

export interface FineTuneExportResponse {
	format: string;
	conversation_count: number;
	message_count: number;
	data: string;
	timestamp: string;
	filters_applied: Record<string, unknown>;
}

export interface FineTunePreviewResponse {
	total_conversations: number;
	total_messages: number;
	format: string;
	sample_data: string;
	sample_count: number;
	quality_scores: FineTuneQualityScore[];
	filters: Record<string, unknown>;
}

export interface FineTuneQualityScore {
	conversation_id: string;
	feedback_score: number;
	benchmark_score: number;
	combined_score: number;
	feedback_count: number;
	has_feedback: boolean;
	has_benchmarks: boolean;
}

export interface FineTuneQualityResponse {
	scores: FineTuneQualityScore[];
	count: number;
}

export interface FineTuneVariant {
	variant_id: string;
	name: string;
	base_model: string;
	variant_model: string;
	status: string;
	created_at: string;
	updated_at: string;
	description: string;
	dataset_size: number;
	epochs: number;
	learning_rate: number;
	loss: number;
	training_duration_seconds: number;
	metadata: Record<string, unknown>;
}

export interface FineTuneVariantCreate {
	name: string;
	base_model: string;
	variant_model: string;
	description?: string;
	dataset_size?: number;
	epochs?: number;
	learning_rate?: number;
	loss?: number;
	training_duration_seconds?: number;
	metadata?: Record<string, unknown>;
}

export interface FineTuneVariantListResponse {
	variants: FineTuneVariant[];
	count: number;
}

export interface FineTuneCompareRequest {
	variant_id: string;
	prompts: string[];
}

export interface FineTuneCompareResponse {
	comparison_id: string;
	variant_id: string;
	base_model: string;
	variant_model: string;
	status: string;
	created_at: string;
	completed_at: string;
	base_wins: number;
	variant_wins: number;
	ties: number;
	base_avg_latency_ms: number;
	variant_avg_latency_ms: number;
	summary: string;
	prompts: FineTuneComparePrompt[];
}

export interface FineTuneComparePrompt {
	prompt: string;
	base_response: string;
	variant_response: string;
	base_latency_ms: number;
	variant_latency_ms: number;
	winner: string;
}

// -- Conversation Branches --

export interface BranchStats {
	message_count: number;
	last_activity: string | null;
	total_tokens: number;
	last_model: string | null;
}

export interface Branch {
	branch_id: string;
	conversation_id: string;
	parent_branch_id: string | null;
	fork_message_id: number;
	name: string;
	color: string;
	created_at: string;
	updated_at: string;
	metadata: Record<string, unknown>;
	stats?: BranchStats | null;
}

export interface BranchMessage {
	id: number;
	branch_id: string;
	conversation_id: string;
	role: string;
	content: string;
	timestamp: string;
	token_estimate: number;
	model: string | null;
	metadata: Record<string, unknown>;
}

export interface BranchTreeNode {
	branch_id: string | null;
	name: string;
	color: string;
	fork_message_id: number | null;
	message_count: number;
	last_model: string | null;
	last_activity: string | null;
	children: BranchTreeNode[];
}

export interface BranchComparison {
	branch_a_id: string | null;
	branch_b_id: string | null;
	branch_a_name: string;
	branch_b_name: string;
	shared_messages: Record<string, unknown>[];
	branch_a_messages: Record<string, unknown>[];
	branch_b_messages: Record<string, unknown>[];
	fork_message_id: number | null;
}

export interface BranchForkRequest {
	conversation_id: string;
	fork_message_id: number;
	name?: string;
	color?: string;
	parent_branch_id?: string;
}

export interface BranchUpdateRequest {
	name?: string;
	color?: string;
	metadata?: Record<string, unknown>;
}

export interface BranchCompareRequest {
	conversation_id: string;
	branch_a_id?: string | null;
	branch_b_id?: string | null;
}

export interface BranchMergeRequest {
	source_branch_id: string;
	target_branch_id: string;
	message_ids?: number[];
}

export interface BranchMergeResponse {
	merged_count: number;
	source_branch_id: string;
	target_branch_id: string;
	messages: BranchMessage[];
}

export interface BranchMessagesResponse {
	branch_id: string;
	messages: Record<string, unknown>[];
	count: number;
}

// -- Auth --

export interface AuthUser {
	user_id: string;
	username: string;
	email: string;
	role: 'admin' | 'user' | 'viewer';
	created_at: number;
	updated_at: number;
	metadata: Record<string, unknown>;
}

export interface AuthTokens {
	access_token: string;
	refresh_token: string;
	token_type: string;
	expires_in: number;
	user_id: string;
}

export interface AuthStatus {
	available: boolean;
	single_user_mode: boolean;
	registration_enabled: boolean;
	user_count: number;
	/** Whether httpOnly cookie mode is enabled for JWT storage. */
	cookie_mode?: boolean;
}

export interface UserSettings {
	user_id: string;
	theme: string;
	default_model: string;
	default_preset: string;
	sidebar_open: boolean;
	language: string;
	created_at: number;
	updated_at: number;
	preferences: Record<string, unknown>;
}

export interface RegisterRequest {
	username: string;
	password: string;
	email?: string;
}

export interface LoginRequest {
	username: string;
	password: string;
}

export interface ProfileUpdateRequest {
	username?: string;
	email?: string;
	metadata?: Record<string, unknown>;
}

export interface PasswordChangeRequest {
	current_password: string;
	new_password: string;
}

export interface SettingsUpdateRequest {
	theme?: string;
	default_model?: string;
	default_preset?: string;
	sidebar_open?: boolean;
	language?: string;
	preferences?: Record<string, unknown>;
}

export interface ShareProjectRequest {
	project_id: string;
	username: string;
	role: 'owner' | 'editor' | 'viewer';
}

export interface ProjectMember {
	project_id: string;
	user_id: string;
	username: string;
	email: string;
	role: string;
	created_at: number;
}

export interface AuditLogEntry {
	id: number;
	user_id: string;
	action: string;
	target_type: string;
	target_id: string;
	details: Record<string, unknown>;
	timestamp: number;
}
// =========================================================================
// RAG v2 -- Knowledge Base
// =========================================================================

export interface RAGCollection {
	name: string;
	description: string;
	document_count: number;
	chunk_count: number;
	created_at: number;
	updated_at: number;
}

export interface RAGCollectionsListResponse {
	collections: RAGCollection[];
	total: number;
}

export interface RAGCollectionCreateRequest {
	name: string;
	description?: string;
}

export interface RAGCollectionDeleteResponse {
	deleted: boolean;
	name: string;
}

export interface RAGDocument {
	doc_id: string;
	collection_name: string;
	source_file: string;
	file_type: string;
	chunk_count: number;
	raw_text_length: number;
	ingested_at: number;
	metadata: Record<string, unknown>;
}

export interface RAGDocumentsListResponse {
	documents: RAGDocument[];
	total: number;
}

export interface RAGDocumentDeleteResponse {
	deleted: boolean;
	doc_id: string;
}

export interface RAGIngestResponse {
	doc_id: string;
	collection_name: string;
	source_file: string;
	file_type: string;
	chunk_count: number;
	raw_text_length: number;
	ingested_at: number;
}

export interface RAGIngestURLRequest {
	url: string;
	collection?: string;
	metadata?: Record<string, unknown>;
}

export interface RAGQueryRequest {
	query: string;
	collection?: string;
	n_results?: number;
	min_score?: number;
	source_filter?: string;
	file_type_filter?: string;
	rerank?: boolean;
	track_citations?: boolean;
}

export interface RAGRetrievalResult {
	content: string;
	score: number;
	source_file: string;
	file_type: string;
	chunk_index: number;
	total_chunks: number;
	parent_doc_id: string;
	collection_name: string;
	section: string | null;
	page: number | null;
}

export interface RAGCitation {
	citation_id: string;
	query: string;
	collection_name: string;
	chunk_id: string;
	parent_doc_id: string;
	source_file: string;
	section: string | null;
	score: number;
	timestamp: number;
}

export interface RAGQueryResponse {
	query: string;
	results: RAGRetrievalResult[];
	citations: RAGCitation[];
	total_results: number;
}

// RAG Dashboard

export interface RAGDashboardStats {
	total_collections: number;
	total_documents: number;
	total_chunks: number;
	total_citations: number;
	total_queries_today: number;
	total_queries_week: number;
	total_queries_all: number;
	avg_score: number;
	storage_bytes: number;
}

export interface RAGUsageDataPoint {
	date: string;
	query_count: number;
	citation_count: number;
	avg_score: number;
}

export interface RAGUsageResponse {
	data: RAGUsageDataPoint[];
	days: number;
}

export interface RAGSourceReliability {
	source_file: string;
	collection_name: string;
	doc_id: string;
	citation_count: number;
	avg_score: number;
	last_cited: number;
	freshness_score: number;
	reliability_score: number;
}

export interface RAGSourcesResponse {
	sources: RAGSourceReliability[];
	total: number;
}

export interface RAGCollectionHealth {
	name: string;
	document_count: number;
	chunk_count: number;
	citation_count: number;
	avg_chunk_size: number;
	file_types: string[];
	last_ingestion: number;
	last_query: number;
	freshness_score: number;
}

export interface RAGHealthResponse {
	collections: RAGCollectionHealth[];
	total: number;
}

export interface RAGRefreshResult {
	checked_at: number;
	sources_checked: number;
	sources_refreshed: number;
	errors: string[];
}

export interface RAGConnectorStatus {
	name: string;
	connector_type: string;
	connected: boolean;
	document_count: number;
	last_query_time_ms: number;
	error: string | null;
}

export interface RAGConnectorsResponse {
	connectors: RAGConnectorStatus[];
	total: number;
}

export interface RAGBackendsResponse {
	backends: Record<string, boolean>;
}

// =========================================================================
// Plugins
// =========================================================================

export interface PluginInfo {
	name: string;
	version: string;
	author: string;
	description: string;
	entry_point: string;
	hooks: string[];
	dependencies: string[];
	permissions: string[];
	state: 'installed' | 'enabled' | 'disabled';
	plugin_dir: string;
	installed_at: number;
	updated_at: number;
}

export interface PluginListResponse {
	plugins: PluginInfo[];
	total: number;
	enabled: number;
}

export interface PluginInstallRequest {
	source_dir: string;
	auto_enable: boolean;
}

export interface PluginInstallResponse {
	success: boolean;
	name: string;
	version: string;
	message: string;
	error: string | null;
}

export interface PluginStateChangeResponse {
	success: boolean;
	name: string;
	state: string;
	message: string;
	error: string | null;
}

export interface PluginUninstallResponse {
	success: boolean;
	name: string;
	message: string;
	error: string | null;
}

export interface PluginConfigResponse {
	name: string;
	config: Record<string, unknown>;
	config_schema: Record<string, unknown>;
}

export interface PluginUpdateConfigResponse {
	success: boolean;
	name: string;
	config: Record<string, unknown>;
	message: string;
	error: string | null;
}

// =========================================================================
// Plugin Marketplace
// =========================================================================

export interface MarketplaceEntry {
	name: string;
	version: string;
	description: string;
	author: string;
	url: string;
	tags: string[];
	hooks: string[];
	permissions: string[];
	min_opti_version: string;
	stars: number;
	downloads: number;
	sha256: string;
	created_at: number;
	updated_at: number;
	average_rating: number;
	review_count: number;
}

export interface MarketplaceListResponse {
	plugins: MarketplaceEntry[];
	total: number;
}

export interface RemoteInstallRequest {
	url: string;
	expected_sha256: string;
	auto_enable: boolean;
}

export interface RemoteInstallResponse {
	success: boolean;
	name: string;
	version: string;
	message: string;
	error: string | null;
}

export interface PluginReview {
	id: number;
	plugin_name: string;
	rating: number;
	title: string;
	text: string;
	author: string;
	created_at: number;
	// REV-2: authenticated owner identity; null on legacy rows.
	user_id?: string | null;
}

export interface ReviewListResponse {
	reviews: PluginReview[];
	total: number;
	average_rating: number;
	rating_distribution: Record<number, number>;
}

export interface AddReviewRequest {
	rating: number;
	title: string;
	text: string;
	author: string;
}

export interface AddReviewResponse {
	success: boolean;
	review: PluginReview | null;
	message: string;
	error: string | null;
}

export interface TemplateRequest {
	name: string;
	author: string;
	description: string;
	version: string;
	hooks: string[];
	permissions: string[];
}

export interface TemplateResponse {
	success: boolean;
	path: string;
	files: string[];
	message: string;
	error: string | null;
}

// ---------------------------------------------------------------------------
// Speculative Decoding
// ---------------------------------------------------------------------------

export interface SpeculativeDecodingConfig {
	enabled: boolean;
	draft_model: string;
	draft_max: number;
	draft_min: number;
	draft_gpu_layers: number;
	auto_select_draft: boolean;
}

export interface SpeculativeDecodingStats {
	total_draft_tokens: number;
	accepted_tokens: number;
	total_runs: number;
	overall_acceptance_rate: number;
	last_acceptance_rate: number;
	last_speedup_factor: number;
	last_updated: number;
}

export interface SpeculativeDecodingStatus {
	config: SpeculativeDecodingConfig;
	stats: SpeculativeDecodingStats;
	available: boolean;
	backend_required: string;
}

export interface DraftCandidate {
	name: string;
	path: string;
	family: string;
	parameter_size_b: number;
	quantization: string;
	estimated_vram_gb: number;
	compatibility_score: number;
}

export interface CompatibleDraftsResponse {
	main_model: string;
	drafts: DraftCandidate[];
	count: number;
}

export interface VRAMBudgetResult {
	fits: boolean;
	main_vram_gb: number;
	draft_vram_gb: number;
	total_vram_gb: number;
	available_vram_gb: number;
	headroom_gb: number;
}

// ---------------------------------------------------------------------------
// Auto-Tuner
// ---------------------------------------------------------------------------

export interface TunerConfig {
	enabled: boolean;
	warmup_runs: number;
	benchmark_tokens: number;
	benchmark_prompt_tokens: number;
	trials_per_param: number;
	auto_apply: boolean;
}

export interface TunerParameterSpace {
	batch_size: number[];
	ubatch_size: number[];
	threads: number[];
	flash_attention: boolean[];
}

export interface TunerStatus {
	config: TunerConfig;
	param_space: TunerParameterSpace;
	active_jobs: Record<string, TunerJob>;
	saved_profiles: string[];
	available: boolean;
}

export interface TunerJob {
	job_id: string;
	model_name: string;
	status: string;
	progress: number;
	current_step: string;
	total_steps: number;
	completed_steps: number;
	started_at: number;
	finished_at: number;
	result: TunerProfile | null;
	error: string;
}

export interface TunerProfile {
	model_name: string;
	best_params: Record<string, unknown>;
	best_tg_speed: number;
	best_pp_speed: number;
	baseline_tg_speed: number;
	baseline_pp_speed: number;
	speedup_factor: number;
	hardware_fingerprint: string;
	timestamp: number;
	all_results: Record<string, unknown>[];
}

export interface TunerResultsResponse {
	results: Record<string, TunerProfile>;
	count: number;
}

// -- Tuner Recommendations --

export interface TunerRecommendation {
	title: string;
	description: string;
	parameter: string;
	current_value: unknown;
	recommended_value: unknown;
	estimated_speedup: number;
	confidence: string;
	category: string;
	applied: boolean;
}

export interface TunerRecommendationsResponse {
	model_name: string;
	recommendations: TunerRecommendation[];
	count: number;
}

// -- Model Lifecycle --

export interface PullProgress {
	status: string;
	digest: string;
	total_bytes: number;
	completed_bytes: number;
	percent: number;
}

export interface PullJob {
	job_id: string;
	model_name: string;
	status: string;
	progress: PullProgress;
	started_at: number;
	completed_at: number;
	error: string;
}

export interface ModelUpdateInfo {
	model_name: string;
	current_digest: string;
	latest_digest: string;
	has_update: boolean;
	checked_at: number;
	error: string;
}

export interface ModelLifecycleStatus {
	available: boolean;
	enabled: boolean;
	ollama_base_url: string;
	max_concurrent_pulls: number;
	active_pulls: number;
	alias_count: number;
	stale_threshold_days: number;
}

export interface ModelEntry {
	name: string;
	size: number;
	size_human: string;
	modified_at: number;
	digest: string;
	details: Record<string, unknown>;
}

// =========================================================================
// RAG Batch Ingestion
// =========================================================================

export interface RAGIngestFileStatus {
	file_id: string;
	job_id: string;
	filepath: string;
	filename: string;
	file_size: number;
	status: 'queued' | 'processing' | 'done' | 'error' | 'skipped';
	doc_id: string | null;
	chunk_count: number;
	error_message: string | null;
	started_at: number | null;
	completed_at: number | null;
}

export interface RAGIngestJob {
	job_id: string;
	status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
	collection: string;
	source_type: 'batch' | 'folder';
	source_path: string | null;
	total_files: number;
	completed_files: number;
	failed_files: number;
	skipped_files: number;
	total_chunks: number;
	progress: number;
	created_at: number;
	started_at: number | null;
	completed_at: number | null;
	error_message: string | null;
	files: RAGIngestFileStatus[];
}

export interface RAGIngestJobsListResponse {
	jobs: RAGIngestJob[];
	total: number;
}

export interface RAGIngestJobDeleteResponse {
	deleted: boolean;
	job_id: string;
}

export interface RAGFolderScanRequest {
	directory: string;
	collection: string;
	recursive: boolean;
}

// ----: Backup / Restore ----

export interface BackupSectionInfo {
	name: string;
	description: string;
	item_count: number;
	available: boolean;
}

export interface BackupSectionsResponse {
	sections: BackupSectionInfo[];
}

export interface BackupDiffItem {
	section: string;
	key: string;
	action: 'add' | 'update' | 'skip';
	current_value?: unknown;
	incoming_value?: unknown;
}

export interface BackupPreviewResponse {
	valid: boolean;
	strategy: string;
	sections: string[];
	diff: BackupDiffItem[];
	errors: string[];
	summary: Record<string, number>;
}

export interface BackupImportResponse {
	success: boolean;
	sections_imported: string[];
	sections_failed: string[];
	errors: string[];
	rolled_back: boolean;
}

export interface BackupMetadata {
	opti_oignon_version: string;
	timestamp: number;
	timestamp_iso: string;
	platform: {
		system: string;
		release: string;
		machine: string;
		python_version: string;
	};
	sections_included: string[];
}

export interface BackupData {
	schema_version: string;
	metadata: BackupMetadata;
	sections: Record<string, unknown>;
}
