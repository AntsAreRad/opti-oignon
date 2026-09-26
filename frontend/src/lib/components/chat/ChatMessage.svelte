<!--
  ChatMessage.svelte
  Displays a user or assistant message.
  Supports streaming content (isStreaming + streamContent).
  Displays inline search results if present in metadata.
  Displays reasoning (thinking) in a collapsible block.
  Displays code verification badges.
  Displays inline tool calls.
  Displays feedback widget (thumbs up/down) on assistant messages.
  Retry button on last assistant message.
  Mobile responsive -- reduced padding.
  An assistant reply renders its markdown through
  markdown/Markdown.svelte (a long finished reply collapses by whole blocks,
  each code block carries its own Copy, and the streaming caret follows the
  reply's last character); any other message, a user's or one of another
  role, stays plain text (markdown/PlainText.svelte), a long one collapsed by
  whole lines.
-->
<script lang="ts">
	import { createEventDispatcher, onMount, onDestroy } from 'svelte';
	import type { MessageItem, VerificationInfo, ToolCallInfo, ReasoningStepInfo, ReasoningMetaInfo, SandboxFileEntry } from '$lib/types';
	import SearchResults from './SearchResults.svelte';
	import ToolCallDisplay from './ToolCallDisplay.svelte';
	import ReasoningDisplay from './ReasoningDisplay.svelte';
	import CorrectionIndicator from './CorrectionIndicator.svelte';
	import RoutingIndicator from './RoutingIndicator.svelte';
	import FeedbackWidget from './FeedbackWidget.svelte';
	import SandboxFileManager from '$lib/components/panels/SandboxFileManager.svelte';
	import CodingAgentInline from './CodingAgentInline.svelte';
	import Icon from '$lib/ds/Icon.svelte';
	import Markdown from './markdown/Markdown.svelte';
	import PlainText from './markdown/PlainText.svelte';

	export let message: MessageItem;
	export let isStreaming: boolean = false;
	export let streamContent: string = '';
	export let streamThinking: string = '';
	export let searchMetadata: {
		results?: { title: string; url: string; snippet: string; source: string; relevance_score: number }[];
		query?: string;
		engine?: string;
		citations?: string[];
	} | null = null;
	export let isLast: boolean = false;
	export let isRetrying: boolean = false;
	// Verification results received during streaming
	export let verificationResults: VerificationInfo[] = [];
	// Tool calls received during streaming
	export let toolCallResults: ToolCallInfo[] = [];
	// Reasoning steps received during streaming
	export let reasoningSteps: ReasoningStepInfo[] = [];
	// Reasoning metadata
	export let reasoningMeta: ReasoningMetaInfo | null = null;
	// Optional routing reason (provided during streaming); falls back to
	// an untyped routing_reason field on the message when present.
	type RoutingReasonFull = {
		model: string;
		display_name: string;
		task_type: string;
		pipeline: string;
		reason: string;
		score: number;
		alternatives: string[];
		profile_used: boolean;
		feedback_adjusted: boolean;
		failover: boolean;
		original_model: string;
	};
	export let routingReason: RoutingReasonFull | null = null;
	// The conversation the message belongs to, for the feedback widget.
	export let conversationId: string = '';
	// A finished message longer than this (in lines of source, about)
	// collapses: a reply to its first whole blocks, plain text to its first
	// lines.
	export let collapseThreshold: number = 500;

	const dispatch = createEventDispatcher<{
		retry: void;
	}>();

	let copied = false;
	let copyTimeout: ReturnType<typeof setTimeout> | null = null;
	let thinkingOpen = false;

	$: isUser = message.role === 'user';
	// Only the assistant's messages are replies, rendered from their markdown.
	$: isReply = message.role === 'assistant';
	$: displayContent = isStreaming ? streamContent : message.content;
	$: showRetry = !isUser && isLast && !isStreaming && !isRetrying;
	// Thinking content (streaming or history)
	$: thinkingContent = isStreaming ? streamThinking : (message.thinking || '');
	$: hasThinking = thinkingContent.length > 0;
	// Verification (streaming or history)
	$: verifications = verificationResults.length > 0 ? verificationResults : (message.verification || []);
	$: hasVerification = verifications.length > 0;
	// Tool calls (streaming or history)
	$: toolCalls = toolCallResults.length > 0 ? toolCallResults : (message.tool_calls || []);
	$: hasToolCalls = toolCalls.length > 0;
	// Reasoning steps (streaming or history)
	$: rSteps = reasoningSteps.length > 0 ? reasoningSteps : (message.reasoning_steps || []);
	$: rMeta = reasoningMeta || message.reasoning_meta || null;
	$: hasReasoning = rSteps.length > 0;
	// Self-correction info reintegrated onto the message
	$: correction = message.correction ?? null;
	$: hasCorrection = !isUser && !!correction?.was_corrected;
	// Routing reason reintegrated; prop wins, else untyped field
	$: effectiveRouting =
		routingReason ??
		(((message as Record<string, unknown>).routing_reason as RoutingReasonFull | null) ?? null);
	$: hasRouting = !isUser && !!effectiveRouting;
	// Sandbox metadata (present when quick sandbox was used)
	$: sandboxMeta = (message as Record<string, unknown>).sandbox_meta as { sandbox_session_id: string; sandbox_files: unknown[]; sandbox_files_created: string[] } | undefined;
	$: hasSandbox = !isStreaming && !!sandboxMeta?.sandbox_session_id;
	// Snapshot from the done metadata: the file manager shows it at once
	// and before any fetch of its own.
	$: sandboxInitialFiles = (sandboxMeta?.sandbox_files ?? []) as SandboxFileEntry[];
	// Chat coding agent metadata (present when coding agent was used)
	$: codingMeta = (message as Record<string, unknown>).coding_meta as {
		chat_coding: boolean;
		coding_result: Record<string, unknown>;
		sandbox_session_id: string;
		sandbox_files: unknown[];
		sandbox_files_created: string[];
		turn_count: number;
	} | undefined;
	$: hasCoding = !isStreaming && !!codingMeta?.chat_coding;

	// Collapsed, a long message keeps about this many lines (whole blocks, for
	// a reply).
	$: collapseKeep = Math.min(20, Math.floor(collapseThreshold / 10));

	function copyContent() {
		navigator.clipboard.writeText(displayContent).then(() => {
			copied = true;
			if (copyTimeout) clearTimeout(copyTimeout);
			copyTimeout = setTimeout(() => { copied = false; }, 1500);
		});
	}

	function handleRetry() {
		dispatch('retry');
	}

	onDestroy(() => {
		if (copyTimeout) clearTimeout(copyTimeout);
	});
</script>

<div class="group flex {isUser ? 'justify-end' : 'justify-start'} animate-message-in">
	<div
		class="relative max-w-[85%] sm:max-w-[85%] rounded-xl px-2.5 sm:px-4 py-2.5 text-sm leading-relaxed"
		style="{isUser
			? 'background-color: var(--oo-msg-user-bg); border: 1px solid var(--oo-msg-user-bd); color: var(--oo-msg-user-fg);'
			: 'background-color: var(--oo-msg-bot-bg); border: 1px solid var(--oo-msg-bot-bd); color: var(--oo-msg-bot-fg);'}"
	>
		<!-- The reply's own heading, for screen readers: its markdown headings
		     start one level below it. -->
		{#if isReply}
			<h2 class="sr-only">{message.model ? `Reply from ${message.model}` : 'Reply'}</h2>
		{/if}

		<!-- Model (assistant only, hidden during streaming) -->
		{#if !isUser && message.model && !isStreaming}
			<div class="text-xs font-mono mb-1" style="color: var(--oo-fg-muted);">{message.model}</div>
		{/if}

		<!-- Vision delegation badge (assistant only, after completion) -->
		{#if !isUser && !isStreaming && message.vision_delegation?.vision_model}
			<div class="inline-flex items-center gap-1 mb-1.5 px-2 py-0.5 rounded-md text-xs"
				style="background-color: var(--oo-bg-base); border: 1px solid var(--oo-bd-subtle); color: var(--oo-fg-muted);">
				<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
					<path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
				</svg>
				Image analyzed by {message.vision_delegation.vision_model}
			</div>
		{/if}

		<!-- Bloc de reflexion retractable (assistant uniquement) -->
		{#if !isUser && hasThinking}
			<details
				class="mb-2 rounded-lg"
				style="background-color: var(--oo-bg-base); border: 1px solid var(--oo-bd-subtle);"
				bind:open={thinkingOpen}
			>
				<summary class="cursor-pointer px-2.5 py-1 text-xs select-none flex items-center gap-1.5"
					style="color: var(--oo-fg-muted);">
					<svg class="w-3 h-3 transition-transform {thinkingOpen ? 'rotate-90' : ''}"
						fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path d="M9 5l7 7-7 7" />
					</svg>
					Thinking{#if isStreaming}<span class="inline-block w-1 h-3 ml-1 animate-cursor-blink" style="background-color: var(--oo-acc-mark);" />{/if}
				</summary>
				<div class="px-2.5 pb-2 text-xs leading-relaxed whitespace-pre-wrap pt-1.5 max-h-60 overflow-y-auto"
					style="color: var(--oo-fg-tertiary); border-top: 1px solid var(--oo-bd-subtle);">
					{thinkingContent}
				</div>
			</details>
		{/if}

		<!-- Inline tool calls (assistant only) -->
		{#if !isUser && hasToolCalls}
			<ToolCallDisplay toolCalls={toolCalls} />
		{/if}

		<!-- Reasoning steps (assistant only) -->
		{#if !isUser && hasReasoning}
			<ReasoningDisplay steps={rSteps} meta={rMeta} {isStreaming} />
		{/if}

		<!-- Routing reason + self-correction indicators (assistant only) -->
		{#if hasRouting}
			<RoutingIndicator routingReason={effectiveRouting} model={message.model ?? ''} />
		{/if}
		{#if hasCorrection}
			<CorrectionIndicator {correction} />
		{/if}

		<!-- Content: a reply's markdown, rendered; any other message as written -->
		<div
			class="break-words msg-content"
			aria-live={isStreaming ? 'polite' : 'off'}
			aria-atomic="false"
		>
			{#if isReply}
				<Markdown
					source={displayContent}
					streaming={isStreaming}
					collapseAbove={collapseThreshold}
					{collapseKeep}
					caret={isStreaming}
				/>
			{:else}
				<PlainText text={displayContent} collapseAbove={collapseThreshold} {collapseKeep} />
			{/if}
		</div>

		<!-- Badges de verification de code -->
		{#if !isUser && !isStreaming && hasVerification}
			<div class="flex flex-wrap gap-1.5 mt-1.5">
				{#each verifications as v}
					<span class="inline-flex items-center gap-1 text-xs px-1.5 py-0.5 rounded-md
						{v.status === 'passed'
							? 'bg-[var(--oo-success-bg)] text-[var(--oo-success)] border border-[var(--oo-success-bd)]'
							: v.status === 'fixed'
								? 'bg-[var(--oo-warning-bg)] text-[var(--oo-warning)] border border-[var(--oo-warning-bd)]'
								: 'bg-[var(--oo-error-bg)] text-[var(--oo-error)] border border-[var(--oo-error-bd)]'}">
						{#if v.status === 'passed'}
							<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
								<path d="M5 13l4 4L19 7" />
							</svg>
							{v.language} verified
						{:else if v.status === 'fixed'}
							<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
								<path d="M5 13l4 4L19 7" />
							</svg>
							{v.language} fixed ({v.iterations} iter.)
						{:else}
							<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
								<path d="M6 18L18 6M6 6l12 12" />
							</svg>
							{v.language} failed
						{/if}
					</span>
				{/each}
			</div>
		{/if}

		<!-- Feedback widget (always visible on assistant messages, not streaming) -- BUG-05 -->
		{#if !isUser && !isStreaming}
			<div class="flex items-center justify-between mt-1">
				<div class="text-xs" style="color: var(--oo-fg-faint);">
					<FeedbackWidget
						conversationId={conversationId}
						messageId={String(message.id ?? '')}
						modelUsed={message.model ?? ''}
						taskType={effectiveRouting?.task_type ?? ''}
						pipelineUsed={effectiveRouting?.pipeline ?? ''}
					/>
				</div>
				{#if message.token_estimate > 0}
					<div class="text-xs" style="color: var(--oo-fg-faint);">
						{message.token_estimate} tokens
					</div>
				{/if}
			</div>
		{/if}

		<!-- Resultats de recherche inline (assistant uniquement) -->
		{#if !isUser && !isStreaming && searchMetadata?.results && searchMetadata.results.length > 0}
			<SearchResults
				results={searchMetadata.results}
				query={searchMetadata.query || ''}
				engine={searchMetadata.engine || ''}
				citations={searchMetadata.citations || []}
			/>
		{/if}

		<!-- Inline chat coding agent display (when coding agent was used) -->
		{#if !isUser && hasCoding && codingMeta}
			<CodingAgentInline
				codingResult={codingMeta.coding_result || {}}
				sandboxSessionId={codingMeta.sandbox_session_id || ''}
				sandboxFiles={codingMeta.sandbox_files || []}
				turnCount={codingMeta.turn_count || 0}
			/>
		{/if}

		<!-- Inline sandbox file manager (when quick sandbox was used, not coding agent) -->
		{#if !isUser && hasSandbox && sandboxMeta && !hasCoding}
			<div class="mt-2 rounded-lg overflow-hidden" style="border: 1px solid var(--oo-bd-default);">
				<SandboxFileManager
					sessionId={sandboxMeta.sandbox_session_id}
					initialFiles={sandboxInitialFiles}
				/>
			</div>
		{/if}

		<!-- Hover actions: retry and copy (branches live in the branch explorer) -->
		{#if displayContent && !isStreaming}
			<div class="absolute -top-2 -right-2 flex items-center gap-0.5
				opacity-0 group-hover:opacity-100 transition-opacity">
				<!-- Retry (last assistant message only) -->
				{#if showRetry}
					<button
						on:click={handleRetry}
						class="p-1 rounded-md transition-colors"
						style="color: var(--oo-fg-muted); background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);"
						title="Regenerate response"
						aria-label="Regenerate response"
					>
						<Icon name="refresh-cw" size="sm" />
					</button>
				{/if}
				<!-- Copy -->
				<button
					on:click={copyContent}
					class="p-1 rounded-md transition-colors"
					style="color: var(--oo-fg-muted); background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);"
					title="Copy"
					aria-label="Copy message"
				>
					{#if copied}
						<Icon name="check" size="sm" />
					{:else}
						<Icon name="copy" size="sm" />
					{/if}
				</button>
			</div>
		{/if}
	</div>
</div>
