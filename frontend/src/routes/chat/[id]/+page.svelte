<!--
  Chat view: messages + streaming + input.
  Handles WebSocket streaming, cancellation and retry.
  Passes chat options (model, preset, temperature) via chatOptions.
  Mobile responsive — scroll FAB, tighter padding, safe-area.
-->
<script lang="ts">
	import { page } from '$app/stores';
	import { onMount, afterUpdate, tick } from 'svelte';
	import {
		activeConversation,
		messages,
		error,
		activeConversationId,
		messagesLoading
	} from '$lib/stores/conversations';
	import {
		isStreaming,
		streamingContent,
		streamingThinking,
		streamingModel,
		streamingError,
		streamingVisionDelegation,
		lastSearchMetadata,
		searchMetadataMap,
		sendMessage,
		retryLastMessage,
		cancelCurrentGeneration,
		isCodingStream
	} from '$lib/stores/chat';
	import { getChatOptions } from '$lib/stores/chatOptions';
	import { toastError } from '$lib/stores/notifications';
	import ChatMessage from '$lib/components/chat/ChatMessage.svelte';
	import ChatInput from '$lib/components/chat/ChatInput.svelte';
	import FileUpload from '$lib/components/chat/FileUpload.svelte';
	import StreamingIndicator from '$lib/components/chat/StreamingIndicator.svelte';
	import LiveMetricsOverlay from '$lib/components/chat/LiveMetricsOverlay.svelte';
	import VisionDelegationIndicator from '$lib/components/chat/VisionDelegationIndicator.svelte';
	import ModelSelector from '$lib/components/chat/ModelSelector.svelte';
	import MessageSkeleton from '$lib/components/chat/MessageSkeleton.svelte';
	import ScrollToBottomFab from '$lib/components/chat/ScrollToBottomFab.svelte';
	import ErrorBoundary from '$lib/components/ui/ErrorBoundary.svelte';
	import BranchExplorer from '$lib/components/chat/BranchExplorer.svelte';
	import CodingAgentProgress from '$lib/components/chat/CodingAgentProgress.svelte';
	import type { AttachedFile } from '$lib/types';

	let messagesContainer: HTMLDivElement;
	let bottomSentinel: HTMLDivElement;
	let shouldAutoScroll = true;
	let showScrollFab = false;
	let attachedFiles: AttachedFile[] = [];
	let selectedMessageId: number | null = null;

	// Is the last message from the assistant (for retry button)?
	$: lastMessageIsAssistant =
		$messages.length > 0 && $messages[$messages.length - 1].role === 'assistant';

	// Active conversation ID (from route)
	$: convId = $page.params?.id ?? null;

	// Combined errors -> toast
	$: if ($streamingError) {
		toastError($streamingError);
	}
	$: if ($error) {
		toastError($error);
	}

	// Placeholder for current streaming message
	$: streamingPlaceholder = {
		id: null,
		role: 'assistant',
		content: '',
		timestamp: new Date().toISOString(),
		model: $streamingModel,
		token_estimate: 0,
	};

	// Extract vision model name from delegation data (avoids TS 'as' cast in template)
	$: delegatedVisionModel = (() => {
		const d = $streamingVisionDelegation;
		if (!d) return '';
		if (d.vision_model) return String(d.vision_model);
		const msg = d.message ? String(d.message) : '';
		const m = msg.match(/with (.+)\.\.\./);
		return m?.[1] ?? '';
	})();

	function scrollToBottom() {
		if (bottomSentinel && shouldAutoScroll) {
			bottomSentinel.scrollIntoView({ behavior: 'smooth' });
		}
	}

	function handleScroll() {
		if (!messagesContainer) return;
		const { scrollTop, scrollHeight, clientHeight } = messagesContainer;
		// Auto-scroll if close to bottom (100px tolerance)
		shouldAutoScroll = scrollHeight - scrollTop - clientHeight < 100;
		// Show scroll-to-bottom FAB when scrolled up beyond 300px
		showScrollFab = scrollHeight - scrollTop - clientHeight > 300;
	}

	// FAB click handler — smooth scroll to bottom
	function handleScrollFabClick() {
		shouldAutoScroll = true;
		showScrollFab = false;
		scrollToBottom();
	}

	async function handleSend(event: CustomEvent<{ text: string; images: string[] }>) {
		if (!convId) return;
		shouldAutoScroll = true;
		const options = getChatOptions();

		// Prepend attached file contents to message
		let messageText = event.detail.text;
		const messageImages = event.detail.images;
		if (attachedFiles.length > 0) {
			const fileBlocks = attachedFiles.map(
				(f) => `[File: ${f.filename}]\n\`\`\`\n${f.content}\n\`\`\``
			).join('\n\n');
			messageText = `${fileBlocks}\n\n${messageText}`;
			attachedFiles = [];
		}

		// Include images in options if present
		if (messageImages && messageImages.length > 0) {
			(options as Record<string, unknown>).images = messageImages;
		}

		await sendMessage(convId, messageText, options);
		await tick();
		scrollToBottom();
	}

	function handleAttach(event: CustomEvent<AttachedFile>) {
		attachedFiles = [...attachedFiles, event.detail];
	}

	function handleRemoveFile(event: CustomEvent<number>) {
		attachedFiles = attachedFiles.filter((_, i) => i !== event.detail);
	}

	async function handleCancel() {
		if (!convId) return;
		await cancelCurrentGeneration(convId);
	}

	async function handleRetry() {
		if (!convId) return;
		shouldAutoScroll = true;
		await retryLastMessage(convId);
		await tick();
		scrollToBottom();
	}

	// Scroll au bas quand le contenu streaming change
	$: if ($streamingContent) {
		tick().then(scrollToBottom);
	}

	// Scroll au bas quand les messages changent
	$: if ($messages) {
		tick().then(scrollToBottom);
	}

	onMount(() => {
		scrollToBottom();
	});
</script>

<div class="h-full flex flex-col">
	<!-- Selecteur modele mobile (visible uniquement sur petits ecrans) -->
	<div class="sm:hidden px-3 py-1.5 border-b border-surface-800/50 flex justify-end">
		<ModelSelector />
	</div>

	<!-- Branch explorer bar -->
	{#if convId}
		<div class="px-3 sm:px-4 py-1.5 border-b" style="border-color: var(--oo-border);">
			<div class="max-w-2xl mx-auto">
				<BranchExplorer
					conversationId={convId}
					currentMessageId={selectedMessageId}
					on:switchBranch={(e) => { selectedMessageId = null; }}
					on:fork={() => { selectedMessageId = null; }}
				/>
			</div>
		</div>
	{/if}

	<!-- Zone de messages — reduced padding on mobile -->
	<div
		bind:this={messagesContainer}
		on:scroll={handleScroll}
		class="flex-1 overflow-y-auto px-2 sm:px-4 py-6 touch-scroll"
		id="main-content"
		role="log"
		aria-label="Chat messages"
	>
		<ErrorBoundary fallbackMessage="Failed to render messages">
			{#if $messagesLoading}
				<div class="max-w-2xl mx-auto">
					<MessageSkeleton count={3} />
				</div>
			{:else if $messages.length === 0 && !$isStreaming}
				<div class="max-w-2xl mx-auto text-center py-12">
					<p class="text-sm text-surface-500">
						No messages yet. Start typing below.
					</p>
				</div>
			{:else}
				<div class="max-w-2xl mx-auto space-y-4">
					<!-- Messages existants -->
					{#each $messages as msg, i (msg.id ?? `${msg.role}-${msg.timestamp}-${i}`)}
						<!-- svelte-ignore a11y-click-events-have-key-events -->
						<!-- svelte-ignore a11y-no-static-element-interactions -->
						<div
							class="message-wrapper"
							class:selected-fork={msg.id != null && msg.id === selectedMessageId}
							on:click={() => { if (msg.id != null) selectedMessageId = msg.id; }}
						>
							<ChatMessage
								message={msg}
								searchMetadata={msg.id != null ? $searchMetadataMap.get(String(msg.id)) ?? null : null}
								isLast={i === $messages.length - 1}
								isRetrying={$isStreaming}
								on:retry={handleRetry}
							/>
						</div>
					{/each}

					<!-- Currently streaming: show partial message -->
					{#if $isStreaming && $streamingContent}
						<ChatMessage
							message={streamingPlaceholder}
							isStreaming={true}
							streamContent={$streamingContent}
							streamThinking={$streamingThinking}
						/>
						{#if $isCodingStream}
							<CodingAgentProgress />
						{/if}
					{:else if $isStreaming && $streamingThinking}
						<ChatMessage
							message={streamingPlaceholder}
							isStreaming={true}
							streamContent={''}
							streamThinking={$streamingThinking}
						/>
					{:else if $isStreaming && $streamingVisionDelegation?.status === 'analyzing'}
						<VisionDelegationIndicator
							visionModel={delegatedVisionModel}
						/>
					{:else if $isStreaming && $isCodingStream}
						<CodingAgentProgress />
					{:else if $isStreaming}
						<StreamingIndicator />
					{/if}
				</div>
			{/if}
		</ErrorBoundary>

		<!-- Sentinelle pour auto-scroll -->
		<div bind:this={bottomSentinel} />
	</div>

	<!-- Scroll-to-bottom floating action button -->
	<ScrollToBottomFab visible={showScrollFab} onClick={handleScrollFabClick} />

	<!-- Zone de saisie — safe-area bottom, tighter mobile padding -->
	<div class="shrink-0 px-2 sm:px-4 py-2 sm:py-3 safe-area-bottom" style="border-top: 1px solid var(--oo-bd-subtle);">
		<div class="max-w-2xl mx-auto">
			<FileUpload
				{attachedFiles}
				disabled={$isStreaming}
				on:attach={handleAttach}
				on:remove={handleRemoveFile}
			>
				<ChatInput
					isStreaming={$isStreaming}
					canRetry={lastMessageIsAssistant && !$isStreaming}
					on:send={handleSend}
					on:cancel={handleCancel}
					on:retry={handleRetry}
				/>
			</FileUpload>
		</div>
	</div>
</div>

<!-- Live performance metrics overlay (auto-shows during inference) -->
<LiveMetricsOverlay />

<style>
	.message-wrapper {
		cursor: pointer;
		border-radius: 8px;
		border: 2px solid transparent;
		transition: border-color 0.15s ease;
	}

	.message-wrapper:hover {
		border-color: var(--oo-bd-strong);
	}

	.message-wrapper.selected-fork {
		border-color: var(--oo-accent);
	}
</style>