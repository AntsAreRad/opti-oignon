<!--
  Chat view: messages + streaming + input.
  Handles WebSocket streaming, cancellation and retry.
  Passes chat options (model, preset, temperature) via chatOptions.
  Mobile responsive -- scroll FAB, tighter padding; the shell keeps the page
  clear of the safe areas.
  The thread is not a live region: StreamingStatus, mounted where the reply
  is written, holds the stream's one status region, its waiting line and a
  run's card.
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
		streamingLoader,
		streamingConversation,
		lastSearchMetadata,
		searchMetadataMap,
		sendMessage,
		retryLastMessage,
		cancelCurrentGeneration,
		isCodingStream
	} from '$lib/stores/chat';
	import { getChatOptions } from '$lib/stores/chatOptions';
	import { scrollBehavior } from '$lib/motion';
	import { toastError } from '$lib/stores/notifications';
	import ChatMessage from '$lib/components/chat/ChatMessage.svelte';
	import ChatInput from '$lib/components/chat/ChatInput.svelte';
	import FileUpload from '$lib/components/chat/FileUpload.svelte';
	import StreamingStatus from '$lib/components/chat/StreamingStatus.svelte';
	import LiveMetricsOverlay from '$lib/components/chat/LiveMetricsOverlay.svelte';
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

	// The loader of the stream this conversation started, and no other's.
	$: loader = convId !== null && $streamingConversation === convId ? $streamingLoader : null;

	function scrollToBottom() {
		if (bottomSentinel && shouldAutoScroll) {
			bottomSentinel.scrollIntoView({ behavior: scrollBehavior() });
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

	// FAB click handler -- scroll to bottom
	function handleScrollFabClick() {
		shouldAutoScroll = true;
		showScrollFab = false;
		scrollToBottom();
	}

	async function handleSend(event: CustomEvent<{ text: string; images: string[]; pasted: [number, number][] }>) {
		if (!convId) return;
		shouldAutoScroll = true;
		const options = getChatOptions();

		// Images travel with the options; an empty list is not sent.
		options.images = event.detail.images;
		// What the user pasted or dropped rather than typed: the server saves
		// it as a document part of the turn. An empty list is not sent.
		options.pasted = event.detail.pasted;
		// Attached files travel beside the typed words, never inside them: the
		// server saves each as a document of its own, and an empty list is not
		// sent.
		options.documents = attachedFiles.map(({ filename, content }) => ({ filename, content }));
		attachedFiles = [];

		await sendMessage(convId, event.detail.text, options);
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

	<!-- Zone de messages -- reduced padding on mobile -->
	<div
		bind:this={messagesContainer}
		on:scroll={handleScroll}
		class="flex-1 overflow-y-auto px-2 sm:px-4 py-6 touch-scroll"
		role="region"
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
								conversationId={convId ?? ''}
								searchMetadata={msg.id != null ? $searchMetadataMap.get(String(msg.id)) ?? null : null}
								isLast={i === $messages.length - 1}
								isRetrying={$isStreaming}
								on:retry={handleRetry}
							/>
						</div>
					{/each}

					<!-- The stream: its waiting line or its run, where the reply is written -->
					<StreamingStatus {loader} quiet={$isStreaming && $isCodingStream} />

					<!-- Currently streaming: show partial message -->
					{#if $isStreaming && ($streamingContent || $streamingThinking)}
						<ChatMessage
							message={streamingPlaceholder}
							conversationId={convId ?? ''}
							isStreaming={true}
							streamContent={$streamingContent}
							streamThinking={$streamingThinking}
							quietThinking={true}
						/>
					{/if}
					{#if $isStreaming && $isCodingStream}
						<CodingAgentProgress />
					{/if}
				</div>
			{/if}
		</ErrorBoundary>

		<!-- Sentinelle pour auto-scroll -->
		<div bind:this={bottomSentinel} />
	</div>

	<!-- Scroll-to-bottom floating action button -->
	<ScrollToBottomFab visible={showScrollFab} onClick={handleScrollFabClick} />

	<!-- The composer, with tighter padding on a phone -->
	<div class="shrink-0 px-2 sm:px-4 py-2 sm:py-3" style="border-top: 1px solid var(--oo-bd-subtle);">
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