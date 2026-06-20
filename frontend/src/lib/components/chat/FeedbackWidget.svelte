<!--
  FeedbackWidget.svelte (S55, refactored S167)
  Inline thumbs up/down feedback for chat messages. Appears alongside the
  copy/retry actions; thumbs-down expands a short text field. Uses the ds
  Icon and Button primitives.
-->
<script lang="ts">
	import { submitFeedback } from '$lib/api/feedback';
	import Icon from '$lib/ds/Icon.svelte';
	import Button from '$lib/ds/Button.svelte';

	export let conversationId: string = '';
	export let messageId: string = '';
	export let modelUsed: string = '';
	export let pipelineUsed: string = '';
	export let taskType: string = '';

	let feedbackState: 'none' | 'up' | 'down' = 'none';
	let showTextInput = false;
	let feedbackText = '';
	let submitting = false;
	let submitted = false;

	async function handleThumb(value: 0 | 1) {
		if (submitted || submitting) return;
		feedbackState = value === 1 ? 'up' : 'down';
		// For thumbs down, show the text input before submitting.
		if (value === 0) {
			showTextInput = true;
			return;
		}
		// Thumbs up: submit immediately.
		await doSubmit(value);
	}

	async function doSubmit(value: number) {
		submitting = true;
		try {
			await submitFeedback({
				conversation_id: conversationId,
				message_id: messageId,
				rating_type: 'thumbs',
				rating_value: value,
				feedback_text: feedbackText,
				model_used: modelUsed,
				pipeline_used: pipelineUsed,
				task_type: taskType,
			});
			submitted = true;
			showTextInput = false;
		} catch (err) {
			// Silently fail - feedback is non-critical.
			console.warn('Feedback submission failed:', err);
		} finally {
			submitting = false;
		}
	}

	function submitWithText() {
		doSubmit(0);
	}

	function cancelText() {
		showTextInput = false;
		feedbackState = 'none';
		feedbackText = '';
	}
</script>

{#if !submitted}
	<div class="inline-flex items-center gap-0.5">
		<!-- Thumbs up -->
		<button
			on:click={() => handleThumb(1)}
			disabled={submitting}
			class="p-1 rounded-md transition-colors"
			style="color: {feedbackState === 'up' ? 'var(--oo-acc-400)' : 'var(--oo-fg-muted)'};
				background-color: {feedbackState === 'up' ? 'var(--oo-accent-bg)' : 'transparent'};"
			title="Good response"
			aria-label="Thumbs up"
		>
			<Icon name="thumbs-up" size="sm" />
		</button>

		<!-- Thumbs down -->
		<button
			on:click={() => handleThumb(0)}
			disabled={submitting}
			class="p-1 rounded-md transition-colors"
			style="color: {feedbackState === 'down' ? 'var(--oo-error)' : 'var(--oo-fg-muted)'};
				background-color: {feedbackState === 'down' ? 'var(--oo-error-bg)' : 'transparent'};"
			title="Poor response"
			aria-label="Thumbs down"
		>
			<Icon name="thumbs-down" size="sm" />
		</button>
	</div>

	<!-- Expanded text input for negative feedback -->
	{#if showTextInput}
		<div class="mt-2 p-2 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
			<textarea
				bind:value={feedbackText}
				placeholder="What could be improved? (optional)"
				rows="2"
				class="w-full px-2 py-1.5 text-xs rounded resize-none"
				style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd);
					color: var(--oo-fg-primary);"
			/>
			<div class="flex justify-end gap-1.5 mt-1.5">
				<Button variant="ghost" size="sm" on:click={cancelText}>Cancel</Button>
				<Button variant="primary" size="sm" loading={submitting} on:click={submitWithText}>
					Submit
				</Button>
			</div>
		</div>
	{/if}
{:else}
	<!-- Submitted state -->
	<div class="inline-flex items-center gap-1 text-xs" style="color: var(--oo-fg-faint);">
		<Icon name="check" size="sm" />
		Thanks
	</div>
{/if}
