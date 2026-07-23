<!--
  StreamingIndicator.svelte
  Animated indicator during LLM response generation.
  Replaced bouncing dots with OnionLoader spinning onion.
  Removed model name display during streaming (shown in message header after completion).
       OnionLoader uses tobacco accent for warm organic feel.
  Shows intermediate status messages (Searching..., Thinking..., etc.)
-->
<script lang="ts">
	import OnionLoader from '$lib/components/OnionLoader.svelte';
	import { streamingStatus } from '$lib/stores/chat';
</script>

<div class="flex justify-start">
	<div class="inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm"
		style="background-color: var(--oo-msg-bot-bg); border: 1px solid var(--oo-msg-bot-bd);">
		<OnionLoader size={20} color="var(--oo-tobacco)" />
		{#if $streamingStatus}
			<span class="text-xs animate-pulse" style="color: var(--oo-fg-muted);">
				{$streamingStatus}
			</span>
		{/if}
	</div>
</div>
