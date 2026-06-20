<!--
  VisionDelegationIndicator.svelte
  S95: Shown during vision delegation when the pipeline is analyzing
  an image with the vision model before passing context to the text model.
  Displays the OnionLoader with an "Analyzing image..." label.
-->
<script lang="ts">
	import OnionLoader from '$lib/components/OnionLoader.svelte';

	export let visionModel: string = '';
</script>

<div class="flex justify-start">
	<div class="inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm"
		style="background-color: var(--oo-msg-bot-bg); border: 1px solid var(--oo-msg-bot-bd);">
		<OnionLoader size={20} color="var(--oo-tobacco)" />
		<span style="color: var(--oo-fg-muted);">
			Analyzing image{visionModel ? ` with ${visionModel}` : ''}...
		</span>
	</div>
</div>
