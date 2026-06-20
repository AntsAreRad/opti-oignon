<!--
  FeatureUnavailable.svelte (S135)
  Displayed when an optional backend module is not available.
  Warm, non-alarming design with icon, message, and optional learn-more link.
  CSS variables only.
-->
<script lang="ts">
	/** Human-readable name of the feature, e.g. "Knowledge Base". */
	export let featureName: string = 'This feature';

	/** Optional longer description of why it might be unavailable. */
	export let description: string =
		'The required backend module is not loaded. This may be because an optional dependency is missing or the module failed to initialize.';

	/** Optional URL for a "Learn more" link. */
	export let learnMoreUrl: string = '';
</script>

<div
	class="flex flex-col items-center justify-center py-10 px-6 text-center rounded-lg"
	style="background-color: var(--oo-bg-elevated); border: 1px dashed var(--oo-bd-subtle);"
	role="status"
>
	<!-- Icon: puzzle piece with question mark -->
	<div
		class="w-12 h-12 rounded-full flex items-center justify-center mb-4"
		style="background-color: var(--oo-bg-overlay);"
	>
		<svg
			class="w-6 h-6"
			style="color: var(--oo-fg-muted);"
			fill="none"
			viewBox="0 0 24 24"
			stroke="currentColor"
			stroke-width="1.5"
		>
			<path
				stroke-linecap="round"
				stroke-linejoin="round"
				d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M12 18h.01"
			/>
			<circle cx="12" cy="12" r="10" stroke-width="1.5" />
		</svg>
	</div>

	<h3 class="text-sm font-medium mb-1" style="color: var(--oo-fg-secondary);">
		{featureName} is not available
	</h3>

	<p class="text-xs max-w-sm mb-3" style="color: var(--oo-fg-muted);">
		{description}
	</p>

	{#if learnMoreUrl}
		<a
			href={learnMoreUrl}
			target="_blank"
			rel="noopener noreferrer"
			class="text-xs underline transition-colors"
			style="color: var(--oo-acc-400);"
		>
			Learn more
		</a>
	{/if}
</div>
