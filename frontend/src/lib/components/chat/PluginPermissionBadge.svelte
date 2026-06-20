<!--
  PluginPermissionBadge.svelte (S125)
  Displays plugin permission badges inline.
  Shows a warning icon when a plugin has inference_content permission
  (can read/modify conversation content during inference).
-->
<script lang="ts">
	/** Plugin name. */
	export let pluginName: string = '';

	/** List of granted permissions. */
	export let permissions: string[] = [];

	$: hasInferenceContent = permissions.includes('inference_content');
	$: hasNetworkOutbound = permissions.includes('network_outbound');
	$: hasFileSystem = permissions.includes('filesystem');

	$: warningLevel = hasInferenceContent ? 'high' : (hasNetworkOutbound || hasFileSystem) ? 'medium' : 'low';

	$: badgeColor = warningLevel === 'high'
		? 'var(--oo-fg-warning)'
		: warningLevel === 'medium'
			? 'var(--oo-fg-muted)'
			: 'var(--oo-sage)';

	$: tooltip = hasInferenceContent
		? `${pluginName}: has inference_content permission (can access conversation data)`
		: hasNetworkOutbound
			? `${pluginName}: has network_outbound permission`
			: `${pluginName}: standard permissions`;
</script>

{#if permissions.length > 0}
	<span
		class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs"
		style="color: {badgeColor}; border: 1px solid {badgeColor}; opacity: 0.85;"
		title={tooltip}
	>
		{#if hasInferenceContent}
			<!-- Warning triangle for inference_content -->
			<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
			</svg>
		{:else}
			<!-- Shield icon for standard permissions -->
			<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
			</svg>
		{/if}
		<span class="truncate max-w-[80px]">{pluginName}</span>
	</span>
{/if}
