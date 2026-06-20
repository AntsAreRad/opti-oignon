<!--
  ProjectContextBadge.svelte (refactored S167)
  Compact badge shown when a conversation is linked to a project. Shows the
  project name; click navigates to the project. A tooltip gives the full
  context, and a pulse indicator marks active context injection. Uses the
  ds Tooltip and Icon primitives.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import Tooltip from '$lib/ds/Tooltip.svelte';
	import Icon from '$lib/ds/Icon.svelte';
	import { conversationProject, conversationProjectId } from '$lib/stores/projects';

	const dispatch = createEventDispatcher<{ openProject: string }>();

	/** Optional trigger level from the last context injection (set externally). */
	export let triggerLevel: string | null = null;

	/** Whether context was injected in the last response. */
	export let contextActive: boolean = false;

	$: project = $conversationProject;
	$: linked = !!$conversationProjectId;
</script>

{#if linked && project}
	<Tooltip content={`Project: ${project.name} (click to open)`}>
		<button
			class="flex items-center gap-1.5 shrink-0 rounded px-1.5 py-0.5 transition-colors group"
			style="background-color: var(--oo-warning-bg); border: 1px solid var(--oo-warning-bd);"
			aria-label={`Open project ${project.name}`}
			on:click={() => dispatch('openProject', project.id)}
		>
			<span style="color: var(--oo-acc-400);"><Icon name="folder" size="sm" /></span>

			<span class="text-[11px] font-medium max-w-[100px] truncate" style="color: var(--oo-acc-400);">
				{project.name}
			</span>

			{#if contextActive}
				<span class="flex items-center gap-0.5">
					<span
						class="w-1.5 h-1.5 rounded-full animate-pulse"
						style="background-color: var(--oo-success);"
					/>
					{#if triggerLevel}
						<span class="text-[9px] font-mono" style="color: var(--oo-fg-faint);">L{triggerLevel}</span>
					{/if}
				</span>
			{/if}
		</button>
	</Tooltip>
{/if}
