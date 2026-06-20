<!--
  ProjectLinker.svelte
  Dropdown selector to link/unlink the current conversation to a project.
  Displayed in the ContextBar alongside the project badge.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		projects,
		conversationProjectId,
		loadProjects,
		linkConversation,
		unlinkConversation,
	} from '$lib/stores/projects';
	import { activeConversationId } from '$lib/stores/conversations';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	let open = false;
	let linking = false;

	$: convId = $activeConversationId;
	$: linkedProjectId = $conversationProjectId;

	onMount(() => {
		// Ensure project list is loaded for the dropdown
		if ($projects.length === 0) {
			loadProjects();
		}
	});

	async function handleLink(projectId: string) {
		if (!convId) return;
		linking = true;
		try {
			await linkConversation(projectId, convId);
			const proj = $projects.find((p) => p.id === projectId);
			toastSuccess(`Linked to "${proj?.name ?? projectId}"`);
			open = false;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to link');
		} finally {
			linking = false;
		}
	}

	async function handleUnlink() {
		if (!convId || !linkedProjectId) return;
		linking = true;
		try {
			await unlinkConversation(linkedProjectId, convId);
			toastSuccess('Unlinked from project');
			open = false;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to unlink');
		} finally {
			linking = false;
		}
	}

	function handleClickOutside(e: MouseEvent) {
		const target = e.target as HTMLElement;
		if (!target.closest('.project-linker-root')) {
			open = false;
		}
	}
</script>

<svelte:window on:click={handleClickOutside} />

{#if convId}
	<div class="relative project-linker-root">
		<button
			on:click|stopPropagation={() => (open = !open)}
			class="flex items-center gap-1 px-1.5 py-0.5 rounded transition-colors text-[11px]"
			style="color: var(--oo-fg-faint); {open ? 'background-color: var(--oo-bg-elevated);' : ''}"
			title={linkedProjectId ? 'Change or unlink project' : 'Link to a project'}
		>
			{#if !linkedProjectId}
				<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101" />
					<path d="M10.172 13.828a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.102 1.101" />
				</svg>
				<span>Link project</span>
			{:else}
				<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101" />
					<path d="M10.172 13.828a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.102 1.101" />
				</svg>
			{/if}
			<svg class="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M19 9l-7 7-7-7" />
			</svg>
		</button>

		<!-- Dropdown -->
		{#if open}
			<div class="absolute top-full left-0 mt-1 w-52 rounded-lg shadow-lg z-50 overflow-hidden"
				style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<!-- Header -->
				<div class="px-3 py-1.5 text-[10px] font-medium"
					style="color: var(--oo-fg-faint); border-bottom: 1px solid var(--oo-bd-subtle);">
					{linkedProjectId ? 'Change or unlink project' : 'Link to a project'}
				</div>

				<!-- Unlink option -->
				{#if linkedProjectId}
					<button
						on:click|stopPropagation={handleUnlink}
						disabled={linking}
						class="w-full text-left px-3 py-1.5 text-xs transition-colors flex items-center gap-2 disabled:opacity-50"
						style="color: var(--oo-error);"
					>
						<svg class="w-3.5 h-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<path d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
						</svg>
						Unlink from project
					</button>
					<hr style="border-color: var(--oo-bd-subtle);" />
				{/if}

				<!-- Project list -->
				<div class="max-h-48 overflow-y-auto">
					{#if $projects.length === 0}
						<p class="px-3 py-3 text-xs text-center" style="color: var(--oo-fg-faint);">
							No projects available
						</p>
					{:else}
						{#each $projects as proj (proj.id)}
							<button
								on:click|stopPropagation={() => handleLink(proj.id)}
								disabled={linking || proj.id === linkedProjectId}
								class="w-full text-left px-3 py-1.5 text-xs transition-colors flex items-center gap-2 disabled:opacity-40"
								style="color: var(--oo-fg-primary);"
							>
								<svg class="w-3.5 h-3.5 shrink-0"
									style="color: {proj.id === linkedProjectId ? 'var(--oo-acc-400)' : 'var(--oo-fg-faint)'};"
									fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
								</svg>
								<span class="truncate">{proj.name}</span>
								{#if proj.id === linkedProjectId}
									<svg class="w-3 h-3 ml-auto shrink-0" style="color: var(--oo-acc-400);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
										<path d="M5 13l4 4L19 7" />
									</svg>
								{/if}
							</button>
						{/each}
					{/if}
				</div>
			</div>
		{/if}
	</div>
{/if}
