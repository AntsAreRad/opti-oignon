<!--
  Projects list page (S169).
  Renders ProjectList inside the shared AppShell (from +layout). ProjectList
  navigates to the promoted /projects/[id] detail route via real links. The
  ?new=1 query (used by the sidebar "New project" affordance) opens the create
  modal on load.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import ProjectList from '$lib/components/panels/ProjectList.svelte';
	import { clearActiveProject } from '$lib/stores/projects';

	$: openCreate = $page.url.searchParams.get('new') === '1';

	onMount(() => {
		// List view owns no active project; clear any lingering detail so the
		// header label and sidebar highlight reset.
		clearActiveProject();
	});
</script>

<ProjectList {openCreate} />
