<!--
  Project detail route (S169).
  Promotes the ProjectDetail panel into a real /projects/[id] route. Loads the
  project on mount and whenever the :id param changes; the back action returns
  to /projects. ProjectDetail keeps all of its behaviour (edit, settings,
  reindex, context preview, file management) and the project APIs are unchanged.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import ProjectDetail from '$lib/components/panels/ProjectDetail.svelte';
	import { selectProject } from '$lib/stores/projects';

	// Re-load whenever the route param changes (covers in-app navigation
	// between two project detail routes without a full remount).
	$: projectId = $page.params.id;
	$: if (projectId) {
		selectProject(projectId);
	}

	function handleBack() {
		goto('/projects');
	}
</script>

<ProjectDetail on:back={handleBack} />
