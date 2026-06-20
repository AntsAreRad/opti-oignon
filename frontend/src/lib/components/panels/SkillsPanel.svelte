<!--
  SkillsPanel.svelte (S177, Theme 3 / Odysseus Core)
  The skills-manager panel for the evolving-skills SKILL.md registry, built on
  the S166 lib/ds primitives (Card, Button, Icon, EmptyState, InlineError). It
  browses the registry over $lib/api/skills -- published skills and the drafts
  the agent proposes -- lets you expand a skill to read its procedure, and
  surfaces the approval-gated write actions: publishing a draft (the human
  approval that turns an agent proposal into a published skill) and deleting one.
  Drafts are clearly marked as awaiting approval. Updates announce through an
  aria-live region. Design-system tokens only (--oo-*); lucide icons through
  Icon. Registered in FRONTEND_REDESIGN_SPEC.md.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { Button, Card, Icon, EmptyState, InlineError } from '$lib/ds';
	import type { IconName } from '$lib/ds';
	import {
		listSkills,
		getSkill,
		publishSkill,
		deleteSkill,
		isDraft,
		type Skill,
		type SkillStatus
	} from '$lib/api/skills';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	type Filter = 'all' | 'published' | 'drafts';

	let skills: Skill[] = [];
	let loading = false;
	let error: string | null = null;
	let filter: Filter = 'all';
	let selectedKey: string | null = null;
	let bodyByKey: Record<string, string> = {};
	let busyKey: string | null = null;

	const STATUS_ICON: Record<SkillStatus, IconName> = {
		draft: 'file-clock',
		published: 'badge-check'
	};

	function keyOf(skill: Skill): string {
		return `${skill.category}/${skill.name}`;
	}

	$: filtered = skills.filter((s) => {
		if (filter === 'published') return s.status === 'published';
		if (filter === 'drafts') return s.status === 'draft';
		return true;
	});

	$: draftCount = skills.filter((s) => s.status === 'draft').length;

	async function load() {
		loading = true;
		error = null;
		try {
			skills = await listSkills(true);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load skills';
		} finally {
			loading = false;
		}
	}

	async function toggleBody(skill: Skill) {
		const key = keyOf(skill);
		if (selectedKey === key) {
			selectedKey = null;
			return;
		}
		selectedKey = key;
		if (bodyByKey[key] === undefined) {
			try {
				const full = await getSkill(skill.category, skill.name);
				bodyByKey = { ...bodyByKey, [key]: full.body ?? '' };
			} catch (e) {
				toastError(e instanceof Error ? e.message : 'Failed to load skill');
			}
		}
	}

	async function handlePublish(skill: Skill) {
		busyKey = keyOf(skill);
		try {
			await publishSkill(skill.category, skill.name);
			toastSuccess(`Published ${skill.name}`);
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to publish skill');
		} finally {
			busyKey = null;
		}
	}

	async function handleDelete(skill: Skill) {
		busyKey = keyOf(skill);
		try {
			await deleteSkill(skill.category, skill.name);
			toastSuccess(`Deleted ${skill.name}`);
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to delete skill');
		} finally {
			busyKey = null;
		}
	}

	onMount(load);
</script>

<section class="skills-panel">
	<header class="skills-header">
		<div class="skills-title">
			<Icon name="book-marked" />
			<h2>Skills</h2>
		</div>
		<Button variant="ghost" on:click={load} disabled={loading}>
			<Icon name="refresh-cw" />
			Refresh
		</Button>
	</header>

	<div class="skills-filters" role="group" aria-label="Filter skills">
		<Button variant={filter === 'all' ? 'primary' : 'ghost'} on:click={() => (filter = 'all')}>
			All
		</Button>
		<Button
			variant={filter === 'published' ? 'primary' : 'ghost'}
			on:click={() => (filter = 'published')}
		>
			Published
		</Button>
		<Button
			variant={filter === 'drafts' ? 'primary' : 'ghost'}
			on:click={() => (filter = 'drafts')}
		>
			Drafts{#if draftCount > 0} ({draftCount}){/if}
		</Button>
	</div>

	{#if draftCount > 0}
		<p class="skills-approval-note" role="note">
			<Icon name="shield-alert" />
			Drafts are agent-proposed and stay unpublished until you approve them below.
		</p>
	{/if}

	{#if error}
		<InlineError message={error} onRetry={load} />
	{/if}

	<div class="skills-list" role="status" aria-live="polite">
		{#if loading && skills.length === 0}
			<p class="skills-loading">Loading skills...</p>
		{:else if filtered.length === 0}
			<EmptyState
				icon="book-marked"
				title="No skills yet"
				description="Skills the agent learns and you approve will appear here."
			/>
		{:else}
			{#each filtered as skill (keyOf(skill))}
				<Card>
					<div class="skill-row">
						<div class="skill-main">
							<Icon name={STATUS_ICON[skill.status]} />
							<div class="skill-meta">
								<button
									type="button"
									class="skill-name"
									aria-expanded={selectedKey === keyOf(skill)}
									on:click={() => toggleBody(skill)}
								>
									{skill.name}
								</button>
								<span class="skill-sub">{skill.category} · v{skill.version} · {skill.source}</span>
							</div>
							<span class="skill-badge skill-badge-{skill.status}">
								{skill.status === 'draft' ? 'Draft - awaiting approval' : 'Published'}
							</span>
						</div>
						<div class="skill-actions">
							{#if isDraft(skill)}
								<Button
									variant="primary"
									on:click={() => handlePublish(skill)}
									disabled={busyKey === keyOf(skill)}
								>
									<Icon name="check" />
									Approve &amp; publish
								</Button>
							{/if}
							<Button
								variant="danger"
								on:click={() => handleDelete(skill)}
								disabled={busyKey === keyOf(skill)}
							>
								<Icon name="trash-2" />
								Delete
							</Button>
						</div>
					</div>
					{#if selectedKey === keyOf(skill)}
						<pre class="skill-body">{bodyByKey[keyOf(skill)] ?? 'Loading...'}</pre>
					{/if}
				</Card>
			{/each}
		{/if}
	</div>
</section>

<style>
	.skills-panel {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
		padding: var(--oo-space-4);
	}
	.skills-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	.skills-title {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}
	.skills-title h2 {
		margin: 0;
		font-size: var(--oo-text-lg);
		color: var(--oo-fg-primary);
	}
	.skills-filters {
		display: flex;
		gap: var(--oo-space-2);
	}
	.skills-approval-note {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		margin: 0;
		padding: var(--oo-space-2) var(--oo-space-3);
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-warning);
		background: var(--oo-warning-bg);
		border: 1px solid var(--oo-warning-bd);
		border-radius: var(--oo-radius-md);
	}
	.skills-list {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}
	.skills-loading {
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}
	.skill-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-3);
	}
	.skill-main {
		display: flex;
		align-items: center;
		gap: var(--oo-space-3);
		min-width: 0;
	}
	.skill-meta {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.skill-name {
		padding: 0;
		border: none;
		background: none;
		text-align: left;
		cursor: pointer;
		font-size: var(--oo-text-base);
		color: var(--oo-fg-primary);
	}
	.skill-name:hover {
		color: var(--oo-acc-600);
	}
	.skill-sub {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}
	.skill-badge {
		white-space: nowrap;
		padding: 2px var(--oo-space-2);
		font-size: var(--oo-text-xs);
		border-radius: var(--oo-radius-sm);
	}
	.skill-badge-draft {
		color: var(--oo-fg-warning);
		background: var(--oo-warning-bg);
	}
	.skill-badge-published {
		color: var(--oo-fg-success);
		background: var(--oo-success-bg);
	}
	.skill-actions {
		display: flex;
		gap: var(--oo-space-2);
		flex-shrink: 0;
	}
	.skill-body {
		margin: var(--oo-space-3) 0 0;
		padding: var(--oo-space-3);
		font-family: var(--oo-font-mono);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-secondary);
		background: var(--oo-bg-subtle);
		border-radius: var(--oo-radius-md);
		white-space: pre-wrap;
		overflow-x: auto;
	}
</style>
