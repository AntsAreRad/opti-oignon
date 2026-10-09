<!--
  SkillsPanel.svelte (Theme 3 / Odysseus Core)
  The skills-manager panel for the evolving-skills SKILL.md registry, built on
  the lib/ds primitives (Card, Button, Icon, EmptyState, InlineError). The
  skills the agent and its teacher write wait at the top as proposals, in the
  review the Memory and Notes panels share: each one whole, every character a
  screen hides written as its escape, accepted by the digest of its text.
  Below, the registry over $lib/api/skills: published skills and the drafts
  left from before proposals, one row each, keyed by its status so a draft
  and the published skill of the same name never share a row. Expanding a row
  shows exactly that item's text; publishing a draft, deleting a draft or a
  published skill, and adopting a published skill's bytes each send the
  digest of what was shown, and every button names its target. Updates
  announce through an aria-live region. Design-system tokens only (--oo-*);
  lucide icons through Icon.
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
		adoptSkill,
		deleteLabel,
		isDraft,
		type Skill,
		type SkillStatus
	} from '$lib/api/skills';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import PendingWritesReview from './PendingWritesReview.svelte';

	type Filter = 'all' | 'published' | 'drafts';

	let skills: Skill[] = [];
	let loading = false;
	let error: string | null = null;
	let filter: Filter = 'all';
	let selectedKey: string | null = null;
	let viewed: Record<string, Skill> = {};
	let busyKey: string | null = null;

	const STATUS_ICON: Record<SkillStatus, IconName> = {
		draft: 'file-clock',
		published: 'badge-check'
	};

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
			viewed = {};
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load skills';
		} finally {
			loading = false;
		}
	}

	async function toggleBody(skill: Skill) {
		if (selectedKey === skill.key) {
			selectedKey = null;
			return;
		}
		selectedKey = skill.key;
		try {
			const full = await getSkill(skill.category, skill.name, skill.status);
			viewed = { ...viewed, [skill.key]: full };
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to load skill');
		}
	}

	async function act(skill: Skill, run: () => Promise<unknown>, done: string, failed: string) {
		busyKey = skill.key;
		try {
			await run();
			toastSuccess(done);
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : failed);
		} finally {
			busyKey = null;
		}
	}

	function handlePublish(skill: Skill, shown: Skill) {
		act(skill, () => publishSkill(skill.category, skill.name, shown.sha256), `Published ${skill.name}`,
			'Failed to publish skill');
	}

	function handleAdopt(skill: Skill, shown: Skill) {
		act(skill, () => adoptSkill(skill.category, skill.name, shown.file_sha256 ?? ''), `Adopted ${skill.name}`,
			'Failed to adopt skill');
	}

	function handleDelete(skill: Skill, shown: Skill) {
		act(skill, () => deleteSkill(skill.category, skill.name, skill.status, shown.sha256), `Deleted ${skill.name}`,
			'Failed to delete skill');
	}

	onMount(load);
</script>

<section class="skills-panel">
	<PendingWritesReview store="skills" on:decided={load} />

	<header class="skills-header">
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
			A draft stays unpublished until you read it below and publish that very text.
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
				description="Skills the agent learns and you accept will appear here."
			/>
		{:else}
			{#each filtered as skill (skill.key)}
				<Card>
					<div class="skill-row" data-skill-key={skill.key}>
						<div class="skill-main">
							<Icon name={STATUS_ICON[skill.status]} />
							<div class="skill-meta">
								<button
									type="button"
									class="skill-name"
									aria-expanded={selectedKey === skill.key}
									on:click={() => toggleBody(skill)}
								>
									{skill.name}
								</button>
								<span class="skill-sub">{skill.category} · v{skill.version} · {skill.source}</span>
							</div>
							<span class="skill-badge skill-badge-{skill.status}">
								{skill.status === 'draft' ? 'Draft - not published' : 'Published'}
							</span>
							{#if skill.prompt_state === 'unadopted'}
								<span class="skill-badge skill-badge-received">
									{skill.sync_state === 'unadopted'
										? 'From a paired device - not adopted here'
										: 'Never adopted here - runs nowhere'}
								</span>
							{/if}
						</div>
					</div>
					{#if selectedKey === skill.key}
						{@const shown = viewed[skill.key]}
						{#if !shown}
							<p class="skills-loading">Loading...</p>
						{:else}
							<pre class="skill-body" data-skill-text={shown.key}>{shown.shown ?? ''}</pre>
							<p class="skill-sub">Digest {shown.sha256}</p>
							<div class="skill-actions">
								<Button
									variant="danger"
									on:click={() => handleDelete(skill, shown)}
									disabled={busyKey === skill.key}
								>
									<Icon name="trash-2" />
									{deleteLabel(skill)}
								</Button>
							</div>
							{#if isDraft(shown)}
								<Button
									variant="primary"
									on:click={() => handlePublish(skill, shown)}
									disabled={busyKey === skill.key}
								>
									<Icon name="check" />
									Publish this text
								</Button>
							{:else if shown.prompt_state === 'unadopted' && shown.raw_shown}
								<p class="skill-sub">Its bytes as they are on disk, every hidden character written out:</p>
								<pre class="skill-body" data-skill-raw={shown.key}>{shown.raw_shown}</pre>
								<p class="skill-sub">File digest {shown.file_sha256}</p>
								<Button
									variant="primary"
									on:click={() => handleAdopt(skill, shown)}
									disabled={busyKey === skill.key}
								>
									<Icon name="check" />
									Adopt these bytes
								</Button>
							{/if}
						{/if}
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
		justify-content: flex-end;
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
		overflow-wrap: anywhere;
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
	.skill-badge-received {
		color: var(--oo-fg-warning);
		background: var(--oo-warning-bg);
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
		overflow-wrap: anywhere;
	}
</style>
