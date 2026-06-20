<!--
  /dev/components -- design-system primitive gallery (spec 10.12).
  Development-only (import.meta.env.DEV). Lets us verify the 10
  primitives across the 5 themes and 3 densities. Not a user route.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { Button, Input, Card, Modal, Select, Switch, Tabs, Tooltip, Icon } from '$lib/ds';
	import Toast from '$lib/ds/Toast.svelte';
	import { addToast } from '$lib/stores/notifications';
	import type { SelectOption, TabItem } from '$lib/ds';

	const isDev = import.meta.env.DEV;

	const themes = [
		{ id: 'anthracite', label: 'Anthracite' },
		{ id: 'parchment', label: 'Parchment' },
		{ id: 'slate', label: 'Slate' },
		{ id: 'linen', label: 'Linen' },
		{ id: 'high-contrast', label: 'High Contrast' }
	];
	const densities = ['compact', 'comfortable', 'spacious'] as const;

	let theme = 'anthracite';
	let density: (typeof densities)[number] = 'comfortable';

	function applyTheme(t: string) {
		theme = t;
		document.documentElement.setAttribute('data-oo-theme', t);
	}
	function applyDensity(d: (typeof densities)[number]) {
		density = d;
		const el = document.documentElement;
		densities.forEach((x) => el.classList.remove(`oo-density-${x}`));
		el.classList.add(`oo-density-${d}`);
	}

	onMount(() => {
		applyTheme(theme);
		applyDensity(density);
	});

	// Demo state
	let modalOpen = false;
	let drawerOpen = false;
	let switchOn = true;
	let textVal = '';
	let selVal = 'a';
	let tabVal = 'one';
	let retryCount = 0;

	const selectOptions: SelectOption[] = [
		{ value: 'a', label: 'Alpha' },
		{ value: 'b', label: 'Beta' },
		{ value: 'c', label: 'Gamma', group: 'Greek' },
		{ value: 'd', label: 'Delta', group: 'Greek' }
	];
	const tabItems: TabItem[] = [
		{ id: 'one', label: 'Overview', icon: 'layout-dashboard' },
		{ id: 'two', label: 'Settings', icon: 'settings' },
		{ id: 'three', label: 'About' }
	];

	async function fakeRetry() {
		retryCount += 1;
		if (retryCount % 2 === 1) throw new Error('still failing');
	}
</script>

{#if isDev}
	<div class="dev-root">
		<Toast />

		<header class="dev-header">
			<h1>Design System — Primitives</h1>
			<div class="dev-controls">
				<div class="dev-seg" role="group" aria-label="Theme">
					{#each themes as t}
						<button class="dev-chip" class:active={theme === t.id} on:click={() => applyTheme(t.id)}>
							{t.label}
						</button>
					{/each}
				</div>
				<div class="dev-seg" role="group" aria-label="Density">
					{#each densities as d}
						<button class="dev-chip" class:active={density === d} on:click={() => applyDensity(d)}>
							{d}
						</button>
					{/each}
				</div>
			</div>
		</header>

		<main class="dev-grid">
			<Card variant="raised" padding="md">
				<h2>Button</h2>
				<div class="row">
					<Button variant="primary">Primary</Button>
					<Button variant="secondary">Secondary</Button>
					<Button variant="ghost">Ghost</Button>
					<Button variant="danger">Danger</Button>
					<Button variant="link">Link</Button>
				</div>
				<div class="row">
					<Button size="sm" iconLeft="plus">Small</Button>
					<Button size="md" iconLeft="plus">Medium</Button>
					<Button size="lg" iconLeft="plus">Large</Button>
					<Button iconOnly="settings" ariaLabel="Settings" />
					<Button loading>Loading</Button>
					<Button disabled>Disabled</Button>
				</div>
			</Card>

			<Card variant="raised" padding="md">
				<h2>Icon</h2>
				<div class="row">
					<Icon name="home" size="sm" />
					<Icon name="settings" size="md" />
					<Icon name="search" size="lg" />
					<Icon name="bell" size="md" />
					<Icon name="user" size="md" />
					<Icon name="check" size="md" />
				</div>
			</Card>

			<Card variant="raised" padding="md">
				<h2>Input</h2>
				<Input label="Text" bind:value={textVal} placeholder="Type..." hint="A helpful hint" iconLeft="search" />
				<Input label="Email" type="email" placeholder="you@example.com" />
				<Input label="With error" error="This field is required" />
				<Input label="Textarea" type="textarea" placeholder="Multiple lines..." />
			</Card>

			<Card variant="raised" padding="md">
				<h2>Select</h2>
				<Select label="Single (grouped)" bind:value={selVal} options={selectOptions} />
			</Card>

			<Card variant="raised" padding="md">
				<h2>Switch</h2>
				<Switch bind:checked={switchOn} label="Enable feature" description="Toggles the thing on or off" />
				<Switch checked={false} label="Disabled toggle" disabled />
			</Card>

			<Card variant="raised" padding="md">
				<h2>Tabs</h2>
				<Tabs bind:value={tabVal} tabs={tabItems}>
					<p>Active panel: <strong>{tabVal}</strong></p>
				</Tabs>
			</Card>

			<Card variant="raised" padding="md">
				<h2>Tooltip</h2>
				<div class="row">
					<Tooltip content="Tooltip on top" placement="top"><Button>Hover (top)</Button></Tooltip>
					<Tooltip content="Tooltip on the right" placement="right"><Button>Hover (right)</Button></Tooltip>
				</div>
			</Card>

			<Card variant="raised" padding="md">
				<h2>Modal &amp; Toast</h2>
				<div class="row">
					<Button on:click={() => (modalOpen = true)}>Open modal</Button>
					<Button on:click={() => (drawerOpen = true)}>Open drawer</Button>
					<Button on:click={() => addToast('Saved successfully', 'success')}>Success toast</Button>
					<Button
						on:click={() =>
							addToast('Upload failed', 'error', 8000, {
								title: 'Network error',
								action: { label: 'Retry', run: fakeRetry }
							})}
					>
						Toast with retry
					</Button>
				</div>
			</Card>

			<Card variant="flat" padding="md">
				<h2>Card (flat)</h2>
				<p>This is a flat card — border only, no shadow.</p>
			</Card>
		</main>

		<Modal open={modalOpen} title="Example modal" size="md" onClose={() => (modalOpen = false)}>
			<p>A center modal rendered through the native &lt;dialog&gt; focus trap.</p>
			<svelte:fragment slot="footer">
				<Button variant="ghost" on:click={() => (modalOpen = false)}>Cancel</Button>
				<Button variant="primary" on:click={() => (modalOpen = false)}>Confirm</Button>
			</svelte:fragment>
		</Modal>

		<Modal open={drawerOpen} variant="drawer-right" title="Example drawer" size="md" onClose={() => (drawerOpen = false)}>
			<p>Drawer variant — collapses to a bottom sheet below 768px.</p>
		</Modal>
	</div>
{:else}
	<p class="dev-disabled">This page is only available in development.</p>
{/if}

<style>
	.dev-root {
		min-height: 100vh;
		padding: var(--oo-space-6);
		background-color: var(--oo-bg-base);
		color: var(--oo-fg-primary);
	}
	.dev-header {
		margin-bottom: var(--oo-space-6);
	}
	.dev-header h1 {
		margin: 0 0 var(--oo-space-4);
		font-size: var(--oo-text-2xl);
		font-weight: 600;
	}
	.dev-controls {
		display: flex;
		flex-wrap: wrap;
		gap: var(--oo-space-4);
	}
	.dev-seg {
		display: inline-flex;
		gap: var(--oo-space-1);
		padding: var(--oo-space-1);
		background-color: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
	}
	.dev-chip {
		border: none;
		background: transparent;
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-xs);
		padding: var(--oo-space-2) var(--oo-space-3);
		border-radius: var(--oo-radius-sm);
		cursor: pointer;
		text-transform: capitalize;
	}
	.dev-chip:hover {
		background-color: var(--oo-bg-hover);
		color: var(--oo-fg-primary);
	}
	.dev-chip.active {
		background-color: var(--oo-acc-500);
		color: var(--oo-fg-on-accent);
	}
	.dev-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(20rem, 1fr));
		gap: var(--oo-space-5);
	}
	.dev-grid :global(h2) {
		margin: 0 0 var(--oo-space-4);
		font-size: var(--oo-text-sm);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: var(--oo-tracking-wide);
		color: var(--oo-fg-tertiary);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--oo-space-3);
		margin-bottom: var(--oo-space-3);
	}
	.row:last-child {
		margin-bottom: 0;
	}
	.dev-disabled {
		padding: var(--oo-space-7);
		color: var(--oo-fg-muted);
	}
</style>
