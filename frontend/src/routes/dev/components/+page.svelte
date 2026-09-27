<!--
  /dev/components -- the design system's gallery.
  Development only (import.meta.env.DEV); not a user route. It shows every
  primitive in its states, and every icon of the inline set by name, in the
  palette and the density chosen at its top. The gallery is a preview: its
  root carries the palette attribute and the density class, and the page's
  own root is left as the theme path set it. It is held to the surface
  rules like the primitives it shows.
-->
<script lang="ts">
	import {
		Button,
		Card,
		Checkbox,
		Icon,
		IconButton,
		Input,
		Menu,
		Modal,
		PanelHeader,
		Select,
		SidePanel,
		Switch,
		Tabs,
		ToggleChip,
		Tooltip,
		ICONS
	} from '$lib/ds';
	import Toast from '$lib/ds/Toast.svelte';
	import { addToast } from '$lib/stores/notifications';
	import type { MenuItem, SelectOption, TabItem } from '$lib/ds';
	import { CHOICE_LABELS, DENSITIES, PALETTES, type Density, type Palette } from '$lib/theme/apply';

	const isDev = import.meta.env.DEV;
	let headerOpen = false;

	const DENSITY_LABELS: Record<Density, string> = {
		compact: 'Compact',
		comfortable: 'Comfortable',
		spacious: 'Spacious'
	};

	let theme: Palette = 'night';
	let density: Density = 'comfortable';

	// Demo state
	let modalOpen = false;
	let drawerOpen = false;
	let switchOn = true;
	let textVal = '';
	let selVal = 'a';
	let tabVal = 'one';
	let pillVal = 'one';
	let retryCount = 0;
	let pinned = true;
	let details = false;
	let webSearch = true;
	let thinking = false;
	let sandbox = false;
	let remember = true;
	let notify = false;
	let allChosen = false;
	let someChosen = true;
	let chosen = 'nothing yet';
	let panelWidth = 260;

	const iconNames = Object.keys(ICONS).sort();

	const selectOptions: SelectOption[] = [
		{ value: 'a', label: 'Alpha' },
		{ value: 'b', label: 'Beta' },
		{ value: 'c', label: 'Gamma', group: 'Greek' },
		{ value: 'd', label: 'Delta', group: 'Greek' }
	];
	const tabItems: TabItem[] = [
		{ id: 'one', label: 'Overview', icon: 'home' },
		{ id: 'two', label: 'Settings', icon: 'sliders' },
		{ id: 'three', label: 'About' }
	];
	const menuItems: MenuItem[] = [
		{ id: 'copy', label: 'Copy', icon: 'copy' },
		{ id: 'export', label: 'Export', icon: 'download' },
		{ id: 'rename', label: 'Rename', icon: 'pencil', disabled: true },
		{ id: 'wipe', label: 'Wipe', icon: 'trash', danger: true }
	];

	async function fakeRetry() {
		retryCount += 1;
		if (retryCount % 2 === 1) throw new Error('still failing');
	}
</script>

{#if isDev}
	<div class="dev-root oo-density-{density}" data-oo-theme={theme}>
		<Toast />

		<header class="dev-header">
			<h1>Design system primitives</h1>
			<div class="dev-controls">
				<div class="dev-seg" role="group" aria-label="Palette">
					{#each PALETTES as t}
						<Button
							variant="ghost"
							shape="pill"
							size="sm"
							pressed={theme === t}
							on:click={() => (theme = t)}>{CHOICE_LABELS[t]}</Button
						>
					{/each}
				</div>
				<div class="dev-seg" role="group" aria-label="Density">
					{#each DENSITIES as d}
						<Button
							variant="ghost"
							shape="pill"
							size="sm"
							pressed={density === d}
							on:click={() => (density = d)}>{DENSITY_LABELS[d]}</Button
						>
					{/each}
				</div>
			</div>
		</header>

		<main id="main-content" class="dev-grid">
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
					<Button loading>Loading</Button>
					<Button disabled>Disabled</Button>
				</div>
				<div class="row">
					<Button shape="pill" variant="secondary" iconLeft="search">Pill</Button>
					<Button shape="round" variant="primary" iconOnly="arrow-up" ariaLabel="Send" />
					<Button shape="round" variant="secondary" iconOnly="more" ariaLabel="More" />
					<Button iconOnly="sliders" ariaLabel="Settings" />
				</div>
				<div class="row">
					<Button variant="secondary" pressed={pinned} on:click={() => (pinned = !pinned)}>Pinned</Button>
					<Button
						variant="ghost"
						iconOnly="pin"
						ariaLabel="Pin"
						pressed={pinned}
						on:click={() => (pinned = !pinned)}
					/>
					<Button
						variant="ghost"
						iconRight={details ? 'chevron-up' : 'chevron-down'}
						expanded={details}
						controls="dev-details"
						on:click={() => (details = !details)}>Details</Button
					>
				</div>
				<p id="dev-details" class="dev-note" hidden={!details}>
					A pressed button draws its check; a button that shows a region says whether it is shown.
				</p>
			</Card>

			<Card variant="raised" padding="md">
				<h2>Icon button</h2>
				<div class="row">
					<IconButton icon="x" label="Close" />
					<IconButton icon="copy" label="Copy" />
					<IconButton icon="retry" label="Retry" />
					<IconButton icon="arrow-up" label="Send" variant="primary" />
					<IconButton icon="trash" label="Delete" disabled />
				</div>
				<div class="row">
					<IconButton icon="panel-right" label="Open the panel" size="lg" />
					<IconButton icon="stop-fill" label="Stop this reply" size="lg" variant="primary" />
				</div>
			</Card>

			<Card variant="raised" padding="md">
				<h2>Toggle chip</h2>
				<div class="row">
					<ToggleChip label="Web search" icon="globe" bind:pressed={webSearch} />
					<ToggleChip label="Thinking" icon="bulb" note="auto" bind:pressed={thinking} />
					<ToggleChip label="Sandbox" icon="box" bind:pressed={sandbox} />
					<ToggleChip label="Code" icon="code" disabled />
				</div>
			</Card>

			<Card variant="raised" padding="md">
				<h2>Menu</h2>
				<div class="row">
					<Menu label="Actions" items={menuItems} on:select={(e) => (chosen = e.detail)} />
					<Menu
						label="More actions"
						icon="more"
						items={menuItems}
						placement="bottom-start"
						on:select={(e) => (chosen = e.detail)}
					/>
				</div>
				<p class="dev-note">Chosen: {chosen}</p>
			</Card>

			<Card variant="raised" padding="md">
				<h2>Checkbox</h2>
				<div class="column">
					<Checkbox label="Remember this device" bind:checked={remember} />
					<Checkbox
						label="Notify me"
						description="A notice when a long task ends"
						bind:checked={notify}
					/>
					<Checkbox label="Select all" bind:checked={allChosen} bind:indeterminate={someChosen} />
					<Checkbox label="Unavailable" disabled />
				</div>
			</Card>

			<Card variant="raised" padding="md">
				<h2>Icons</h2>
				<ul class="dev-icons">
					{#each iconNames as name}
						<li class="dev-icon">
							<Icon {name} size="lg" />
							<span class="dev-icon-name">{name}</span>
						</li>
					{/each}
				</ul>
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
				<Tabs bind:value={pillVal} tabs={tabItems} variant="pill">
					<p>Active pill: <strong>{pillVal}</strong></p>
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
				<h2>Modal and toast</h2>
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

			<Card variant="raised" padding="md">
				<h2>Panel header</h2>
				<PanelHeader title="Cache" description="A plain heading, with its description." />
				<PanelHeader
					title="Model health"
					description="A disclosure: the whole row opens it."
					expanded={headerOpen}
					controls="dev-header-body"
					on:toggle={() => (headerOpen = !headerOpen)}
				/>
				<div id="dev-header-body" hidden={!headerOpen}>
					<p>The region the header shows or hides.</p>
				</div>
			</Card>

			<Card variant="flat" padding="md">
				<h2>Card (flat)</h2>
				<p>A flat card: set apart by its tone, with no shadow.</p>
			</Card>
		</main>

		<!-- The side panel is a complementary landmark, which stands at the top
		     level: its demonstration sits outside the page's main region. -->
		<div class="dev-grid dev-after-main">
			<Card variant="raised" padding="md">
				<h2>Side panel</h2>
				<div class="dev-frame">
					<div class="dev-frame-main">The page</div>
					<SidePanel label="Demo panel" bind:width={panelWidth} min={160} max={320}>
						<p class="dev-frame-panel">
							Beside the page, {panelWidth} px. Focus its edge and use the arrows, or press it
							without dragging.
						</p>
					</SidePanel>
				</div>
			</Card>
		</div>

		<Modal open={modalOpen} title="Example modal" size="md" onClose={() => (modalOpen = false)}>
			<p>A center modal rendered through the native &lt;dialog&gt; focus trap.</p>
			<svelte:fragment slot="footer">
				<Button variant="ghost" on:click={() => (modalOpen = false)}>Cancel</Button>
				<Button variant="primary" on:click={() => (modalOpen = false)}>Confirm</Button>
			</svelte:fragment>
		</Modal>

		<Modal open={drawerOpen} variant="drawer-right" title="Example drawer" size="md" onClose={() => (drawerOpen = false)}>
			<p>The drawer variant: a bottom sheet below 768px.</p>
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
		flex-wrap: wrap;
		gap: var(--oo-space-1);
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
		color: var(--oo-fg-secondary);
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
	.column {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}
	.dev-note {
		margin: var(--oo-space-3) 0 0;
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-secondary);
	}
	.dev-after-main {
		margin-top: var(--oo-space-5);
	}
	.dev-frame {
		display: flex;
		height: 12rem;
		border-radius: var(--oo-radius-md);
		background-color: var(--oo-bg-base);
		overflow: hidden;
	}
	.dev-frame-main {
		flex: 1;
		min-width: 0;
		padding: var(--oo-space-4);
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
	}
	.dev-frame-panel {
		margin: 0;
		padding: var(--oo-space-4);
		font-size: var(--oo-text-sm);
	}
	.dev-icons {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(6rem, 1fr));
		gap: var(--oo-space-3);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.dev-icon {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--oo-space-2);
		padding: var(--oo-space-2);
		color: var(--oo-fg-primary);
	}
	.dev-icon-name {
		font-family: var(--oo-font-mono);
		font-size: var(--oo-text-2xs);
		color: var(--oo-fg-secondary);
	}
	.dev-disabled {
		padding: var(--oo-space-7);
		color: var(--oo-fg-muted);
	}
</style>
