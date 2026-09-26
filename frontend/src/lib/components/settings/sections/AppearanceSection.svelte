<!--
  AppearanceSection.svelte
  The Appearance section of the consolidated /settings hub (spec 5.5, 9.2).

  Appearance is the one new section: it did not exist among the 12 legacy
  tabs. It owns the visual preferences in the `preferences` store -- the
  palette choice ("Match system", day, night, high contrast), 3 density
  modes, the text size, and the motion preference -- and applies each
  immediately with a toast (spec 5.9 immediate-apply). Each group offers a
  reset-to-default (spec 5.9).

  It also opens the keyboard bindings (ShortcutSettings) in a right-side
  drawer.

  Each palette's preview is an element carrying that palette's attribute,
  so it shows the palette's own tokens and no colour of its own; "Match
  system" shows day and night side by side.
-->
<script lang="ts">
	import SettingsGroup from '$lib/components/settings/SettingsGroup.svelte';
	import Modal from '$lib/ds/Modal.svelte';
	import Button from '$lib/ds/Button.svelte';
	import ShortcutSettings from '$lib/components/settings/ShortcutSettings.svelte';
	import {
		palette,
		CHOICES,
		CHOICE_LABELS,
		setPalette,
		density,
		DENSITIES,
		DENSITY_LABELS,
		setDensity,
		typeScale,
		TYPE_SCALES,
		TYPE_SCALE_LABELS,
		setTypeScale,
		motionPref,
		MOTION_PREFS,
		MOTION_LABELS,
		setMotionPref,
		type Choice,
		type Density,
		type TypeScale,
		type MotionPref
	} from '$lib/stores/preferences';
	import { toastSuccess } from '$lib/stores/notifications';

	function choosePalette(p: Choice) {
		if ($palette === p) return;
		setPalette(p);
		toastSuccess(`Theme set to ${CHOICE_LABELS[p]}`);
	}

	function chooseDensity(d: Density) {
		if ($density === d) return;
		setDensity(d);
		toastSuccess(`Density set to ${DENSITY_LABELS[d]}`);
	}

	function chooseTypeScale(t: TypeScale) {
		if ($typeScale === t) return;
		setTypeScale(t);
		toastSuccess(`Text size set to ${TYPE_SCALE_LABELS[t]}`);
	}

	function chooseMotion(m: MotionPref) {
		if ($motionPref === m) return;
		setMotionPref(m);
		toastSuccess(`Motion set to ${MOTION_LABELS[m]}`);
	}

	function resetTheme() {
		setPalette('system');
		toastSuccess('Theme reset to default');
	}
	function resetDensity() {
		setDensity('comfortable');
		toastSuccess('Density reset to default');
	}
	function resetTypeScale() {
		setTypeScale('default');
		toastSuccess('Text size reset to default');
	}
	function resetMotion() {
		setMotionPref('system');
		toastSuccess('Motion reset to default');
	}

	const DENSITY_HINT: Record<Density, string> = {
		compact: 'Tighter spacing, more on screen',
		comfortable: 'Balanced default',
		spacious: 'Looser spacing, larger touch targets'
	};

	const PALETTE_HINT: Record<Choice, string> = {
		system: 'Night, or day when your system is light; high contrast when it asks for more contrast',
		day: 'Light',
		night: 'Dark',
		'high-contrast': 'Dark, strongest contrast'
	};

	const MOTION_HINT: Record<MotionPref, string> = {
		system: 'Follow your operating system setting',
		reduced: 'Minimize animations and transitions',
		full: 'Keep animations even if the system asks for less'
	};

	let showShortcuts = false;
</script>

<div class="oo-appearance">
	<SettingsGroup
		id="appearance-theme"
		title="Theme"
		description="The palette applies instantly across the whole interface. Match system follows your system's light or dark setting, and its contrast setting."
		onReset={resetTheme}
	>
		<div class="oo-swatch-grid" role="radiogroup" aria-label="Theme palette">
			{#each CHOICES as p (p)}
				<button
					type="button"
					class="oo-swatch"
					class:oo-swatch-active={$palette === p}
					role="radio"
					aria-checked={$palette === p}
					on:click={() => choosePalette(p)}
				>
					<span class="oo-swatch-preview" aria-hidden="true">
						{#if p === 'system'}
							<span class="oo-swatch-half" data-oo-theme="day"><span class="oo-swatch-surface"></span><span class="oo-swatch-fg"></span></span>
							<span class="oo-swatch-half" data-oo-theme="night"><span class="oo-swatch-surface"></span><span class="oo-swatch-fg"></span></span>
						{:else}
							<span class="oo-swatch-half" data-oo-theme={p}><span class="oo-swatch-surface"></span><span class="oo-swatch-fg"></span></span>
						{/if}
					</span>
					<span class="oo-swatch-meta">
						<span class="oo-swatch-name">{CHOICE_LABELS[p]}</span>
						<span class="oo-swatch-mode">{PALETTE_HINT[p]}</span>
					</span>
				</button>
			{/each}
		</div>
	</SettingsGroup>

	<SettingsGroup
		id="appearance-density"
		title="Density"
		description="Controls spacing and control sizes throughout the app."
		onReset={resetDensity}
	>
		<div class="oo-opt-row oo-opt-row-3" role="radiogroup" aria-label="Interface density">
			{#each DENSITIES as d (d)}
				<button
					type="button"
					class="oo-opt"
					class:oo-opt-active={$density === d}
					role="radio"
					aria-checked={$density === d}
					on:click={() => chooseDensity(d)}
				>
					<span class="oo-opt-name">{DENSITY_LABELS[d]}</span>
					<span class="oo-opt-hint">{DENSITY_HINT[d]}</span>
				</button>
			{/each}
		</div>
	</SettingsGroup>

	<SettingsGroup
		id="appearance-typography"
		title="Text size"
		description="Scales every text size across the app. Composes with density."
		onReset={resetTypeScale}
	>
		<div class="oo-opt-row oo-opt-row-4" role="radiogroup" aria-label="Text size">
			{#each TYPE_SCALES as t (t)}
				<button
					type="button"
					class="oo-opt oo-opt-center"
					class:oo-opt-active={$typeScale === t}
					role="radio"
					aria-checked={$typeScale === t}
					on:click={() => chooseTypeScale(t)}
				>
					<span class="oo-opt-name">{TYPE_SCALE_LABELS[t]}</span>
				</button>
			{/each}
		</div>
		<p class="oo-type-preview">
			The quick brown fox jumps over the lazy dog.
		</p>
	</SettingsGroup>

	<SettingsGroup
		id="appearance-motion"
		title="Motion"
		description="How much the interface animates."
		onReset={resetMotion}
	>
		<div class="oo-opt-row oo-opt-row-3" role="radiogroup" aria-label="Motion preference">
			{#each MOTION_PREFS as m (m)}
				<button
					type="button"
					class="oo-opt"
					class:oo-opt-active={$motionPref === m}
					role="radio"
					aria-checked={$motionPref === m}
					on:click={() => chooseMotion(m)}
				>
					<span class="oo-opt-name">{MOTION_LABELS[m]}</span>
					<span class="oo-opt-hint">{MOTION_HINT[m]}</span>
				</button>
			{/each}
		</div>
	</SettingsGroup>

	<SettingsGroup
		id="appearance-advanced"
		title="Advanced"
		description="Keyboard shortcuts."
	>
		<div class="oo-adv-actions">
			<Button variant="secondary" iconLeft="keyboard" on:click={() => (showShortcuts = true)}>
				Keyboard shortcuts
			</Button>
		</div>
	</SettingsGroup>
</div>

<Modal
	open={showShortcuts}
	variant="drawer-right"
	size="lg"
	title="Keyboard shortcuts"
	onClose={() => (showShortcuts = false)}
>
	{#if showShortcuts}
		<ShortcutSettings />
	{/if}
</Modal>

<style>
	.oo-appearance {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
	}

	.oo-swatch-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
		gap: var(--oo-space-3);
	}

	.oo-swatch {
		display: flex;
		align-items: center;
		gap: var(--oo-space-3);
		padding: var(--oo-space-2);
		border-radius: var(--oo-radius-md);
		border: 1.5px solid var(--oo-bd-subtle);
		background-color: var(--oo-bg-elevated);
		cursor: pointer;
		text-align: left;
		transition:
			border-color 0.12s ease,
			background-color 0.12s ease;
	}

	.oo-swatch:hover {
		border-color: var(--oo-bd-strong);
	}

	.oo-swatch-active {
		border-color: var(--oo-acc-ink);
		background-color: var(--oo-accent-bg);
	}

	.oo-swatch-preview {
		width: 44px;
		height: 32px;
		border-radius: var(--oo-radius-sm);
		border: 1px solid var(--oo-bd-default);
		flex-shrink: 0;
		overflow: hidden;
		display: flex;
	}

	.oo-swatch-half {
		position: relative;
		flex: 1;
		background-color: var(--oo-bg-base);
	}

	.oo-swatch-surface {
		position: absolute;
		left: 4px;
		top: 6px;
		right: 4px;
		height: 12px;
		border-radius: 2px;
		background-color: var(--oo-bg-surface);
	}

	.oo-swatch-fg {
		position: absolute;
		left: 4px;
		bottom: 6px;
		width: 10px;
		height: 4px;
		border-radius: 2px;
		background-color: var(--oo-fg-primary);
	}

	.oo-swatch-meta {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}

	.oo-swatch-name {
		font-size: var(--oo-text-sm);
		font-weight: 500;
		color: var(--oo-fg-primary);
	}

	.oo-swatch-mode {
		font-size: var(--oo-text-2xs);
		color: var(--oo-fg-muted);
		text-transform: uppercase;
		letter-spacing: var(--oo-tracking-wide);
	}

	.oo-opt-row {
		display: grid;
		gap: var(--oo-space-2);
	}

	.oo-opt-row-3 {
		grid-template-columns: repeat(3, 1fr);
	}

	.oo-opt-row-4 {
		grid-template-columns: repeat(4, 1fr);
	}

	.oo-opt {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: var(--oo-space-3);
		border-radius: var(--oo-radius-md);
		border: 1.5px solid var(--oo-bd-subtle);
		background-color: var(--oo-bg-elevated);
		cursor: pointer;
		text-align: left;
		transition:
			border-color 0.12s ease,
			background-color 0.12s ease;
	}

	.oo-opt-center {
		align-items: center;
		text-align: center;
	}

	.oo-opt:hover {
		border-color: var(--oo-bd-strong);
	}

	.oo-opt-active {
		border-color: var(--oo-acc-ink);
		background-color: var(--oo-accent-bg);
	}

	.oo-opt-name {
		font-size: var(--oo-text-sm);
		font-weight: 500;
		color: var(--oo-fg-primary);
	}

	.oo-opt-hint {
		font-size: var(--oo-text-2xs);
		color: var(--oo-fg-muted);
		line-height: var(--oo-leading-snug);
	}

	.oo-type-preview {
		margin: var(--oo-space-2) 0 0;
		padding: var(--oo-space-3);
		border-radius: var(--oo-radius-md);
		background-color: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
		font-size: var(--oo-text-base);
		color: var(--oo-fg-secondary);
	}

	.oo-adv-actions {
		display: flex;
		flex-wrap: wrap;
		gap: var(--oo-space-2);
	}

	@media (max-width: 640px) {
		.oo-opt-row-3,
		.oo-opt-row-4 {
			grid-template-columns: 1fr 1fr;
		}
	}
</style>
