<!--
  ThemeCustomizer.svelte (S152)
  Full theme customization panel.

  Features:
  - 3 color input methods: hex field, hue wheel (canvas), hue slider
  - Independent modifier sliders: saturation, lightness, warmth
  - Separate controls for primary and secondary accent
  - Preset grid: built-in (non-deletable) + custom user presets
  - Save current as custom preset, delete custom presets
  - Export all custom presets as JSON file download
  - Import custom presets from JSON file
  - Live preview via CSS variable injection
  - WCAG contrast indicator

  All styles use --oo-* CSS variables exclusively.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { darkMode } from '$lib/stores/ui';
	import {
		getThemeConfig,
		saveThemeConfig,
		getThemePresets,
		createCustomPreset,
		deleteCustomPreset,
		exportCustomPresets,
		importCustomPresets,
		type ThemeConfig,
		type ThemePreset,
	} from '$lib/api/theme';
	import {
		SAMPLE_ACCENT_HEX,
		SAMPLE_SECONDARY_HEX,
		FALLBACK_BG_SURFACE,
	} from './themeCustomizerConstants';

	// -- State: color parameters --
	let accentHue = 35;
	let accentSaturation = 70;
	let accentLightnessOffset = 0;
	let accentWarmth = 0;
	let secondaryHue = 130;
	let secondarySaturation = 30;
	let secondaryLightnessOffset = 0;
	let secondaryWarmth = 0;
	let presetId: string | null = 'default';

	// -- State: hex input --
	let accentHexInput = '';
	let secondaryHexInput = '';

	// -- State: UI --
	let loading = true;
	let saving = false;
	let error = '';
	let successMsg = '';
	let presets: ThemePreset[] = [];
	let variables: Record<string, string> = {};
	let contrastRatio = 0;
	let contrastPasses = false;

	// -- State: save preset dialog --
	let showSaveDialog = false;
	let newPresetName = '';
	let newPresetDesc = '';

	// -- State: which accent section is expanded --
	let activeSection: 'primary' | 'secondary' = 'primary';

	// -- Canvas refs --
	let primaryWheelCanvas: HTMLCanvasElement;
	let secondaryWheelCanvas: HTMLCanvasElement;

	// -- Lifecycle --

	onMount(async () => {
		await loadPresets();
		await loadTheme();
		drawWheels();
	});

	async function loadPresets() {
		try {
			presets = await getThemePresets();
		} catch {
			presets = [];
		}
	}

	async function loadTheme() {
		loading = true;
		error = '';
		try {
			const config = await getThemeConfig();
			applyConfigToState(config);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load theme';
		} finally {
			loading = false;
		}
	}

	function applyConfigToState(config: ThemeConfig) {
		accentHue = config.accent_hue;
		accentSaturation = config.accent_saturation;
		accentLightnessOffset = config.accent_lightness_offset;
		accentWarmth = config.accent_warmth;
		secondaryHue = config.secondary_hue;
		secondarySaturation = config.secondary_saturation;
		secondaryLightnessOffset = config.secondary_lightness_offset;
		secondaryWarmth = config.secondary_warmth;
		presetId = config.preset_id;
		variables = config.variables;
		accentHexInput = hueToHex(accentHue, accentSaturation);
		secondaryHexInput = hueToHex(secondaryHue, secondarySaturation);
		applyVariables(config.variables);
		updateContrast();
	}

	// -- CSS variable injection --

	function applyVariables(vars: Record<string, string>) {
		const root = document.documentElement;
		for (const [key, value] of Object.entries(vars)) {
			root.style.setProperty(`--${key}`, value);
		}
	}

	/**
	 * Live --oo-bg-surface of the CURRENT palette (S197, DS-02): the
	 * contrast badge and the wheel cutout follow whichever of the five
	 * palettes is active instead of assuming a hardcoded pair. Falls back
	 * to FALLBACK_BG_SURFACE when computed styles are unavailable or the
	 * declared value is not a #rrggbb literal (relativeLuminance needs one).
	 */
	function liveBgSurface(): string {
		if (typeof document === 'undefined') return FALLBACK_BG_SURFACE;
		const v = getComputedStyle(document.documentElement)
			.getPropertyValue('--oo-bg-surface')
			.trim();
		return /^#[0-9a-fA-F]{6}$/.test(v) ? v : FALLBACK_BG_SURFACE;
	}

	function updateContrast() {
		const accent = variables['oo-acc-500'] || SAMPLE_ACCENT_HEX;
		const bg = liveBgSurface();
		contrastRatio = computeContrastRatio(accent, bg);
		contrastPasses = contrastRatio >= 3.0;
	}

	function computeContrastRatio(hex1: string, hex2: string): number {
		const lum1 = relativeLuminance(hex1);
		const lum2 = relativeLuminance(hex2);
		const lighter = Math.max(lum1, lum2);
		const darker = Math.min(lum1, lum2);
		return (lighter + 0.05) / (darker + 0.05);
	}

	function relativeLuminance(hex: string): number {
		const r = parseInt(hex.slice(1, 3), 16) / 255;
		const g = parseInt(hex.slice(3, 5), 16) / 255;
		const b = parseInt(hex.slice(5, 7), 16) / 255;
		const lin = (c: number) =>
			c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
		return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
	}

	// -- Color conversion helpers --

	function hueToHex(h: number, s: number = 70, l: number = 50): string {
		const sn = s / 100;
		const ln = l / 100;
		const a = sn * Math.min(ln, 1 - ln);
		const f = (n: number) => {
			const k = (n + h / 30) % 12;
			const color = ln - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
			return Math.round(255 * color).toString(16).padStart(2, '0');
		};
		return `#${f(0)}${f(8)}${f(4)}`;
	}

	function hexToHsl(hex: string): { h: number; s: number; l: number } | null {
		const m = hex.match(/^#([0-9A-Fa-f]{6})$/);
		if (!m) {
			const m3 = hex.match(/^#([0-9A-Fa-f]{3})$/);
			if (!m3) return null;
			hex = `#${m3[1][0]}${m3[1][0]}${m3[1][1]}${m3[1][1]}${m3[1][2]}${m3[1][2]}`;
		}
		const r = parseInt(hex.slice(1, 3), 16) / 255;
		const g = parseInt(hex.slice(3, 5), 16) / 255;
		const b = parseInt(hex.slice(5, 7), 16) / 255;
		const max = Math.max(r, g, b), min = Math.min(r, g, b);
		const l = (max + min) / 2;
		if (max === min) return { h: 0, s: 0, l: Math.round(l * 100) };
		const d = max - min;
		const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
		let h = 0;
		if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
		else if (max === g) h = ((b - r) / d + 2) / 6;
		else h = ((r - g) / d + 4) / 6;
		return { h: Math.round(h * 360), s: Math.round(s * 100), l: Math.round(l * 100) };
	}

	$: accentPreview = hueToHex(accentHue, accentSaturation);
	$: secondaryPreview = hueToHex(secondaryHue, secondarySaturation);

	// -- Hex input handler --

	function handleHexInput(which: 'primary' | 'secondary') {
		const hex = which === 'primary' ? accentHexInput : secondaryHexInput;
		const hsl = hexToHsl(hex);
		if (!hsl) return;
		if (which === 'primary') {
			accentHue = hsl.h;
			accentSaturation = hsl.s;
		} else {
			secondaryHue = hsl.h;
			secondarySaturation = hsl.s;
		}
		presetId = null;
		debouncedLiveUpdate();
	}

	// -- Color wheel drawing --

	function drawWheel(canvas: HTMLCanvasElement | undefined) {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const size = canvas.width;
		const cx = size / 2;
		const cy = size / 2;
		const radius = size / 2 - 4;

		ctx.clearRect(0, 0, size, size);
		for (let angle = 0; angle < 360; angle++) {
			const startAngle = ((angle - 1) * Math.PI) / 180;
			const endAngle = ((angle + 1) * Math.PI) / 180;
			ctx.beginPath();
			ctx.moveTo(cx, cy);
			ctx.arc(cx, cy, radius, startAngle, endAngle);
			ctx.closePath();
			ctx.fillStyle = `hsl(${angle}, 70%, 50%)`;
			ctx.fill();
		}
		// Inner circle cutout for donut shape
		ctx.beginPath();
		ctx.arc(cx, cy, radius * 0.55, 0, Math.PI * 2);
		ctx.fillStyle = liveBgSurface();
		ctx.fill();
	}

	function drawWheels() {
		drawWheel(primaryWheelCanvas);
		drawWheel(secondaryWheelCanvas);
	}

	function handleWheelClick(e: MouseEvent, which: 'primary' | 'secondary') {
		const canvas = which === 'primary' ? primaryWheelCanvas : secondaryWheelCanvas;
		if (!canvas) return;
		const rect = canvas.getBoundingClientRect();
		const x = e.clientX - rect.left - canvas.width / 2;
		const y = e.clientY - rect.top - canvas.height / 2;
		const dist = Math.sqrt(x * x + y * y);
		const maxR = canvas.width / 2 - 4;
		if (dist < maxR * 0.55 || dist > maxR) return;
		let angle = Math.atan2(y, x) * (180 / Math.PI);
		if (angle < 0) angle += 360;
		const hue = Math.round(angle);
		if (which === 'primary') {
			accentHue = hue;
			accentHexInput = hueToHex(hue, accentSaturation);
		} else {
			secondaryHue = hue;
			secondaryHexInput = hueToHex(hue, secondarySaturation);
		}
		presetId = null;
		debouncedLiveUpdate();
	}

	// -- Live update with debounce --

	let debounceTimer: ReturnType<typeof setTimeout> | null = null;
	function debouncedLiveUpdate() {
		if (debounceTimer) clearTimeout(debounceTimer);
		debounceTimer = setTimeout(() => liveUpdate(), 250);
	}

	onDestroy(() => {
		if (debounceTimer) clearTimeout(debounceTimer);
	});

	async function liveUpdate() {
		try {
			const config = await saveThemeConfig({
				accent_hue: accentHue,
				accent_saturation: accentSaturation,
				secondary_hue: secondaryHue,
				secondary_saturation: secondarySaturation,
				accent_lightness_offset: accentLightnessOffset,
				secondary_lightness_offset: secondaryLightnessOffset,
				accent_warmth: accentWarmth,
				secondary_warmth: secondaryWarmth,
				mode: $darkMode ? 'dark' : 'light',
				preset_id: presetId,
			});
			variables = config.variables;
			applyVariables(config.variables);
			updateContrast();
		} catch {
			// Non-blocking for live preview
		}
	}

	function onSliderChange() {
		presetId = null;
		accentHexInput = hueToHex(accentHue, accentSaturation);
		secondaryHexInput = hueToHex(secondaryHue, secondarySaturation);
		debouncedLiveUpdate();
	}

	// -- Preset actions --

	async function selectPreset(preset: ThemePreset) {
		saving = true;
		error = '';
		successMsg = '';
		try {
			const config = await saveThemeConfig({
				accent_hue: preset.accent_hue,
				accent_saturation: preset.accent_saturation,
				secondary_hue: preset.secondary_hue,
				secondary_saturation: preset.secondary_saturation,
				accent_lightness_offset: preset.accent_lightness_offset,
				secondary_lightness_offset: preset.secondary_lightness_offset,
				accent_warmth: preset.accent_warmth,
				secondary_warmth: preset.secondary_warmth,
				mode: $darkMode ? 'dark' : 'light',
				preset_id: preset.id,
			});
			applyConfigToState(config);
			presetId = preset.id;
			successMsg = `Applied "${preset.name}"`;
			setTimeout(() => (successMsg = ''), 2000);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to apply preset';
		} finally {
			saving = false;
		}
	}

	async function resetToDefault() {
		const def = presets.find((p) => p.id === 'default');
		if (def) await selectPreset(def);
	}

	// -- Custom preset CRUD --

	async function saveAsPreset() {
		if (!newPresetName.trim()) return;
		saving = true;
		error = '';
		try {
			const created = await createCustomPreset({
				name: newPresetName.trim(),
				description: newPresetDesc.trim(),
				accent_hue: accentHue,
				accent_saturation: accentSaturation,
				secondary_hue: secondaryHue,
				secondary_saturation: secondarySaturation,
				accent_lightness_offset: accentLightnessOffset,
				secondary_lightness_offset: secondaryLightnessOffset,
				accent_warmth: accentWarmth,
				secondary_warmth: secondaryWarmth,
			});
			presets = [...presets, created];
			presetId = created.id;
			showSaveDialog = false;
			newPresetName = '';
			newPresetDesc = '';
			successMsg = `Saved "${created.name}"`;
			setTimeout(() => (successMsg = ''), 2000);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save preset';
		} finally {
			saving = false;
		}
	}

	async function handleDeletePreset(id: string) {
		saving = true;
		error = '';
		try {
			await deleteCustomPreset(id);
			presets = presets.filter((p) => p.id !== id);
			if (presetId === id) presetId = null;
			successMsg = 'Preset deleted';
			setTimeout(() => (successMsg = ''), 2000);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete preset';
		} finally {
			saving = false;
		}
	}

	// -- Import/Export --

	async function handleExport() {
		try {
			const jsonStr = await exportCustomPresets();
			const blob = new Blob([jsonStr], { type: 'application/json' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = 'opti-oignon-theme-presets.json';
			a.click();
			URL.revokeObjectURL(url);
			successMsg = 'Presets exported';
			setTimeout(() => (successMsg = ''), 2000);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Export failed';
		}
	}

	let fileInput: HTMLInputElement;

	function triggerImport() {
		fileInput?.click();
	}

	async function handleImportFile(e: Event) {
		const target = e.target as HTMLInputElement;
		const file = target.files?.[0];
		if (!file) return;
		try {
			const text = await file.text();
			const data = JSON.parse(text);
			if (!Array.isArray(data)) {
				error = 'Import file must contain a JSON array';
				return;
			}
			const all = await importCustomPresets(data);
			presets = all;
			successMsg = `Imported ${data.length} preset(s)`;
			setTimeout(() => (successMsg = ''), 2000);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Import failed';
		} finally {
			target.value = '';
		}
	}
</script>

<div class="theme-customizer">
	<h3 class="tc-title">Theme Customizer</h3>

	{#if loading}
		<p class="tc-loading">Loading theme...</p>
	{:else}
		<!-- Preset grid -->
		<div class="tc-section">
			<div class="tc-section-header">
				<label class="tc-label">Presets</label>
				<div class="tc-preset-actions">
					<button class="tc-btn-sm" on:click={() => (showSaveDialog = !showSaveDialog)} title="Save current as preset">Save as...</button>
					<button class="tc-btn-sm" on:click={handleExport} title="Export custom presets">Export</button>
					<button class="tc-btn-sm" on:click={triggerImport} title="Import presets from JSON">Import</button>
					<input bind:this={fileInput} type="file" accept=".json" on:change={handleImportFile} class="tc-hidden" />
				</div>
			</div>
			<div class="tc-presets-grid">
				{#each presets as preset}
					<div class="tc-preset-item" class:tc-preset-active={presetId === preset.id}>
						<button
							class="tc-preset-swatch"
							style="--swatch-color: {hueToHex(preset.accent_hue, preset.accent_saturation)};
							       --swatch-secondary: {hueToHex(preset.secondary_hue, preset.secondary_saturation)}"
							on:click={() => selectPreset(preset)}
							title={preset.description || preset.name}
							disabled={saving}
						>
							<span class="tc-swatch-dot"></span>
							<span class="tc-swatch-dot tc-swatch-dot-secondary"></span>
						</button>
						<span class="tc-swatch-name">{preset.name}</span>
						{#if !preset.builtin}
							<button
								class="tc-delete-btn"
								on:click|stopPropagation={() => handleDeletePreset(preset.id)}
								title="Delete this preset"
							>x</button>
						{/if}
					</div>
				{/each}
			</div>

			{#if showSaveDialog}
				<div class="tc-save-dialog">
					<input
						bind:value={newPresetName}
						placeholder="Preset name"
						maxlength="50"
						class="tc-input"
					/>
					<input
						bind:value={newPresetDesc}
						placeholder="Description (optional)"
						maxlength="200"
						class="tc-input"
					/>
					<div class="tc-save-dialog-actions">
						<button class="tc-btn tc-btn-primary" on:click={saveAsPreset} disabled={!newPresetName.trim() || saving}>Save</button>
						<button class="tc-btn tc-btn-secondary" on:click={() => (showSaveDialog = false)}>Cancel</button>
					</div>
				</div>
			{/if}
		</div>

		<!-- Tab toggle: primary / secondary -->
		<div class="tc-tabs">
			<button class="tc-tab" class:tc-tab-active={activeSection === 'primary'} on:click={() => (activeSection = 'primary')}>
				<span class="tc-tab-dot" style="background: {accentPreview}"></span>
				Primary
			</button>
			<button class="tc-tab" class:tc-tab-active={activeSection === 'secondary'} on:click={() => (activeSection = 'secondary')}>
				<span class="tc-tab-dot" style="background: {secondaryPreview}"></span>
				Secondary
			</button>
		</div>

		<!-- Primary accent controls -->
		{#if activeSection === 'primary'}
			<div class="tc-section">
				<!-- Hex input -->
				<label class="tc-label" for="accent-hex">Hex color</label>
				<div class="tc-hex-row">
					<input
						id="accent-hex"
						bind:value={accentHexInput}
						on:change={() => handleHexInput('primary')}
						placeholder={SAMPLE_ACCENT_HEX}
						maxlength="7"
						class="tc-input tc-hex-input"
					/>
					<span class="tc-hue-badge" style="background: {accentPreview}"></span>
				</div>

				<!-- Color wheel -->
				<div class="tc-wheel-container">
					<canvas
						bind:this={primaryWheelCanvas}
						width="160"
						height="160"
						class="tc-wheel"
						on:click={(e) => handleWheelClick(e, 'primary')}
					></canvas>
					<div class="tc-wheel-indicator" style="--indicator-angle: {accentHue}deg; --indicator-color: {accentPreview}"></div>
				</div>

				<!-- Hue slider -->
				<label class="tc-label" for="accent-hue-slider">
					Hue <span class="tc-val">{accentHue}</span>
				</label>
				<input id="accent-hue-slider" type="range" min="0" max="359" step="1" bind:value={accentHue} on:input={onSliderChange} class="tc-slider tc-slider-hue" />

				<!-- Saturation -->
				<label class="tc-label" for="accent-sat">
					Saturation <span class="tc-val">{accentSaturation}%</span>
				</label>
				<input id="accent-sat" type="range" min="0" max="100" step="1" bind:value={accentSaturation} on:input={onSliderChange} class="tc-slider" />

				<!-- Lightness offset -->
				<label class="tc-label" for="accent-light">
					Lightness <span class="tc-val">{accentLightnessOffset > 0 ? '+' : ''}{accentLightnessOffset}</span>
				</label>
				<input id="accent-light" type="range" min="-50" max="50" step="1" bind:value={accentLightnessOffset} on:input={onSliderChange} class="tc-slider" />

				<!-- Warmth -->
				<label class="tc-label" for="accent-warmth">
					Warmth <span class="tc-val">{accentWarmth > 0 ? '+' : ''}{accentWarmth}</span>
				</label>
				<input id="accent-warmth" type="range" min="-30" max="30" step="1" bind:value={accentWarmth} on:input={onSliderChange} class="tc-slider" />
			</div>
		{/if}

		<!-- Secondary accent controls -->
		{#if activeSection === 'secondary'}
			<div class="tc-section">
				<label class="tc-label" for="secondary-hex">Hex color</label>
				<div class="tc-hex-row">
					<input
						id="secondary-hex"
						bind:value={secondaryHexInput}
						on:change={() => handleHexInput('secondary')}
						placeholder={SAMPLE_SECONDARY_HEX}
						maxlength="7"
						class="tc-input tc-hex-input"
					/>
					<span class="tc-hue-badge" style="background: {secondaryPreview}"></span>
				</div>

				<div class="tc-wheel-container">
					<canvas
						bind:this={secondaryWheelCanvas}
						width="160"
						height="160"
						class="tc-wheel"
						on:click={(e) => handleWheelClick(e, 'secondary')}
					></canvas>
					<div class="tc-wheel-indicator" style="--indicator-angle: {secondaryHue}deg; --indicator-color: {secondaryPreview}"></div>
				</div>

				<label class="tc-label" for="secondary-hue-slider">
					Hue <span class="tc-val">{secondaryHue}</span>
				</label>
				<input id="secondary-hue-slider" type="range" min="0" max="359" step="1" bind:value={secondaryHue} on:input={onSliderChange} class="tc-slider tc-slider-hue" />

				<label class="tc-label" for="secondary-sat">
					Saturation <span class="tc-val">{secondarySaturation}%</span>
				</label>
				<input id="secondary-sat" type="range" min="0" max="100" step="1" bind:value={secondarySaturation} on:input={onSliderChange} class="tc-slider" />

				<label class="tc-label" for="secondary-light">
					Lightness <span class="tc-val">{secondaryLightnessOffset > 0 ? '+' : ''}{secondaryLightnessOffset}</span>
				</label>
				<input id="secondary-light" type="range" min="-50" max="50" step="1" bind:value={secondaryLightnessOffset} on:input={onSliderChange} class="tc-slider" />

				<label class="tc-label" for="secondary-warmth">
					Warmth <span class="tc-val">{secondaryWarmth > 0 ? '+' : ''}{secondaryWarmth}</span>
				</label>
				<input id="secondary-warmth" type="range" min="-30" max="30" step="1" bind:value={secondaryWarmth} on:input={onSliderChange} class="tc-slider" />
			</div>
		{/if}

		<!-- WCAG contrast indicator -->
		<div class="tc-section">
			<div class="tc-contrast" class:tc-contrast-pass={contrastPasses} class:tc-contrast-fail={!contrastPasses}>
				<span class="tc-contrast-label">Contrast:</span>
				<span class="tc-contrast-value">{contrastRatio.toFixed(1)}:1</span>
				<span class="tc-contrast-badge">{contrastPasses ? 'AA pass' : 'Low contrast'}</span>
			</div>
		</div>

		<!-- Actions -->
		<div class="tc-actions">
			<button class="tc-btn tc-btn-secondary" on:click={resetToDefault} disabled={saving}>Reset to default</button>
		</div>

		{#if error}
			<p class="tc-error">{error}</p>
		{/if}
		{#if successMsg}
			<p class="tc-success">{successMsg}</p>
		{/if}
	{/if}
</div>

<style>
	.theme-customizer {
		padding: 1rem;
		background: var(--oo-bg-surface);
		border-radius: var(--oo-radius-lg);
		border: 1px solid var(--oo-bd-default);
		max-width: 440px;
	}

	.tc-title {
		margin: 0 0 0.75rem;
		font-size: var(--oo-text-lg);
		color: var(--oo-fg-primary);
		font-weight: 600;
	}

	.tc-loading {
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
	}

	.tc-section {
		margin-bottom: 1rem;
	}

	.tc-section-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.4rem;
	}

	.tc-label {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
		margin-bottom: 0.3rem;
		font-weight: 500;
	}

	.tc-val {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
		font-family: var(--oo-font-mono);
	}

	.tc-hidden { display: none; }

	/* -- Tabs -- */

	.tc-tabs {
		display: flex;
		gap: 0.25rem;
		margin-bottom: 0.75rem;
		border-bottom: 1px solid var(--oo-bd-default);
		padding-bottom: 0.5rem;
	}

	.tc-tab {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		padding: 0.35rem 0.65rem;
		border: none;
		border-radius: var(--oo-radius-sm);
		background: transparent;
		color: var(--oo-fg-tertiary);
		font-size: var(--oo-text-sm);
		font-weight: 500;
		cursor: pointer;
		transition: background-color var(--oo-transition-fast);
	}

	.tc-tab:hover { background: var(--oo-bg-elevated); }
	.tc-tab-active {
		background: var(--oo-bg-elevated);
		color: var(--oo-fg-primary);
	}

	.tc-tab-dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		border: 1px solid var(--oo-bd-strong);
	}

	/* -- Hex input -- */

	.tc-hex-row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.75rem;
	}

	.tc-input {
		padding: 0.35rem 0.5rem;
		border-radius: var(--oo-radius-sm);
		border: 1px solid var(--oo-bd-default);
		background: var(--oo-bg-elevated);
		color: var(--oo-fg-primary);
		font-size: var(--oo-text-sm);
		font-family: inherit;
		width: 100%;
	}

	.tc-input:focus {
		outline: none;
		border-color: var(--oo-accent-primary);
		box-shadow: 0 0 0 2px var(--oo-input-focus);
	}

	.tc-hex-input {
		width: 100px;
		font-family: var(--oo-font-mono);
	}

	.tc-hue-badge {
		display: inline-block;
		width: 20px;
		height: 20px;
		border-radius: var(--oo-radius-sm);
		border: 1px solid var(--oo-bd-strong);
		flex-shrink: 0;
	}

	/* -- Color wheel -- */

	.tc-wheel-container {
		position: relative;
		width: 160px;
		height: 160px;
		margin: 0.5rem auto 0.75rem;
	}

	.tc-wheel {
		border-radius: 50%;
		cursor: crosshair;
		display: block;
	}

	.tc-wheel-indicator {
		position: absolute;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		border: 2px solid var(--oo-fg-primary);
		background: var(--indicator-color);
		top: 50%;
		left: 50%;
		transform-origin: center center;
		transform:
			translate(-50%, -50%)
			rotate(var(--indicator-angle))
			translateX(56px);
		pointer-events: none;
		box-shadow: var(--oo-shadow-sm);
	}

	/* -- Sliders -- */

	.tc-slider {
		width: 100%;
		height: 6px;
		appearance: none;
		-webkit-appearance: none;
		background: var(--oo-bg-elevated);
		border-radius: 3px;
		outline: none;
		cursor: pointer;
		margin-bottom: 0.6rem;
	}

	.tc-slider::-webkit-slider-thumb {
		appearance: none;
		-webkit-appearance: none;
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: var(--oo-accent-primary);
		border: 2px solid var(--oo-bg-base);
		cursor: pointer;
		box-shadow: var(--oo-shadow-sm);
	}

	.tc-slider::-moz-range-thumb {
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: var(--oo-accent-primary);
		border: 2px solid var(--oo-bg-base);
		cursor: pointer;
		box-shadow: var(--oo-shadow-sm);
	}

	.tc-slider-hue {
		background: linear-gradient(
			to right,
			hsl(0, 70%, 50%), hsl(60, 70%, 50%), hsl(120, 70%, 50%),
			hsl(180, 70%, 50%), hsl(240, 70%, 50%), hsl(300, 70%, 50%),
			hsl(360, 70%, 50%)
		);
	}

	/* -- Preset grid -- */

	.tc-preset-actions {
		display: flex;
		gap: 0.3rem;
	}

	.tc-btn-sm {
		padding: 0.2rem 0.45rem;
		border-radius: var(--oo-radius-sm);
		border: 1px solid var(--oo-bd-default);
		background: var(--oo-bg-elevated);
		color: var(--oo-fg-tertiary);
		font-size: var(--oo-text-xs);
		cursor: pointer;
		transition: background-color var(--oo-transition-fast);
	}

	.tc-btn-sm:hover {
		background: var(--oo-bg-overlay);
		color: var(--oo-fg-secondary);
	}

	.tc-presets-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(68px, 1fr));
		gap: 0.4rem;
	}

	.tc-preset-item {
		display: flex;
		flex-direction: column;
		align-items: center;
		position: relative;
		padding: 0.35rem 0.2rem;
		border: 2px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		background: var(--oo-bg-elevated);
		transition: border-color var(--oo-transition-fast);
	}

	.tc-preset-item:hover { border-color: var(--oo-bd-strong); }
	.tc-preset-active { border-color: var(--oo-accent-primary); }

	.tc-preset-swatch {
		display: flex;
		align-items: center;
		gap: 0.15rem;
		padding: 0.2rem;
		border: none;
		background: transparent;
		cursor: pointer;
	}

	.tc-swatch-dot {
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: var(--swatch-color);
	}

	.tc-swatch-dot-secondary {
		width: 13px;
		height: 13px;
		background: var(--swatch-secondary);
	}

	.tc-swatch-name {
		font-size: 0.6rem;
		color: var(--oo-fg-tertiary);
		text-align: center;
		line-height: 1.2;
		margin-top: 0.15rem;
	}

	.tc-delete-btn {
		position: absolute;
		top: 2px;
		right: 2px;
		width: 14px;
		height: 14px;
		border-radius: 50%;
		border: none;
		background: var(--oo-error-bg);
		color: var(--oo-error);
		font-size: 0.55rem;
		line-height: 1;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		opacity: 0;
		transition: opacity var(--oo-transition-fast);
	}

	.tc-preset-item:hover .tc-delete-btn { opacity: 1; }

	/* -- Save preset dialog -- */

	.tc-save-dialog {
		margin-top: 0.5rem;
		padding: 0.6rem;
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		background: var(--oo-bg-elevated);
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}

	.tc-save-dialog-actions {
		display: flex;
		gap: 0.35rem;
		margin-top: 0.2rem;
	}

	/* -- Contrast indicator -- */

	.tc-contrast {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.35rem 0.5rem;
		border-radius: var(--oo-radius-sm);
		font-size: var(--oo-text-sm);
	}

	.tc-contrast-pass {
		background: var(--oo-success-bg);
		border: 1px solid var(--oo-success-bd);
	}

	.tc-contrast-fail {
		background: var(--oo-error-bg);
		border: 1px solid var(--oo-error-bd);
	}

	.tc-contrast-label { color: var(--oo-fg-secondary); }

	.tc-contrast-value {
		font-family: var(--oo-font-mono);
		font-weight: 600;
		color: var(--oo-fg-primary);
	}

	.tc-contrast-badge { font-size: var(--oo-text-xs); font-weight: 600; }
	.tc-contrast-pass .tc-contrast-badge { color: var(--oo-success); }
	.tc-contrast-fail .tc-contrast-badge { color: var(--oo-error); }

	/* -- Action buttons -- */

	.tc-actions {
		display: flex;
		gap: 0.5rem;
	}

	.tc-btn {
		padding: 0.35rem 0.7rem;
		border-radius: var(--oo-radius-sm);
		font-size: var(--oo-text-sm);
		cursor: pointer;
		border: none;
		transition: background-color var(--oo-transition-fast);
	}

	.tc-btn:disabled { opacity: 0.5; cursor: not-allowed; }

	.tc-btn-primary {
		background: var(--oo-btn-primary-bg);
		color: var(--oo-btn-primary-fg);
	}

	.tc-btn-primary:hover:not(:disabled) { background: var(--oo-btn-primary-hover); }

	.tc-btn-secondary {
		background: var(--oo-btn-secondary-bg);
		color: var(--oo-btn-secondary-fg);
	}

	.tc-btn-secondary:hover:not(:disabled) { background: var(--oo-btn-secondary-hover); }

	/* -- Messages -- */

	.tc-error {
		color: var(--oo-error);
		font-size: var(--oo-text-sm);
		margin-top: 0.5rem;
	}

	.tc-success {
		color: var(--oo-success);
		font-size: var(--oo-text-sm);
		margin-top: 0.5rem;
	}
</style>
