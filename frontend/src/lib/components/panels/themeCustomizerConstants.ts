/**
 * ThemeCustomizer constants (S197, DS-02) -- the only hex literals the
 * customizer needs, kept in a TS module so the .svelte source stays free
 * of raw hex (the S83 design-system invariant; same convention as
 * PALETTE_SWATCH in stores/preferences.ts).
 */

/** Sample primary accent for the hex-input placeholder and the accent
 * fallback when no variable is loaded (matches anthracite --oo-acc-500). */
export const SAMPLE_ACCENT_HEX = '#C48838';

/** Sample secondary accent for the secondary hex-input placeholder. */
export const SAMPLE_SECONDARY_HEX = '#8A9A8A';

/**
 * Fallback for the live --oo-bg-surface read, used only when computed
 * styles are unavailable or the declared value is not a #rrggbb literal
 * (anthracite surface value).
 */
export const FALLBACK_BG_SURFACE = '#27272A';
