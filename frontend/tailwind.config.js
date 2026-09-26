/**
 * Tailwind resolves to the design tokens: no colour is written here.
 *
 * The older `surface-*` and `accent-*` utilities keep their names but mean
 * a role per kind of utility, so the same class reads right in every
 * palette: a background is a ground or the accent fill, a text colour is
 * one of the two text levels or the accent ink, a border is a quiet or a
 * required boundary. An opacity modifier (`bg-surface-800/50`) mixes the
 * token with transparent.
 *
 * The base layer Tailwind generates reads its colours from the theme too:
 * the default border, the ring and its offset, and a field's placeholder,
 * which the base layer takes from the gray 400 step. Each is a token here,
 * so the base layer draws no colour of its own.
 */

/** A token, with Tailwind's opacity modifier applied by mixing. */
function token(name) {
	return `color-mix(in srgb, var(${name}) calc(<alpha-value> * 100%), transparent)`;
}

/** The same token for every step. */
function steps(name, keys) {
	return Object.fromEntries(keys.map((key) => [key, token(name)]));
}

const STEPS = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900', '950'];

const grounds = {
	surface: {
		...steps('--oo-bg-overlay', ['50', '100', '200', '300', '400', '500']),
		600: token('--oo-bg-subtle'),
		700: token('--oo-bg-overlay'),
		800: token('--oo-bg-elevated'),
		900: token('--oo-bg-base'),
		950: token('--oo-sidebar-bg')
	},
	accent: {
		...steps('--oo-acc-fill', STEPS),
		// The lighter step is the fill's hover, as in `bg-accent-600 hover:bg-accent-500`.
		500: token('--oo-acc-fill-hover'),
		50: token('--oo-bg-tint-1'),
		100: token('--oo-bg-tint-1'),
		200: token('--oo-bg-tint-1'),
		900: token('--oo-bg-tint-1')
	}
};

const inks = {
	surface: {
		...steps('--oo-fg-primary', ['50', '100', '200']),
		300: token('--oo-fg-secondary'),
		400: token('--oo-fg-tertiary'),
		500: token('--oo-fg-muted'),
		600: token('--oo-fg-faint'),
		...steps('--oo-fg-faint', ['700', '800', '900', '950'])
	},
	accent: steps('--oo-acc-ink', STEPS)
};

const boundaries = {
	surface: {
		...steps('--oo-bd-strong', ['50', '100', '200', '300', '400', '500']),
		...steps('--oo-bd-default', ['600', '700']),
		...steps('--oo-bd-subtle', ['800', '900', '950'])
	},
	accent: steps('--oo-focus-ink', STEPS)
};

/** The ring's default, with the ring opacity Tailwind applies to it. */
function ring({ opacityValue }) {
	return `color-mix(in srgb, var(--oo-focus-ink) calc(${opacityValue ?? 1} * 100%), transparent)`;
}

/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	darkMode: 'class',
	theme: {
		extend: {
			backgroundColor: grounds,
			gradientColorStops: grounds,
			textColor: inks,
			placeholderColor: inks,
			colors: { gray: { 400: 'var(--oo-fg-muted)' } },
			borderColor: { ...boundaries, DEFAULT: 'var(--oo-bd-default)' },
			divideColor: boundaries,
			outlineColor: boundaries,
			ringColor: { ...boundaries, DEFAULT: ring },
			ringOffsetColor: { ...grounds, DEFAULT: 'var(--oo-bg-surface)' },
			fontFamily: {
				sans: ['var(--oo-font-sans)'],
				mono: ['var(--oo-font-mono)'],
				serif: ['var(--oo-font-serif)']
			},
			borderRadius: {
				none: 'var(--oo-radius-none)',
				sm: 'var(--oo-radius-sm)',
				DEFAULT: 'var(--oo-radius-sm)',
				md: 'var(--oo-radius-md)',
				lg: 'var(--oo-radius-lg)',
				xl: 'var(--oo-radius-xl)',
				'2xl': 'var(--oo-radius-2xl)',
				'3xl': 'var(--oo-radius-3xl)',
				full: 'var(--oo-radius-full)'
			}
		}
	},
	plugins: []
};
