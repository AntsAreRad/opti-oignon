<!--
  PixelStrip.svelte (components/pixel) -- draws a strip of pixel frames and
  plays it. The frames stand side by side in one group, one <path> per
  colour across the whole strip, and the SVG is a window one frame wide. A
  motion moves the group a whole frame at a time with CSS steps(), by one
  class per motion whose steps and durations are the frame tables' own: no
  script timer, no requestAnimationFrame, no SMIL.

  Frame 0 is the still. Under reduced motion, and wherever the motion
  preference cannot be read, only frame 0 is drawn; a preference changed
  later stops the motion by CSS. While the tab is hidden, a class set on
  visibilitychange pauses it. A motion that stopped resumes without its
  lead: the root is the entry of a running step, and the flowering is
  played once.

  Integer scales only: an art pixel is round(scale x dpr) / dpr CSS pixels,
  so every edge falls on a device pixel. Every colour is an app token the
  frame tables name. Decorative: the words beside it carry the state.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { scrollBehavior } from '$lib/motion';
	import { artPixel } from '$lib/pixel/plantFrames';
	import type { Grid, Motion, Strip } from '$lib/pixel/plantFrames';

	/** The frames and how they move. */
	export let strip: Strip;
	/** The app token each character is drawn in. */
	export let inks: Readonly<Record<string, string>>;
	/** Art pixels per CSS pixel before the device ratio: a whole number. */
	export let scale = 2;
	/** The device pixel ratio an art pixel is rounded to. */
	export let dpr = 1;
	/** Whether the strip plays; false holds frame 0. */
	export let moving = true;
	/** Reduced motion, when the caller knows it; null reads the preference once mounted. */
	export let reduced: boolean | null = null;

	interface Ink {
		token: string;
		d: string;
		/** A ground drawn behind the plant: the page's own colour in forced colours. */
		ground: boolean;
		/** A quiet line (the soil). */
		quiet: boolean;
	}

	// Until the page can be read (on the server, before mount), the still.
	let preferStill = true;
	let hidden = false;

	onMount(() => {
		// The motion preference is decided once, in lib/motion.ts: its scroll
		// behaviour is 'auto' exactly when motion is reduced or cannot be read.
		preferStill = scrollBehavior() === 'auto';
		const onVisibility = () => {
			hidden = document.hidden;
		};
		onVisibility();
		document.addEventListener('visibilitychange', onVisibility);
		return () => document.removeEventListener('visibilitychange', onVisibility);
	});

	/** One path per token: every run of one character, frame after frame. */
	function inksOf(grids: readonly Grid[], table: Readonly<Record<string, string>>): Ink[] {
		const runs = new Map<string, string[]>();
		grids.forEach((grid, index) => {
			grid.forEach((line, y) => {
				let x = 0;
				while (x < line.length) {
					let end = x + 1;
					while (end < line.length && line[end] === line[x]) end += 1;
					const token = line[x] === '.' ? undefined : table[line[x]];
					if (token) {
						const width = end - x;
						const list = runs.get(token) ?? [];
						list.push(`M${index * line.length + x} ${y}h${width}v1h-${width}z`);
						runs.set(token, list);
					}
					x = end;
				}
			});
		});
		return [...runs].map(([token, parts]) => ({
			token,
			d: parts.join(''),
			ground: token.startsWith('--oo-bg-') || token === '--oo-mark-ground',
			quiet: token === '--oo-bd-subtle',
		}));
	}

	let current: Strip | null = null;
	let wasPlaying = false;
	let resumed = false;

	/** A new strip enters with its lead; the same strip, stopped and started again, without. */
	function follow(next: Strip, playing: boolean): void {
		if (next !== current) {
			current = next;
			resumed = false;
		} else if (wasPlaying && !playing) {
			resumed = true;
		}
		wasPlaying = playing;
	}

	function resumedAs(motion: Motion): Motion | null {
		if (motion === 'root') return 'sway';
		if (motion === 'bloom') return null;
		return motion;
	}

	$: still = reduced ?? preferStill;
	$: frames = still ? strip.frames.slice(0, 1) : strip.frames;
	$: paths = inksOf(frames, inks);
	$: width = strip.frames[0][0].length;
	$: height = strip.frames[0].length;
	$: px = artPixel(scale, dpr);
	$: playing = !still && moving && strip.motion !== 'still';
	$: follow(strip, playing);
	$: motion = playing ? (resumed ? resumedAs(strip.motion) : strip.motion) : null;
</script>

<svg
	class="oo-pixel-strip"
	width={width * px}
	height={height * px}
	viewBox="0 0 {width} {height}"
	shape-rendering="crispEdges"
	aria-hidden="true"
	focusable="false"
>
	<g
		class="oo-pixel-frames"
		class:oo-pixel-turn={motion === 'turn'}
		class:oo-pixel-sway={motion === 'sway'}
		class:oo-pixel-root={motion === 'root'}
		class:oo-pixel-bloom={motion === 'bloom'}
		class:oo-pixel-paused={hidden}
	>
		{#each paths as ink (ink.token)}
			<path d={ink.d} fill="var({ink.token})" class:oo-pixel-ground={ink.ground} class:oo-pixel-quiet={ink.quiet} />
		{/each}
	</g>
</svg>

<style>
	.oo-pixel-strip {
		display: block;
		flex-shrink: 0;
		overflow: hidden;
		image-rendering: pixelated;
		shape-rendering: crispEdges;
	}

	/* One class per motion. Its steps and durations are the frame tables'
	   own (a contract reads them here and holds them equal): the turn, ten
	   frames of 360 ms; the sway, two of 800 ms; the root's frame for 120 ms,
	   then the sway; the flowering's two frames of 120 ms, then frame 0.
	   Every frame is 11 art pixels wide, so the group moves 11 user units a
	   step. The fill mode stays none: a played lead returns to frame 0. */
	.oo-pixel-turn {
		animation: oo-pixel-turn 3600ms steps(10) infinite;
	}

	.oo-pixel-sway {
		animation: oo-pixel-sway 1600ms steps(2) infinite;
	}

	.oo-pixel-root {
		animation:
			oo-pixel-root 120ms steps(1),
			oo-pixel-sway 1600ms steps(2) 120ms infinite;
	}

	.oo-pixel-bloom {
		animation: oo-pixel-bloom 240ms steps(2);
	}

	.oo-pixel-frames.oo-pixel-paused {
		animation-play-state: paused;
	}

	@keyframes oo-pixel-turn {
		from {
			transform: translateX(0);
		}
		to {
			transform: translateX(-110px);
		}
	}

	@keyframes oo-pixel-sway {
		from {
			transform: translateX(0);
		}
		to {
			transform: translateX(-22px);
		}
	}

	@keyframes oo-pixel-root {
		from {
			transform: translateX(-22px);
		}
		to {
			transform: translateX(-33px);
		}
	}

	@keyframes oo-pixel-bloom {
		from {
			transform: translateX(-11px);
		}
		to {
			transform: translateX(-33px);
		}
	}

	/* Reduced motion, chosen or asked by the system, holds frame 0. */
	:global(html.oo-reduce-motion) .oo-pixel-frames {
		animation: none;
	}

	@media (prefers-reduced-motion: reduce) {
		:global(html:not(.oo-motion-full)) .oo-pixel-frames {
			animation: none;
		}
	}

	/* Forced colours: the drawing in the text colour, its grounds in the
	   page's, its quiet lines grey; the shapes say the rest. */
	@media (forced-colors: active) {
		.oo-pixel-strip {
			forced-color-adjust: none;
		}

		path {
			fill: CanvasText;
		}

		path.oo-pixel-quiet {
			fill: GrayText;
		}

		path.oo-pixel-ground {
			fill: Canvas;
		}
	}
</style>
