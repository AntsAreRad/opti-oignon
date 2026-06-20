<!--
  NotesDrawingCanvas.svelte (S254, Notes feature N.7)
  SVG drawing canvas for the active note: a minimal stroke/shape model
  (pen / line / rect / ellipse over a named-colour palette, so the no-raw-hex
  discipline holds by construction) edited with pointer events on an 800x600
  viewBox, serialized by lib/drawing/svgDrawing.ts to a standalone SVG
  document and uploaded through the S253 client as an encrypted attachment of
  kind "drawing" (sealed server-side under a per-attachment subkey; nothing
  plaintext touches disk). Thumbnails decrypt in memory through short-lived
  object URLs, revoked on removal and destroy. Re-edit fetches the blob and
  parses our own data-oo-* format back into strokes; a foreign SVG is refused
  as not an editable drawing, never guessed at. Replacing an edited drawing
  is fail-safe: upload first; remove the old version only after success, so
  an interrupted save never loses the previous version. Design-system tokens
  only (--oo-*); lucide-svelte icons through Icon.
-->
<script lang="ts">
	import { onDestroy } from 'svelte';
	import { Button, Card, Icon, EmptyState, InlineError } from '$lib/ds';
	import {
		drawingAttachments,
		mediaLoading,
		mediaError,
		loadAttachments,
		uploadNoteAttachment,
		removeAttachment
	} from '$lib/stores/attachments';
	import { fetchAttachmentBlob, type AttachmentRecord } from '$lib/api/attachments';
	import {
		parseDrawing,
		drawingToBlob,
		DRAWING_COLORS,
		type DrawingStroke,
		type DrawingTool
	} from '$lib/drawing/svgDrawing';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	export let noteId: string;

	const CANVAS_W = 800;
	const CANVAS_H = 600;
	const STROKE_WIDTHS = [2, 3, 6];

	let editorOpen = false;
	let tool: DrawingTool = 'pen';
	let color: string = DRAWING_COLORS[0];
	let strokeWidth = 3;
	let strokes: DrawingStroke[] = [];
	let current: DrawingStroke | null = null;
	let editingId: string | null = null;
	let saving = false;
	let svgEl: SVGSVGElement;

	/** Thumbnail object URLs keyed by attachment id (in-memory only). */
	let thumbs: Record<string, string> = {};
	const pendingThumbs = new Set<string>();

	$: if (noteId) {
		loadAttachments(noteId);
	}

	$: renderStrokes = current ? [...strokes, current] : strokes;

	// Keep the thumbnail map in step with the manifest list: fetch missing
	// thumbs, and revoke + drop URLs for attachments no longer present.
	$: syncThumbs($drawingAttachments);

	function syncThumbs(list: AttachmentRecord[]): void {
		for (const att of list) {
			void ensureThumb(att.id);
		}
		const live = new Set(list.map((a) => a.id));
		for (const id of Object.keys(thumbs)) {
			if (!live.has(id)) {
				URL.revokeObjectURL(thumbs[id]);
				const next = { ...thumbs };
				delete next[id];
				thumbs = next;
			}
		}
	}

	async function ensureThumb(id: string): Promise<void> {
		if (thumbs[id] || pendingThumbs.has(id)) {
			return;
		}
		pendingThumbs.add(id);
		try {
			const blob = await fetchAttachmentBlob(id);
			thumbs = { ...thumbs, [id]: URL.createObjectURL(blob) };
		} catch {
			// The thumbnail is cosmetic; the row still renders without it.
		} finally {
			pendingThumbs.delete(id);
		}
	}

	onDestroy(() => {
		for (const id of Object.keys(thumbs)) {
			URL.revokeObjectURL(thumbs[id]);
		}
	});

	function openNew(): void {
		strokes = [];
		current = null;
		editingId = null;
		editorOpen = true;
	}

	function closeEditor(): void {
		editorOpen = false;
		current = null;
	}

	function undoStroke(): void {
		strokes = strokes.slice(0, -1);
	}

	function clearStrokes(): void {
		strokes = [];
		current = null;
	}

	function toCanvasPoint(ev: PointerEvent): [number, number] {
		const rect = svgEl.getBoundingClientRect();
		const x = ((ev.clientX - rect.left) / rect.width) * CANVAS_W;
		const y = ((ev.clientY - rect.top) / rect.height) * CANVAS_H;
		return [x, y];
	}

	function handlePointerDown(ev: PointerEvent): void {
		if (ev.button !== 0) {
			return;
		}
		svgEl.setPointerCapture(ev.pointerId);
		const p = toCanvasPoint(ev);
		current = {
			tool,
			color,
			width: strokeWidth,
			points: tool === 'pen' ? [p] : [p, p]
		};
	}

	function handlePointerMove(ev: PointerEvent): void {
		if (!current) {
			return;
		}
		const p = toCanvasPoint(ev);
		if (current.tool === 'pen') {
			const last = current.points[current.points.length - 1];
			const dx = p[0] - last[0];
			const dy = p[1] - last[1];
			if (dx * dx + dy * dy < 2) {
				return;
			}
			current = { ...current, points: [...current.points, p] };
		} else {
			current = { ...current, points: [current.points[0], p] };
		}
	}

	function handlePointerUp(ev: PointerEvent): void {
		if (!current) {
			return;
		}
		svgEl.releasePointerCapture(ev.pointerId);
		strokes = [...strokes, current];
		current = null;
	}

	async function saveDrawing(): Promise<void> {
		if (strokes.length === 0) {
			toastError('Nothing to save yet');
			return;
		}
		saving = true;
		try {
			const blob = drawingToBlob({
				width: CANVAS_W,
				height: CANVAS_H,
				strokes
			});
			const replacingId = editingId;
			const filename = 'drawing-' + Date.now() + '.svg';
			// Fail-safe replace: upload first; remove the old version only after success.
			await uploadNoteAttachment(noteId, 'drawing', blob, filename);
			if (replacingId) {
				await removeAttachment(replacingId);
				toastSuccess('Drawing replaced');
			} else {
				toastSuccess('Drawing saved');
			}
			closeEditor();
		} catch {
			toastError('Failed to save drawing');
		} finally {
			saving = false;
		}
	}

	async function editAttachment(record: AttachmentRecord): Promise<void> {
		try {
			const blob = await fetchAttachmentBlob(record.id);
			const text = await blob.text();
			const model = parseDrawing(text);
			if (model === null) {
				toastError('This attachment is not an editable drawing');
				return;
			}
			strokes = model.strokes;
			current = null;
			editingId = record.id;
			editorOpen = true;
		} catch {
			toastError('Failed to load drawing');
		}
	}

	async function deleteDrawing(id: string): Promise<void> {
		try {
			await removeAttachment(id);
			if (editingId === id) {
				closeEditor();
				editingId = null;
			}
			toastSuccess('Drawing deleted');
		} catch {
			toastError('Failed to delete drawing');
		}
	}

	function pointsAttr(s: DrawingStroke): string {
		return s.points.map((p) => p[0] + ',' + p[1]).join(' ');
	}

	function rectOf(s: DrawingStroke): { x: number; y: number; w: number; h: number } {
		const [a, b] = s.points;
		return {
			x: Math.min(a[0], b[0]),
			y: Math.min(a[1], b[1]),
			w: Math.abs(b[0] - a[0]),
			h: Math.abs(b[1] - a[1])
		};
	}

	function ellOf(s: DrawingStroke): { cx: number; cy: number; rx: number; ry: number } {
		const [a, b] = s.points;
		return {
			cx: (a[0] + b[0]) / 2,
			cy: (a[1] + b[1]) / 2,
			rx: Math.abs(b[0] - a[0]) / 2,
			ry: Math.abs(b[1] - a[1]) / 2
		};
	}

	const TOOLS: Array<{ id: DrawingTool; icon: string; label: string }> = [
		{ id: 'pen', icon: 'pen-tool', label: 'Pen' },
		{ id: 'line', icon: 'minus', label: 'Line' },
		{ id: 'rect', icon: 'square', label: 'Rectangle' },
		{ id: 'ellipse', icon: 'circle', label: 'Ellipse' }
	];
</script>

<Card>
	<div class="drawing-section">
		<div class="drawing-header">
			<span class="drawing-title">
				<Icon name="pen-tool" size="sm" />
				Drawings
			</span>
			{#if !editorOpen}
				<Button variant="secondary" size="sm" iconLeft="plus" on:click={openNew}>
					New drawing
				</Button>
			{/if}
		</div>

		{#if $mediaError}
			<InlineError message={$mediaError} />
		{/if}

		{#if editorOpen}
			<div class="drawing-editor">
				<div class="drawing-toolbar">
					<div class="drawing-tools" role="group" aria-label="Drawing tools">
						{#each TOOLS as t (t.id)}
							<button
								type="button"
								class="drawing-tool"
								class:active={tool === t.id}
								aria-label={t.label}
								title={t.label}
								on:click={() => (tool = t.id)}
							>
								<Icon name={t.icon} size="sm" />
							</button>
						{/each}
					</div>
					<div class="drawing-palette" role="group" aria-label="Colours">
						{#each DRAWING_COLORS as c (c)}
							<button
								type="button"
								class="drawing-swatch"
								class:active={color === c}
								style="background: {c}"
								aria-label={'Colour ' + c}
								title={c}
								on:click={() => (color = c)}
							></button>
						{/each}
					</div>
					<div class="drawing-widths" role="group" aria-label="Stroke width">
						{#each STROKE_WIDTHS as w (w)}
							<button
								type="button"
								class="drawing-width"
								class:active={strokeWidth === w}
								aria-label={'Stroke width ' + w}
								title={'Width ' + w}
								on:click={() => (strokeWidth = w)}
							>
								<span class="drawing-width-dot" style="width: {w * 2}px; height: {w * 2}px"
								></span>
							</button>
						{/each}
					</div>
				</div>

				<svg
					bind:this={svgEl}
					class="drawing-surface"
					viewBox="0 0 800 600"
					role="img"
					aria-label="Drawing editor"
					on:pointerdown={handlePointerDown}
					on:pointermove={handlePointerMove}
					on:pointerup={handlePointerUp}
				>
					{#each renderStrokes as s, i (i)}
						{#if s.tool === 'pen'}
							<polyline
								points={pointsAttr(s)}
								fill="none"
								stroke={s.color}
								stroke-width={s.width}
								stroke-linecap="round"
								stroke-linejoin="round"
							/>
						{:else if s.tool === 'line'}
							<line
								x1={s.points[0][0]}
								y1={s.points[0][1]}
								x2={s.points[1][0]}
								y2={s.points[1][1]}
								stroke={s.color}
								stroke-width={s.width}
								stroke-linecap="round"
							/>
						{:else if s.tool === 'rect'}
							<rect
								x={rectOf(s).x}
								y={rectOf(s).y}
								width={rectOf(s).w}
								height={rectOf(s).h}
								fill="none"
								stroke={s.color}
								stroke-width={s.width}
								stroke-linejoin="round"
							/>
						{:else}
							<ellipse
								cx={ellOf(s).cx}
								cy={ellOf(s).cy}
								rx={ellOf(s).rx}
								ry={ellOf(s).ry}
								fill="none"
								stroke={s.color}
								stroke-width={s.width}
							/>
						{/if}
					{/each}
				</svg>

				<div class="drawing-actions">
					<Button
						variant="ghost"
						size="sm"
						iconLeft="undo-2"
						disabled={strokes.length === 0}
						on:click={undoStroke}
					>
						Undo
					</Button>
					<Button
						variant="ghost"
						size="sm"
						iconLeft="eraser"
						disabled={strokes.length === 0}
						on:click={clearStrokes}
					>
						Clear
					</Button>
					<span class="drawing-spacer"></span>
					<Button variant="ghost" size="sm" on:click={closeEditor}>Cancel</Button>
					<Button
						variant="primary"
						size="sm"
						iconLeft="save"
						disabled={saving || strokes.length === 0}
						on:click={saveDrawing}
					>
						{saving ? 'Saving...' : editingId ? 'Save (replace)' : 'Save'}
					</Button>
				</div>
			</div>
		{/if}

		{#if $mediaLoading && $drawingAttachments.length === 0}
			<p class="drawing-hint">Loading drawings...</p>
		{:else if $drawingAttachments.length === 0}
			{#if !editorOpen}
				<EmptyState
					icon="pen-tool"
					title="No drawings yet"
					description="Sketch an idea right inside the note."
					size="sm"
				/>
			{/if}
		{:else}
			<div class="drawing-grid">
				{#each $drawingAttachments as att (att.id)}
					<div class="drawing-item">
						{#if thumbs[att.id]}
							<img class="drawing-thumb" src={thumbs[att.id]} alt="Drawing attachment" />
						{:else}
							<div class="drawing-thumb drawing-thumb-empty">
								<Icon name="image" size="md" />
							</div>
						{/if}
						<div class="drawing-item-actions">
							<Button
								variant="ghost"
								size="sm"
								iconLeft="pencil"
								on:click={() => editAttachment(att)}
							>
								Edit
							</Button>
							<Button
								variant="ghost"
								size="sm"
								iconLeft="trash-2"
								on:click={() => deleteDrawing(att.id)}
							>
								Delete
							</Button>
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>
</Card>

<style>
	.drawing-section {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.drawing-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
	}

	.drawing-title {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
		font-weight: 600;
		color: var(--oo-fg-primary, currentColor);
	}

	.drawing-hint {
		margin: 0;
		font-size: 0.85rem;
		color: var(--oo-fg-secondary, gray);
	}

	.drawing-editor {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.drawing-toolbar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.75rem;
	}

	.drawing-tools,
	.drawing-palette,
	.drawing-widths {
		display: inline-flex;
		align-items: center;
		gap: 0.25rem;
	}

	.drawing-tool,
	.drawing-width {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 1.9rem;
		height: 1.9rem;
		padding: 0;
		border: 1px solid var(--oo-border-subtle, lightgray);
		border-radius: 6px;
		background: var(--oo-bg-elevated, transparent);
		color: var(--oo-fg-secondary, gray);
		cursor: pointer;
	}

	.drawing-tool.active,
	.drawing-width.active {
		border-color: var(--oo-accent, steelblue);
		color: var(--oo-accent, steelblue);
	}

	.drawing-swatch {
		width: 1.4rem;
		height: 1.4rem;
		padding: 0;
		border: 2px solid var(--oo-border-subtle, lightgray);
		border-radius: 50%;
		cursor: pointer;
	}

	.drawing-swatch.active {
		border-color: var(--oo-accent, steelblue);
		outline: 2px solid var(--oo-accent, steelblue);
		outline-offset: 1px;
	}

	.drawing-width-dot {
		display: inline-block;
		border-radius: 50%;
		background: currentColor;
	}

	.drawing-surface {
		width: 100%;
		height: auto;
		touch-action: none;
		cursor: crosshair;
		border: 1px solid var(--oo-border-subtle, lightgray);
		border-radius: 8px;
		background: var(--oo-bg-base, white);
	}

	.drawing-actions {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}

	.drawing-spacer {
		flex: 1;
	}

	.drawing-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
		gap: 0.6rem;
	}

	.drawing-item {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		padding: 0.4rem;
		border: 1px solid var(--oo-border-subtle, lightgray);
		border-radius: 8px;
		background: var(--oo-bg-elevated, transparent);
	}

	.drawing-thumb {
		width: 100%;
		aspect-ratio: 4 / 3;
		object-fit: contain;
		border-radius: 6px;
		background: var(--oo-bg-base, white);
	}

	.drawing-thumb-empty {
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--oo-fg-secondary, gray);
	}

	.drawing-item-actions {
		display: flex;
		justify-content: space-between;
		gap: 0.25rem;
	}
</style>
