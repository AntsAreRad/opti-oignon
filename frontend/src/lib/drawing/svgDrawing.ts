/**
 * svgDrawing.ts (S254, Notes feature N.7) -- the drawing model.
 *
 * A minimal stroke/shape model for the NotesDrawingCanvas component,
 * serialized to a STANDALONE SVG document (vector, editable, sync-friendly)
 * and parsed back for re-editing. Only our own format round-trips: the
 * serializer stamps the root with data-oo-drawing="1" and every element with
 * data-oo-tool, and the parser refuses (returns null) any SVG that does not
 * carry the marker -- a foreign SVG is never guessed at, it is simply not an
 * editable drawing. The saved document is a plain attachment of kind
 * "drawing": the S249 route seals it under a per-attachment subkey like any
 * other blob, and this module never touches storage or the network.
 *
 * Colours are NAMED CSS colours only (the palette below), guarded by
 * SAFE_COLOR on both serialize and parse, so no raw hex ever enters the
 * component or the document and no attribute injection is possible: every
 * serialized attribute value is either a rounded number or a guarded
 * lowercase colour name.
 */

/** The drawing tools. 'pen' is a freehand polyline; the rest are shapes. */
export type DrawingTool = 'pen' | 'line' | 'rect' | 'ellipse';

/**
 * One stroke. For 'pen' the points are the polyline vertices; for the shape
 * tools the points are the two drag corners [[x1, y1], [x2, y2]].
 */
export interface DrawingStroke {
	tool: DrawingTool;
	color: string;
	width: number;
	points: Array<[number, number]>;
}

/** A whole drawing: the canvas dimensions and the stroke list, in order. */
export interface DrawingModel {
	width: number;
	height: number;
	strokes: DrawingStroke[];
}

/** The MIME type a saved drawing is uploaded under. */
export const DRAWING_MIME = 'image/svg+xml';

/** The palette: NAMED CSS colours only, never hex (the --oo-* discipline). */
export const DRAWING_COLORS: readonly string[] = [
	'black',
	'crimson',
	'steelblue',
	'seagreen',
	'darkorange',
	'rebeccapurple'
];

/** Lowercase named-colour guard; anything else falls back to 'black'. */
const SAFE_COLOR = /^[a-z]{3,30}$/;

function safeColor(color: string): string {
	return SAFE_COLOR.test(color) ? color : 'black';
}

/** Round to two decimals so the serialized document stays compact. */
function round2(n: number): number {
	return Math.round(n * 100) / 100;
}

function strokeAttrs(stroke: DrawingStroke): string {
	return (
		'stroke="' +
		safeColor(stroke.color) +
		'" stroke-width="' +
		round2(stroke.width) +
		'"'
	);
}

function serializeStroke(stroke: DrawingStroke): string {
	const attrs = strokeAttrs(stroke);
	if (stroke.tool === 'pen') {
		const pts = stroke.points
			.map((p) => round2(p[0]) + ',' + round2(p[1]))
			.join(' ');
		return (
			'<polyline points="' +
			pts +
			'" fill="none" ' +
			attrs +
			' stroke-linecap="round" stroke-linejoin="round" data-oo-tool="pen"/>'
		);
	}
	const [a, b] = [stroke.points[0] ?? [0, 0], stroke.points[1] ?? [0, 0]];
	if (stroke.tool === 'line') {
		return (
			'<line x1="' +
			round2(a[0]) +
			'" y1="' +
			round2(a[1]) +
			'" x2="' +
			round2(b[0]) +
			'" y2="' +
			round2(b[1]) +
			'" ' +
			attrs +
			' stroke-linecap="round" data-oo-tool="line"/>'
		);
	}
	if (stroke.tool === 'rect') {
		const x = round2(Math.min(a[0], b[0]));
		const y = round2(Math.min(a[1], b[1]));
		const w = round2(Math.abs(b[0] - a[0]));
		const h = round2(Math.abs(b[1] - a[1]));
		return (
			'<rect x="' +
			x +
			'" y="' +
			y +
			'" width="' +
			w +
			'" height="' +
			h +
			'" fill="none" ' +
			attrs +
			' stroke-linejoin="round" data-oo-tool="rect"/>'
		);
	}
	const cx = round2((a[0] + b[0]) / 2);
	const cy = round2((a[1] + b[1]) / 2);
	const rx = round2(Math.abs(b[0] - a[0]) / 2);
	const ry = round2(Math.abs(b[1] - a[1]) / 2);
	return (
		'<ellipse cx="' +
		cx +
		'" cy="' +
		cy +
		'" rx="' +
		rx +
		'" ry="' +
		ry +
		'" fill="none" ' +
		attrs +
		' data-oo-tool="ellipse"/>'
	);
}

/** Serialize a model to a standalone SVG document (our editable format). */
export function serializeDrawing(model: DrawingModel): string {
	const w = round2(model.width);
	const h = round2(model.height);
	const body = model.strokes.map(serializeStroke).join('\n  ');
	return (
		'<?xml version="1.0" encoding="UTF-8"?>\n' +
		'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' +
		w +
		' ' +
		h +
		'" width="' +
		w +
		'" height="' +
		h +
		'" data-oo-drawing="1">\n  ' +
		body +
		'\n</svg>\n'
	);
}

function num(el: Element, attr: string, fallback = 0): number {
	const v = parseFloat(el.getAttribute(attr) ?? '');
	return Number.isFinite(v) ? v : fallback;
}

function parsePen(el: Element, color: string, width: number): DrawingStroke {
	const raw = el.getAttribute('points') ?? '';
	const points: Array<[number, number]> = [];
	for (const pair of raw.trim().split(/\s+/)) {
		const [x, y] = pair.split(',').map((s) => parseFloat(s));
		if (Number.isFinite(x) && Number.isFinite(y)) {
			points.push([x, y]);
		}
	}
	return { tool: 'pen', color, width, points };
}

/**
 * Parse a serialized drawing back into a model. Returns null for anything
 * that is not our own format: an unparseable document, a non-SVG root, or an
 * SVG without the data-oo-drawing marker (a foreign SVG is not editable).
 * Elements without a recognised data-oo-tool are skipped, never guessed.
 */
export function parseDrawing(svgText: string): DrawingModel | null {
	let doc: Document;
	try {
		doc = new DOMParser().parseFromString(svgText, 'image/svg+xml');
	} catch {
		return null;
	}
	if (doc.querySelector('parsererror') !== null) {
		return null;
	}
	const root = doc.documentElement;
	if (root.tagName.toLowerCase() !== 'svg') {
		return null;
	}
	if (root.getAttribute('data-oo-drawing') !== '1') {
		return null;
	}
	let width = num(root, 'width', 800);
	let height = num(root, 'height', 600);
	const viewBox = (root.getAttribute('viewBox') ?? '').trim().split(/\s+/);
	if (viewBox.length === 4) {
		const vw = parseFloat(viewBox[2]);
		const vh = parseFloat(viewBox[3]);
		if (Number.isFinite(vw) && Number.isFinite(vh)) {
			width = vw;
			height = vh;
		}
	}
	const strokes: DrawingStroke[] = [];
	for (const el of Array.from(root.children)) {
		const tool = el.getAttribute('data-oo-tool');
		const color = safeColor(el.getAttribute('stroke') ?? 'black');
		const width2 = num(el, 'stroke-width', 3);
		if (tool === 'pen') {
			strokes.push(parsePen(el, color, width2));
		} else if (tool === 'line') {
			strokes.push({
				tool: 'line',
				color,
				width: width2,
				points: [
					[num(el, 'x1'), num(el, 'y1')],
					[num(el, 'x2'), num(el, 'y2')]
				]
			});
		} else if (tool === 'rect') {
			const x = num(el, 'x');
			const y = num(el, 'y');
			strokes.push({
				tool: 'rect',
				color,
				width: width2,
				points: [
					[x, y],
					[x + num(el, 'width'), y + num(el, 'height')]
				]
			});
		} else if (tool === 'ellipse') {
			const cx = num(el, 'cx');
			const cy = num(el, 'cy');
			const rx = num(el, 'rx');
			const ry = num(el, 'ry');
			strokes.push({
				tool: 'ellipse',
				color,
				width: width2,
				points: [
					[cx - rx, cy - ry],
					[cx + rx, cy + ry]
				]
			});
		}
		// Any other element is skipped, never guessed.
	}
	return { width, height, strokes };
}

/** Wrap a serialized drawing in a Blob ready for the S253 upload client. */
export function drawingToBlob(model: DrawingModel): Blob {
	return new Blob([serializeDrawing(model)], { type: DRAWING_MIME });
}
