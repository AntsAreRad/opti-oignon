/**
 * Where a dialog opens: the element marked for it (`autofocus`,
 * `data-autofocus`) wherever it sits, else the first focusable element that
 * is not among the head's actions. The head holds Stop all in a dialog of
 * the shell; opening there would make Enter start the stop's confirmation
 * instead of the dialog's own action.
 *
 * A pure function, with no dependency: the ds Modal calls it with the
 * dialog's focusable elements in document order, and it runs under Node as
 * it is.
 */

/** The element a dialog gives its first focus, or undefined when it has none. */
export function firstFocus<T>(
	candidates: readonly T[],
	marked: T | null | undefined,
	inActions: (candidate: T) => boolean
): T | undefined {
	if (marked) return marked;
	return candidates.find((candidate) => !inActions(candidate));
}
