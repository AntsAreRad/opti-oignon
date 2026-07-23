/**
 * focusTrap.ts — Reusable Svelte action for modal focus management.
 *
 * Usage in a Svelte component:
 *   import { focusTrap } from '$lib/actions/focusTrap';
 *   <div use:focusTrap={{ onEscape: () => dispatch('close') }}>
 *
 * Features:
 * - Traps Tab/Shift+Tab within the element
 * - Closes on Escape key press (configurable callback)
 * - Saves and restores previously focused element
 * - Auto-focuses the first focusable child on mount
 */

/** Selector for elements that can receive focus. */
const FOCUSABLE_SELECTOR =
	'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export interface FocusTrapOptions {
	/** Called when Escape is pressed inside the trap. */
	onEscape?: () => void;
	/** If true, do not auto-focus the first element on mount. */
	noAutoFocus?: boolean;
}

/**
 * Svelte action that traps focus within a DOM node.
 */
export function focusTrap(
	node: HTMLElement,
	options: FocusTrapOptions = {}
) {
	let opts = options;
	const previouslyFocused = document.activeElement as HTMLElement | null;

	function getFocusable(): HTMLElement[] {
		return Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			e.preventDefault();
			e.stopPropagation();
			opts.onEscape?.();
			return;
		}

		if (e.key === 'Tab') {
			const focusable = getFocusable();
			if (focusable.length === 0) {
				e.preventDefault();
				return;
			}

			const first = focusable[0];
			const last = focusable[focusable.length - 1];

			if (e.shiftKey && document.activeElement === first) {
				e.preventDefault();
				last.focus();
			} else if (!e.shiftKey && document.activeElement === last) {
				e.preventDefault();
				first.focus();
			}
		}
	}

	// Mount: attach listener and auto-focus
	node.addEventListener('keydown', handleKeydown);
	if (!opts.noAutoFocus) {
		requestAnimationFrame(() => {
			const focusable = getFocusable();
			if (focusable.length > 0) {
				focusable[0].focus();
			}
		});
	}

	return {
		update(newOptions: FocusTrapOptions) {
			opts = newOptions;
		},
		destroy() {
			node.removeEventListener('keydown', handleKeydown);
			// Restore focus to previous element
			if (previouslyFocused && typeof previouslyFocused.focus === 'function') {
				previouslyFocused.focus();
			}
		},
	};
}
