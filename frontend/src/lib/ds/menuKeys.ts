/**
 * menuKeys.ts (lib/ds) -- what a key does in a menu, as a pure function.
 *
 * Menu.svelte hands every key pressed in its list, or on its trigger while
 * the menu is open, to menuKey, and every key pressed on its closed trigger
 * to triggerKey, and does what the answers say; it holds no key logic of
 * its own.
 *
 *   - ArrowDown and ArrowUp move to the next or the previous item, wrapping
 *     at either end and skipping disabled items; from no item (-1) they
 *     land on the first or the last.
 *   - Home and End move to the first and the last enabled item.
 *   - Escape closes the menu (handled: the key goes no further).
 *   - Tab closes it too, and is not handled, so focus moves on as it would.
 *   - Any other key is left alone (Enter and Space activate the focused
 *     item natively).
 *
 * On the closed trigger, the down arrow opens the menu on its first enabled
 * item and the up arrow on its last; no other key opens it (Enter and Space
 * open it as a click does).
 *
 * When no item can take focus (none, or all disabled), a move lands on -1.
 * Dependency-free: the contracts run it under Node.
 */

export interface MenuKeyResult {
	/** The index of the item that should have focus, or -1 for none. */
	active: number;
	/** Whether the menu should close. */
	close: boolean;
	/** Whether the key was the menu's: its default action is prevented. */
	handled: boolean;
}

/** The next enabled index from `from` in direction `step` (1 or -1), wrapping; -1 if none. */
function next(from: number, step: number, count: number, disabled?: readonly boolean[]): number {
	if (count <= 0) return -1;
	let index = from;
	for (let tried = 0; tried < count; tried += 1) {
		if (index < 0 || index >= count) {
			index = step > 0 ? 0 : count - 1;
		} else {
			index = (index + step + count) % count;
		}
		if (!disabled || !disabled[index]) return index;
	}
	return -1;
}

/**
 * What `key` does to a menu of `count` items whose focused item is
 * `active` (-1 for none); `disabled[i]` marks an item that cannot take focus.
 */
export function menuKey(
	key: string,
	active: number,
	count: number,
	disabled?: readonly boolean[]
): MenuKeyResult {
	switch (key) {
		case 'ArrowDown':
			return { active: next(active, 1, count, disabled), close: false, handled: true };
		case 'ArrowUp':
			return { active: next(active, -1, count, disabled), close: false, handled: true };
		case 'Home':
			return { active: next(-1, 1, count, disabled), close: false, handled: true };
		case 'End':
			return { active: next(-1, -1, count, disabled), close: false, handled: true };
		case 'Escape':
			return { active, close: true, handled: true };
		case 'Tab':
			return { active, close: true, handled: false };
		default:
			return { active, close: false, handled: false };
	}
}

/**
 * The item a key on the closed trigger opens the menu on (-1 when no item
 * can take focus), or null for a key that does not open it.
 */
export function triggerKey(key: string, count: number, disabled?: readonly boolean[]): number | null {
	switch (key) {
		case 'ArrowDown':
			return next(-1, 1, count, disabled);
		case 'ArrowUp':
			return next(-1, -1, count, disabled);
		default:
			return null;
	}
}
