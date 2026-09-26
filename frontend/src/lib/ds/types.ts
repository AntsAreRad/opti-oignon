// Shared types for the Opti-Oignon design-system primitives (lib/ds).
// Type declarations only; no runtime code.

/** An icon name in kebab-case or PascalCase (e.g. 'plus', 'ChevronDown'): one the inline set
 * draws (icons.ts), or, until it is drawn there, one the older icon package draws. */
export type IconName = string;

/** Standard control size scale. */
export type Size = 'sm' | 'md' | 'lg';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'link';

/** A button's outline: a rounded rectangle, a pill, or a circle. */
export type ButtonShape = 'rect' | 'pill' | 'round';

/** What a control opens, as aria-haspopup says it. */
export type PopupKind = 'menu' | 'listbox' | 'dialog' | 'true';

/** An icon button's box: 36 px on a desktop, 44 px as a phone's touch target. */
export type IconButtonSize = 'md' | 'lg';

export type IconButtonVariant = 'ghost' | 'primary';

export type ToastVariant = 'success' | 'warning' | 'error' | 'info';

export type ModalVariant = 'center' | 'drawer-right' | 'drawer-bottom';

export type TooltipPlacement = 'top' | 'bottom' | 'left' | 'right';

/** Where a menu opens, beside its trigger. */
export type MenuPlacement = 'bottom-start' | 'bottom-end' | 'top-start' | 'top-end';

export interface MenuItem {
	id: string;
	label: string;
	icon?: IconName;
	disabled?: boolean;
	/** The item destroys something: drawn in the stop ink. */
	danger?: boolean;
}

export interface SelectOption {
	value: string;
	label: string;
	group?: string;
	disabled?: boolean;
}

export interface TabItem {
	id: string;
	label: string;
	icon?: IconName;
}
