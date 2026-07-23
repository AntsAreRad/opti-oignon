// Shared types for the Opti-Oignon design-system primitives (lib/ds).
// Type declarations only; no runtime code.

/** A lucide-svelte icon name in kebab-case or PascalCase (e.g. 'plus', 'Settings'). */
export type IconName = string;

/** Standard control size scale. */
export type Size = 'sm' | 'md' | 'lg';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'link';

export type ToastVariant = 'success' | 'warning' | 'error' | 'info';

export type ModalVariant = 'center' | 'drawer-right' | 'drawer-bottom';

export type TooltipPlacement = 'top' | 'bottom' | 'left' | 'right';

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
