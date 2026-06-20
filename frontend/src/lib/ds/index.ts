// Opti-Oignon design-system primitives (lib/ds) -- S166.
// Import as: import { Button, Modal } from '$lib/ds';

export { default as Button } from './Button.svelte';
export { default as Input } from './Input.svelte';
export { default as Card } from './Card.svelte';
export { default as Modal } from './Modal.svelte';
export { default as Toast } from './Toast.svelte';
export { default as Select } from './Select.svelte';
export { default as Switch } from './Switch.svelte';
export { default as Tabs } from './Tabs.svelte';
export { default as Tooltip } from './Tooltip.svelte';
export { default as Icon } from './Icon.svelte';
export { default as EmptyState } from './EmptyState.svelte';
export { default as InlineError } from './InlineError.svelte';

export type {
	IconName,
	Size,
	ButtonVariant,
	ToastVariant,
	ModalVariant,
	TooltipPlacement,
	SelectOption,
	TabItem
} from './types';
