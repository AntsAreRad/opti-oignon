// Opti-Oignon design-system primitives (lib/ds).
// Import as: import { Button, Modal } from '$lib/ds';

export { default as Button } from './Button.svelte';
export { default as IconButton } from './IconButton.svelte';
export { default as TextButton } from './TextButton.svelte';
export { default as ToggleChip } from './ToggleChip.svelte';
export { default as Checkbox } from './Checkbox.svelte';
export { default as Input } from './Input.svelte';
export { default as Card } from './Card.svelte';
export { default as Menu } from './Menu.svelte';
export { default as Modal } from './Modal.svelte';
export { default as ConfirmDialog } from './ConfirmDialog.svelte';
export { default as SidePanel } from './SidePanel.svelte';
export { default as Toast } from './Toast.svelte';
export { default as Select } from './Select.svelte';
export { default as Switch } from './Switch.svelte';
export { default as Tabs } from './Tabs.svelte';
export { default as Tooltip } from './Tooltip.svelte';
export { default as Icon } from './Icon.svelte';
export { default as EmptyState } from './EmptyState.svelte';
export { default as InlineError } from './InlineError.svelte';
export { default as PanelHeader } from './PanelHeader.svelte';

export { ICONS, iconKey, inlineIcon } from './icons';

export type {
	IconName,
	Size,
	ButtonVariant,
	ButtonShape,
	PopupKind,
	IconButtonSize,
	IconButtonVariant,
	ToastVariant,
	ModalVariant,
	TooltipPlacement,
	MenuPlacement,
	MenuItem,
	SelectOption,
	TabItem
} from './types';
export type { IconShape } from './icons';
