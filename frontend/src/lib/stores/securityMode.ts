/**
 * Security mode store for Opti-Oignon.
 *
 * Reactive store managing the Daily/Bulbe security mode state,
 * including pending downgrade ceremony tracking.
 */

import { writable, derived } from 'svelte/store';
import type { SecurityModeStatus, PendingDowngrade, ModePolicy } from '../api/securityMode';

/** Full security mode state. */
export const securityModeStatus = writable<SecurityModeStatus>({
  mode: 'daily',
  available: false,
});

/** Derived: current mode string. */
export const currentMode = derived(securityModeStatus, ($s) => $s.mode);

/** Derived: is Bulbe mode active? */
export const isBulbe = derived(securityModeStatus, ($s) => $s.mode === 'bulbe');

/** Derived: is the security mode system available? */
export const securityModeAvailable = derived(securityModeStatus, ($s) => $s.available);

/** Derived: current policy. */
export const modePolicy = derived(securityModeStatus, ($s) => $s.policy ?? null);

/** Derived: pending downgrade state. */
export const pendingDowngrade = derived(
  securityModeStatus,
  ($s) => $s.pending_downgrade ?? null
);
