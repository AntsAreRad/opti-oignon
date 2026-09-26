/**
 * Security mode store for Opti-Oignon.
 *
 * Reactive store managing the Daily/Bulbe security mode state,
 * including pending downgrade ceremony tracking.
 */

import { writable, derived } from 'svelte/store';
import { getSecurityMode } from '../api/securityMode';
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

/**
 * Reads the mode from the server into the store; a failed read keeps what
 * the store holds.
 */
export async function refreshSecurityMode(): Promise<void> {
  try {
    securityModeStatus.set(await getSecurityMode());
  } catch {
    // The mode cannot be read now: keep the last one known.
  }
}
