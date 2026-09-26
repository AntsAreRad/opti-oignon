<!--
  Root layout: the stylesheets, the preferences, sign-in, the first-run
  overlay, the global shortcuts, the toasts, the one skip link and the
  route announcer. The shell of both spaces is mounted one level down, by
  the (app) layout; sign-in and registration sit outside it.

  The announcer names the destination a page belongs to, read from the
  destination table (lib/nav/destinations.ts) through lib/nav/active.ts; it
  holds no map of its own.
-->
<script>
	import { onMount, tick } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import { get } from 'svelte/store';
	import '../app.css';
	import KeyboardShortcuts from '$lib/components/ui/KeyboardShortcuts.svelte';
	import OnboardingOverlay from '$lib/components/ui/OnboardingOverlay.svelte';
	import Toast from '$lib/ds/Toast.svelte';
	import { initReducedMotion, sidebarOpen, toggleSidebar } from '$lib/stores/ui';
	import { initPreferences, toggleDayNight } from '$lib/stores/preferences';
	import { activeConversationId, activeConversation, createNewConversation } from '$lib/stores/conversations';
	import { toastError } from '$lib/stores/notifications';
	import { initAuth, authLoading, currentUser, isSingleUserMode } from '$lib/stores/auth';
	import { openExportDialog } from '$lib/stores/exportDialog';
	import { DESTINATIONS, spaceHome, visibleDestinations } from '$lib/nav/destinations';
	import { destinationFor } from '$lib/nav/active';

	/** Auth public routes that don't require login. */
	const PUBLIC_ROUTES = ['/login', '/register'];

	// The componion's own switch arrives with its page; until then its entry
	// is not ready, and the visible list never names it.
	const visibility = { componion: true };

	// The appearance choices were applied before the first paint; this
	// applies them again with the stores and follows the system from now on.
	onMount(() => {
		const stopFollowing = initPreferences();
		initReducedMotion();
		initAuth();
		return stopFollowing;
	});

	// Auth guard: redirect to /login if multi-user mode and not authenticated
	$: pathname = $page.url.pathname;
	$: isPublicRoute = PUBLIC_ROUTES.some((r) => pathname.startsWith(r));

	// Announce route changes for screen readers: the destination the page
	// belongs to, or the page's path when it belongs to none.
	let announcement = '';
	$: {
		const destination = destinationFor(pathname, visibleDestinations(DESTINATIONS, visibility));
		announcement = `Navigated to ${destination ? destination.label : pathname}`;
	}

	$: {
		if (!$authLoading && !$isSingleUserMode && !$currentUser && !isPublicRoute) {
			goto('/login');
		}
		// Redirect away from login/register if already authenticated
		if (!$authLoading && $currentUser && isPublicRoute) {
			goto(spaceHome('use', DESTINATIONS));
		}
	}

	// -- Global shortcut callbacks --
	async function handleNewConversation() {
		try {
			const id = await createNewConversation();
			goto(`/chat/${id}`);
		} catch {
			toastError('Failed to create conversation');
		}
	}

	// The export dialog is the shell's, opened through its store on every page.
	function handleExportConversation() {
		const id = get(activeConversationId);
		if (id) openExportDialog(id, get(activeConversation)?.title || 'conversation');
	}

	function handleGoToPreferences() {
		goto(DESTINATIONS.find((d) => d.id === 'preferences')?.href ?? '/');
	}

	// Focus the sidebar's search field, which sits in the form carrying
	// data-oo-search; a collapsed sidebar or a shut drawer opens first.
	async function handleToggleSearch() {
		sidebarOpen.set(true);
		await tick();
		const box = document.querySelector('[data-oo-search]');
		const field = box ? box.querySelector('input') : null;
		if (field instanceof HTMLInputElement) {
			field.focus();
		}
	}
</script>

<!-- The one skip link; every page renders one main-content landmark for it -->
<a href="#main-content" class="oo-skip-link">Skip to main content</a>

<!-- Route change announcements for screen readers -->
<div class="sr-only" aria-live="polite" aria-atomic="true" id="oo-route-announcer">{announcement}</div>

<OnboardingOverlay />

<KeyboardShortcuts
	onNewConversation={handleNewConversation}
	onExportConversation={handleExportConversation}
	onGoToSettings={handleGoToPreferences}
	onToggleSearch={handleToggleSearch}
	onToggleTheme={toggleDayNight}
	onToggleSidebar={toggleSidebar}
/>

<slot />

<!-- Global toast notifications (ds primitive, single mount) -->
<Toast />
