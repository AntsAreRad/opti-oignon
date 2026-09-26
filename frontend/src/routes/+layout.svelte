<!--
  Root layout: CSS imports, theme init, auth init, global keyboard shortcuts.
-->
<script>
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import { get } from 'svelte/store';
	import '../app.css';
	import KeyboardShortcuts from '$lib/components/ui/KeyboardShortcuts.svelte';
	import OnboardingOverlay from '$lib/components/ui/OnboardingOverlay.svelte';
	import Toast from '$lib/ds/Toast.svelte';
	import { initReducedMotion, toggleSidebar } from '$lib/stores/ui';
	import { initPreferences, toggleDayNight } from '$lib/stores/preferences';
	import { activeConversationId, activeConversation, createNewConversation } from '$lib/stores/conversations';
	import { toastError } from '$lib/stores/notifications';
	import { initAuth, authLoading, currentUser, isSingleUserMode } from '$lib/stores/auth';

	/** Auth public routes that don't require login. */
	const PUBLIC_ROUTES = ['/login', '/register'];

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

	// Announce route changes for screen readers
	$: if (typeof document !== 'undefined' && pathname) {
		const announcer = document.getElementById('oo-route-announcer');
		if (announcer) {
			const routeNames = {
				'/chat': 'Chat',
				'/settings': 'Settings',
				'/login': 'Login',
				'/register': 'Register'
			};
			const name = Object.entries(routeNames).find(([r]) => pathname.startsWith(r));
			announcer.textContent = `Navigated to ${name ? name[1] : pathname}`;
		}
	}

	$: {
		if (!$authLoading && !$isSingleUserMode && !$currentUser && !isPublicRoute) {
			goto('/login');
		}
		// Redirect away from login/register if already authenticated
		if (!$authLoading && $currentUser && isPublicRoute) {
			goto('/chat');
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

	function handleExportConversation() {
		// Dispatch custom event that chat layout can capture
		const convId = get(activeConversationId);
		if (convId) {
			window.dispatchEvent(new CustomEvent('opti-export-conversation', {
				detail: { id: convId, title: get(activeConversation)?.title || 'conversation' }
			}));
		}
	}

	function handleGoToSettings() {
		goto('/settings');
	}

	function handleToggleSearch() {
		// Focus the sidebar's search field, which carries data-oo-search
		const el = document.querySelector('[data-oo-search]');
		if (el instanceof HTMLInputElement) {
			el.focus();
		}
	}
</script>

<!-- The one skip link; every page renders one main-content landmark for it -->
<a href="#main-content" class="oo-skip-link">Skip to main content</a>

<OnboardingOverlay />

<KeyboardShortcuts
	onNewConversation={handleNewConversation}
	onExportConversation={handleExportConversation}
	onGoToSettings={handleGoToSettings}
	onToggleSearch={handleToggleSearch}
	onToggleTheme={toggleDayNight}
	onToggleSidebar={toggleSidebar}
/>

<slot />

<!-- Global toast notifications (ds primitive, single mount) -->
<Toast />
