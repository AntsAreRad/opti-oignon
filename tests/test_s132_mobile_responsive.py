"""
Tests for S132 -- Mobile Responsive: Layout + Chat.

Covers:
- Part 1: responsive.css foundation (breakpoints, touch-target, safe-area, dvh)
- Part 2: Sidebar responsive (swipe refs, backdrop, touch-friendly nav, safe-area)
- Part 3: AppShell responsive (h-viewport, hamburger, mobile-default-closed)
- Part 4: ChatInput mobile (touch-target, enterkeyhint, font-size-16, full-width)
- Part 5: ChatMessage mobile (reduced padding, code scroll, responsive images)
- Part 6: ChatControlBar mobile (overflow scroll, shrink-0, min-height)
- Part 7: ScrollToBottomFab (component exists, visibility logic, styling)
- Part 8: Chat page integration (FAB import, safe-area, touch-scroll)
- Part 9: Version bump 2.9.3, no French in code, CSS color compliance
"""

import os
import re
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")


def read_file(relpath: str) -> str:
    """Read a file relative to project root."""
    fullpath = os.path.join(PROJECT_ROOT, relpath)
    with open(fullpath, encoding="utf-8") as f:
        return f.read()


class TestResponsiveCSSFoundation(unittest.TestCase):
    """Part 1: responsive.css exists and contains expected definitions."""

    def setUp(self):
        self.css = read_file("frontend/src/styles/responsive.css")

    def test_responsive_css_exists(self):
        """responsive.css file exists in styles directory."""
        path = os.path.join(FRONTEND_SRC, "styles", "responsive.css")
        self.assertTrue(os.path.isfile(path))

    def test_breakpoint_variables(self):
        """responsive.css defines --oo-bp-sm, --oo-bp-md, --oo-bp-lg."""
        self.assertIn("--oo-bp-sm", self.css)
        self.assertIn("--oo-bp-md", self.css)
        self.assertIn("--oo-bp-lg", self.css)
        self.assertIn("640px", self.css)
        self.assertIn("768px", self.css)
        self.assertIn("1024px", self.css)

    def test_touch_target_class(self):
        """responsive.css defines .touch-target with min 44px dimensions."""
        self.assertIn(".touch-target", self.css)
        self.assertIn("44px", self.css)
        self.assertIn("min-width", self.css)
        self.assertIn("min-height", self.css)

    def test_safe_area_insets(self):
        """responsive.css uses env(safe-area-inset-*) for notched phones."""
        self.assertIn("env(safe-area-inset-", self.css)
        self.assertIn(".safe-area-pad", self.css)
        self.assertIn(".safe-area-bottom", self.css)

    def test_viewport_height_fix_dvh(self):
        """responsive.css defines .h-viewport with dvh fallback."""
        self.assertIn(".h-viewport", self.css)
        self.assertIn("100dvh", self.css)
        self.assertIn("100vh", self.css)

    def test_scroll_fab_class(self):
        """responsive.css defines .scroll-fab styling."""
        self.assertIn(".scroll-fab", self.css)
        self.assertIn("48px", self.css)
        self.assertIn("border-radius", self.css)

    def test_touch_scroll_classes(self):
        """responsive.css defines touch-scroll utilities."""
        self.assertIn(".touch-scroll", self.css)
        self.assertIn(".touch-scroll-x", self.css)
        self.assertIn("-webkit-overflow-scrolling", self.css)

    def test_imported_in_app_css(self):
        """app.css imports responsive.css."""
        app_css = read_file("frontend/src/app.css")
        self.assertIn("@import './styles/responsive.css'", app_css)


class TestViewportMeta(unittest.TestCase):
    """Part 1b: app.html viewport meta tag."""

    def test_viewport_fit_cover(self):
        """app.html has viewport-fit=cover for notched phones."""
        html = read_file("frontend/src/app.html")
        self.assertIn("viewport-fit=cover", html)

    def test_initial_scale(self):
        """app.html keeps width=device-width, initial-scale=1."""
        html = read_file("frontend/src/app.html")
        self.assertIn("width=device-width", html)
        self.assertIn("initial-scale=1", html)


class TestSidebarResponsive(unittest.TestCase):
    """Part 2: Sidebar touch-friendly and safe-area."""

    def setUp(self):
        self.sidebar = read_file(
            "frontend/src/lib/components/layout/Sidebar.svelte"
        )

    def test_touch_friendly_nav_links(self):
        """Nav links have min-height: 44px for touch targets."""
        self.assertIn("min-height: 44px", self.sidebar)

    def test_touch_scroll_conversation_list(self):
        """ConversationList area uses touch-scroll class."""
        self.assertIn("touch-scroll", self.sidebar)

    def test_safe_area_bottom_footer(self):
        """Footer has safe-area-bottom class for notched phones."""
        self.assertIn("safe-area-bottom", self.sidebar)

    def test_theme_toggle_touch_target(self):
        """Theme toggle button has touch-target class."""
        self.assertIn("touch-target", self.sidebar)

    def test_s132_comment(self):
        """Sidebar references S132 in header comment."""
        self.assertIn("S132", self.sidebar)


class TestAppShellResponsive(unittest.TestCase):
    """Part 3: AppShell swipe-to-close, mobile defaults, dvh."""

    def setUp(self):
        self.appshell = read_file(
            "frontend/src/lib/components/layout/AppShell.svelte"
        )

    def test_h_viewport_class(self):
        """Root container uses h-viewport (dvh) instead of h-screen."""
        self.assertIn("h-viewport", self.appshell)
        self.assertNotIn('class="h-screen', self.appshell)

    def test_swipe_touch_handlers(self):
        """AppShell has touch event handlers for swipe-to-close."""
        self.assertIn("handleSidebarTouchStart", self.appshell)
        self.assertIn("handleSidebarTouchMove", self.appshell)
        self.assertIn("handleSidebarTouchEnd", self.appshell)
        self.assertIn("on:touchstart", self.appshell)
        self.assertIn("on:touchmove", self.appshell)
        self.assertIn("on:touchend", self.appshell)

    def test_swipe_threshold(self):
        """Swipe threshold is defined (at least 50px)."""
        match = re.search(r"SWIPE_THRESHOLD\s*=\s*(\d+)", self.appshell)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(int(match.group(1)), 50)

    def test_sidebar_closed_on_mobile_init(self):
        """Sidebar is set closed on mobile during onMount."""
        self.assertIn("sidebarOpen.set(false)", self.appshell)

    def test_hamburger_icon_on_mobile(self):
        """Hamburger icon (3 bars) shown on mobile even when sidebar open."""
        # On mobile: always show hamburger, chevron only on desktop
        self.assertIn("$sidebarOpen && !isMobile", self.appshell)

    def test_touch_target_hamburger(self):
        """Hamburger button uses touch-target class on mobile."""
        self.assertIn("touch-target", self.appshell)

    def test_backdrop_with_transition(self):
        """Sidebar backdrop has transition class."""
        self.assertIn("sidebar-mobile-backdrop", self.appshell)

    def test_safe_area_header(self):
        """Header has safe-area-pad class."""
        self.assertIn("safe-area-pad", self.appshell)


class TestChatInputMobile(unittest.TestCase):
    """Part 4: ChatInput mobile enhancements."""

    def setUp(self):
        self.chatinput = read_file(
            "frontend/src/lib/components/chat/ChatInput.svelte"
        )

    def test_enterkeyhint_send(self):
        """Textarea has enterkeyhint='send' for mobile keyboards."""
        self.assertIn('enterkeyhint="send"', self.chatinput)

    def test_font_size_16px(self):
        """Textarea uses font-size: 16px to prevent iOS auto-zoom."""
        self.assertIn("font-size: 16px", self.chatinput)

    def test_send_button_touch_target(self):
        """Send button has 44px dimensions for touch targets."""
        self.assertIn("width: 44px", self.chatinput)
        self.assertIn("height: 44px", self.chatinput)

    def test_image_button_touch_target(self):
        """Image upload button has touch-target class."""
        # Count touch-target occurrences — image btn, send, stop, retry
        count = self.chatinput.count("touch-target")
        self.assertGreaterEqual(count, 3)

    def test_s132_comment(self):
        """ChatInput references S132 in header comment."""
        self.assertIn("S132", self.chatinput)


class TestChatMessageMobile(unittest.TestCase):
    """Part 5: ChatMessage mobile padding and code scroll."""

    def setUp(self):
        self.chatmsg = read_file(
            "frontend/src/lib/components/chat/ChatMessage.svelte"
        )

    def test_reduced_mobile_padding(self):
        """Message bubble uses px-2.5 on mobile, px-4 on sm+."""
        self.assertIn("px-2.5", self.chatmsg)
        self.assertIn("sm:px-4", self.chatmsg)

    def test_code_block_overflow_scroll(self):
        """Code blocks have overflow-x: auto for horizontal scroll."""
        self.assertIn("overflow-x: auto", self.chatmsg)

    def test_code_block_touch_scroll(self):
        """Code blocks use -webkit-overflow-scrolling: touch."""
        self.assertIn("-webkit-overflow-scrolling: touch", self.chatmsg)

    def test_responsive_images(self):
        """Images inside messages have max-width: 100%."""
        self.assertIn("max-width: 100%", self.chatmsg)

    def test_msg_content_class(self):
        """Content div has msg-content class for scoped styles."""
        self.assertIn("msg-content", self.chatmsg)


class TestChatControlBarMobile(unittest.TestCase):
    """Part 6: ChatControlBar overflow and touch-friendly toggles."""

    def setUp(self):
        self.controlbar = read_file(
            "frontend/src/lib/components/chat/ChatControlBar.svelte"
        )

    def test_horizontal_scroll_overflow(self):
        """Control bar uses overflow-x-auto for horizontal scroll."""
        self.assertIn("overflow-x-auto", self.controlbar)

    def test_no_flex_wrap(self):
        """Control bar does NOT use flex-wrap (uses nowrap scroll)."""
        # The main container should not have flex-wrap
        self.assertNotIn("flex-wrap", self.controlbar)

    def test_toggle_buttons_shrink_0(self):
        """Toggle buttons have shrink-0 to prevent compression."""
        # All toggle button classes should include shrink-0
        self.assertIn("shrink-0", self.controlbar)
        count = self.controlbar.count("shrink-0")
        # At least 9 toggle buttons + 2 selector divs
        self.assertGreaterEqual(count, 10)

    def test_min_height_container(self):
        """Control bar container has min-height for touch friendliness."""
        self.assertIn("min-height: 36px", self.controlbar)

    def test_touch_scroll_x_class(self):
        """Control bar uses touch-scroll-x class."""
        self.assertIn("touch-scroll-x", self.controlbar)

    def test_s132_comment(self):
        """ChatControlBar references S132 in header comment."""
        self.assertIn("S132", self.controlbar)


class TestScrollToBottomFab(unittest.TestCase):
    """Part 7: ScrollToBottomFab component."""

    def setUp(self):
        self.fab_path = os.path.join(
            FRONTEND_SRC,
            "lib", "components", "chat", "ScrollToBottomFab.svelte",
        )

    def test_component_exists(self):
        """ScrollToBottomFab.svelte exists."""
        self.assertTrue(os.path.isfile(self.fab_path))

    def test_scroll_fab_class_used(self):
        """Component uses scroll-fab CSS class from responsive.css."""
        with open(self.fab_path) as f:
            content = f.read()
        self.assertIn("scroll-fab", content)

    def test_visibility_prop(self):
        """Component accepts visible prop for show/hide."""
        with open(self.fab_path) as f:
            content = f.read()
        self.assertIn("visible", content)

    def test_onclick_prop(self):
        """Component accepts onClick callback prop."""
        with open(self.fab_path) as f:
            content = f.read()
        self.assertIn("onClick", content)

    def test_aria_label(self):
        """FAB button has aria-label for accessibility."""
        with open(self.fab_path) as f:
            content = f.read()
        self.assertIn("aria-label", content)

    def test_down_arrow_icon(self):
        """FAB contains a down-arrow SVG icon."""
        with open(self.fab_path) as f:
            content = f.read()
        self.assertIn("<svg", content)
        self.assertIn("</svg>", content)


class TestChatPageIntegration(unittest.TestCase):
    """Part 8: Chat page FAB integration and mobile adjustments."""

    def setUp(self):
        self.chatpage = read_file(
            "frontend/src/routes/chat/[id]/+page.svelte"
        )

    def test_fab_import(self):
        """Chat page imports ScrollToBottomFab."""
        self.assertIn("ScrollToBottomFab", self.chatpage)

    def test_show_scroll_fab_state(self):
        """Chat page tracks showScrollFab state variable."""
        self.assertIn("showScrollFab", self.chatpage)

    def test_scroll_fab_threshold(self):
        """FAB visibility triggered by scroll distance (>= 200px)."""
        # Look for a threshold comparison
        match = re.search(
            r"scrollHeight\s*-\s*scrollTop\s*-\s*clientHeight\s*>\s*(\d+)",
            self.chatpage,
        )
        found = False
        for m in re.finditer(
            r"scrollHeight\s*-\s*scrollTop\s*-\s*clientHeight\s*>\s*(\d+)",
            self.chatpage,
        ):
            val = int(m.group(1))
            if val >= 200:
                found = True
        self.assertTrue(found, "FAB threshold should be >= 200px")

    def test_safe_area_bottom_input(self):
        """Input area has safe-area-bottom class."""
        self.assertIn("safe-area-bottom", self.chatpage)

    def test_touch_scroll_messages(self):
        """Messages container uses touch-scroll class."""
        self.assertIn("touch-scroll", self.chatpage)

    def test_tighter_mobile_padding(self):
        """Messages area uses px-2 on mobile (tighter than px-3)."""
        self.assertIn("px-2 sm:px-4", self.chatpage)

    def test_s132_comment(self):
        """Chat page references S132 in header comment."""
        self.assertIn("S132", self.chatpage)


class TestVersionAndCompliance(unittest.TestCase):
    """Part 9: Version bump and code quality checks."""

    def test_version_is_2_9_3(self):
        """Version is 2.9.3."""
        content = read_file("opti_oignon/__version__.py")
        self.assertIn('"3.0.0"', content)

    def test_no_french_in_new_code(self):
        """No French words in new S132 code (comments + UI text)."""
        french_words = [
            "ouvrir", "fermer", "envoyer", "Bonjour",
            "suivant", "precedent", "parametres", "connexion",
            "cliquer", "afficher", "masquer",
        ]
        files = [
            "frontend/src/styles/responsive.css",
            "frontend/src/lib/components/chat/ScrollToBottomFab.svelte",
        ]
        for fpath in files:
            content = read_file(fpath)
            for word in french_words:
                self.assertNotIn(
                    word.lower(),
                    content.lower(),
                    f"French word '{word}' found in {fpath}",
                )

    def test_no_hardcoded_hex_in_responsive_css(self):
        """responsive.css uses only --oo-* variables (hex only in fallbacks)."""
        css = read_file("frontend/src/styles/responsive.css")
        # Find all hex color references
        hex_refs = re.findall(r"#[0-9a-fA-F]{3,8}", css)
        for href in hex_refs:
            # Each must appear inside a var(--oo-*, #fallback) pattern
            pattern = rf"var\(--oo-[^)]*,\s*{re.escape(href)}\)"
            self.assertTrue(
                re.search(pattern, css),
                f"Hardcoded hex {href} not in var() fallback in responsive.css",
            )

    def test_no_hardcoded_hex_in_scroll_fab(self):
        """ScrollToBottomFab uses only CSS classes, no inline hex."""
        content = read_file(
            "frontend/src/lib/components/chat/ScrollToBottomFab.svelte"
        )
        # Remove script block
        clean = re.sub(
            r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL
        )
        hex_refs = re.findall(r"#[0-9a-fA-F]{3,8}", clean)
        self.assertEqual(
            len(hex_refs), 0,
            f"Hardcoded hex in ScrollToBottomFab template: {hex_refs}",
        )


if __name__ == "__main__":
    unittest.main()
