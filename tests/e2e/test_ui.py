"""
Playwright E2E tests for Velqua dashboard UI (v3.2+).

Tests the real browser against a running server, verifying:
- Page shell: title, header, hero metrics
- Dashboard panels: provider, import, connection
- Onboarding wizard
- Toast notifications
- API health endpoints

Run:
    python -m pytest tests/e2e/ -v           (headless)
    python -m pytest tests/e2e/ -v --headed  (visual)
"""
import re
import pytest
from playwright.sync_api import Page, expect


# ============================================================
# App Shell
# ============================================================

class TestAppShell:
    def test_page_loads_with_title(self, page: Page):
        expect(page).to_have_title("Velqua")

    def test_header_visible(self, page: Page):
        expect(page.locator(".topbar")).to_be_visible()
        expect(page.locator(".brand")).to_be_visible()

    def test_hero_card_renders(self, page: Page):
        expect(page.locator(".hero-card")).to_be_visible()
        expect(page.locator("#heroFacts")).to_be_visible()
        expect(page.locator("#heroEpisodes")).to_be_visible()
        expect(page.locator("#heroProviders")).to_be_visible()

    def test_hero_metrics_populated(self, page: Page):
        """Metrics should show a value or '—' (loading placeholder)."""
        for el_id in ("#heroFacts", "#heroEpisodes", "#heroProviders"):
            text = page.locator(el_id).inner_text()
            assert text is not None and len(text) > 0, f"{el_id} is empty"

    def test_no_js_errors_on_load(self, page: Page):
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.reload()
        page.wait_for_load_state("networkidle")
        assert errors == [], f"JS errors on load: {errors}"

    def test_onboarding_dismissed(self, page: Page):
        """Wizard overlay should exist and not crash."""
        overlay = page.locator("#onboardingOverlay")
        assert overlay.count() > 0


# ============================================================
# Provider Panel
# ============================================================

class TestProviderPanel:
    def test_provider_select_present(self, page: Page):
        expect(page.locator("#providerSelect")).to_be_visible()

    def test_provider_status_visible(self, page: Page):
        expect(page.locator("#providerStatusDot")).to_be_visible()
        expect(page.locator("#providerStatusText")).to_be_visible()

    def test_provider_status_populated(self, page: Page):
        text = page.locator("#providerStatusText").inner_text()
        assert text is not None and len(text) > 0
        # Should not still say "Loading" after page settles
        assert "Loading" not in text, f"Provider still loading: {text}"


# ============================================================
# Import Panel
# ============================================================

class TestImportPanel:
    def test_file_input_present(self, page: Page):
        expect(page.locator("#memoryFileInput")).to_be_visible()

    def test_import_button_present(self, page: Page):
        expect(page.locator("#importMemoryBtn")).to_be_visible()

    def test_import_status_text_present(self, page: Page):
        expect(page.locator("#importStatus")).to_be_visible()


# ============================================================
# Connection Panel
# ============================================================

class TestConnectionPanel:
    def test_proxy_url_displayed(self, page: Page):
        el = page.locator("#proxyBaseUrl")
        expect(el).to_be_visible()
        text = el.inner_text()
        assert "localhost" in text or "127.0.0.1" in text

    def test_copy_button_present(self, page: Page):
        expect(page.locator("#copyProxyUrlBtn")).to_be_visible()


# ============================================================
# Dashboard Actions
# ============================================================

class TestDashboardActions:
    def test_refresh_button_present(self, page: Page):
        expect(page.locator("#refreshDashboardBtn")).to_be_visible()

    def test_onboarding_button_present(self, page: Page):
        expect(page.locator("#openOnboardingBtn")).to_be_visible()

    def test_refresh_button_clickable(self, page: Page):
        """Clicking refresh should not crash and metrics remain visible."""
        page.click("#refreshDashboardBtn", force=True)
        page.wait_for_timeout(1000)
        expect(page.locator("#heroFacts")).to_be_visible()


# ============================================================
# Onboarding Wizard
# ============================================================

class TestOnboarding:
    def test_wizard_opens(self, page: Page):
        # Clear onboarding flag so wizard shows
        page.evaluate("() => localStorage.removeItem('velqua_onboarding_done')")
        page.reload()
        page.wait_for_load_state("networkidle")

        overlay = page.locator("#onboardingOverlay")
        expect(overlay).to_be_visible()

    def test_wizard_has_steps(self, page: Page):
        page.evaluate("() => localStorage.removeItem('velqua_onboarding_done')")
        page.reload()
        page.wait_for_load_state("networkidle")

        steps = page.locator(".wizard-step")
        assert steps.count() >= 3, f"Expected >=3 wizard steps, got {steps.count()}"

    def test_wizard_close_button(self, page: Page):
        page.evaluate("() => localStorage.removeItem('velqua_onboarding_done')")
        page.reload()
        page.wait_for_load_state("networkidle")

        page.click("#closeOnboardingBtn", force=True)
        page.wait_for_timeout(500)
        # Overlay should hide
        expect(page.locator("#onboardingOverlay")).to_be_hidden()

    def test_wizard_skip_button(self, page: Page):
        page.evaluate("() => localStorage.removeItem('velqua_onboarding_done')")
        page.reload()
        page.wait_for_load_state("networkidle")

        page.click("#wizardSkipBtn", force=True)
        page.wait_for_timeout(500)
        expect(page.locator("#onboardingOverlay")).to_be_hidden()

    def test_wizard_next_navigates(self, page: Page):
        page.evaluate("() => localStorage.removeItem('velqua_onboarding_done')")
        page.reload()
        page.wait_for_load_state("networkidle")

        # First panel should be active
        panel_0 = page.locator('.wizard-panel[data-step-panel="0"]')
        expect(panel_0).to_have_class(re.compile("active"))

        page.click("#wizardNextBtn", force=True)
        page.wait_for_timeout(500)

        # Second panel should now be active
        panel_1 = page.locator('.wizard-panel[data-step-panel="1"]')
        expect(panel_1).to_have_class(re.compile("active"))


# ============================================================
# Toast Notifications
# ============================================================

class TestToast:
    def test_toast_root_present(self, page: Page):
        assert page.locator("#toastRoot").count() > 0


# ============================================================
# API Health (backend endpoints, no browser UI needed)
# ============================================================

class TestApiHealth:
    def test_health_endpoint(self, velqua_url):
        import httpx
        r = httpx.get(velqua_url + "/health", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "facts_count" in data

    def test_license_status(self, velqua_url):
        import httpx
        r = httpx.get(velqua_url + "/license/status", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "trial"
        assert data["is_trial"] is True

    def test_providers_endpoint(self, velqua_url):
        import httpx
        r = httpx.get(velqua_url + "/settings/providers", timeout=10)
        assert r.status_code == 200
        data = r.json()
        # Response is {"providers": [...]}
        assert "providers" in data
        assert isinstance(data["providers"], list)

    def test_update_check_endpoint(self, velqua_url):
        import httpx
        r = httpx.get(velqua_url + "/update/check", timeout=10)
        assert r.status_code in (200, 503)

    def test_settings_endpoint(self, velqua_url):
        import httpx
        r = httpx.get(velqua_url + "/settings", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "budget" in data
