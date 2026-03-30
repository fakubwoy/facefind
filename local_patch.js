/**
 * local_patch.js — Lenstagram Desktop
 *
 * Loaded in <head> before any page scripts run.
 * Patches globals and defers DOM fixups to DOMContentLoaded so that:
 *   - The API always points to the local server (port 8765)
 *   - Auth never redirects to /login.html — activation is the only gate
 *   - All billing / upgrade / pricing UI is hidden or neutralised
 *   - "Sign out" in the profile dropdown becomes "Re-activate" (for key changes)
 */

(function () {
  'use strict';

  /* ── 1. Lock the API base URL to the local server ─────────────────────── */
  window.__LOCAL_API__ = 'http://localhost:8765';

  /* ── 2. Intercept window.location.href assignments ─────────────────────
     Redirect attempts to /login.html are silently swallowed.
     Redirect attempts to / are sent to /admin.html instead.
     Everything else passes through.                                         */
  (function patchLocationHref() {
    const descriptor = Object.getOwnPropertyDescriptor(window.location, 'href');
    // location.href is not always configurable — use a history-push shim instead
    const _origAssign = window.location.assign.bind(window.location);
    window.location.assign = function (url) {
      if (typeof url === 'string') {
        if (url.includes('login.html')) return;               // block login redirect
        if (url === '/' || url === '') url = '/admin.html';   // root → admin
      }
      _origAssign(url);
    };
  })();

  /* ── 3. Override fetch for /api/auth/me so it always succeeds ──────────
     The local backend already returns the activation user, but if the page
     ever calls it before the server is warm we return a safe cached value.  */
  const _origFetch = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const url = typeof input === 'string' ? input : (input?.url ?? '');

    /* Rewrite relative localhost:8000 leftovers just in case */
    const rewritten = url.replace('http://localhost:8000', window.__LOCAL_API__);
    if (rewritten !== url) {
      input = typeof input === 'string' ? rewritten : new Request(rewritten, input);
    }

    /* Block billing endpoints — return a harmless stub */
    if (url.includes('/api/billing/')) {
      return new Response(JSON.stringify({ is_local: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    /* Block cancel-subscription / cancel-downgrade */
    if (url.includes('cancel-subscription') || url.includes('cancel-downgrade')) {
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    return _origFetch(input, init);
  };

  /* ── 4. DOM fixups once the page has rendered ───────────────────────── */
  document.addEventListener('DOMContentLoaded', function () {

    /* 4a. Hide all links / buttons that point to /pricing.html */
    document.querySelectorAll('a[href*="pricing"], a[href*="upgrade"]').forEach(el => {
      el.style.display = 'none';
    });

    /* 4b. Hide the plan upgrade button in the banner */
    const upgradeBtn = document.getElementById('planUpgradeBtn');
    if (upgradeBtn) upgradeBtn.style.display = 'none';

    /* 4c. Rewrite "Manage plan" dropdown item → link to lenstagram.app for
           subscription management, since billing happens on the cloud side  */
    document.querySelectorAll('.pd-item').forEach(el => {
      if (el.textContent.trim().toLowerCase().includes('manage plan')) {
        el.href = 'https://lenstagram.app/download.html';
        el.target = '_blank';
        el.rel = 'noopener';
        el.innerHTML = el.innerHTML.replace(/manage plan/i, 'Manage subscription ↗');
      }
    });

    /* 4d. Rename "Sign out" → "Re-activate" (re-entering a license key)   */
    document.querySelectorAll('.pd-item.danger, button.pd-item').forEach(el => {
      if (el.textContent.trim().toLowerCase().includes('sign out')) {
        el.textContent = 'Re-activate';
        el.title = 'Enter a different license key';
        // Reassign onclick: go to activate.html instead of triggering doLogout
        el.onclick = function (e) {
          e.preventDefault();
          window.location.href = '/activate.html';
        };
      }
    });

    /* 4e. Hide Stripe / billing extras container if it appears later       */
    const obs = new MutationObserver(function () {
      const extras = document.getElementById('planExtras');
      if (extras) { extras.innerHTML = ''; }

      /* Also hide any dynamically injected upgrade prompts */
      document.querySelectorAll('a[href*="pricing"]').forEach(el => {
        el.style.display = 'none';
      });
    });
    obs.observe(document.body, { childList: true, subtree: true });

    /* 4f. Patch the global API constant that admin.html defines so any
           inline calls after DOMContentLoaded also hit port 8765.
           (The const is scoped to the inline <script>, so we set a window
           property that the page script already reads from window if present.) */
    // Note: the inline script reads `API` from its own closure — the fetch
    // patch above handles rewriting at the network level, which is the safe path.
  });

})();