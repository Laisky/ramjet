# Operator HTTP endpoints

SMTP proxy POSTs and explicit stored-account Twitter fetch POSTs require exactly one Authorization header containing Bearer followed by the privately configured OPERATOR_API_TOKEN. Authentication runs before body parsing, sender construction, account loading, or worker submission. Missing server configuration returns HTTP 503; absent, invalid or ambiguous credentials return HTTP 401. Comparisons are constant-time and responses never include credential values.

The committed default is empty and disables these privileged HTTP actions. Set the value privately through the existing settings/prd.py configuration and update trusted callers to send the bearer. This change creates no credential and changes no running service configuration. Scheduled crawling retains its existing behavior.

Authorized SMTP callers retain custom hosts and optional TLS exactly as before. Authorized tweet fetches retain the existing URL/ID normalization and stored-account job.

The unchanged handlers accepted anonymous HTTP requests and constructed an inert SMTP sender or submitted an inert stored-account job. Retained tests use actual handler methods behind disposable local HTTP servers and replace every external effect. They cover anonymous/wrong/duplicate credentials, malformed bodies rejected before parsing, missing configuration, all optional TLS values, and tweet normalization. Run: `python -m unittest tests.test_operator_boundaries -v` (eight tests).

Refs #241 and #244.

## Interactive Twitter caller

The Twitter form now submits to its same-origin relative fetch endpoint with a
transient operator credential. The credential input has no form-data name and is
cleared immediately on submission. The caller sends exactly one bearer header,
omits cookies and referrers, rejects redirects and cross-origin targets, and
never writes credentials to URLs, logs, localStorage or sessionStorage. Pending
submissions cannot queue a second job. Status messages use text only.

This operator page uses local JavaScript and styles and disables inherited remote
modules. The new default-preserving template block leaves the search page's
rendered module output byte-for-byte identical. Other pages keep their existing
module behavior. If JavaScript is unavailable, the ordinary form remains denied
and never falls back to submitting a credential in the request body.

The old actual multipart HTTP caller returned 401 with no privileged effects;
the unchanged browser form sent no Authorization header. Retained qualification
now covers eight HTTP/template contracts and seven Chromium cases using only
inert local fixtures and intercepted requests. Run the browser suite after
rendering the actual inherited template to a local file:

```sh
OPERATOR_HTML_FIXTURE=/path/to/rendered-operator-page.html \
  PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
  node scripts/security/twitter-operator-browser.mjs
```

The existing automatic fast gate is unchanged. No operator credential is
created, no running configuration is read or changed, and no service is deployed.
