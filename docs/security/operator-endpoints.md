# Operator HTTP endpoints

SMTP proxy POSTs and explicit stored-account Twitter fetch POSTs require exactly one Authorization header containing Bearer followed by the privately configured OPERATOR_API_TOKEN. Authentication runs before body parsing, sender construction, account loading, or worker submission. Missing server configuration returns HTTP 503; absent, invalid or ambiguous credentials return HTTP 401. Comparisons are constant-time and responses never include credential values.

The committed default is empty and disables these privileged HTTP actions. Set the value privately through the existing settings/prd.py configuration and update trusted callers to send the bearer. This change creates no credential and changes no running service configuration. Scheduled crawling retains its existing behavior.

Authorized SMTP callers retain custom hosts and optional TLS exactly as before. Authorized tweet fetches retain the existing URL/ID normalization and stored-account job.

The unchanged handlers accepted anonymous HTTP requests and constructed an inert SMTP sender or submitted an inert stored-account job. Retained tests use actual handler methods behind disposable local HTTP servers and replace every external effect. They cover anonymous/wrong/duplicate credentials, malformed bodies rejected before parsing, missing configuration, all optional TLS values, and tweet normalization. Run: `python -m unittest tests.test_operator_boundaries -v` (six tests).

Refs #241 and #244.
