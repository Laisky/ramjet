# Upload admission

GPT uploads share eight queued/running job permits per server process. A ninth job receives HTTP 429 before multipart parsing, even when callers use different user IDs. The existing per-user limit remains in effect.

A permit lasts through actual worker completion. Success, worker failure, cancellation, form/body errors, and submission failure release it exactly once. Worker-owned file descriptors remain independent of request cleanup. This is a process capacity bound; authenticating identities and entitlements is a separate concern (Refs #242).

Behavioral reproduction on the unchanged handler admitted nine distinct user IDs. The retained regression now verifies only eight forms and jobs are admitted, and completion makes one slot reusable. The actual recovery decorator preserves HTTP 429 and body-limit responses.

Local qualification: `python -m unittest tests.test_upload_jobs tests.test_upload_bounds -v` (21 tests). Tests use inert schedulers, disposable files and bounded local executors; no provider, SMTP, storage or account calls.
