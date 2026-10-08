# OAuth session migration

Twitter request tokens now use a strict JSON schema with two bounded strings and
a ten-minute issuance window. Callback processing removes the state before any
provider/account operation, checks the callback token and verifier, and rejects
legacy pickle strings without decoding them. Provider rejection cannot write an
account. Encrypted client-side cookies cannot enforce global replay prevention:
a captured old cookie can be replayed, while the OAuth provider's single-use
request-token exchange must reject a second redemption. A server-side nonce store
would be required for application-owned global replay prevention.

Session setup rejects missing, short, and the tracked public default secrets.
Configure a deployment-specific random secret of at least 32 bytes. The optional
SESSION_SECRET_KEY can separate cookie credentials from SECRET_KEY; otherwise a
validated SECRET_KEY is used with a cookie-specific HMAC-SHA256 derivation.
The new derivation invalidates existing cookies upon application rollout.
Users will sign in again. Never reuse the public test values as deployment keys.

Local acceptance uses harmless reducer sentinels, mocked OAuth/account calls,
and a disposable aiohttp cookie server. No production settings, credentials,
OAuth exchange, live account write, or production request is required:

    python -m unittest tests.test_twitter_oauth -v

Changing the repository does not rotate any deployed key or application. Rollout
and real key rotation are separate operator actions; the push workflow deploys
production, so this security PR must not be merged without rollout authorization.
