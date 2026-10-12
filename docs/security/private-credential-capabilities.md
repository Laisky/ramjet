# Private credential capability boundary

Ramjet previously accepted a caller-selected UID and free/paid flag. Go supplied the first 15 key characters as that UID. Neither active request path verifies an account or enrolls a provider key. The unused JWT verifier and commented account lookup cannot supply ownership authority.

The repaired contract authorizes new private storage through possession of the complete HTTP-safe model credential. Both services derive:

    ckv2- + hex(SHA256("ramjet-private-key-capability-v2\0" + canonical_key))

Ramjet recomputes this value from the raw or Bearer Authorization credential. Incoming UID and free/paid headers grant no private access or paid capacity. Go computes its informational forwarding UID after its existing constant-time credential checks, and keeps generic cached identity unchanged. Raw provider credentials and intentional configured HTTP(S) backends remain supported.

The capability is not an assertion of provider validity, account ownership, paid status, or human identity. Anyone who shares or learns the complete credential shares this capability. Weak or publicly known credentials remain weak capabilities. Provider routing does not change storage scope; rotation creates a distinct scope, with no automatic account or key alias.

## New namespace initialization

A ckv2-looking name is insufficient proof: historical callers could supply arbitrary UIDs. Before private access, Ramjet authenticates __ownership_v2/marker.json under the namespace using a domain-separated HMAC of its scope with the complete credential.

A valid upload may initialize only an empty namespace, after request validation and existing bounded admission. An occupied unmarked namespace, malformed/copied marker, failed marker write/readback, or unknown storage error grants no access. An empty uninitialized namespace lists as empty and deletion is a no-op; private chat/build/activation/publication require an initialized namespace. Metadata lies outside all single-component dataset paths.

The marker authenticates new namespace initialization. It does not authenticate individual historical records or repair unauthorized old writers. Empty-prefix inspection and marker creation are not atomic against a legacy unrestricted writer.

**Any separately authorized rollout must stop all unrestricted legacy writers before initializing new namespaces.** Mixed old/new private writers are unsupported. No production rollout, inventory, migration, account action or configuration change is performed by this source work.

Historical records stay in place. SHA1, truncated-key and arbitrary-UID names are never aliased automatically. Even a historical ckv2-looking namespace fails closed without valid initialization proof. Migration needs an independently verified binding of principal/complete credential to each exact record and permitted operations, plus conflict handling for collisions. Namespace resemblance, a provider-shaped key, or a password for one encrypted record cannot establish ownership of an entire legacy namespace.

## Operations and compatibility

Dataset list/upload/delete, chatbot names/build/activation/publication, private query and password-bound caches use only the server-derived private scope. Object names use full-component validation; traversal, separators, trailing newlines and malformed current-pointer names are rejected before object access or scheduling.

Merged PR288's AES password proof remains required on private warm and cold cache reads. Password knowledge alone cannot select another key's cache. The marker remains necessary before reading a warm cache.

Published links keep query uid and chatbot_name. A legacy UID header can select an owner only on the explicit public-share read route; query uid takes precedence. The reader retrieves named chatbot-share-v2 plaintext archives, never a private current pointer or encrypted private store. Public cache keys are owner/name tuples. Reader credentials, provider and admission remain bound to the reader, rather than the public owner.

Legacy encrypted-original downloads retain their independent object-address/password capability and authenticated decryption. Public/content-addressed files, content-supplied chunk/summary caches, Go sync encryption, memory session IDs and generic anonymous free-tier quota semantics are unchanged. No new claim of a memory read leak or anonymous anti-Sybil control is made.

Unverified Ramjet callers receive the conservative free chunk budget and share a correctly initialized process admission bucket of three permits. The existing eight queued/running upload ceiling remains. Caller UID/free flags, fresh keys, or max_chunks cannot grant a paid budget.

## Local qualification

Run retained ordinary regressions with synthetic storage and model transports:

    python -m unittest tests.test_namespace_ownership tests.test_namespace_provenance tests.test_namespace_public tests.test_private_cache_password tests.test_byok_routing tests.test_byok_image tests.test_upload_jobs

The Go companion adds actual proxy/chunk forwarding controls, fixed scope vectors, forged-cache controls and an opt-in cross-language resolver contract. Heavy suites remain local; existing formatting and fast-unit CI workload is unchanged.

Historical characterization receipts retain their original source revisions. Current manual selectors assert repaired behavior. No test contacts production storage, providers, accounts or quota services.
