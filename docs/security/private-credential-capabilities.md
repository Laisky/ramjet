# Private ownership and separate model credentials

Private data now requires an independently verified Laisky SSO account and an exact server-local registry grant. Model API keys remain transient provider credentials. Sharing a provider key does not share a namespace; rotating that key does not move an account's namespace. Caller UID/free headers, guessed keys, legacy names and object-store markers provide no ownership authority.

## Session verification

Private entries accept one explicit bearer token in X-Laisky-SSO-Token. Ramjet verifies the original token independently of Go. Local admission requires EdDSA, issuer laisky-sso, finite unexpired expiration, a valid issue time, matching canonical nonzero UUID sub and uid, and the current eight-field claim contract. Legacy/algorithm fallback, malformed or duplicate claim fields and caller-supplied identity headers are refused before issuer contact.

Every incoming private request then asks WhoAmI at fixed https://sso.laisky.com/query. Transport uses HTTPS, a five-second total timeout, a 64 KiB response limit, no environment proxy and no redirects. The returned stable account ID must match the admitted claims. Issuer failure, malformed responses, revocation or identity mismatch fail closed. Successful sessions are not cached. The worker receives a verified principal and expiry, never the session token, and rechecks expiry and its exact registry grant before private work.

Go uses the existing CV WhoAmI contract through a shared verifier while applying the stricter private admission policy. It forwards the original session only for declared private operations to the configured Ramjet backend with redirects disabled. Public share/model/OneAPI requests strip this header. Both services reject identifiable Laisky SSO sessions as model keys or provider URL options, while preserving intentional configured HTTP(S) backends and unrelated provider JWTs. Compound credential/option strings and case-preserved URL authorities are checked too; URL components decode once, without recursive decoding or prompt/content inspection. The synthetic admission regressions enforce this separation policy and do not claim production disclosure.

## Operator registry and fresh namespaces

PRIVATE_NAMESPACE_REGISTRY_PATH is an optional server setting. It is unset by default; business requests never create a registry file or schema. An unavailable registry returns a fixed failure before any session transmission, object access or job submission. No runtime registry or setting was created in this task.

The SQLite authority lives outside all caller-writable object paths. It validates its exact versioned schema, file ownership and write permissions, protected parent permissions, and pinned bucket/root target. Each grant binds a canonical verified principal, an independently generated pns3 UUID namespace, one exact operation and the storage target. Object-store ownership markers are never read or trusted.

Only upload may initialize a fresh namespace, after multipart validation and bounded admission. Fresh initialization requires an explicit persisted operator policy, disabled by default. BEGIN IMMEDIATE serializes ownership creation; an existing exact upload grant is reused. A pre-existing owner without upload permission cannot escalate. The server generates the candidate independently of account IDs and keys, checks that exact object prefix is empty, and atomically records the owner and fresh-operation grants. Occupied prefixes, collisions, failed inspection, absent grants and invalid authority fail closed.

SQLite and S3 do not form a distributed transaction. Any separately authorized rollout must stop unrestricted legacy writers and drain their jobs before allowing fresh initialization. The runtime issuer availability/redirect contract, protected registry placement, old-writer drain and independently audited historical account mappings are rollout gates. They do not authorize configuration, deployment or migration in this source task.

Historical grants and import are unsupported and disabled. Existing arbitrary-UID, truncated-key, SHA1 and ckv2 objects remain unchanged and are never adopted automatically, even if a caller knows a weak old model credential or has an SSO account. Future migration must separately bind each exact historical namespace and permitted operation to an independently verified account, resolve collisions, and preserve records. Namespace resemblance or one encrypted object's password cannot supply that binding.

## Operations and retained compatibility

Dataset list/upload/delete, chatbot list/build/activation/publication and private queries require their exact grants. List/delete/chatbot-list metadata needs only SSO; model-using operations also need separate BYOK credentials. SSO does not grant paid status. Callers retain conservative free chunk capacity, three shared free admission permits and the existing eight queued/running upload ceiling.

Private object names and stored current-pointer names are validated before access or scheduling. Store loaders/savers validate their own component inputs. Local chatbot archive staging uses fixed filenames rather than request-selected filesystem paths. Named objects and archive bytes remain compatible.

Merged PR288's AES password proof remains required on private warm and cold cache reads. Password knowledge alone does not authorize a namespace. Private cache and processing state use versioned tuples containing the registry namespace, excluding legacy string and previous key-derived cache entries.

Published links retain query uid and chatbot_name. A legacy UID header can select an owner only on the explicit public-share read route; query uid takes precedence. That route reads named plaintext chatbot-share-v2 archives and never private current pointers or encrypted stores. Public cache keys remain owner/name tuples. Reader model credentials and limits belong to the reader.

Encrypted-original downloads preserve their separate exact-object/password capability and authenticated decryption. Public/content-addressed files, content-supplied chunk/summary caches, Go sync encryption, memory session IDs and generic anonymous free-tier quota semantics retain their existing contracts.

## Regression evidence

Ordinary regressions use synthetic signed sessions, an in-process issuer, disposable registries, storage and model transports:

    python -m unittest tests.test_sso_identity tests.test_private_registry tests.test_namespace_ownership tests.test_namespace_provenance tests.test_namespace_public tests.test_private_cache_password tests.test_byok_routing tests.test_byok_image tests.test_upload_jobs

The retained cases cover forged/legacy/expired/revoked sessions, issuer failure, stable-account mismatches, guessed weak keys without or with another account, shared provider keys, key rotation, model/session separation, atomic creation, conflicts, missing operation grants, old markers/caches and public/password compatibility. Go adds actual proxy transport controls; frontend tests cover callback state, session scrubbing and stale dataset response rejection.

Historical characterization receipts retain their original source heads. The weak-key capability reproduction applies only to prior head 1bbcc7f9ab1d70cca23385fb05c5d93182c4b8cb and is permanently skipped on this account-bound source. It is evidence of the retired contract, not a passing acceptance criterion. Current retained tests require rejection of that attack.

Heavy qualification remains local. Existing formatting and fast-unit CI workload is unchanged. No test uses production accounts, storage, model providers, live credentials or blockchain transactions. This change remains source-only and does not create issuer credentials, migrate data, modify host security settings or deploy services.
