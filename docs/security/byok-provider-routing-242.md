# Ramjet BYOK routing and credential audit

Base: Ramjet master `34e2a014a1c61e2e457690c1719de33421b378d6`.
Related report: [issue 242](https://github.com/Laisky/ramjet/issues/242).

## Product decision and scope

Every client-triggered Ramjet model operation must carry an explicit client API
key. Server settings and SDK environment variables must not provide a replacement.
The selected backend is request data and must remain consistent through chat,
embedding, restoration, retrieval and request-triggered background work.

The internal Tailscale architecture and key forwarding are retained. This audit
does not introduce a provider allowlist, a gateway-only trust requirement or a
private-address prohibition. The issue's arbitrary-provider finding is therefore
not resolved by silently restricting intentional backends. User identifiers,
entitlement headers and dataset ownership rules are not migrated here.

## Historical behavior and confirmed defects

BYOK profiles date to `e42f616` (2023-09-14). Provider overrides through
`X-Laisky-Openai-Api-Base` date to `0e04183` (2023-09-23).
The reported scan `9e495a9` and the audited current master retain this design.

Behavioral reproduction used actual repository functions and actual installed
LangChain/OpenAI SDK clients, with synthetic credentials and HTTPX mock transport.
No production key, model request, live account action or external socket was used.

| Path | Reproduced behavior before the fix | Result |
| --- | --- | --- |
| Fresh chat | Caller key and chosen provider are preserved | Preserved |
| Chunk-cache restoration | Caller key retained; backend and embedding model dropped | Request options retained |
| Encrypted/shared chatbot restoration | Server key, SDK default backend/model selected | Explicit caller options required |
| Cached private chain | Prior request key and provider reused | Request-local embedding client |
| Prebuilt query and search | Startup/server embedding credentials used | Request-local embedding client |
| Provider query components | SDK appended operation path after query values | Explicit SDK request query hooks |
| User embedding helper | Selected provider omitted | Shared explicit options |
| Missing index/new-store key | SDK environment key fallback | Rejected before model construction |
| Summary background work | Credential validation was deferred | Rejected before jobs are queued |
| Diagnostics | Provider URL or reflected SDK error could expose a synthetic key | Raw values omitted |
| Shared-index persistence | A serializable embedding adapter's key was included in a pickle | Vector/document archive only |

The plaintext persistence reproduction used a serializable legacy/custom adapter.
It is not evidence that today's SDK client can itself be pickled: current clients
contain non-picklable locks. A separate actual-FAISS regression verifies that its
vector/document archive excludes the real SDK object's synthetic credential.

## Shared resolver and request binding

`ramjet.tasks.gptchat.credentials.resolve_model_credentials(api_key, api_base)`
returns explicit `api_key` and `base_url` SDK options. Keys must be nonempty,
opaque printable ASCII without whitespace; missing values, placeholder sentinels
and control characters fail with generic errors. No provider-specific key prefix
or minimum length is imposed by Ramjet.

`resolve_request_credentials(authorization, api_base)` accepts the existing
bearer and legacy raw-key forms. A root provider URL gains `/v1` once; an existing
`/v1` suffix is retained. HTTP(S) internal hosts and provider URL components are
preserved. Syntax failures do not fall back to another backend or echo the URL.

Chat, classification, summarization and embedding constructors use the shared
SDK adapter `resolve_sdk_credentials`. It separates provider query components
from SDK operation paths and uses public HTTPX request hooks to preserve raw query
bytes, including duplicate/blank values, for synchronous and asynchronous calls.
Fragments are excluded from the HTTP destination. The canonical caller resolver retains the selected URL. Index restoration requires an explicit key. Startup prebuilt data has
an unbound embedding guard instead of a server-key client. Retrieval copies the
FAISS store and binds a fresh request client without mutating cached vectors or
another caller's embedding client. Restored user indices retain
`text-embedding-3-small`, matching new user indices.

Package initialization no longer installs model key/backend settings into
process-global SDK environment variables. Request-triggered jobs carry the
existing user profile in the in-memory executor; no new credential file, queue
secret store or persistent credential storage is introduced. Stored index data
contains vectors, documents, index mappings and scanned-file metadata, not the
embedding client.

Handler failures return a generic error and log the exception category. Known
HTTPX/OpenAI model diagnostics retain severity and HTTP status while omitting
URLs, SDK payloads and exception bodies; reflected-key headers and provider-query
credentials are covered by actual SDK logging regressions.
Image background errors likewise omit upstream text from logs and stored error
objects. Health/static route behavior is unchanged.

Legacy indices with different vector dimensions/models are not migrated. Existing
plaintext whole-store pickle objects also are not automatically migrated to the
archive format; the previous plaintext loader already expected that archive.

## Caller qualification and rollout dependencies

Confirmed caller work is independently reviewable:

- Go Ramjet frontend/proxy: isolated BYOK gate, explicit provider propagation and
  synthetic caller-to-actual-Ramjet-resolver contracts; [Draft PR 80](https://github.com/Laisky/go-ramjet/pull/80).
- [HelloWorld Draft PR 97](https://github.com/Laisky/HelloWorld/pull/97):
  both PDF scanners explicitly pass their existing configured key and provider,
  reject missing keys before store work, and avoid reflected-key error logging.
  It depends on this shared resolver.

The complete account census contains 108 repositories with readable default
heads, including private repositories. The source audit uses exact default-head
snapshots and explicit per-file exclusions; large repository source gaps are
still being resolved. The full private-safe inventory and local evidence remain
outside this public document. Other branches, excluded credential/generated/
binary files and deployed endpoint aliases are not exhaustively verified.

Tracked-source checks of Blog v2, GraphQL and the Cloudflare Workers found no
direct executable Ramjet caller. GraphQL contains API documentation; the speech
Worker uses its own Cloudflare AI binding. Configured service endpoints and
external/compiled alternate clients require runtime route mapping before they
can be excluded as indirect Ramjet callers. No global Ramjet key is substituted
to make such a dependency appear complete.

Caller-first rollout, after all dependencies and product decisions are resolved:

1. Qualify and publish caller changes and resolver dependency as Draft PRs.
2. Review the full caller inventory, configured aliases, existing key routes and
   push-triggered delivery effects before any merge.
3. In a separately authorized rollout, install the resolver API compatibly before
   dependent notebooks and update callers to send their existing explicit keys
   and selected backends.
4. Confirm synthetic propagation and missing-key rejection through each deployed
   caller route, then enable server enforcement.

This task performs no merge, production rollout or deployment. Go and Ramjet
master-push delivery triggers make unqualified merges consequential.

## Remaining decisions and topology limits

Image generation still uses the removed legacy SDK API and a default hardcoded
Azure route. Its error disclosure is fixed here, but image backend modernization
is pending a product decision: caller-selected OpenAI-compatible generation with
existing request parameters, or explicitly configured Azure deployment/version
handling. This PR remains Draft until that path and caller/system review finish.

Go's legacy BYOK identifier contains the first 15 key characters and its accepted
key formats are narrower than Ramjet's opaque-key resolver. Changing those
dataset/quota identities requires a separate migration decision; no UID migration
is performed here. Redirect behavior also remains characterized rather than
changed: a mocked SDK 307 cross-origin redirect strips Authorization but forwards
the embedding query body.

Earlier read-only checks observed the service listener at the internal Tailscale
endpoint with no wildcard/IPv6 host listener, and a harmless health request from
dev succeeded. Repository nginx configuration also declares a forwarded route to
that endpoint; configuration alone is not proof of current public exposure.
A fresh effective-ACL inspection was blocked by SSH host-key verification.
No host-key, ACL, network-security or production setting was changed.

## Validation

Retained tests:

```sh
python -m unittest tests.test_byok_routing tests.test_byok_provider_components -v
python -m pytest -q -p no:cacheprovider tests
```

The expanded focused suite passes 21 tests with real SDK/mock transport and no
external requests. Full offline validation before the last regression additions
passed 100 tests on Python 3.12, including exact dependency-export checks.
Final exact-commit qualification is recorded in the PR.

The shared nonblocking heavy-validation lock serializes suites on dev.
Qualification containers use CPU/memory/time bounds and no network. Pytest tooling
is scoped to its own packages so it cannot override production dependency pins.
No automatic CI suite or production dependency manifest is enlarged.
