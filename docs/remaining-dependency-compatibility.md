# Remaining dependency compatibility

Baseline: Ramjet master `528bfb26071d2900be9385695eeed1f307c6d1fb`.
The qualified and deployed graph uses Tweepy 4.16.0, OAuthlib 3.3.1,
Kipp 0.3.2, xxhash 1.4.4 and LangSmith 0.6.3.

## OAuthlib proposal #265

[The official Tweepy 4.17.0 release metadata](https://pypi.org/pypi/tweepy/4.17.0/json)
still requires `oauthlib>=3.2.0,<4`; replacing only OAuthlib with 4.0.0
violates that contract. Current Tweepy master retains the same cap.
[Tweepy PR2245](https://github.com/tweepy/tweepy/pull/2245) proposes widening it
to permit OAuthlib 4, but it is not merged or released.

[OAuthlib 4 release notes](https://github.com/oauthlib/oauthlib/releases/tag/v4.0.0)
identify provider-side breaking changes. The
[PKCE timing advisory](https://github.com/oauthlib/oauthlib/security/advisories/GHSA-xpv3-w29h-x7cv)
concerns the OAuth2 Authorization Code grant's provider implementation.
Ramjet uses Tweepy's OAuth1 client, request/access-token exchange and API client;
it does not instantiate that OAuth2 grant provider. This excludes an asserted
direct Ramjet exploit path for that advisory, and does not remove the affected
package or dismiss the dependency alert.

The new dependency-runtime tests exercise the real client's prepared request,
a frozen public HMAC-SHA1 signature, callback, request-token parsing,
access-token exchange, authenticated API response and error handling.
Only the HTTP transport is replaced; no provider or account is contacted.
They supplement the existing callback-state security tests.

## LangSmith proposal #274

[LangSmith 0.8.18 metadata](https://pypi.org/pypi/langsmith/0.8.18/json) requires
`xxhash>=3.0.0`. [Kipp 0.3.2 metadata](https://pypi.org/pypi/kipp/0.3.2/json)
requires `xxhash~=1.3`, which excludes all 3.x releases. LangSmith also requires
`websockets>=15` (new to this graph) and `uuid-utils>=0.12` (already satisfied
by the deployed 0.17.1). Neither overriding
Kipp's cap nor removing Kipp is an acceptable generated-lock repair.

Kipp's actual hash consumer is `calculate_args_hash`, used by its timeout
cache. It calls the established `xxh32(...).hexdigest()` API. A maintainer
patch must qualify that API and preserve the existing seven golden cache keys,
argument separation, hits and expiry across the supported hash versions.
The Ramjet tests retain these contracts independently of any metadata change.

[Current LangSmith 0.14.6 metadata](https://pypi.org/pypi/langsmith/0.14.6/json)
also introduces a different HTTP client dependency.
It is not a substitute for qualifying the requested 0.8.18 graph against
Ramjet's existing LangChain/OpenAI stack.

## Qualification results

All behavioral checks used official, SHA-256-verified PyPI artifacts and
blocked provider networking. The Kipp source patch changes only its declared
xxhash cap to `>=1.3,<4`.

| Check | Python 3.10.14 | Python 3.12.15 |
| --- | --- | --- |
| Ramjet complete baseline suite, before final tracing contract addition | 86 passed | 86 passed |
| OAuthlib 4.0.0 isolated behavior suite | 85 passed | 85 passed |
| LangSmith 0.8.18 + xxhash 3.8.1 + websockets 15.0.1 + minimum uuid-utils 0.12.0 isolated behavior suite | 85 passed | 85 passed |
| Kipp cache/decorator suite with xxhash 1.4.4, 2.0.2 and 3.8.1 | 71 passed per version | 71 passed per version |
| Final six SDK/cache/tracing contracts across baseline, OAuthlib 4, Tweepy 4.17 + OAuthlib 4, and LangSmith 0.8.18 profiles | 6 passed per profile | 6 passed per profile |
| Normal complete graph resolution using published Kipp | Rejected incompatible cap | Rejected incompatible cap |
| Normal complete graph resolution using the actual patched maintainer Kipp wheel | Passed | Passed |

The experimental suites intentionally exclude only
`test_installed_dependencies_match_export`: the production export remains
unchanged and must reject experimental version differences. Baseline
qualification retains that test. Cap failures were reproduced separately with
Ramjet's real dependency validator; normal resolution uses all frozen baseline
pins except the reviewed LangSmith/xxhash changes and the patched Kipp source,
retains uuid-utils 0.17.1, and adds websockets 15.0.1. The maintained Kipp source
also declares zipp, resolved normally. These are disposable qualification
inputs; they do not create a production PDM lock.

The final runtime contracts retain baseline uuid-utils 0.17.1 and additionally
serialize a real LangSmith trace
through its HTTP client, retaining public Unicode inputs, outputs, run identity
and project association. Only the transport is replaced. Passing the isolated
OAuthlib experiments does not make Tweepy's published cap disappear.

## Qualification boundary

The normal dependency validator must continue checking upstream declarations
on Python 3.10 through 3.14. Do not commit a lock or export produced by ignoring
a transitive cap. Until the compatible source is selected and qualified, keep
the valid deployed dependency graph and the two original proposals visible.
Experimental wheel overlays are only isolated offline test environments; they
are never a production lock or a deployment instruction.

Full image/behavior qualification uses the shared nonblocking
`/tmp/laisky-heavy-validation-20261008.lock`, bounded CPU, memory and time,
synthetic settings and blocked provider networking. CI remains formatting and
fast units. Hosted SSH diagnosis is separate in
[hosted-ssh-diagnosis.md](hosted-ssh-diagnosis.md).
