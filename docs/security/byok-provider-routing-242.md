# Ramjet BYOK routing audit for issue 242

Audited Ramjet master `34e2a014a1c61e2e457690c1719de33421b378d6`.
This change preserves the internal Tailscale architecture and caller API-key
forwarding. It does not decide which caller-selected provider destinations should
be permitted.

## Historical and current behavior

- `e42f616` (2023-09-14) constructs the user profile from the supplied bearer key
  and derives the default user identifier from its hash.
- `0e04183` (2023-09-23) adds `X-Laisky-Openai-Api-Base`; the parser appends
  `/v1` to the supplied root URL.
- Scanned `9e495a9` and current master retain this BYOK and provider override.
  Neither parser verifies the key with the provider before constructing the
  profile. The upstream provider validates the key when an operation calls it.
- Current root-URL normalization also appends another `/v1` when callers supply
  a URL already ending in `/v1`. This pre-existing compatibility behavior is
  unchanged.
- A direct caller that can reach Ramjet can select an arbitrary HTTP(S)
  provider through the header. Internal placement alone does not establish that
  every selected destination is trusted. No new destination allowlist, gateway
  requirement or private-address prohibition is introduced here.

## Confirmed routing defects and fix

Retained tests execute the repository parser, cache restoration and request
methods with synthetic settings. They use actual LangChain/OpenAI SDK clients
and an HTTPX mock transport; no socket or production credential is used.

1. Fresh chat correctly sends the caller's key to the selected provider.
2. A chunk-cache hit keeps the key but drops the provider URL and embedding model.
3. Encrypted and shared chatbot restoration explicitly selects the server
   `OPENAI_TOKEN`, loses the provider URL and falls back to the SDK embedding model.
4. An already cached private chain retains the previous request's embedding
   credential and destination.
5. Provider URL diagnostics can expose credentials embedded in URL user info.

Cache and user-index restoration now retain the caller key, provider and
`text-embedding-3-small`, matching the existing fresh embedding path. A request
uses a shallow store copy with its own embedding client; it shares stored vectors
and documents without modifying the shared cached client. User data identifiers,
dataset selection, quotas, provider normalization and authorization policy stay
unchanged. Legacy server-owned index deserialization retains its existing
server-key, SDK provider and model defaults when caller options are omitted.
Diagnostic messages omit raw provider URLs.

The model fix restores the model already used for new user indices. Legacy
indices created with different embedding dimensions/models are not migrated by
this change.

## Caller tracing

The current Go Ramjet frontend and proxy are the concrete callers:
`go-ramjet` master `bb26ec77f68fe5a28535f6def19635dc6014b8bb`.

- Modern frontend `web/src/pages/gptchat/utils/api.ts` sends the supplied API
  token in Authorization for upload/list/delete/chatbot operations and sends
  an optional `X-Laisky-Api-Base`.
- The legacy `templates/js/chat.js` also passes its configured API token.
- `getUserByToken` recognizes BYOK keys, keeps the supplied OpenAI and image
  credentials, and defaults to the configured OneAPI provider. A valid BYOK
  override accepts HTTP(S), while rejecting user info, query and fragment.
- `setUserAuth` forwards the resolved key and maps the provider into
  `X-Laisky-Openai-Api-Base`. `RamjetProxyHandler` sends the request to its
  server-configured `RamjetURL`; the provider header is not the proxy destination.
  The embedding chunk caller also applies this helper.
- Freetier intentionally uses the configured server credential and ignores BYOK
  provider overrides. This is distinct from accidental substitution during
  restored BYOK retrieval.
- Go's existing BYOK identifier is the first 15 key characters. Changing it
  would migrate dataset/quota identities, so this patch does not change it.
  Separate Go logging/error regressions are being qualified.

Tracked-source searches found no executable Ramjet caller in Blog v2
`d6f59f62fc91ccc86e41b32aad0aa4169e7ece2b`, GraphQL
`3952e43041eebd117321030ea3615143690d51ea`, or Cloudflare Workers
`4d57b1e67f576319c9389bf859b588aa1f5eeb75`. GraphQL has a Ramjet API reference
document. The speech Worker uses a Cloudflare AI binding, not a Ramjet API.

## Redirects and topology limits

The installed SDK's default HTTP client follows redirects. A retained synthetic
307 test confirms that a cross-origin redirect strips Authorization but forwards
the embedding query body. Redirect policy is characterized here, not changed.

Earlier read-only deployment checks found the Ramjet host listener bound to
Tailscale address `100.69.166.78:22280`, with no wildcard/IPv6 host listener, and a
harmless dev-to-tailnet health request succeeded. This confirms the observed
internal service endpoint, not tailnet-wide ACL authorization or absence of
every external gateway.

A new read-only attempt to inspect home2's effective Tailscale packet filter was
blocked by SSH host-key verification. No host-key, network-security or ACL setting
was changed. Tailnet ACL scope remains unverified.

## Validation

The original cache/restore reproduction produced three failing assertions
(cache, encrypted restore and shared restore), with fresh chat passing.
The additional cached-private-chain reproduction failed before request binding.
The raw-URL log reproduction observed the synthetic credential in the actual
diagnostic function before redaction.

Run retained transport contracts with:

```sh
python -m unittest tests.test_byok_routing -v
```

The test suite uses synthetic defaults, checks that caller credentials are not
replaced, verifies diagnostic privacy, preserves legacy defaults, and records
redirect behavior. Full offline tests run separately under the shared
nonblocking heavy-validation lock. No automatic CI suite is enlarged.
