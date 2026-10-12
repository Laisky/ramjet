# Independent request credential helper

Import `resolve_request_credentials` from `ramjet.credentials` to validate an explicit caller key and configured provider URL before loading model tasks.

The standard-library module exports the existing `require_api_key`, `resolve_model_credentials` and `resolve_request_credentials` contracts. It never reads environment variables, imports task/settings/SDK modules, constructs clients, logs keys or stores them. HTTP and HTTPS backends, including configured internal addresses and query/fragment values, remain supported. The key is returned separately from the URL. Omitted backend arguments retain the existing OpenAI default; applications requiring an explicit destination must reject omission before calling.

This helper does not authenticate a principal, authorize a UID or grant access to an index or object. It does not adopt the pending BYOK router/provider changes or resolve their ownership prerequisites. Existing task initialization remains unchanged. Callers should capture validated options before importing existing model tasks, which can still change process-global SDK environment values.

The module is included by the existing PDM `ramjet` package inclusion; no dependency, lock, version or deployment change is needed to build it. Installing a compatible source/package is a consumer prerequisite.

Local verification:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_request_credentials.py -v
```

The retained subprocess import contract rejects all task/settings imports, preserves a synthetic environment and uses no production settings or networking. Heavy suites remain local; automatic CI is unchanged.
