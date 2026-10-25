# CI testing policy (2026-10-08)

The maintainer requests minimal automatic checks: Black 26.3.1 checks Python
files changed since the event's base commit, plus the gate and packaging-test
files, and four existing independent packaging contracts run on Python 3.12.
The explicit test names protect license metadata, exclusion of private production
settings, Docker dependency-layer ordering, and lock/export consistency.
Twenty-four historical Python files failed a repository-wide Black check at
adoption; changed-file formatting makes this debt explicit rather than rewriting
unrelated application files. The runner verifies the base commit exists.

```sh
python -m pip install black==26.3.1 pytest==8.4.2 PyYAML==6.0.3
python -m unittest discover -s .scripts -p 'test_*.py' -v
python .scripts/fast_ci.py --base "$(git rev-parse HEAD^)" --evidence /tmp/ramjet-fast-ci
```

These four packaging tests need no runtime settings or external AI dependencies.
`--noconftest` deliberately isolates this named packaging-only subset from the
full suite's aiohttp/LangChain synthetic fixtures; no other tests are claimed to
run by this packaging subset. Collection must find all four exact node IDs,
execution must exit zero, and the completion report must contain exactly four
passing tests with no skips/errors/failures. Native exits and timings are retained
on failure too.

The same quick job runs the local failure-reporting, dependency-updater, and
build-input regression tests. PyYAML parses both supported workflow extensions
and normal YAML layouts; it is a small test dependency already used by the
application. The build-input tests inspect configuration and exercise inert
local wheels and Git repositories. They do not execute GitHub Actions or add
networked image builds to automatic CI.

The prior complete Python 3.10/3.12 offline suite and frozen production-image
acceptance jobs are unchanged in `offline-full.yml` and now run through Actions
> complete offline qualification > Run workflow. Run relevant full qualification
manually on dev/staging before promotion. This explicit cadence amendment
supersedes prior descriptions of the complete suite as automatic.

```sh
python -m pip install -r requirements.txt pytest==8.4.2 tomli==2.3.0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export TIKTOKEN_CACHE_DIR=/tmp/ramjet-tokenizer-assets
python -c 'import tiktoken; tiktoken.get_encoding("gpt2")'
python -m pytest -q tests
```

For image qualification, manually dispatch `offline-full.yml`; its read-only,
network-disabled acceptance commands and all fixture/assertion behavior remain
intact. `ci.yml`, publishing, deployment health/failure/rollback logic, secrets
and security/protection settings are unchanged. The quick job has a five-minute
timeout; actual timing and setup limitations are recorded in the PR.
