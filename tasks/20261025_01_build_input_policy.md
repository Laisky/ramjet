# 20261025-01: Repair direct build-input regression checks

## Request and scope

Follow-up to merged PR #286 and issue #240. Base:
`f5f72574fa9007f7b718ff2b2f3ae7b6134d584a`.

The current action/image/tool pins are retained. This change repairs the policy
checks that could miss future regressions and documents repeatable maintenance.
It does not change application source, application dependencies, the Dockerfile,
the bootstrap graph, the release workflow, or deployment commands.

## Reproduction before implementation

Added retained tests to the existing policy suite before replacing its helpers.
The initial expanded suite ran 20 test methods and failed 32 assertions, including
subtests. Existing policy and local wheel/Git behaviors continued to pass.
Concrete failures included:

- `.yaml` workflows and YAML flow/quoted-key forms escaped action inspection.
- Lowercase or indented later `FROM` stages escaped image inspection.
- Comments or echo output supplied the expected strings while the real pip
  invocation stopped consuming the reviewed bootstrap file.
- Missing/late/wrong bootstrap copies and a later unhashed PDM reinstall passed.
- Correctly pinned version updates and several equivalent syntax forms failed.

During independent review of the fix, quoted shell `&&` text and abbreviated pip
binary options exposed two additional checker defects. They were reproduced,
corrected, and added to the retained negative fixtures. Valid unrelated `/tmp/`
copies, `COPY --link`, isolated pip, JSON instructions, quoting, and continuations
are also covered to prevent new false positives.

## Implementation

`.scripts/build_input_policy.py` centralizes parsed YAML references, normalized
Docker stages, exact hashed requirements, and executable bootstrap wiring.
Identity checks are independent of the selected Python/PDM versions. Git fixture
commands have timeouts and explicitly disable signing inherited from user config.

The existing quick workflow gains only PyYAML 6.0.3, matching the application's
existing pin. It keeps the same jobs and test invocation. No new automatic image
build, release gate, or privileged operation is added.

## Validation

Clean temporary environment: Linux x86_64, CPython 3.12.14, Black 26.3.1,
pytest 8.4.2, PyYAML 6.0.3; no inherited application packages.

```sh
python -m unittest discover -s .scripts -p 'test_*.py' -v
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest \
  -p no:cacheprovider -q tests/test_packaging_metadata.py tests/test_deployment.py
python -m black --check .scripts/build_input_policy.py .scripts/test_build_input_policy.py
git diff --check
```

- All 32 script unit tests passed, including 20 build-input policy methods.
- All 11 packaging/deployment-shell tests passed.
- Independent targeted review accepted all 19 expected positive/negative probes.
- Formatting and whitespace checks passed. Black's worker pool required execution
  outside this environment's process-isolated sandbox; its source checks passed.
- The HOWTO's pip-tools regeneration preserved all 38 package pins and 190 wheel
  hashes. All 38 Linux/amd64 wheels downloaded with hash enforcement.
- Docker was unavailable, so no new full image build or production test is claimed.

See [the update HOWTO](../docs/HOWTO_build_input_updates.md) for the exact bootstrap
maintenance commands and platform limitations.

## Compatibility and rollback

The changes affect repository validation and documentation. No data migration or
runtime configuration change is required. Reverting this follow-up restores the
older checks while retaining PR #286's original pins. Issue #240 remains open for
the separately identified nested inputs, deployment authority, and artifact
selection/provenance work.
