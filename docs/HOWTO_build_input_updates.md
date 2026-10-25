# HOWTO: Update direct build inputs

The Python base digest, workflow action commits, and PDM bootstrap requirements
have separate identities from the application graph in `pdm.lock`. Keep their
updates explicit and reviewable. See [the policy scope](security/build-input-identities.md).

## Regenerate the PDM bootstrap

Run these commands from the repository root in Bash on Linux with CPython 3.12.
They use [pip-tools](https://pip-tools.readthedocs.io/en/stable/) in a temporary
environment and download candidate wheels without installing PDM or application
packages. The input retains every current exact pin, including compatibility
entries that are not needed by Python 3.12, so regeneration does not silently
upgrade packages or prune the existing 38-package graph.

```sh
bootstrap_tmp="$(mktemp -d)"
python3.12 -m venv "$bootstrap_tmp/tools"
"$bootstrap_tmp/tools/bin/python" -m pip install --index-url https://pypi.org/simple \
  pip==26.2.2 pip-tools==7.6.2 PyYAML==6.0.3
awk '/^[A-Za-z0-9_.-]+==/ { print $1 }' .scripts/pdm-bootstrap.txt \
  > "$bootstrap_tmp/pdm-bootstrap.in"

"$bootstrap_tmp/tools/bin/pip-compile" \
  --no-config --generate-hashes --allow-unsafe --strip-extras \
  --no-header --no-annotate --no-emit-options \
  --index-url https://pypi.org/simple \
  --pip-args='--only-binary=:all: --no-cache-dir' \
  --cache-dir "$bootstrap_tmp/cache" \
  --output-file "$bootstrap_tmp/pdm-bootstrap.txt" \
  "$bootstrap_tmp/pdm-bootstrap.in"

"$bootstrap_tmp/tools/bin/python" -m pip --isolated download \
  --index-url https://pypi.org/simple --no-cache-dir \
  --only-binary=:all: --require-hashes \
  -r "$bootstrap_tmp/pdm-bootstrap.txt" --dest "$bootstrap_tmp/wheels"

diff -u .scripts/pdm-bootstrap.txt "$bootstrap_tmp/pdm-bootstrap.txt"
```

Review version and hash changes; comments and blank lines may differ. To make an
intentional upgrade, edit the selected exact pins in the temporary `.in` file and
rerun compilation and download. Resolve incompatible constraints explicitly;
avoid a broad `--upgrade`. If changing PDM itself, also update the matching
version in `.scripts/update_dependencies.py` and dependency-maintenance examples.
After accepting the candidate, copy it to `.scripts/pdm-bootstrap.txt`, retain
its scope comments, and run the local policy tests:

```sh
"$bootstrap_tmp/tools/bin/python" -m unittest discover -s .scripts -p test_build_input_policy.py -v
```

On 2026-10-25, this procedure with CPython 3.12.14, pip 26.2.2, and pip-tools
7.6.2 regenerated exactly the same 38 pins and 190 wheel hashes. All 38 wheels
downloaded successfully with hash enforcement on Linux/amd64. No package from
the bootstrap or application graph was installed for this check.

## Update action commits and the Python base

Resolve a chosen action release to its full commit in the official upstream
repository, inspect the change, and retain a readable version comment. Update
all usages found with `rg -n 'uses:' .github/workflows`. For Python, verify the
chosen official image's manifest digest and target platforms, then update the
`FROM` reference with both its readable tag and digest. Run the same local policy
tests after either change. A full commit or digest fixes identity; it does not
prove that the selected code is trustworthy or that nested downloads are pinned.

## Platform and validation boundaries

The current bootstrap has compatible published wheels for Linux/amd64 and
Linux/arm64 on CPython 3.12. Only amd64 wheel downloads were exercised in the
check above; arm64 metadata availability is not a full image build. Although
the base image index also includes arm/v7 and ppc64le, `msgpack==1.2.3` lacks
compatible CPython 3.12 wheels there. The wheel-only bootstrap therefore cannot
build on those targets as written. Prior full application support on those
targets is unverified; the release workflow does not currently request them.

For a changed tool graph or base image, use a bounded local image build and
relevant offline application checks to assess compatibility. Metadata and hash
checks alone do not demonstrate runtime behavior. Hosted CI remains limited to
the existing fast checks; this procedure adds no release gate or deployment step.

Issue #240 still tracks broader supply-chain work: nested action downloads,
apt and application build inputs, deployment authority, and source-to-image
verification. This procedure does not change the delivery script or close that
remaining scope. Application dependency changes use
[the separate maintenance procedure](dependency-maintenance.md).
