# Immutable direct build inputs

Refs #240.

The existing Python 3.12.15 Bookworm base now names the public multi-platform
manifest digest. Privileged workflow action references name the full official
source commits resolved from the existing release tags. This preserves action
versions and the existing delivery script.

PDM was already fixed at 2.26.2; the finding's unversioned-PDM example no longer
matched the current Dockerfile. Its installation now requires exact versions and
published SHA256 wheel identities for the complete 38-package bootstrap graph.
Pip must use wheels and verify hashes before installation. The application's
pyproject.toml, pdm.lock and requirements.txt are unchanged.

Fast offline policy checks inspect `.yml` and `.yaml` workflows as YAML, including
quoted and flow-style references and reusable workflows. External repository
actions require full commit SHAs; container actions require SHA256 digests.
Local action paths remain supported. Docker checks cover external build stages
without fixing policy to one Python version and allow local stage reuse.

The bootstrap policy checks exact hashed requirements and the executable pip
install command that consumes them. Comments containing security flags do not
satisfy the check. Negative fixtures exercise the repository checks themselves,
alongside the retained moved-tag and tampered-wheel behaviors: a local tag can
change content, whereas its recorded commit remains stable, and pip rejects an
inert wheel whose bytes no longer match the reviewed hash. These tests execute
no remote action, privileged workflow or deployment.

The Docker checks cover the repository's direct installation convention:
literal image references, standard backslash continuations, and direct pip or
`python -m pip` commands in shell or exec form. Unsupported bootstrap wrappers
produce a diagnostic. The checker does not execute arbitrary shell code or
prove the effects of arbitrary filesystem writes.

Use [HOWTO: Update direct build inputs](../HOWTO_build_input_updates.md) for the
locally validated bootstrap regeneration procedure, action/base updates, and
platform limitations. The bootstrap file and application dependency graph
remain unchanged by the policy-check improvements.

These pins cover the repository's direct inputs. Action implementations can
still select nested inputs (for example the SSH action's container and default
Buildx/QEMU downloads), and apt repositories and application build inputs have
their existing policies. The project's `pdm-backend>=2.4.0` build requirement is
also outside this bootstrap lock. These checks do not establish reproducibility
of the entire transitive supply chain or close #240's remaining scope, including
deployment authority and source-to-image verification. They do not change the
release/deployment script or add a mandatory release gate.
