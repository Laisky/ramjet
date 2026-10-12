# Immutable first-party build inputs

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

The retained offline regression moves a local Git tag between two benign
commits: the tag changes content while the commit reference stays stable. Fast
policy checks reject short action hashes, mutable refs including inline YAML
steps, digestless Python bases, floating tool versions and missing hashes.
A disposable inert wheel is downloaded successfully with its reviewed hash, then
changed locally; pip rejects the changed bytes before copying the artifact.
No remote action, privileged workflow or deployment is executed by these tests.

These pins cover the repository's direct inputs. Action implementations can
still select nested inputs (for example the SSH action's container and default
Buildx/QEMU downloads), and apt repositories and application distributions have
their existing policies. This PR does not claim reproducibility of that entire
transitive supply chain or close the remaining scope of #240. It does not change
the release/deployment script.
