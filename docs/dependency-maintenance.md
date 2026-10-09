# Dependency maintenance

Production installs `pdm.lock` with `--frozen-lockfile`. `requirements.txt` is a
generated export retained for dependency visibility and security update proposals.
A requirements-only bot PR does not update production and must not be merged as a
completed fix. The fast packaging contract rejects lock/export drift.

## Existing security coverage and supported update route

Retain existing Dependabot alerts and security proposals. Do not add ignore rules,
exclude the generated export, disable security updates, or give a workflow broader
write permissions to make those proposals green. Dependabot documents PEP 621 and
requirements support, but its PDM lockfile support request remains open:

- [GitHub supported Python manifests](https://docs.github.com/en/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories#pip-and-pip-compile)
- [Dependabot PDM support request](https://github.com/dependabot/dependabot-core/issues/3190)

In an isolated worktree based on current master, inspect each proposal, upstream
release notes, supported Python/server versions, and dependency constraints. Use
explicit reviewed stable pins with the same PDM version as the Dockerfile:

```sh
# Print the update plan without changing files.
uvx --from pdm==2.26.2 python .scripts/update_dependencies.py \
  pyjwt==2.15.1 pypdf==6.19.0 urllib3==2.8.0 multidict==6.9.1 pymongo==4.18.3

# Resolve selected versions, reuse unaffected pins, and regenerate the export.
uvx --from pdm==2.26.2 python .scripts/update_dependencies.py --apply \
  pyjwt==2.15.1 pypdf==6.19.0 urllib3==2.8.0 multidict==6.9.1 pymongo==4.18.3
```

The command refuses dirty dependency metadata and unknown production packages,
checks declared lock dependencies across Python 3.10–3.14 before export (PDM overrides
can otherwise bypass upstream version caps), does not sync/install runtime
dependencies, and restores metadata after a failed
resolver/export/check, and preserves the checkout's saved interpreter selection.
It uses supported [PDM targeted updates and overrides](https://pdm-project.org/en/latest/reference/cli/#update)
and [production export](https://pdm-project.org/en/latest/usage/lockfile/#export-locked-packages-to-alternative-formats).
Review all transitive changes; resolver success alone is not qualification.

Build the actual frozen Dockerfile with bounded local resources and run the offline
suite on production Python and the supported Python 3.10 baseline. Exercise the
application paths touched by the updates, including JWT validation, PDF ingestion,
HTTP headers/transport, and MongoDB/BSON. Check deployment server compatibility:
PyMongo 4.18 drops MongoDB 4.2 support; the current deployment was read-only verified
as MongoDB 6.0.28. Other deployments must verify their own server version.
Keep hosted CI to formatting and fast offline contracts.

Only close original security proposals after the equivalent qualified lock update
is merged, citing the replacement commit/PR and exact versions. Verify published
image and running service separately. Delete only redundant proposal branches
after exact-head and recoverable Git-history verification.

## Reviewable alternatives

[Renovate's PEP 621 manager](https://docs.renovatebot.com/modules/manager/pep621/)
officially supports PDM lockfile maintenance. Adopting it would require a separate
approved app/service decision and export integration; it has not been installed.
Migrating to Dependabot-supported `uv.lock` is another separately reviewed package
manager migration. Neither alternative is required for the current local route,
and neither justifies removing existing security coverage.
