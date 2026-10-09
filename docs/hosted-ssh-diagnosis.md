# Hosted SSH deployment diagnosis

This note records the read-only investigation for Ramjet master
`528bfb26071d2900be9385695eeed1f307c6d1fb` and CI run
[37869159270](https://github.com/Laisky/ramjet/actions/runs/37869159270).

## What failed

Both image-publication jobs succeeded. The deploy job ran on a GitHub-hosted
`ubuntu-latest` runner (runner group `GitHub Actions`). Its SSH action exited
after the configured 30-second connection timeout with
`dial tcp ***:***: i/o timeout`. The remote deployment script did not start;
authentication, Compose, application health and dependency compatibility were
not tested by this failed job.

The workflow reads the destination and port from the existing `TARGET_HOST`
and `TARGET_HOST_SSH_PORT` secrets. This investigation did not retrieve or
change those secret values. The masked log does not establish which endpoint
the runner attempted.

## Working production route

The existing Windows route to home uses a private Tailscale address on port 22, follows the
existing host-specific Tailscale interface route, and matches the saved
SSH host key with strict verification enabled. The qualified release was
accepted through that route at 2026-10-09 15:56:11 UTC. It recreated only
`ramjet`; Compose, runtime settings and the 20 unrelated containers were
unchanged.

The verified commands, executed in `/home/laisky/repo/laisky/VPS`, were:

```sh
docker compose -f home-docker-compose.yml pull ramjet
docker compose -f home-docker-compose.yml up -d --no-deps --force-recreate --pull never ramjet
docker compose -f home-docker-compose.yml exec -T ramjet python /app/scripts/check_container_health.py
```

The image and rollback image were verified before recreation. The final
`--pull never` uses the already-pulled and verified image rather than resolving
the mutable tag again during recreation.

## What can be concluded

If the hosted job's destination is this private Tailscale address, the current
workflow does not establish the private-network route required to reach it.
This is an inference: the secret-backed destination mapping remains unknown.
If it points to a different public endpoint, that endpoint's existing routing
and port mapping must instead be verified by the operator.

[GitHub's private-networking documentation](https://docs.github.com/en/actions/concepts/runners/private-networking)
describes private-network access as an additional runner setup. No such setup
appears in the current deploy job. Increasing the SSH timeout, changing the
remote Compose command, changing Python packages, or retrying authentication
cannot establish a missing route.

## Scope boundary and next action

Preserve the healthy production service and its existing configuration.
Do not rotate credentials, edit endpoint secrets, add an overlay/VPN, change
firewalls, or substitute an unverified self-hosted runner as part of dependency
consolidation. There is no evidenced workflow-only repair within that boundary.
Before a future hosted deployment, the operator must confirm that the existing
secret destination maps to the intended host and that an approved runner route
already reaches that destination. Run the existing health helper after any
authorized service-only recreation.

No diagnostic command should print private keys, environment values, tokens,
passwords or secret-backed connection parameters.
