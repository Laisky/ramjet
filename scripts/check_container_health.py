#!/usr/bin/env python3
"""Accept Ramjet only after its existing local health route returns the expected body."""
from pathlib import Path
import sys
import urllib.request


def listening_ports(tables):
    """Return TCP listening ports without inspecting application settings."""
    ports = set()
    for table in tables:
        for line in table.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 4 and fields[3] == "0A":
                ports.add(int(fields[1].rsplit(":", 1)[1], 16))
    return sorted(ports)


def healthy(ports):
    """Check only container-loopback routes for Ramjet's existing health response."""
    for port in ports:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if (
                    response.status == 200
                    and response.read(64).strip() == b"hello, world"
                ):
                    return True
        except (OSError, ValueError):
            continue
    return False


def main():
    """Probe local listeners without loading production configuration."""
    tables = [Path(name).read_text() for name in ("/proc/net/tcp", "/proc/net/tcp6")]
    if not healthy(listening_ports(tables)):
        print("Ramjet health response not accepted.", file=sys.stderr)
        return 1
    print("Ramjet health accepted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
