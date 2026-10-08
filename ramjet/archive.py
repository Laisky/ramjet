"""Bounded ZIP extraction for locally published prototype uploads."""

import stat
import struct
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


def _entry_path(info):
    """Return a portable relative path while preserving legacy ZIP encodings."""
    name = info.filename
    if not info.flag_bits & 0x800:
        raw = name.encode("cp437")
        try:
            name = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                name = raw.decode("gbk")
            except UnicodeDecodeError as exc:
                raise ValueError("unsupported ZIP filename encoding") from exc
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or PureWindowsPath(name).drive
        or ".." in path.parts
        or not path.parts
        or "\x00" in name
    ):
        raise ValueError("unsafe ZIP path")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode) or info.flag_bits & 1:
        raise ValueError("unsupported ZIP entry")
    return path


def bounded_extract(
    source,
    destination,
    *,
    max_compressed,
    max_expanded,
    max_entries,
    max_ratio,
    timeout,
):
    """Validate and extract an archive within compressed, expanded and time budgets."""
    deadline = time.monotonic() + timeout

    def check_time():
        """Fail extraction when its cooperative wall-clock budget expires."""
        if time.monotonic() > deadline:
            raise ValueError("ZIP extraction timed out")

    with tempfile.TemporaryFile() as compressed:
        copied = 0
        while True:
            check_time()
            chunk = source.read(min(1024 * 1024, max_compressed - copied + 1))
            if not chunk:
                break
            copied += len(chunk)
            if copied > max_compressed:
                raise ValueError("ZIP compressed size limit exceeded")
            compressed.write(chunk)
        # Inspect the fixed end record before ZipFile allocates entry metadata.
        compressed.seek(max(0, copied - 65557))
        tail = compressed.read()
        offset = tail.rfind(b"PK\x05\x06")
        if offset < 0 or len(tail) - offset < 22:
            raise ValueError("invalid ZIP archive")
        end = struct.unpack_from("<4s4H2LH", tail, offset)
        if (
            end[1] != 0
            or end[2] != 0
            or end[3] != end[4]
            or end[4] == 65535
            or end[4] > max_entries
            or offset + 22 + end[7] != len(tail)
        ):
            raise ValueError("unsupported ZIP metadata or entry count")
        compressed.seek(0)
        try:
            with zipfile.ZipFile(compressed) as archive:
                entries = archive.infolist()
                if not entries or len(entries) > max_entries:
                    raise ValueError("ZIP entry count limit exceeded")
                declared = 0
                paths = set()
                planned = []
                for info in entries:
                    check_time()
                    path = _entry_path(info)
                    if path in paths:
                        raise ValueError("duplicate ZIP path")
                    paths.add(path)
                    declared += info.file_size
                    if declared > max_expanded:
                        raise ValueError("ZIP expanded size limit exceeded")
                    if info.file_size > max_ratio * max(info.compress_size, 1):
                        raise ValueError("ZIP compression ratio limit exceeded")
                    planned.append((info, path))
                root = Path(destination)
                root.mkdir(parents=True, exist_ok=True)
                expanded = 0
                for info, path in planned:
                    check_time()
                    target = root.joinpath(*path.parts)
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as entry, target.open("xb") as output:
                        entry_size = 0
                        while True:
                            check_time()
                            chunk = entry.read(
                                min(1024 * 1024, max_expanded - expanded + 1)
                            )
                            if not chunk:
                                break
                            expanded += len(chunk)
                            entry_size += len(chunk)
                            if expanded > max_expanded:
                                raise ValueError("ZIP expanded size limit exceeded")
                            if entry_size > max_ratio * max(info.compress_size, 1):
                                raise ValueError("ZIP compression ratio limit exceeded")
                            output.write(chunk)
        except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
            raise ValueError("invalid ZIP archive") from exc
