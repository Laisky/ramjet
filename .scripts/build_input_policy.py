"""Inspect direct build inputs without executing workflows or Docker commands."""

import json
from pathlib import Path, PurePosixPath
import re
import shlex

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
import yaml

LOCK_SOURCE = ".scripts/pdm-bootstrap.txt"
LOCK_TARGET = "/tmp/pdm-bootstrap.txt"
DIGEST = re.compile(r"[^@\s]+@sha256:[0-9a-fA-F]{64}")


def workflow_files(root: Path) -> list[Path]:
    """workflow_files returns both supported workflow extensions below root."""
    directory = root / ".github/workflows"
    return sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")])


def mutable_actions(source: str) -> list[str]:
    """mutable_actions returns movable job/step refs, raising for malformed YAML."""
    try:
        document = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid workflow YAML: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("jobs"), dict):
        raise ValueError("workflow jobs must be a mapping")
    if not document["jobs"]:
        raise ValueError("workflow must define at least one job")
    references = []
    for name, job in document["jobs"].items():
        if not isinstance(job, dict):
            raise ValueError(f"job {name} must be a mapping")
        entries = [job]
        if "steps" in job:
            if not isinstance(job["steps"], list):
                raise ValueError(f"job {name} steps must be a list")
            entries.extend(job["steps"])
        elif "uses" not in job:
            raise ValueError(f"job {name} must define steps or uses")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"job {name} steps must contain mappings")
            if "uses" not in entry:
                continue
            reference = entry["uses"]
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"job {name} uses must be a nonempty string")
            reference = reference.strip()
            if reference.startswith("./"):
                continue
            if reference.startswith("docker://"):
                pinned = DIGEST.fullmatch(reference.removeprefix("docker://"))
            else:
                pinned = re.fullmatch(r"[^@\s]+@[0-9a-fA-F]{40}", reference)
            if not pinned:
                references.append(reference)
    return references


def _logical_lines(source: str):
    """_logical_lines yields backslash-continued lines, removing full-line comments."""
    pending = ""
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        yield pending + line
        pending = ""
    if pending:
        raise ValueError("unfinished line continuation")


def _instructions(source: str):
    """_instructions yields normalized Docker instructions for standard escapes."""
    if re.search(r"^\s*#\s*escape\s*=\s*[^\\\s]", source, re.I | re.M):
        raise ValueError("unsupported Docker escape directive; use backslash escapes")
    for line in _logical_lines(source):
        match = re.fullmatch(r"([A-Za-z]+)\s+(.+)", line)
        if not match:
            raise ValueError(f"malformed Docker instruction: {line}")
        instruction, arguments = match.groups()
        if instruction.upper() in {"RUN", "COPY"} and not arguments.startswith("["):
            lexer = shlex.shlex(arguments, posix=False, punctuation_chars="<")
            lexer.whitespace_split = True
            if "<<" in lexer:
                raise ValueError(
                    "Docker heredocs need an explicit policy implementation"
                )
        yield instruction.upper(), arguments


def _base(arguments: str) -> tuple[str, str | None]:
    """_base returns a FROM image and optional stage alias, rejecting unknown forms."""
    words = shlex.split(arguments)
    if words and words[0].startswith("--platform="):
        words.pop(0)
    if len(words) == 1 and not words[0].startswith("--"):
        return words[0], None
    if len(words) == 3 and words[1].upper() == "AS":
        return words[0], words[2].lower()
    raise ValueError("FROM must name an image and optional AS stage")


def immutable_base(source: str) -> bool:
    """immutable_base checks all external stages for digests, allowing stage reuse."""
    aliases = set()
    found = False
    try:
        for instruction, arguments in _instructions(source):
            if instruction != "FROM":
                continue
            found = True
            base, alias = _base(arguments)
            if (
                base != "scratch"
                and base.lower() not in aliases
                and not DIGEST.fullmatch(base)
            ):
                return False
            if alias:
                if alias in aliases:
                    return False
                aliases.add(alias)
    except ValueError:
        return False
    return found


def hashed_bootstrap(source: str) -> bool:
    """hashed_bootstrap checks exact versions and SHA256 hashes, including PDM."""
    found_pdm = False
    hashes = re.compile(r"(?<!\S)--hash(?:=|\s+)sha256:[0-9a-fA-F]{64}(?=\s|$)")
    try:
        for line in _logical_lines(source):
            line = line.split("#", 1)[0].strip()
            if not hashes.search(line):
                return False
            requirement = Requirement(hashes.sub("", line).strip())
            specifiers = list(requirement.specifier)
            if (
                requirement.url
                or len(specifiers) != 1
                or specifiers[0].operator != "=="
            ):
                return False
            Version(specifiers[0].version)
            found_pdm |= (
                canonicalize_name(requirement.name) == "pdm"
                and requirement.marker is None
            )
    except (ValueError, InvalidRequirement, InvalidVersion):
        return False
    return found_pdm


def _words(arguments: str) -> list[str]:
    """_words reads a shell-style argument list or Docker JSON array of strings."""
    if arguments.startswith("["):
        words = json.loads(arguments)
        if not isinstance(words, list) or not all(
            isinstance(word, str) for word in words
        ):
            raise ValueError("Docker JSON arguments must be a list of strings")
        return words
    return shlex.split(arguments)


def _run_commands(arguments: str) -> tuple[list[list[str]], bool]:
    """_run_commands extracts direct commands; only AND chains preserve bootstrap exits."""
    if arguments.startswith("["):
        return [_words(arguments)], True
    lexer = shlex.shlex(arguments, posix=False, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    commands = [[]]
    supported = True
    for word in lexer:
        if re.fullmatch(r"[;&|()<>]+", word):
            supported &= word == "&&"
            commands.append([])
        else:
            commands[-1].append(word)
    return [
        shlex.split(" ".join(command), comments=True) for command in commands if command
    ], supported


def _pip_install(words: list[str]) -> list[str] | None:
    """_pip_install returns arguments for direct pip or python -m pip installations."""
    if not words:
        return None
    executable = PurePosixPath(words[0]).name
    if re.fullmatch(r"pip(?:\d+(?:\.\d+)?)?", executable):
        arguments = words[1:]
    elif re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable) and words[1:3] == [
        "-m",
        "pip",
    ]:
        arguments = words[3:]
    else:
        return None
    while arguments and arguments[0] in {
        "--isolated",
        "--disable-pip-version-check",
        "--no-input",
        "--no-cache-dir",
        "--quiet",
        "--verbose",
        "-q",
        "-v",
    }:
        arguments = arguments[1:]
    if arguments[:1] == ["install"]:
        return arguments[1:]
    return None


def _option_values(words: list[str], long: str, short: str = "") -> list[str]:
    """_option_values returns explicit values of one pip long/optional short option."""
    values = []
    for index, word in enumerate(words):
        if word == long or (short and word == short):
            values.append(words[index + 1] if index + 1 < len(words) else "")
        elif word.startswith(long + "="):
            values.append(word[len(long) + 1 :])
        elif short and word.startswith(short) and len(word) > len(short):
            values.append(word[len(short) :])
    return values


def bootstrap_errors(source: str) -> list[str]:
    """bootstrap_errors checks direct PDM installs consume the copied, hashed lock.

    Standard COPY and direct pip/python RUN forms are supported. Shell wrappers,
    redirects and conditional recovery around bootstrap are reported, not evaluated.
    This checks bootstrap wiring, not arbitrary shell behavior or filesystem writes.
    """
    errors = []
    copied = False
    found = False
    stage = None
    stage_copies = {}
    try:
        for instruction, arguments in _instructions(source):
            if instruction == "FROM":
                if stage:
                    stage_copies[stage] = copied
                base, stage = _base(arguments)
                copied = stage_copies.get(base.lower(), False)
            elif instruction == "COPY":
                words = _words(arguments)
                options = []
                while words and words[0].startswith("--"):
                    options.append(words.pop(0))
                if len(words) < 2:
                    raise ValueError("COPY requires sources and a destination")
                sources, destination = words[:-1], words[-1]
                targets = sources if destination == LOCK_TARGET else []
                if destination == "/tmp/":
                    targets = [
                        source
                        for source in sources
                        if PurePosixPath(source).name == "pdm-bootstrap.txt"
                    ]
                if targets:
                    copied = (
                        len(targets) == 1
                        and str(PurePosixPath(targets[0])) == LOCK_SOURCE
                        and not any(
                            option.startswith(("--from=", "--parents", "--exclude"))
                            for option in options
                        )
                    )
            elif instruction == "RUN":
                commands, supported = _run_commands(arguments)
                for command in commands:
                    install = _pip_install(command)
                    text = " ".join(command)
                    mentions_pdm = bool(
                        re.search(
                            r"\bpdm(?:[<>=!~\[\s]|$)|pdm-bootstrap\.txt", text, re.I
                        )
                    )
                    if install is None:
                        if (
                            mentions_pdm
                            and re.search(r"\bpip(?:\d+(?:\.\d+)?)?\b", text)
                            and command[0] not in {"echo", "printf"}
                        ):
                            errors.append(
                                "unsupported wrapped pip bootstrap; use a direct RUN invocation"
                            )
                        continue
                    if not mentions_pdm:
                        continue
                    found = True
                    if not supported:
                        errors.append(
                            "bootstrap RUN supports direct commands joined only by &&"
                        )
                    if not copied:
                        errors.append(
                            "copy .scripts/pdm-bootstrap.txt to /tmp/pdm-bootstrap.txt before installation"
                        )
                    if _option_values(install, "--requirement", "-r") != [LOCK_TARGET]:
                        errors.append(
                            "bootstrap pip install must consume only /tmp/pdm-bootstrap.txt"
                        )
                    if "--require-hashes" not in install:
                        errors.append("bootstrap pip install must use --require-hashes")
                    protected_options = (
                        "--require-hashes",
                        "--requirement",
                        "--only-binary",
                        "--no-binary",
                    )
                    for word in install:
                        name = word.split("=", 1)[0]
                        if name.startswith("--") and any(
                            option.startswith(name) and option != name
                            for option in protected_options
                        ):
                            errors.append(
                                "spell bootstrap hash, requirement and binary options in full"
                            )
                    if _option_values(install, "--only-binary") != [
                        ":all:"
                    ] or _option_values(install, "--no-binary"):
                        errors.append(
                            "bootstrap pip install must use --only-binary=:all:"
                        )
                    if "--" in install or any(
                        "$" in word or "`" in word for word in install
                    ):
                        errors.append("bootstrap arguments must be literal pip options")
    except ValueError as exc:
        errors.append(str(exc))
    if not found:
        errors.append(
            "missing executable PDM bootstrap installation from the reviewed lock"
        )
    return errors
