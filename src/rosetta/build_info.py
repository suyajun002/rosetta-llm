"""Load and validate non-secret build provenance for startup diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_DISTRIBUTION = "rosetta-llm"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class BuildInfo:
    version: str
    commit: str
    source_dirty: bool | None
    source_digest: str
    provenance_status: str

    def log_fields(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "commit": self.commit,
            "source_dirty": self.source_dirty,
            "source_digest": self.source_digest,
            "provenance_status": self.provenance_status,
        }

    def display(self) -> str:
        short_commit = self.commit[:12] if self.commit != "unknown" else self.commit
        return f"{_DISTRIBUTION} {self.version} (commit {short_commit}, {self.provenance_status})"


def _installed_version() -> str:
    try:
        return version(_DISTRIBUTION)
    except PackageNotFoundError:
        return "unknown"


def _package_digest(root: Path) -> str:
    files: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        files.append((relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    digest = hashlib.sha256()
    for relative, file_hash in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _valid_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == 1
        and payload.get("distribution") == _DISTRIBUTION
        and isinstance(payload.get("version"), str)
        and isinstance(payload.get("commit"), str)
        and _COMMIT_RE.fullmatch(payload["commit"]) is not None
        and isinstance(payload.get("source_dirty"), bool)
        and isinstance(payload.get("source_digest"), str)
        and _DIGEST_RE.fullmatch(payload["source_digest"]) is not None
    )


def load_build_info(provenance_path: Path | None = None) -> BuildInfo:
    installed_version = _installed_version()
    if provenance_path is None:
        configured = os.environ.get("ROSETTA_BUILD_PROVENANCE")
        provenance_path = (
            Path(configured)
            if configured
            else Path.home() / ".local" / "share" / "claude-sky" / "rosetta-build.json"
        )

    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return BuildInfo(installed_version, "unknown", None, "unknown", "missing")
    except (OSError, json.JSONDecodeError):
        return BuildInfo(installed_version, "unknown", None, "unknown", "invalid")

    if not _valid_payload(payload):
        return BuildInfo(installed_version, "unknown", None, "unknown", "invalid")

    commit = payload["commit"]
    source_dirty = payload["source_dirty"]
    source_digest = payload["source_digest"]
    if payload["version"] != installed_version:
        return BuildInfo(
            installed_version, commit, source_dirty, source_digest, "version_mismatch"
        )
    if _package_digest(Path(__file__).resolve().parent) != source_digest:
        return BuildInfo(installed_version, commit, source_dirty, source_digest, "digest_mismatch")
    status = "verified_dirty" if source_dirty else "verified"
    return BuildInfo(installed_version, commit, source_dirty, source_digest, status)
