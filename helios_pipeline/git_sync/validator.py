"""Pre-commit security and weight validator to prevent accidental leakages to Git."""

from __future__ import annotations

import re
from pathlib import Path

BLOCKED_EXTENSIONS = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".onnx",
    ".parquet",
    ".arrow",
    ".tar",
    ".gz",
    ".zip",
    ".7z",
}

BLOCKED_DIRECTORIES = {
    "datasets",
    "checkpoints",
    ".tmp_",
    "cache",
    "venv",
    ".venv",
}

SECRET_PATTERNS = [
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
    re.compile(r"(?i)hf_[A-Za-z0-9]{30,}"),
    re.compile(r"(?i)ghp_[A-Za-z0-9]{36,}"),
    re.compile(r"-----BEGIN (RSA|EC|OPENSSH|PRIVATE) KEY-----"),
]

MAX_ALLOWED_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


class GitStagingViolationError(Exception):
    """Raised when a staged file violates git safety policies."""


class StagingValidator:
    """Validates files before git staging to block large model weights, datasets, and secrets."""

    def __init__(self, repo_root: Path | str = ".") -> None:
        self.repo_root = Path(repo_root).resolve()

    def is_file_allowed(self, file_path: Path | str) -> tuple[bool, str]:
        """Check whether a single file is safe to commit.

        Returns (is_allowed, reason).
        """
        p = Path(file_path).resolve()
        try:
            rel_path = p.relative_to(self.repo_root)
        except ValueError:
            rel_path = p

        # 1. Directory boundary check
        for part in rel_path.parts:
            if part in BLOCKED_DIRECTORIES or part.startswith(".tmp_"):
                return False, f"File resides in blocked directory '{part}': {rel_path}"

        # 2. Extension check
        suffix = p.suffix.lower()
        if suffix in BLOCKED_EXTENSIONS:
            return False, f"Prohibited binary/weight extension '{suffix}': {rel_path}"

        # 3. Size check
        if p.exists() and p.is_file():
            size = p.stat().st_size
            if size > MAX_ALLOWED_FILE_SIZE_BYTES:
                return (
                    False,
                    f"File size ({size / (1024 * 1024):.2f} MB) exceeds 5MB limit: {rel_path}",
                )

            # 4. Secret scanning on text files
            if suffix in {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".toml", ".sh"}:
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    for pattern in SECRET_PATTERNS:
                        if pattern.search(content):
                            return False, f"Potential credential/secret detected in {rel_path}"
                except Exception as e:
                    return False, f"Failed to scan file content for secrets: {e}"

        return True, "Safe to commit"

    def validate_file_list(self, files: list[Path | str]) -> list[str]:
        """Validate a list of candidate files. Returns list of violation messages."""
        violations: list[str] = []
        for f in files:
            allowed, reason = self.is_file_allowed(f)
            if not allowed:
                violations.append(reason)
        return violations
