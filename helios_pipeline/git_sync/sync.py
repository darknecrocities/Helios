"""Safe Git synchronization engine for pipeline metadata and documentation."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from helios_pipeline.git_sync.validator import GitStagingViolationError, StagingValidator

logger = logging.getLogger(__name__)


class GitSyncEngine:
    """Safely synchronizes pipeline documentation and manifests to Git."""

    def __init__(
        self,
        repo_root: Path | str = ".",
        target_branch: str = "automation/helios-training-history",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.target_branch = target_branch
        self.validator = StagingValidator(repo_root=self.repo_root)

    def _run_git(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Execute git command within repo root."""
        return subprocess.run(
            ["git", *args],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )

    def get_current_branch(self) -> str:
        """Return the currently checked out branch name."""
        proc = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        return proc.stdout.strip()

    def sync_files(
        self,
        files_to_sync: list[Path | str],
        commit_message: str,
        push: bool = False,
    ) -> bool:
        """Validate, stage, commit, and optionally push metadata files."""
        # 1. Pre-commit validation
        violations = self.validator.validate_file_list(files_to_sync)
        if violations:
            logger.error("Git sync rejected due to pre-commit safety violations:")
            for v in violations:
                logger.error(f"  - {v}")
            raise GitStagingViolationError(
                f"Staging rejected with {len(violations)} violations: {violations[0]}"
            )

        # 2. Stage verified files
        resolved_paths: list[str] = []
        for f in files_to_sync:
            p = Path(f).resolve()
            if p.exists():
                try:
                    rel = str(p.relative_to(self.repo_root))
                except ValueError:
                    rel = str(p)
                resolved_paths.append(rel)

        if not resolved_paths:
            logger.info("No existing files to stage for Git sync.")
            return True

        stage_res = self._run_git(["add", *resolved_paths])
        if stage_res.returncode != 0:
            logger.error(f"Failed to stage files: {stage_res.stderr}")
            return False

        # 3. Check if anything is actually staged
        diff_res = self._run_git(["diff", "--cached", "--quiet"])
        if diff_res.returncode == 0:
            logger.info("No modifications staged to commit.")
            return True

        # 4. Commit
        commit_res = self._run_git(["commit", "-m", commit_message])
        if commit_res.returncode != 0:
            logger.error(f"Commit failed: {commit_res.stderr}")
            return False

        logger.info(f"Committed metadata updates: {commit_message}")

        # 5. Push if requested
        if push:
            curr_branch = self.get_current_branch()
            push_res = self._run_git(["push", "origin", curr_branch])
            if push_res.returncode != 0:
                logger.warning(f"Git push failed (will retry next iteration): {push_res.stderr}")
                return False
            logger.info(f"Pushed metadata updates to remote branch {curr_branch}")

        return True
