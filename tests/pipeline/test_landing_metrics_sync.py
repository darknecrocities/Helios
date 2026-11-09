"""Unit tests for landing page pretrain metrics synchronization and CI/CD verification."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.sync_pretrain_metrics_to_landing import (
    EMAIL_REGEX,
    OUTPUT_JSON_PATH,
    PLATFORMS_CATALOG,
    build_pretrain_metrics_payload,
    parse_topics_doc,
    sanitize_privacy_dict,
    sync_metrics,
    verify_sync,
)


class TestLandingMetricsSync:
    """Verifies pretraining metrics compilation, privacy safety, and CI verification."""

    def test_metrics_payload_invariants(self) -> None:
        payload = build_pretrain_metrics_payload()

        assert payload["version"] == "0.3.0"
        curriculum = payload["curriculum"]
        assert curriculum["totalDatasets"] == 177
        assert curriculum["totalRecords"] == 120616
        assert curriculum["totalTokens"] == 90376089
        assert curriculum["tokensFormatted"] == "90.38M"
        assert curriculum["duplicationRatePct"] == 0.0
        assert "HEALTHY" in curriculum["checkpointIntegrity"]

    def test_creator_identity_and_zero_email(self) -> None:
        payload = build_pretrain_metrics_payload()
        creator = payload["creator"]

        assert "Arron Kian Parejas" in creator["name"]
        assert "Lead Architect" in creator["role"]

        # Assert no email address anywhere in the entire JSON payload
        raw_json = json.dumps(payload)
        assert EMAIL_REGEX.search(raw_json) is None
        assert "@" not in creator["name"]

    def test_topics_distribution(self) -> None:
        topics_doc = Path("docs/TRAINED_TOPICS_HISTORY.md")
        topics = parse_topics_doc(topics_doc)

        assert len(topics) >= 5
        topic_names = [t["name"] for t in topics]
        assert "Coding" in topic_names
        assert "Math" in topic_names
        assert "Agents" in topic_names

        # Check that each topic has sample prompts
        for t in topics:
            assert len(t["samplePrompts"]) >= 1

    def test_platforms_catalog(self) -> None:
        assert len(PLATFORMS_CATALOG) == 8
        platform_ids = [p["id"] for p in PLATFORMS_CATALOG]
        assert "apple-mps" in platform_ids
        assert "nvidia-cuda" in platform_ids
        assert "arm-cpu" in platform_ids
        assert "ollama" in platform_ids
        assert "huggingface" in platform_ids
        assert "openclaw" in platform_ids
        assert "nous-hermes" in platform_ids
        assert "antigravity-mcp" in platform_ids

        for p in PLATFORMS_CATALOG:
            assert len(p["title"]) > 0
            assert len(p["command"]) > 0
            assert len(p["statusBadge"]) > 0

    def test_privacy_sanitizer_removes_emails(self) -> None:
        dirty_input = {
            "name": "Arron",
            "developer_email": "test@example.com",
            "bio": "Contact me at dev@helios.ai for queries",
            "nested": {"contact_email": "admin@gmail.com", "valid": "ok"},
        }
        cleaned = sanitize_privacy_dict(dirty_input)
        assert "developer_email" not in cleaned
        assert "contact_email" not in cleaned["nested"]
        assert "[redacted-privacy-guard]" in cleaned["bio"]
        assert EMAIL_REGEX.search(json.dumps(cleaned)) is None

    def test_sync_and_ci_verify(self, tmp_path: Path) -> None:
        test_out = tmp_path / "pretrain_metrics.json"
        sync_metrics(test_out)
        assert test_out.exists()
        assert verify_sync(test_out) is True

    def test_existing_output_json_in_sync(self) -> None:
        assert OUTPUT_JSON_PATH.exists()
        assert verify_sync(OUTPUT_JSON_PATH) is True
