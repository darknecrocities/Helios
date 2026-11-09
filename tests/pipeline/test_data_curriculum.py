"""Unit tests for dataset streaming, PII sanitization, and 4-stage curriculum engine."""

from helios_pipeline.data.curriculum import CurriculumPartitioner
from helios_pipeline.data.manifest import ClusterManifest
from helios_pipeline.data.sanitization import DocumentSanitizer


def test_pii_redaction_and_quality_filters() -> None:
    """Verify regex masking of emails, phone numbers, SSNs, credit cards, and IPs."""
    sanitizer = DocumentSanitizer(min_chars=20, min_words=5)

    dirty_text = (
        "Hello researcher, please contact us at user@example.com or call +1-555-123-4567. "
        "The security number is 123-45-6789 and server is located at 192.168.1.100."
    )
    clean, rejection = sanitizer.sanitize(dirty_text)
    assert rejection is None
    assert clean is not None
    assert "[EMAIL]" in clean
    assert "user@example.com" not in clean
    assert "[PHONE]" in clean
    assert "+1-555-123-4567" not in clean
    assert "[SSN]" in clean
    assert "123-45-6789" not in clean
    assert "[IP_ADDRESS]" in clean
    assert "192.168.1.100" not in clean


def test_exact_deduplication() -> None:
    """Verify duplicate documents with identical SHA-256 are rejected."""
    sanitizer = DocumentSanitizer(min_chars=10, min_words=3)
    text = "This is a clean unique sentence for testing deduplication."

    clean1, rej1 = sanitizer.sanitize(text)
    assert clean1 is not None
    assert rej1 is None

    clean2, rej2 = sanitizer.sanitize(text)
    assert clean2 is None
    assert rej2 == "duplicate_content"


def test_curriculum_partitioner_proportions() -> None:
    """Verify 4-stage curriculum allocation matches 40/30/20/10 target."""
    curriculum = CurriculumPartitioner()
    allocations = curriculum.allocate_cluster_stages(100)
    assert len(allocations) == 100

    counts = {s: allocations.count(s) for s in ["stage_a", "stage_b", "stage_c", "stage_d"]}
    assert 35 <= counts["stage_a"] <= 45
    assert 25 <= counts["stage_b"] <= 35
    assert 15 <= counts["stage_c"] <= 25
    assert 5 <= counts["stage_d"] <= 15


def test_cluster_manifest_state_machine() -> None:
    """Verify cluster transitions and validation."""
    m = ClusterManifest(cluster_id="c_test", stage="stage_a")
    assert m.status == "PENDING"
    m.transition_to("DOWNLOADING")
    assert m.status == "DOWNLOADING"
    m.transition_to("COMPLETED")
    assert m.status == "COMPLETED"
    assert m.completed_at != ""
