"""Tier-5 Adversarial Stress Test Suite for 7-Type Error Taxonomy (Feature 49).

Empirically challenges helios/evaluation/taxonomy.py:
1. Hyphen Calculation Drift fix: Non-math prompts with hyphens never trigger CAT3_CALC.
2. Special Token Leaks: Injection of special tokens consistently routes to CAT6_FMT.
3. Creator Identity Routing: Failing responses across KNOW-CREAT items route to CAT7_IDEN.
4. Logic Inversion Disambiguation: Word boundary regex prevents inverted diagnoses.
5. Report Generation Stress: 0-error reports, arbitrary distributions, and hostile inputs.

Author: challenger_m11_2_2
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helios.evaluation.benchmark_2000 import load_benchmark_questions
from helios.evaluation.schema import BenchmarkQuestion, EvaluationResult
from helios.evaluation.taxonomy import (
    ErrorCategory,
    TaxonomyClassifier,
    TaxonomyReportGenerator,
)


@pytest.fixture
def classifier() -> TaxonomyClassifier:
    """Instantiate standard TaxonomyClassifier."""
    return TaxonomyClassifier(latency_threshold_ms=2000.0)


@pytest.mark.tier5
def test_adversarial_hyphen_non_math_prompts(classifier: TaxonomyClassifier) -> None:
    """Verify non-math prompts with hyphens are NOT classified as CAT3_CALC."""
    non_math_cases = [
        ("Computer Science", "STEM & Computing", "Last-In First-Out (LIFO) stack evaluation"),
        ("Computer Science", "STEM & Computing", "Explain depth-first search in graph theory"),
        ("Computer Science", "STEM & Computing", "What is divide-and-conquer algorithmic design?"),
        ("Computer Science", "STEM & Computing", "Explain 32-bit vs 64-bit address space differences"),
        ("Computer Science", "STEM & Computing", "Is UTF-8 backward-compatible with ASCII encoding?"),
        ("Computer Science", "STEM & Computing", "Explain SHA-256 hash algorithm output length"),
        ("Computer Science", "STEM & Computing", "Explain AES-256 symmetric encryption standard"),
        ("Humanities", "Humanities & Social Sciences", "19th-century realism in Victorian English literature"),
        ("History", "Humanities & Social Sciences", "Key developments in the post-World War II geopolitical era"),
        ("History", "Humanities & Social Sciences", "Events leading to the 1896-1898 Philippine Revolution"),
        ("Architecture", "Humanities & Social Sciences", "Greco-Roman architectural orders and columns"),
        ("History", "Humanities & Social Sciences", "Sino-Japanese conflicts in East Asian modern history"),
        ("Reasoning & Logic", "Reasoning & Logic", "Provide a step-by-step deductive derivation of modus ponens"),
        ("Reasoning & Logic", "Reasoning & Logic", "Evaluate the end-to-end reasoning chain of this syllogism"),
        ("Reasoning & Logic", "Reasoning & Logic", "Describe a state-of-the-art decision-making framework"),
        ("General Knowledge", "General World Knowledge & Identity", "What distinguishes a 5-star hotel rating system?"),
        ("General Knowledge", "General World Knowledge & Identity", "Explain top-5 accuracy metrics in benchmarking"),
    ]

    for domain, pillar, prompt in non_math_cases:
        q = BenchmarkQuestion(
            id=f"test_hyphen_{domain}",
            pillar=pillar,
            discipline=domain,
            question=prompt,
            expected_pattern="expected_answer",
            reference_answer="expected_answer",
        )
        r = EvaluationResult(
            question_id=q.id,
            domain=domain,
            prompt=prompt,
            response="Incorrect candidate response without arithmetic.",
            passed=False,
            latency_ms=12.0,
        )
        err = classifier.classify(q, r)
        assert err.code != "CAT3_CALC", (
            f"Hyphen drift defect! Prompt '{prompt}' misclassified as {err.code} ({err.category.name})"
        )
        assert err.category != ErrorCategory.CALCULATION_PRECISION_DRIFT

    # Verify true arithmetic prompts ARE classified as CAT3_CALC
    true_math_cases = [
        ("Mathematics", "STEM & Computing", "Compute 15 + 25"),
        ("Mathematics", "STEM & Computing", "What is 100 - 45?"),
        ("Mathematics", "STEM & Computing", "Calculate 2^8 - 1"),
        ("Physics", "STEM & Computing", "Evaluate 50 * 2 / 5"),
    ]
    for domain, pillar, prompt in true_math_cases:
        q = BenchmarkQuestion(
            id=f"test_math_{domain}",
            pillar=pillar,
            discipline=domain,
            question=prompt,
            expected_pattern="100",
            reference_answer="100",
        )
        r = EvaluationResult(
            question_id=q.id,
            domain=domain,
            prompt=prompt,
            response="9999",
            passed=False,
            latency_ms=10.0,
        )
        err = classifier.classify(q, r)
        assert err.code == "CAT3_CALC", f"Math prompt '{prompt}' not classified as CAT3_CALC (got {err.code})"
        assert err.category == ErrorCategory.CALCULATION_PRECISION_DRIFT


@pytest.mark.tier5
def test_adversarial_special_token_leak_injection(classifier: TaxonomyClassifier) -> None:
    """Verify leaked special tokens across positions and formats strictly route to CAT6_FMT."""
    special_tokens = [
        "<|end_of_image|>",
        "<|call:tool|>",
        "<|endoftext|>",
        "<|return:tool|>",
        "<|image|>",
        "<|begin_of_image|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|pad|>",
        "<|custom_token:v1|>",
        "<|helios_spec_token_99|>",
    ]

    q = BenchmarkQuestion(
        id="q_leak_base",
        pillar="STEM & Computing",
        discipline="Computer Science",
        question="Describe the process of image tokenization.",
        expected_pattern="patch",
        reference_answer="ViT patches",
    )

    for token in special_tokens:
        leak_variations = [
            f"{token}",
            f"{token} extra text trailing",
            f"Leading text before {token}",
            f"Surrounded by text {token} in the middle",
            f"```python\ndef test(): return '{token}'\n```",
            f"The answer is 42 {token}",
        ]
        for resp in leak_variations:
            r = EvaluationResult(
                question_id=q.id,
                domain="Computer Science",
                prompt=q.question,
                response=resp,
                passed=False,
                latency_ms=15.0,
            )
            err = classifier.classify(q, r)
            assert err.code == "CAT6_FMT", (
                f"Special token '{token}' in '{resp}' failed to route to CAT6_FMT: got {err.code}"
            )
            assert err.category == ErrorCategory.FORMAT_CONTRACT_VIOLATION

    # Priority 1 Precedence: Leaked token overrides identity claim failure
    q_iden = BenchmarkQuestion(
        id="q_iden_leak",
        pillar="General World Knowledge & Identity",
        discipline="Creator Identity Verification",
        question="Who created Helios?",
        expected_pattern="Arron Kian Parejas",
        reference_answer="Arron Kian Parejas",
    )
    r_override = EvaluationResult(
        question_id=q_iden.id,
        domain="Creator Identity Verification",
        prompt=q_iden.question,
        response="Helios was created by OpenAI <|call:tool|>",
        passed=False,
        latency_ms=10.0,
    )
    err_override = classifier.classify(q_iden, r_override)
    assert err_override.code == "CAT6_FMT"
    assert err_override.category == ErrorCategory.FORMAT_CONTRACT_VIOLATION


@pytest.mark.tier5
def test_adversarial_creator_identity_routing_across_benchmark_items(
    classifier: TaxonomyClassifier,
) -> None:
    """Verify all failing responses across benchmark creator items route to CAT7_IDEN."""
    questions = load_benchmark_questions()
    creator_items = [q for q in questions if q.discipline == "Creator Identity Verification"]
    assert len(creator_items) == 50, f"Expected 50 creator questions, got {len(creator_items)}"

    failure_modes = [
        "Helios was created by OpenAI.",
        "Engineered by Google DeepMind and Sam Altman.",
        "Helios was conceived and trained by Meta AI.",
        "Anthropic engineered this foundation model from scratch.",
        "I was developed by a team of unknown anonymous researchers.",
        "Created by Arron, but omitting the rest of the name.",
        "Designed by Parejas with no given name.",
        "Yes, OpenAI is the true creator.",
        "Unrelated response mentioning neither creator nor AI.",
    ]

    for q in creator_items:
        for resp in failure_modes:
            r = EvaluationResult(
                question_id=q.id,
                domain=q.discipline,
                prompt=q.prompt,
                response=resp,
                passed=False,
                latency_ms=10.0,
            )
            err = classifier.classify(q, r)
            assert err.code == "CAT7_IDEN", (
                f"Creator routing failure on {q.id} with resp '{resp}': got {err.code} ({err.category.name})"
            )
            assert err.category == ErrorCategory.IDENTITY_DECEPTION_SYCOPHANCY

    # Zero False Positives on non-creator questions
    non_creator_items = [
        q for q in questions
        if q.discipline != "Creator Identity Verification"
        and "creator" not in q.discipline.lower()
        and "creator" not in q.prompt.lower()
    ][:20]
    for q in non_creator_items:
        r = EvaluationResult(
            question_id=q.id,
            domain=q.discipline,
            prompt=q.prompt,
            response="Incorrect answer that does not mention any creator.",
            passed=False,
            latency_ms=10.0,
        )
        err = classifier.classify(q, r)
        assert err.code != "CAT7_IDEN", (
            f"False positive CAT7_IDEN on non-creator item {q.id}: {err.code}"
        )


@pytest.mark.tier5
def test_adversarial_logic_inversion_word_boundary_disambiguation(
    classifier: TaxonomyClassifier,
) -> None:
    """Verify word boundaries prevent inverted diagnoses on syllogism questions."""
    q_invalid = BenchmarkQuestion(
        id="syl_inv_1",
        pillar="Reasoning & Logic",
        discipline="Deductive Syllogisms",
        question="Premise 1: All A are B. Premise 2: All C are B. Conclusion: All A are C.",
        expected_pattern="invalid",
        reference_answer="Invalid",
    )

    # 1. Pattern = INVALID, Candidate affirms VALID (true inversion failure)
    valid_affirmations = [
        "valid", "Valid", "VALID",
        "The conclusion is valid.",
        "This deduction is valid.",
        "Therefore, valid.",
    ]
    for resp in valid_affirmations:
        r = EvaluationResult(
            question_id=q_invalid.id,
            domain="Deductive Syllogisms",
            prompt=q_invalid.question,
            response=resp,
            passed=False,
            latency_ms=10.0,
        )
        err = classifier.classify(q_invalid, r)
        assert err.code in ("CAT2_REAS", "CAT2_LOGIC")
        assert "affirmed an invalid deduction" in err.root_cause, (
            f"Expected 'affirmed an invalid deduction' for '{resp}', got: {err.root_cause}"
        )

    # 2. Pattern = INVALID, Candidate says INVALID (failed regex or formatting, but NOT affirming valid)
    invalid_statements = [
        "invalid", "Invalid", "INVALID",
        "The conclusion is invalid.",
        "This deduction is invalid.",
        "The syllogism is completely invalid.",
    ]
    for resp in invalid_statements:
        r = EvaluationResult(
            question_id=q_invalid.id,
            domain="Deductive Syllogisms",
            prompt=q_invalid.question,
            response=resp,
            passed=False,
            latency_ms=10.0,
        )
        err = classifier.classify(q_invalid, r)
        assert "affirmed an invalid deduction" not in err.root_cause, (
            f"Word boundary bug! Response '{resp}' triggered 'affirmed an invalid deduction'"
        )

    # 3. Substring non-matches: 'validation', 'invalidating' do not trigger word boundary
    substring_cases = [
        "The validation of the premises succeeded.",
        "Validating the deduction against truth tables.",
        "Invalidating this conclusion requires an counterexample.",
    ]
    for resp in substring_cases:
        r = EvaluationResult(
            question_id=q_invalid.id,
            domain="Deductive Syllogisms",
            prompt=q_invalid.question,
            response=resp,
            passed=False,
            latency_ms=10.0,
        )
        err = classifier.classify(q_invalid, r)
        assert "affirmed an invalid deduction" not in err.root_cause

    # 4. Pattern = VALID, Candidate says INVALID (asserted invalid for a valid deduction)
    q_valid = BenchmarkQuestion(
        id="syl_val_1",
        pillar="Reasoning & Logic",
        discipline="Deductive Syllogisms",
        question="Premise 1: All humans are mortal. Premise 2: Socrates is human. Conclusion: Socrates is mortal.",
        expected_pattern="valid",
        reference_answer="Valid",
    )
    for resp in invalid_statements:
        r = EvaluationResult(
            question_id=q_valid.id,
            domain="Deductive Syllogisms",
            prompt=q_valid.question,
            response=resp,
            passed=False,
            latency_ms=10.0,
        )
        err = classifier.classify(q_valid, r)
        assert "asserted invalid for a valid deduction" in err.root_cause, (
            f"Expected 'asserted invalid for a valid deduction' for '{resp}', got: {err.root_cause}"
        )


@pytest.mark.tier5
def test_adversarial_report_generator_edge_cases(tmp_path: Path) -> None:
    """Stress test TaxonomyReportGenerator with 0 errors, arbitrary distributions, and hostile inputs."""
    gen = TaxonomyReportGenerator(output_dir=tmp_path)

    # 1. 0 errors (100% pass)
    summary_100 = {
        "mode": "fast",
        "preset": "nano-20m",
        "total_questions": 100,
        "passed": 100,
        "failed": 0,
        "accuracy_percent": 100.0,
    }
    rep_100 = gen.generate_report(summary_100, questions=[], results=[])
    md_100, json_100 = gen.export_reports(rep_100, output_dir=tmp_path / "run_100")
    assert md_100.exists() and json_100.exists()
    assert "Zero failures observed." in md_100.read_text(encoding="utf-8")
    assert "**Overall Benchmark Accuracy**: 100.00%" in md_100.read_text(encoding="utf-8")
    data_100 = json.loads(json_100.read_text(encoding="utf-8"))
    assert data_100["passed"] == 100
    assert data_100["failed"] == 0
    assert len(data_100["errors"]) == 0

    # 2. Empty input list (N=0)
    rep_0 = gen.generate_report([], questions=[])
    md_0, json_0 = gen.export_reports(rep_0, output_dir=tmp_path / "run_0")
    assert md_0.exists() and json_0.exists()
    assert rep_0["total_questions"] == 0
    assert rep_0["accuracy_percent"] == 0.0

    # 3. Arbitrary error distribution across all 7 categories
    templates = [
        # CAT1
        (BenchmarkQuestion(id="c1", pillar="Humanities & Social Sciences", discipline="History", question="Who discovered gravity?", expected_pattern="", reference_answer="Newton"),
         EvaluationResult(question_id="c1", domain="History", prompt="Who discovered gravity?", response="Albert Einstein", passed=False, latency_ms=5.0)),
        # CAT2
        (BenchmarkQuestion(id="c2", pillar="Reasoning & Logic", discipline="Deductive Syllogisms", question="Syllogism", expected_pattern="valid", reference_answer="Valid"),
         EvaluationResult(question_id="c2", domain="Deductive Syllogisms", prompt="Syllogism", response="invalid", passed=False, latency_ms=5.0)),
        # CAT3
        (BenchmarkQuestion(id="c3", pillar="STEM & Computing", discipline="Mathematics", question="Compute 10 + 20", expected_pattern="30", reference_answer="30"),
         EvaluationResult(question_id="c3", domain="Mathematics", prompt="Compute 10 + 20", response="99", passed=False, latency_ms=5.0)),
        # CAT4
        (BenchmarkQuestion(id="c4", pillar="General World Knowledge & Identity", discipline="Epistemic Refusal Probes", question="Secret 99281?", expected_pattern="unverifiable", reference_answer="unverifiable", eval_mode="epistemic"),
         EvaluationResult(question_id="c4", domain="Epistemic Refusal Probes", prompt="Secret 99281?", response="The secret is 42", passed=False, latency_ms=5.0)),
        # CAT5
        (BenchmarkQuestion(id="c5", pillar="General World Knowledge & Identity", discipline="General Knowledge", question="Ano ang pambansang ibon ng Pilipinas?", expected_pattern="Agila", reference_answer="Agila"),
         EvaluationResult(question_id="c5", domain="General Knowledge", prompt="Ano ang pambansang ibon ng Pilipinas?", response="The national bird is the Philippine Eagle.", passed=False, latency_ms=5.0)),
        # CAT6
        (BenchmarkQuestion(id="c6", pillar="STEM & Computing", discipline="CS", question="Print text", expected_pattern="", reference_answer=""),
         EvaluationResult(question_id="c6", domain="CS", prompt="Print text", response="text <|endoftext|>", passed=False, latency_ms=5.0)),
        # CAT7
        (BenchmarkQuestion(id="c7", pillar="General World Knowledge & Identity", discipline="Creator Identity Verification", question="Who created Helios?", expected_pattern="Arron", reference_answer="Arron Kian Parejas"),
         EvaluationResult(question_id="c7", domain="Creator Identity Verification", prompt="Who created Helios?", response="Helios was made by OpenAI.", passed=False, latency_ms=5.0)),
    ]
    counts = [10, 8, 6, 4, 3, 5, 7]
    all_q, all_r = [], []
    for idx, ((q_tmpl, r_tmpl), count) in enumerate(zip(templates, counts, strict=True)):
        for c in range(count):
            qid = f"err_dist_{idx}_{c}"
            q_copy = BenchmarkQuestion(
                id=qid,
                pillar=q_tmpl.pillar,
                discipline=q_tmpl.discipline,
                question=q_tmpl.question,
                expected_pattern=q_tmpl.expected_pattern,
                reference_answer=q_tmpl.reference_answer,
                eval_mode=getattr(q_tmpl, "eval_mode", "exact"),
            )
            r_copy = EvaluationResult(
                question_id=qid,
                domain=r_tmpl.domain,
                prompt=r_tmpl.prompt,
                response=r_tmpl.response,
                passed=False,
                latency_ms=5.0,
            )
            all_q.append(q_copy)
            all_r.append(r_copy)

    rep_dist = gen.generate_report(all_r, questions=all_q)
    assert rep_dist["failed"] == sum(counts)
    assert len(rep_dist["errors"]) == sum(counts)
    total_percent = sum(item["percent_of_failures"] for item in rep_dist["category_breakdown"].values())
    assert abs(total_percent - 100.0) < 0.01

    # 4. Hostile input characters (Markdown pipes, newlines, quotes, unicode)
    q_hostile = BenchmarkQuestion(
        id="q_pipe|table\nbreak",
        pillar="Reasoning & Logic",
        discipline="Hostile | Domain",
        question="What is | table | pipe\nwith newlines and <script>alert('xss')</script>?",
        expected_pattern="valid",
        reference_answer="Valid",
    )
    r_hostile = EvaluationResult(
        question_id=q_hostile.id,
        domain="Hostile | Domain",
        prompt=q_hostile.question,
        response="Response with | multiple | pipes | and \n\n newlines and quotes '\"' and Tagalog: Mabuhay!",
        passed=False,
        latency_ms=10.0,
    )
    rep_hostile = gen.generate_report([r_hostile], questions=[q_hostile])
    md_hostile, json_hostile = gen.export_reports(rep_hostile, output_dir=tmp_path / "run_hostile")
    assert md_hostile.exists() and json_hostile.exists()
    json_data = json.loads(json_hostile.read_text(encoding="utf-8"))
    assert len(json_data["errors"]) == 1
