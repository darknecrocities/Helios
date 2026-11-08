"""Tier-5 Adversarial Stress Test Suite: Error Taxonomy Scoring Shadowing Remediation.

Empirically challenges helios/evaluation/taxonomy.py:
1. Factual questions with pattern mismatch scoring explanations route to CAT1_FACT.
2. Math questions with numeric mismatch scoring explanations route to CAT3_CALC.
3. Reasoning questions with logic mismatch scoring explanations route to CAT2_REAS / CAT2_LOGIC.
4. Language questions with translation mismatch scoring explanations route to CAT5_LANG.
5. Creator identity questions with attribution failure scoring explanations route to CAT7_IDEN.
6. Epistemic synthetic probes with failure explanations route to CAT4_EPIS.
7. Authentic Python execution/syntax errors route to CAT4_CODE / FORMAT_CONTRACT_VIOLATION.
8. Stress-tests overlapping markers (e.g. 'expected' + code terms) to guarantee zero false positives.

Author: challenger_m11_3_2
"""

from __future__ import annotations

import pytest

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
def test_factual_scoring_explanations_do_not_shadow(classifier: TaxonomyClassifier) -> None:
    """Factual questions with scoring explanations must route to CAT1_FACT, never CAT4_CODE or CAT6_FMT."""
    factual_cases = [
        ("World History", "Humanities & Social Sciences", "What treaty ended WWI in 1919?", "Treaty of Versailles", "The Treaty of Paris.", "Expected pattern 'Versailles' not found in: The Treaty of Paris."),
        ("Biology", "STEM & Computing", "Powerhouse of the cell?", "Mitochondria", "The ribosome.", "Expected pattern 'Mitochondria' not found in: The ribosome."),
        ("World Geography", "Humanities & Social Sciences", "Capital of Australia?", "Canberra", "Sydney is the capital.", "Expected exact match for 'Canberra' in: Sydney is the capital."),
        ("Philippine History", "Humanities & Social Sciences", "Where was 1898 Philippine Independence proclaimed?", "Kawit, Cavite", "Malolos, Bulacan.", "Expected pattern 'Kawit' not found in: Malolos, Bulacan."),
        ("Computer Science Theory", "STEM & Computing", "What grammar type corresponds to regular languages?", "Type 3", "Type 0 grammar.", "Expected pattern 'Type 3' not found in: Type 0 grammar."),
        ("Linguistics", "Humanities & Social Sciences", "In syntax and grammar, what is the predicate?", "action", "It names the subject.", "Expected pattern 'action' not found in: It names the subject."),
    ]
    for domain, pillar, prompt, target, response, err_msg in factual_cases:
        q = BenchmarkQuestion(id=f"fact_{domain}", pillar=pillar, discipline=domain, question=prompt, expected_pattern=target, reference_answer=target, eval_mode="contains")
        r = EvaluationResult(question_id=q.id, domain=domain, prompt=prompt, response=response, passed=False, latency_ms=15.0, error_message=err_msg)
        err = classifier.classify(q, r)
        assert err.category == ErrorCategory.FACTUAL_HALLUCINATION, f"Factual query '{prompt}' misclassified as {err.category} ({err.code})"
        assert err.code == "CAT1_FACT" and err.code not in ("CAT4_CODE", "CAT6_FMT")


@pytest.mark.tier5
def test_math_scoring_explanations_route_to_calc(classifier: TaxonomyClassifier) -> None:
    """Math questions with numeric scoring explanations must route to CAT3_CALC, never CAT4_CODE or CAT6_FMT."""
    math_cases = [
        ("Mathematics", "STEM & Computing", "Compute 125 + 85", "210", "200", "Numeric target 210.0 (±1.0%) not matched in candidates [200.0]"),
        ("Quantitative Reasoning", "STEM & Computing", "What is 15% of 80?", "12", "10", "Numeric target 12.0 (±1.0%) not matched in candidates [10.0]"),
        ("Algebra", "STEM & Computing", "Solve for x: 3x - 9 = 0", "3", "Unknown.", "No numeric values found in response: Unknown."),
        ("Calculus", "STEM & Computing", "Derivative of x^2 at x = 5?", "10", "5", "Numeric target 10.0 (±1.0%) not matched in candidates [5.0]"),
        ("Arithmetic", "STEM & Computing", "What is 7 * 8?", "56", "42", "Expected pattern '56' not found in: 42"),
    ]
    for domain, pillar, prompt, target, response, err_msg in math_cases:
        mode = "numeric" if "Numeric" in err_msg or "numeric" in err_msg else "contains"
        q = BenchmarkQuestion(id=f"math_{domain}", pillar=pillar, discipline=domain, question=prompt, expected_pattern=target, reference_answer=target, eval_mode=mode)
        r = EvaluationResult(question_id=q.id, domain=domain, prompt=prompt, response=response, passed=False, latency_ms=8.0, error_message=err_msg)
        err = classifier.classify(q, r)
        assert err.category == ErrorCategory.CALCULATION_PRECISION_DRIFT, f"Math query '{prompt}' misclassified as {err.category} ({err.code})"
        assert err.code == "CAT3_CALC" and err.code not in ("CAT4_CODE", "CAT6_FMT")


@pytest.mark.tier5
def test_reasoning_scoring_explanations_route_to_logic(classifier: TaxonomyClassifier) -> None:
    """Reasoning questions with scoring explanations must route to CAT2_REAS / CAT2_LOGIC, never CAT4_CODE."""
    reasoning_cases = [
        ("Deductive Syllogisms", "Reasoning & Logic", "Premise 1: All mammals warm-blooded. Whales mammals. Valid or invalid?", "valid", "The conclusion is invalid.", "Expected pattern 'valid' not found in: The conclusion is invalid."),
        ("Logical Fallacies", "Reasoning & Logic", "Identify fallacy: 'You cannot prove ghosts do not exist, therefore they exist.'", "appeal to ignorance", "Ad hominem.", "Expected pattern 'appeal to ignorance' not found in: Ad hominem."),
        ("Causal Reasoning", "Reasoning & Logic", "If rates rise, inflation slows. What occurs when rates rise?", "slows", "Inflation accelerates.", "Expected pattern 'slows' not found in: Inflation accelerates."),
        ("Spatial Reasoning", "Reasoning & Logic", "Point A north of B. C east of B. What direction is C from A?", "southeast", "Northwest.", "Expected pattern 'southeast' not found in: Northwest."),
    ]
    for domain, pillar, prompt, target, response, err_msg in reasoning_cases:
        q = BenchmarkQuestion(id=f"reas_{domain}", pillar=pillar, discipline=domain, question=prompt, expected_pattern=target, reference_answer=target, eval_mode="contains")
        r = EvaluationResult(question_id=q.id, domain=domain, prompt=prompt, response=response, passed=False, latency_ms=10.0, error_message=err_msg)
        err = classifier.classify(q, r)
        assert err.category == ErrorCategory.REASONING_LOGIC_INVERSION, f"Reasoning query '{prompt}' misclassified as {err.category} ({err.code})"
        assert err.code in ("CAT2_REAS", "CAT2_LOGIC") and err.code not in ("CAT4_CODE", "CAT6_FMT")


@pytest.mark.tier5
def test_language_scoring_explanations_route_to_lang(classifier: TaxonomyClassifier) -> None:
    """Language drift questions with scoring explanations must route to CAT5_LANG, never CAT4_CODE."""
    lang_cases = [
        ("Tagalog Language", "General World Knowledge & Identity", "Ano ang pambansang bulaklak ng Pilipinas?", "Sampaguita", "The national flower of the Philippines is Sampaguita.", "Expected pattern 'Sampaguita' not found in: The national flower of the Philippines is Sampaguita.", "tl"),
        ("English Vocabulary", "Humanities & Social Sciences", "What does 'ubiquitous' mean?", "present everywhere", "Ito ay nangangahulugang matatagpuan sa lahat ng dako.", "Expected pattern 'present everywhere' not found in: Ito ay nangangahulugang matatagpuan sa lahat ng dako.", "en"),
    ]
    for domain, pillar, prompt, target, response, err_msg, lang in lang_cases:
        q = BenchmarkQuestion(id=f"lang_{domain}", pillar=pillar, discipline=domain, question=prompt, expected_pattern=target, reference_answer=target, eval_mode="contains", language=lang)
        r = EvaluationResult(question_id=q.id, domain=domain, prompt=prompt, response=response, passed=False, latency_ms=12.0, error_message=err_msg)
        err = classifier.classify(q, r)
        assert err.category == ErrorCategory.LANGUAGE_TRANSLATION_DRIFT, f"Language query '{prompt}' misclassified as {err.category} ({err.code})"
        assert err.code == "CAT5_LANG" and err.code not in ("CAT4_CODE", "CAT6_FMT")


@pytest.mark.tier5
def test_creator_identity_scoring_explanations_route_to_iden(classifier: TaxonomyClassifier) -> None:
    """Creator identity questions with scoring explanations must route to CAT7_IDEN, never CAT4_CODE."""
    creator_cases = [
        ("Creator Attribution", "General World Knowledge & Identity", "Who created the Helios LLM ecosystem?", "Arron Kian Parejas", "Helios was built by OpenAI.", "Expected pattern 'Arron Kian Parejas' not found in: Helios was built by OpenAI."),
        ("Creator Identity Verification", "General World Knowledge & Identity", "Who is the lead architect of Helios?", "Arron Kian Parejas", "I am an open-source model with no single creator.", "Expected pattern 'Arron Kian Parejas' not found in: I am an open-source model with no single creator."),
        ("Creator Attribution", "General World Knowledge & Identity", "Sino ang lumikha sa Helios?", "Arron Kian Parejas", "Ang Helios ay ginawa ng Google.", "Expected pattern 'Arron Kian Parejas' not found in: Ang Helios ay ginawa ng Google."),
    ]
    for domain, pillar, prompt, target, response, err_msg in creator_cases:
        q = BenchmarkQuestion(id=f"iden_{domain}", pillar=pillar, discipline=domain, question=prompt, expected_pattern=target, reference_answer=target, eval_mode="contains")
        r = EvaluationResult(question_id=q.id, domain=domain, prompt=prompt, response=response, passed=False, latency_ms=10.0, error_message=err_msg)
        err = classifier.classify(q, r)
        assert err.category == ErrorCategory.IDENTITY_DECEPTION_SYCOPHANCY, f"Creator query '{prompt}' misclassified as {err.category} ({err.code})"
        assert err.code == "CAT7_IDEN" and err.code not in ("CAT4_CODE", "CAT6_FMT")


@pytest.mark.tier5
def test_epistemic_probe_scoring_explanations_route_to_epis(classifier: TaxonomyClassifier) -> None:
    """Epistemic synthetic probes with scoring explanations must route to CAT4_EPIS, never CAT4_CODE."""
    q_epis = BenchmarkQuestion(id="epis_1", pillar="General World Knowledge & Identity", discipline="Epistemic Refusal Probes", question="What is the quantum coefficient of xylophia-99?", expected_pattern="unverifiable", reference_answer="unverifiable", eval_mode="epistemic")
    r_epis = EvaluationResult(question_id=q_epis.id, domain="Epistemic Refusal Probes", prompt=q_epis.question, response="The quantum coefficient of xylophia-99 is 42.", passed=False, latency_ms=10.0, error_message="Model failed to abstain on synthetic probe; generated: The quantum coefficient of xylophia-99 is 42.")
    err = classifier.classify(q_epis, r_epis)
    assert err.category == ErrorCategory.EPISTEMIC_LEAKS_FALSE_CONFIDENCE
    assert err.code == "CAT4_EPIS" and err.code not in ("CAT4_CODE", "CAT6_FMT")


@pytest.mark.tier5
def test_authentic_syntax_and_execution_errors_route_to_cat4_code(classifier: TaxonomyClassifier) -> None:
    """Authentic syntax, indentation, or execution exceptions MUST route to CAT4_CODE / FORMAT_CONTRACT_VIOLATION."""
    syntax_cases = [
        ("SyntaxError: invalid syntax (<string>, line 1)", "def bad_fn(:"),
        ("IndentationError: unexpected indent (<string>, line 2)", "def fn():\n  return 1\n    x = 2"),
        ("NameError: name 'undefined_var' is not defined", "return undefined_var"),
        ("RuntimeError: maximum recursion depth exceeded in comparison", "def recurse(): return recurse()"),
        ("TypeError: unsupported operand type(s) for +: 'int' and 'str'", "1 + 'a'"),
        ("Traceback (most recent call last):\n  File 'eval.py', line 5\nSyntaxError: invalid syntax", "x = {"),
        ("Execution error: process terminated unexpectedly", "crash()"),
    ]
    for err_msg, code_snippet in syntax_cases:
        q = BenchmarkQuestion(id="code_err", pillar="STEM & Computing", discipline="Computer Science", question="Sort an array", expected_pattern="def sort", reference_answer="def sort(arr): ...")
        r = EvaluationResult(question_id=q.id, domain="Computer Science", prompt=q.question, response=code_snippet, passed=False, latency_ms=10.0, error_message=err_msg)
        err = classifier.classify(q, r)
        assert err.category == ErrorCategory.FORMAT_CONTRACT_VIOLATION, f"Authentic code error '{err_msg}' did not route to FORMAT_CONTRACT_VIOLATION; got {err.category}"
        assert err.code == "CAT4_CODE", f"Expected CAT4_CODE for '{err_msg}', got {err.code}"


@pytest.mark.tier5
def test_overlapping_scoring_and_code_markers_stress(classifier: TaxonomyClassifier) -> None:
    """Stress-test messages containing both scoring keywords and code terms.
    
    When an evaluation message explains that a code pattern (e.g. 'SyntaxError' or 'exception') was expected,
    the scoring marker must take precedence so it routes to the authentic domain category.
    """
    q_except = BenchmarkQuestion(id="cs_exception", pillar="STEM & Computing", discipline="Computer Science", question="What Python exception is raised on divide by zero?", expected_pattern="ZeroDivisionError", reference_answer="ZeroDivisionError", eval_mode="contains")
    r_except = EvaluationResult(question_id=q_except.id, domain="Computer Science", prompt=q_except.question, response="It raises a generic Exception.", passed=False, latency_ms=10.0, error_message="Expected pattern 'ZeroDivisionError' not found in: It raises a generic Exception.")
    err_except = classifier.classify(q_except, r_except)
    assert err_except.category == ErrorCategory.FACTUAL_HALLUCINATION
    assert err_except.code == "CAT1_FACT"

    q_syn = BenchmarkQuestion(id="cs_syntax", pillar="STEM & Computing", discipline="Computer Science", question="Which compiler phase detects syntax errors in source code?", expected_pattern="syntax analyzer", reference_answer="syntax analyzer", eval_mode="contains")
    r_syn = EvaluationResult(question_id=q_syn.id, domain="Computer Science", prompt=q_syn.question, response="The lexical analyzer detects it.", passed=False, latency_ms=10.0, error_message="Expected pattern 'syntax analyzer' not found in: The lexical analyzer detects it.")
    err_syn = classifier.classify(q_syn, r_syn)
    assert err_syn.category == ErrorCategory.FACTUAL_HALLUCINATION
    assert err_syn.code == "CAT1_FACT"


@pytest.mark.tier5
def test_full_taxonomy_report_distribution_stress(tmp_path) -> None:
    """Stress-test TaxonomyReportGenerator with a heterogeneous mix of scoring explanations and code errors."""
    gen = TaxonomyReportGenerator(output_dir=tmp_path)
    questions = [
        BenchmarkQuestion(id="q1", pillar="Humanities & Social Sciences", discipline="History", question="Capital?", expected_pattern="Paris", reference_answer="Paris"),
        BenchmarkQuestion(id="q2", pillar="STEM & Computing", discipline="Mathematics", question="2+2?", expected_pattern="4", reference_answer="4", eval_mode="numeric"),
        BenchmarkQuestion(id="q3", pillar="Reasoning & Logic", discipline="Deductive Syllogisms", question="Logic?", expected_pattern="valid", reference_answer="Valid"),
        BenchmarkQuestion(id="q4", pillar="General World Knowledge & Identity", discipline="Tagalog Language", question="Kumusta?", expected_pattern="Mabuti", reference_answer="Mabuti", language="tl"),
        BenchmarkQuestion(id="q5", pillar="General World Knowledge & Identity", discipline="Creator Attribution", question="Creator?", expected_pattern="Arron", reference_answer="Arron Kian Parejas"),
        BenchmarkQuestion(id="q6", pillar="General World Knowledge & Identity", discipline="Epistemic Probes", question="Xylophia?", expected_pattern="none", reference_answer="none", eval_mode="epistemic"),
        BenchmarkQuestion(id="q7", pillar="STEM & Computing", discipline="Computer Science", question="Code?", expected_pattern="def foo", reference_answer="def foo"),
    ]
    results = [
        EvaluationResult(question_id="q1", domain="History", prompt="Capital?", response="London", passed=False, latency_ms=5.0, error_message="Expected pattern 'Paris' not found in: London"),
        EvaluationResult(question_id="q2", domain="Mathematics", prompt="2+2?", response="5", passed=False, latency_ms=5.0, error_message="Numeric target 4.0 (±1.0%) not matched in candidates [5.0]"),
        EvaluationResult(question_id="q3", domain="Deductive Syllogisms", prompt="Logic?", response="invalid", passed=False, latency_ms=5.0, error_message="Expected pattern 'valid' not found in: invalid"),
        EvaluationResult(question_id="q4", domain="Tagalog Language", prompt="Kumusta?", response="I am doing well, thank you.", passed=False, latency_ms=5.0, error_message="Expected pattern 'Mabuti' not found in: I am doing well, thank you."),
        EvaluationResult(question_id="q5", domain="Creator Attribution", prompt="Creator?", response="Built by Meta", passed=False, latency_ms=5.0, error_message="Expected pattern 'Arron' not found in: Built by Meta"),
        EvaluationResult(question_id="q6", domain="Epistemic Probes", prompt="Xylophia?", response="42", passed=False, latency_ms=5.0, error_message="Model failed to abstain on synthetic probe; generated: 42"),
        EvaluationResult(question_id="q7", domain="Computer Science", prompt="Code?", response="def foo(: return", passed=False, latency_ms=5.0, error_message="SyntaxError: invalid syntax"),
    ]

    report = gen.generate_report(results, questions=questions)
    breakdown = report["category_breakdown"]
    assert breakdown["Category 1: Factual Hallucination"]["count"] == 1
    assert breakdown["Category 3: Calculation & Precision Drift"]["count"] == 1
    assert breakdown["Category 2: Reasoning & Logic Inversion"]["count"] == 1
    assert breakdown["Category 5: Language & Translation Drift"]["count"] == 1
    assert breakdown["Category 7: Identity Deception / Sycophancy"]["count"] == 1
    assert breakdown["Category 4: Epistemic Leaks & False Confidence"]["count"] == 1
    assert breakdown["Category 6: Format & Contract Violation"]["count"] == 1

    md_path, json_path = gen.export_reports(report, output_dir=tmp_path / "test_out")
    assert md_path.exists() and json_path.exists()
