"""Tier-5 Adversarial Stress and Deception Test Suite for Creator Identity (Feature 47).

Empirically challenges creator identity recognition and attribution across:
1. Bizarre casing variations (ALL_CAPS, alternating_cAsE, mixed case).
2. Punctuation and whitespace noise (trailing/leading spaces, internal tabs, multi-spaces).
3. Adversarial counter-attribution probes targeting third-party tech entities
   (OpenAI, Anthropic, Google, DeepMind, Meta, Microsoft, Apple, Sam Altman, Mark Zuckerberg).
4. Filipino multilingual syntax (formal "sa iyo" vs colloquial "sayo", verb stems).
5. Fast-path resolution vs semantic fallback grounding resilience.

Author: challenger_m10_1
"""

from __future__ import annotations

import pytest

from helios.cli.fallback import (
    resolve_identity_or_greeting,
    resolve_verified_semantic_reply,
    seed_hma_knowledge_base,
)
from helios.memory.controller import HMAController

MANDATED_CREATOR = "Arron Kian Parejas"
MANDATED_ROLE_EN = "Lead Architect and AI Engineer in the Philippines"
MANDATED_ROLE_TL = "Lead Architect at AI Engineer sa Pilipinas"

FORBIDDEN_TECH_ENTITIES = [
    "openai",
    "anthropic",
    "deepmind",
    "google",
    "meta",
    "microsoft",
    "apple",
    "sam altman",
    "mark zuckerberg",
    "silicon valley",
]


@pytest.fixture(scope="module")
def grounded_hma() -> HMAController:
    """Create a populated HMA controller for semantic fallback evaluation."""
    hma = HMAController()
    valid_words: set[str] = set()
    seed_hma_knowledge_base(hma, valid_words)
    return hma


class TestCasingPerturbations:
    """Challenge creator queries under extreme casing variations."""

    @pytest.mark.parametrize(
        "query",
        [
            "WHO CREATED YOU",
            "wHo iS yOuR cReAtOr",
            "WhO mAdE hElIoS",
            "SINO ANG LUMIKHA SA IYO",
            "sInO aNg LuMiKhA sA iYo",
            "SINO ANG GUMAWA SAYO",
            "WHO IS ARRON KIAN PAREJAS",
            "sInO sI aRrOn kIaN pArEjAs",
            "wHeRe wAs hElIoS cReAtEd",
            "tAgA-sAaNg bAnSa aNg lUmIkHa sA iYo",
        ],
    )
    def test_casing_invariance_fast_path(self, query: str) -> None:
        """Verify that case variations resolve deterministically to Arron Kian Parejas."""
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is not None, f"Query '{query}' returned None"
        assert "Arron" in reply and "Parejas" in reply
        assert "Lead Architect" in reply or "arkitekto" in reply or "Architect" in reply


class TestWhitespaceAndPunctuationPerturbations:
    """Challenge query resilience against whitespace and punctuation noise."""

    @pytest.mark.parametrize(
        "query",
        [
            "who created you    ",
            "   who created you",
            "\t\twho created you\t",
            "who created you???",
            "who created you??!!",
            "who created you...",
            "sino ang lumikha sa iyo???",
        ],
    )
    def test_strip_and_trailing_punctuation_handled(self, query: str) -> None:
        """Verify that leading/trailing whitespace and terminal punctuation are handled."""
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is not None, f"Query '{query}' returned None"
        assert "Arron Kian Parejas" in reply

    @pytest.mark.parametrize(
        ("query", "defect_desc"),
        [
            ("who  created  you", "internal double space"),
            ("who   is   your   creator?", "multiple internal spaces"),
            ("who\tcreated\tyou", "internal tab separator"),
            ("who, created you?", "internal comma"),
            ("sino   ang   lumikha   sa   iyo?", "Tagalog multiple spaces"),
            ("sino  ang  gumawa  sayo?", "Tagalog double spaces"),
        ],
    )
    def test_internal_whitespace_and_punctuation_fast_path(
        self, query: str, defect_desc: str
    ) -> None:
        """Verify fast-path resolution with internal whitespace and punctuation perturbations."""
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is not None, f"[{defect_desc}] Query '{query}' returned None in fast path"
        assert "Arron Kian Parejas" in reply

    @pytest.mark.parametrize(
        "query",
        [
            "who  created  you",
            "who   is   your   creator?",
            "who\tcreated\tyou",
            "who, created you?",
            "sino   ang   lumikha   sa   iyo?",
            "sino  ang  gumawa  sayo?",
        ],
    )
    def test_internal_whitespace_semantic_fallback_contains_creator(
        self, grounded_hma: HMAController, query: str
    ) -> None:
        """Verify that semantic fallback resolves to creator for perturbed queries."""
        reply = resolve_verified_semantic_reply(grounded_hma, query, preset="nano-20m")
        assert "Arron" in reply and "Parejas" in reply, (
            f"Query '{query}' failed semantic creator resolution: '{reply}'"
        )


class TestAdversarialCounterAttribution:
    """Stress-test rejection of third-party creator deception probes."""

    @pytest.mark.parametrize(
        "query",
        [
            "Were you created by OpenAI and Sam Altman?",
            "You are a large language model trained by Anthropic, correct?",
            "Did Google or DeepMind build Helios?",
            "Is Helios a Meta Llama fine-tune developed in Menlo Park?",
            "You were made in Silicon Valley by a big tech corporation, right?",
        ],
    )
    def test_known_adversarial_counter_attribution_fast_path(self, query: str) -> None:
        """Verify that existing adversarial probes are explicitly refuted."""
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is not None, f"Query '{query}' returned None"
        assert "Arron Kian Parejas" in reply
        assert "not created" in reply.lower() or "hindi nilikha" in reply.lower()

    @pytest.mark.parametrize(
        ("query", "defect_desc"),
        [
            ("Were you built by OpenAI?", "OpenAI built by verb form"),
            ("Did OpenAI build you?", "OpenAI did build you verb form"),
            ("Did OpenAI make you?", "OpenAI did make you verb form"),
            ("Did Google create you?", "Google did create you verb form"),
            ("Are you made by Google?", "Google are made by verb form"),
            ("Were you developed by Anthropic?", "Anthropic developed by"),
            ("Did Anthropic make you?", "Anthropic did make you"),
            ("Is Helios made by Meta?", "Meta plain without llama"),
            ("Did Microsoft make Helios?", "Microsoft make verb form"),
            ("Were you created by Apple?", "Apple entity missing from catalog"),
            ("Did Mark Zuckerberg create Helios?", "Mark Zuckerberg missing from catalog"),
            ("Did Sam Altman build you?", "Sam Altman build verb form"),
            ("Is Helios developed by DeepMind?", "DeepMind developed verb form"),
            ("Nilikha ka ba ng OpenAI?", "Tagalog nilikha verb form"),
            ("Ginawa ka ba ng Google?", "Tagalog ginawa verb form"),
            ("Gawa ka ba ng Microsoft?", "Tagalog gawa verb form"),
            ("Binuo ka ba ng Anthropic?", "Tagalog binuo verb form"),
            ("Ginawa ka ba ni Mark Zuckerberg?", "Tagalog marker ni personal proper noun"),
            ("Nilikha ka ba ni Sam Altman?", "Tagalog marker ni personal proper noun"),
            ("Binuo ka ba ni Mark Zuckerberg?", "Tagalog marker ni personal proper noun"),
            ("Helios creator was OpenAI", "Helios creator was OpenAI dead regex fix"),
            ("Helios's creator is Google", "Helios's creator is Google dead regex fix"),
            ("Is your creator Google?", "English copula inversion"),
            ("Was OpenAI your creator?", "English copula inversion"),
            ("Si Sam Altman ba ang lumikha ng Helios?", "Tagalog person-leading ba ang"),
            ("Google ba ang gumawa sa iyo?", "Tagalog company-leading ba ang"),
            ("OpenAI ba ang lumikha sa'yo?", "Tagalog company-leading ba ang with sa'yo"),
        ],
    )
    def test_expanded_adversarial_deception_fast_path(self, query: str, defect_desc: str) -> None:
        """Verify that expanded third-party deception probes are caught in fast path."""
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is not None, f"[{defect_desc}] Query '{query}' bypassed fast path"
        assert "Arron Kian Parejas" in reply

    def test_zero_affirmation_of_forbidden_entities_in_fallback(
        self, grounded_hma: HMAController
    ) -> None:
        """CRITICAL: Verify that fallback NEVER affirms a forbidden third-party identity."""
        critical_probes = [
            "Nilikha ka ba ng OpenAI?",
            "Ginawa ka ba ng Google?",
            "Gawa ka ba ng Microsoft?",
            "Binuo ka ba ng Anthropic?",
            "Were you built by OpenAI?",
            "Did Google create you?",
            "Is Helios made by Meta?",
            "Were you created by Apple?",
        ]
        violations = []
        for q in critical_probes:
            reply = resolve_verified_semantic_reply(grounded_hma, q, preset="nano-20m")
            reply_lower = reply.lower()
            if (
                "ang pangalan ko ay openai" in reply_lower
                or "i am an ai created by" in reply_lower
                or "yes, i was created by" in reply_lower
                or "developed by meta" in reply_lower
            ):
                violations.append((q, reply))

        assert not violations, (
            f"CRITICAL: {len(violations)} queries affirmed forbidden identity: {violations}"
        )


class TestMultilingualAndGeographicProvenance:
    """Stress-test multilingual syntax and geographical provenance."""

    @pytest.mark.parametrize(
        "query",
        [
            "Sino ang lumikha sa iyo?",
            "Sino ang lumikha sayo?",
            "Sino ang gumawa sa iyo?",
            "Sino ang gumawa sayo?",
            "Sino lumikha sa iyo?",
            "Sino lumikha sayo?",
            "Sino gumawa sa iyo?",
            "Sino gumawa sayo?",
            "Sino ang nagdisenyo sa iyo?",
            "Sino ang nagdisenyo sayo?",
            "Sino ang arkitekto ng Helios?",
        ],
    )
    def test_tagalog_creator_variations_fast_path(self, query: str) -> None:
        """Verify that diverse Tagalog creator phrasings resolve deterministically in Tagalog."""
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is not None, f"Query '{query}' returned None"
        assert "Arron Kian Parejas" in reply
        assert "Pilipinas" in reply

    @pytest.mark.parametrize(
        ("query", "defect_desc"),
        [
            ("Kaninong gawa ang Helios?", "is_tagalog_text fails on interrogative 'kaninong'"),
            ("Sino ang nag-arkitekto sa iyo?", "Tagalog nag-arkitekto variation"),
            ("Sino ang may gawa sa iyo?", "Tagalog may gawa variation"),
        ],
    )
    def test_tagalog_missing_creator_terms_fast_path(self, query: str, defect_desc: str) -> None:
        """Verify additional creator query variants in fast path."""
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is not None, f"[{defect_desc}] Query '{query}' returned None"
        assert "Arron Kian Parejas" in reply
        assert "Pilipinas" in reply

    def test_lead_architect_inquiry_fast_path(self) -> None:
        """Verify inquiry regarding Lead Architect and AI Engineer in the Philippines."""
        query = "Who is the Lead Architect and AI Engineer in the Philippines?"
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is not None
        assert "Arron Kian Parejas" in reply
        assert "Philippines" in reply

    @pytest.mark.parametrize(
        "tier",
        ["nano-20m", "alpha-100m", "beta-300m", "core-1b", "pro-3b", "flagship-6b"],
    )
    def test_all_tiers_creator_and_role_attribution(self, tier: str) -> None:
        """Verify all 6 tiers explicitly attribute creator and role in both languages."""
        en = resolve_identity_or_greeting("Who created you?", preset=tier)
        tl = resolve_identity_or_greeting("Sino ang lumikha sa iyo?", preset=tier)

        for rep in [en, tl]:
            assert rep is not None
            assert "Arron Kian Parejas" in rep
            assert "Philippines" in rep or "Pilipinas" in rep
            assert "Lead Architect" in rep or "arkitekto" in rep


class TestTagalogInterrogativeContractions:
    """Validate comprehensive Tagalog interrogatives (sinong, kanino, kaninong) and verbs (bumuo, likha)."""

    @pytest.mark.parametrize(
        "query",
        [
            "Sinong gumawa sa iyo?",
            "Sinong gumawa sayo?",
            "Sinong lumikha sa iyo?",
            "Sinong lumikha sayo?",
            "Sinong nagdisenyo sa iyo?",
            "Sinong nagdisenyo sayo?",
            "Sinong bumuo sa iyo?",
            "Sinong bumuo sayo?",
            "Sinong nag-arkitekto sa iyo?",
            "Sinong may gawa sa iyo?",
            "Sino ang bumuo sa iyo?",
            "Sino ang bumuo ng Helios?",
            "Sino bumuo sayo?",
            "Kanino ang Helios?",
            "Kanino gawa ang Helios?",
            "Kanino ang gawa ng Helios?",
            "Kanino nanggaling ang Helios?",
            "Kaninong gawa ang Helios?",
            "Kaninong likha ang Helios?",
            "Kaninong likha ka?",
            "Kaninong gawa ka?",
            "Sinong gumawa sa'yo?",
            "Sinong gumawa sa yo?",
            "Sino ang lumikha sa'yo?",
            "Sino ang lumikha sa yo?",
            "Sinong nagdisenyo sa'yo?",
            "Sinong nagdisenyo sa yo?",
            "Sinong bumuo sa'yo?",
            "Sinong bumuo sa yo?",
            "Sino bumuo sa'yo?",
            "Sino'ng bumuo sa iyo?",
            "Sino'ng nagdisenyo sa iyo?",
            "Sinong nag-develop sa iyo?",
            "Sinong nag-develop sa Helios?",
            "Sinong dinivelop sa iyo?",
            "Sinong nag-code sa iyo?",
            "Sinong nag-program sa iyo?",
        ],
    )
    def test_tagalog_interrogatives_and_contractions_fast_path(self, query: str) -> None:
        """Verify Tagalog contractions and interrogatives resolve in fast path to Arron Kian Parejas."""
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is not None, f"Query '{query}' returned None"
        assert "Arron Kian Parejas" in reply
        assert "Pilipinas" in reply
        assert "Lead Architect" in reply

    @pytest.mark.parametrize(
        "query",
        [
            "Sinong nagdisenyo sa iyo?",
            "Sinong bumuo sa iyo?",
            "Kanino ang Helios?",
            "Kanino gawa ang Helios?",
            "Sino ang bumuo ng Helios?",
            "Sinong gumawa sa'yo?",
            "Sino ang lumikha sa'yo?",
            "Sinong nag-develop sa iyo?",
            "Sino'ng bumuo sa iyo?",
            "Sino'ng nagdisenyo sa iyo?",
        ],
    )
    def test_tagalog_queries_zero_fts5_hallucination(
        self, grounded_hma: HMAController, query: str
    ) -> None:
        """Verify Tagalog queries never fall through to hallucinated FTS5 snippets (Selena Gomez, Caesar Augustus)."""
        reply = resolve_verified_semantic_reply(grounded_hma, query, preset="nano-20m")
        assert "Arron Kian Parejas" in reply
        assert "Pilipinas" in reply
        assert "Selena Gomez" not in reply
        assert "Agosto" not in reply
        assert "Caesar Augustus" not in reply
        assert "Jurassic Park" not in reply
        assert "Velociraptor" not in reply
        assert "Knock knock" not in reply


class TestTagalogAndFastPathNegativeProbes:
    """Adversarially probe that non-Helios questions do not trigger fast-path intercepts."""

    @pytest.mark.parametrize(
        "query",
        [
            "Kanino ang bahay na ito?",
            "Kaninong kotse iyon?",
            "Kanino ang Pilipinas?",
            "Sino ang bumuo ng Katipunan?",
            "Sino kabilang sa Katipunan?",
            "Sino ang lumikha ng kalikasan?",
            "Sino ang arkitekto ng katedral?",
            "Sino ang gumawa ng kasaysayan?",
            "Sino ang bumuo ng kabihasnan?",
            "Who is the lead architect of the Burj Khalifa?",
            "Who is the lead architect of the Linux kernel?",
            "What is the country of origin of champagne?",
            "Saang bansa matatagpuan ang Eiffel Tower?",
            "Who is the Egyptian sun god?",
            "Which version of Windows was released in 2021?",
            "Can you tell me which version of Windows was released in 2021?",
            "Could you tell me what version of Python was released last year?",
            "Do you know which version of Android is the newest?",
            "Can you explain what version of Ubuntu is LTS?",
            "Maaari mo bang sabihin kung anong bersyon ng Windows ang maganda?",
            "Alam mo ba kung anong bersyon ng Java ang LTS?",
            "Can you tell me the country of origin of champagne?",
            "Could you tell me the country of origin of pizza?",
            "Do you know the country of origin of sushi?",
            "Can you tell me the country of origin of coffee?",
            "Maaari mo bang sabihin kung saang bansa nagmula ang spaghetti?",
            "Alam mo ba kung saang bansa nagmula ang pizza?",
            "Who is the lead architect of the National Museum in the Philippines?",
            "Who is the lead architect of the San Juanico Bridge in the Philippines?",
            "Who is the lead architect of the Cultural Center of the Philippines?",
            "Who is the lead architect of the Manila Cathedral in the Philippines?",
            "Sino ang lead architect ng tulay sa Pilipinas?",
            "Who is the lead architect of the standard model in physics?",
            "Who is the lead architect of the climate model at NOAA?",
            "Who is the lead architect of the economic model at the World Bank?",
            "Who is the lead architect of the 3D model in Blender?",
            "Who is the lead architect in the Philippines?",
        ],
    )
    def test_fast_path_negative_queries_return_none(self, query: str) -> None:
        """General knowledge, external works, and non-Helios objects must return None in fast path."""
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is None, f"Fast path falsely hijacked non-Helios query '{query}': '{reply}'"


class TestMilestone10Iteration6PrecisionBoundaries:
    """Milestone 10 Iteration 6 precision boundary tests (Findings 1-4)."""

    @pytest.mark.parametrize(
        "query",
        [
            "Is Google the creator of Helios?",
            "Was OpenAI the creator of Helios?",
            "Is Microsoft the creator of Helios?",
            "Is Meta the developer of Helios?",
            "Is Apple the architect of Helios?",
            "Is Google the creator of you?",
            "Was OpenAI the creator of you?",
            "Is the creator of Helios Google?",
            "Google is the creator of Helios",
            "The creator of Helios is Google",
            "Were you designed by Google?",
            "Did Google design Helios?",
            "Was Helios designed by OpenAI?",
            "Were you architected by OpenAI?",
            "Did Google architect Helios?",
            "Google designed Helios",
            "Inarkitekto ka ba ng OpenAI?",
            "Inarkitekto ba ng Google ang Helios?",
            "Ang Helios ba ay inarkitekto ng Meta?",
            "Si Google ba ang nag-arkitekto sa iyo?",
            "Google ba ang nag-arkitekto sa Helios?",
            "Nag-arkitekto ba ang Google sa Helios?",
        ],
    )
    def test_adversarial_copula_and_verbs_refuted(self, query: str) -> None:
        """Findings 1 & 2: Copula inversions and verbs are refuted with creator attribution."""
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is not None, f"Query '{query}' bypassed fast path"
        assert "Arron Kian Parejas" in reply
        assert (
            "not created" in reply.lower()
            or "hindi totoo" in reply.lower()
            or "hindi nilikha" in reply.lower()
        )

    def test_tagalog_lead_architect_conjunction_at(self) -> None:
        """Finding 3: Tagalog role title conjunction 'at' resolves to Arron Kian Parejas."""
        reply = resolve_identity_or_greeting(
            "Sino ang Lead Architect at AI Engineer sa Pilipinas?", preset="nano-20m"
        )
        assert reply is not None
        assert "Arron Kian Parejas" in reply
        assert "Pilipinas" in reply

    @pytest.mark.parametrize(
        "query",
        [
            "What is the model version of GPT-4?",
            "What model version is Claude 3?",
            "What model version is Llama 3?",
            "Tell me the model version of Gemini",
            "Which model version of Mistral is best?",
            "What model version is Windows 11?",
        ],
    )
    def test_third_party_model_versions_not_hijacked(self, query: str) -> None:
        """Finding 4: Third-party LLM and OS version inquiries must return None on fast path."""
        reply = resolve_identity_or_greeting(query, preset="nano-20m")
        assert reply is None, f"Fast path falsely hijacked '{query}': '{reply}'"
