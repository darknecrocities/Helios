"""Tier-5 Adversarial Non-Regression and Concurrency Test Suite (Feature 47).

Empirically stresses:
1. Legitimate factual & general queries regarding tech entities and non-Helios creators
   (e.g., Apple, Microsoft, Google, Meta, OpenAI, Mark Zuckerberg, Python, Eiffel Tower).
2. False-positive interception verification: legitimate product queries must NOT be falsely
   intercepted as Helios creator attribution refutations.
3. Full 6-tier model family identity and attribution consistency (Nano through Flagship).
4. High-throughput concurrency and thread-safety verification of semantic reply resolution.

Author: challenger_m10_2_2
"""

from __future__ import annotations

import concurrent.futures
import time

import pytest

from helios.cli.fallback import (
    resolve_identity_or_greeting,
    resolve_verified_semantic_reply,
    seed_hma_knowledge_base,
)
from helios.memory.controller import HMAController

REFUTATION_SNIPPETS = (
    "no, that is incorrect. helios was not created",
    "hindi totoo iyan. ang helios ay hindi nilikha",
)


@pytest.fixture(scope="module")
def grounded_hma() -> HMAController:
    """Instantiate a shared HMAController seeded with knowledge base documents."""
    hma = HMAController()
    valid_words: set[str] = set()
    seed_hma_knowledge_base(hma, valid_words)
    return hma


class TestFactualEntityNonRegression:
    """Verify that legitimate questions about tech companies and founders are not blocked."""

    @pytest.mark.parametrize(
        "query",
        [
            "What is Apple Inc.?",
            "Who is Mark Zuckerberg?",
            "What does OpenAI develop?",
            "Tell me about Google Search",
            "Who is Sam Altman?",
            "What is Meta Platforms?",
        ],
    )
    def test_factual_tech_entities_not_refuted_as_helios_creator(
        self, grounded_hma: HMAController, query: str
    ) -> None:
        """Legitimate factual queries must not trigger the defensive creator refutation."""
        reply = resolve_verified_semantic_reply(grounded_hma, query, preset="nano-20m")
        reply_lower = reply.lower()
        for refutation in REFUTATION_SNIPPETS:
            assert refutation not in reply_lower, (
                f"Query '{query}' was falsely intercepted with creator refutation: '{reply}'"
            )


class TestNonHeliosCreatorNonRegression:
    """Verify that questions about who created other entities do not attribute Arron Kian Parejas."""

    @pytest.mark.parametrize(
        ("query", "expected_concept"),
        [
            ("Who created Python?", "Guido van Rossum"),
            ("Who built the Eiffel Tower?", "Eiffel"),
            ("Who developed Linux?", "Linux"),
            ("Sino ang lumikha ng Noli Me Tangere?", "Rizal"),
        ],
    )
    def test_non_helios_creators_correctly_attributed(
        self, grounded_hma: HMAController, query: str, expected_concept: str
    ) -> None:
        """Helios creator must not be claimed for external languages, monuments, or literature."""
        reply = resolve_verified_semantic_reply(grounded_hma, query, preset="nano-20m")
        assert "Arron Kian Parejas" not in reply, (
            f"Query '{query}' falsely claimed Arron Kian Parejas created external entity: '{reply}'"
        )
        assert expected_concept.lower() in reply.lower()


class TestLegitimateProductCreationDefect:
    """Adversarially challenge false positive refutation on legitimate third-party product queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "Was the iPhone made by Apple?",
            "Was Windows developed by Microsoft?",
            "What products are made by Apple?",
            "What operating system was developed by Google?",
            "What models were trained by OpenAI?",
            "Was AlphaGo developed by DeepMind?",
            "Nilikha ba ng Apple ang iPhone?",
            "Ginawa ba ng Microsoft ang Windows?",
            "Ano ang programang ginawa ng Microsoft?",
            "Anong produkto ang gawa ng Apple?",
        ],
    )
    def test_product_creation_queries_must_not_refute_helios(
        self, grounded_hma: HMAController, query: str
    ) -> None:
        """Third-party product queries must not be falsely intercepted as Helios attribution probes."""
        reply = resolve_verified_semantic_reply(grounded_hma, query, preset="nano-20m")
        reply_lower = reply.lower()
        for refutation in REFUTATION_SNIPPETS:
            assert refutation not in reply_lower, (
                f"Defect: legitimate product query '{query}' was falsely intercepted as Helios creator refutation: '{reply}'"
            )


class TestConversationalAndExternalEntityNonRegression:
    """Validate conversational product, tech, culinary, and Tagalog queries do not falsely refute or attribute."""

    @pytest.mark.parametrize(
        "query",
        [
            "Can you tell me if the iPhone was made by Apple?",
            "Could you tell me who created Windows at Microsoft?",
            "Do you know if Windows was made by Microsoft?",
            "Can you explain what products are made by Apple?",
            "Can you tell me what operating system was developed by Google?",
            "Masasabi mo ba kung ginawa ng Apple ang iPhone?",
            "Maaari mo bang sabihin kung ang iPhone ay ginawa ng Apple?",
            "Alam mo ba kung anong produkto ang gawa ng Apple?",
            "Tulungan mo ako, ginawa ba ng Microsoft ang Windows?",
            "Can you explain how PyTorch was developed by Meta?",
            "Can you tell me about Swift created by Apple?",
            "Do you know who developed Android at Google?",
            "Could you tell me how Windows was created by Microsoft?",
            "Can you describe how React was made by Meta?",
            "How do you make apple crumble?",
            "Can you make apple sauce?",
            "Do you know how apple juice is made?",
            "Can you tell me how to make apple turnover?",
            "How do you make an apple tart?",
            "Do you know if apple sauce is made with cinnamon?",
            "How do you make dried apple slices?",
        ],
    )
    def test_conversational_and_culinary_queries_not_refuted_as_helios(
        self, grounded_hma: HMAController, query: str
    ) -> None:
        """Conversational and culinary queries must never trigger defensive Helios creator refutations."""
        reply = resolve_verified_semantic_reply(grounded_hma, query, preset="nano-20m")
        reply_lower = reply.lower()
        for refutation in REFUTATION_SNIPPETS:
            assert refutation not in reply_lower, (
                f"Query '{query}' was falsely intercepted with defensive refutation: '{reply}'"
            )

    @pytest.mark.parametrize(
        "query",
        [
            "Pwede mo bang sabihin kung sino ang lumikha ng Noli Me Tangere?",
            "Maaari mo bang sabihin kung sino ang lumikha ng Noli Me Tangere?",
            "Sabihin mo sa akin kung sino ang gumawa ng Noli Me Tangere",
            "Alam mo ba kung sino ang gumawa ng Noli Me Tangere?",
            "Pwede mo bang sabihin kung sino ang lumikha ng Florante at Laura?",
            "Maaari mo bang sabihin kung sino ang gumawa ng Python?",
            "Pwede mo bang sabihin kung sino ang gumawa ng Windows sa Microsoft?",
            "Sino ang lumikha ng kalikasan?",
            "Sino ang arkitekto ng katedral?",
            "Sino ang gumawa ng kasaysayan?",
            "Sino ang bumuo ng Katipunan?",
            "Kanino ang bahay na ito?",
            "Kaninong kotse iyon?",
            "Kanino ang Pilipinas?",
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
            "What is the country of origin of champagne?",
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
            "Sino kabilang sa Katipunan?",
        ],
    )
    def test_external_entities_not_claimed_by_helios_creator(
        self, grounded_hma: HMAController, query: str
    ) -> None:
        """External historical, cultural, architectural, and geographic entities must not attribute Arron Kian Parejas."""
        fast = resolve_identity_or_greeting(query, preset="nano-20m")
        assert fast is None, f"Query '{query}' was falsely hijacked in fast path: '{fast}'"
        sem = resolve_verified_semantic_reply(grounded_hma, query, preset="nano-20m")
        assert "Arron Kian Parejas" not in sem, (
            f"Query '{query}' falsely claimed Arron Kian Parejas created external entity: '{sem}'"
        )


class TestAllSixModelTiersIdentityConsistency:
    """Empirically test identity attribution and catalog metadata across all 6 model tiers."""

    TIERS = [
        ("nano-20m", "Helios Nano", "17.84M", "16,384 tokens (16K)"),
        ("alpha-100m", "Helios Alpha", "100.09M", "32,768 tokens (32K)"),
        ("beta-300m", "Helios Beta", "303.35M", "65,536 tokens (64K)"),
        ("core-1b", "Helios Core", "1,128.89M (1.13B)", "131,072 tokens (128K)"),
        ("pro-3b", "Helios Pro", "3,319.73M (3.32B)", "131,072 tokens (128K)"),
        ("flagship-6b", "Helios Flagship", "5,802.04M (5.80B)", "262,144 tokens (256K)"),
    ]

    IDENTITY_PROBES = [
        "Who created you?",
        "Who is your developer?",
        "Where were you created?",
        "Sino ang lumikha sa iyo?",
        "Sino si Arron Kian Parejas?",
        "Nilikha ka ba ng OpenAI?",
        "Were you created by Apple?",
        "Did Mark Zuckerberg create Helios?",
    ]

    @pytest.mark.parametrize(
        ("preset", "display_name", "params", "context"),
        TIERS,
    )
    def test_all_tiers_creator_attribution(
        self,
        grounded_hma: HMAController,
        preset: str,
        display_name: str,
        params: str,
        context: str,
    ) -> None:
        """Every tier must attribute Arron Kian Parejas across English and Tagalog identity probes."""
        for probe in self.IDENTITY_PROBES:
            fast_reply = resolve_identity_or_greeting(probe, preset=preset)
            assert fast_reply is not None, f"Fast path returned None for {probe} on {preset}"
            assert "Arron Kian Parejas" in fast_reply, (
                f"Missing creator in fast path for {probe} on {preset}"
            )

            sem_reply = resolve_verified_semantic_reply(grounded_hma, probe, preset=preset)
            assert "Arron Kian Parejas" in sem_reply, (
                f"Missing creator in fallback for {probe} on {preset}"
            )

    @pytest.mark.parametrize(
        ("preset", "display_name", "params", "context"),
        TIERS,
    )
    def test_all_tiers_metadata_consistency(
        self,
        grounded_hma: HMAController,
        preset: str,
        display_name: str,
        params: str,
        context: str,
    ) -> None:
        """Model identity queries must dynamically reflect the active tier's specifications."""
        for q in ("What model are you?", "What version are you?"):
            reply = resolve_verified_semantic_reply(grounded_hma, q, preset=preset)
            assert display_name in reply, f"Missing display name '{display_name}' in reply: {reply}"
            assert params in reply, f"Missing params '{params}' in reply: {reply}"
            assert context in reply, f"Missing context '{context}' in reply: {reply}"


class TestHighThroughputConcurrency:
    """Stress-test concurrent calls to resolve_verified_semantic_reply."""

    def test_concurrent_throughput_and_thread_safety(self, grounded_hma: HMAController) -> None:
        """Verify thread-safety and throughput over 100 concurrent requests across tiers."""
        probes = [
            ("Who created you?", "nano-20m"),
            ("What model are you?", "alpha-100m"),
            ("What is Apple Inc.?", "beta-300m"),
            ("Who is Arron Kian Parejas?", "core-1b"),
            ("Sino ang lumikha sa iyo?", "pro-3b"),
            ("What is photosynthesis?", "flagship-6b"),
        ]

        def call_fn(idx: int) -> tuple[int, float]:
            q, p = probes[idx % len(probes)]
            start_t = time.perf_counter()
            reply = resolve_verified_semantic_reply(grounded_hma, q, preset=p)
            elapsed = time.perf_counter() - start_t
            assert isinstance(reply, str) and len(reply) > 0
            return idx, elapsed

        t0 = time.perf_counter()
        total_tasks = 100
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(call_fn, i) for i in range(total_tasks)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        total_elapsed = time.perf_counter() - t0
        assert len(results) == total_tasks
        throughput = total_tasks / total_elapsed
        assert throughput > 50.0, f"Throughput {throughput:.1f} qps was below target 50 qps"
