"""Omni-Domain Curated Pretraining & Grounding Builder for Helios Foundation Models.

Provides high-precision encyclopedic, analytical, and step-by-step reasoning pairs across:
1. Technology, Cloud & Distributed Systems
2. World History & Historical Timelines
3. Ancient Civilizations & Archaeology
4. Complex Mathematics & Formal Proofs
5. Biology, Genetics & Life Sciences
6. Nature, Ecology & Earth Sciences
7. Places, World Geography & Landmarks
8. Coding, Algorithms & Software Engineering
9. Skills & Autonomous Agentic Workflows
10. Model Context Protocol (MCP) & JSON-RPC Orchestration
11. Technical Documentation, RFCs & API Specifications
12. Scientific Research Methodology & Literature Analysis

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from typing import Any

OMNI_DOMAIN_RECORDS: list[tuple[str, str, str]] = [
    # 1. Ancient Civilizations & Archaeology
    (
        "Ancient Civilizations & Archaeology",
        "What is Mesopotamia and why is it called the Cradle of Civilization?",
        "Mesopotamia is the historical region in Western Asia situated between the Tigris and Euphrates rivers (modern-day Iraq, eastern Syria, and southeastern Turkey). It is called the Cradle of Civilization because the Sumerians, Akkadians, Babylonians, and Assyrians pioneered urban city-states (such as Uruk and Ur), cuneiform writing (~3200 BCE), the wheel, irrigation agriculture, and codified law such as the Code of Hammurabi.",
    ),
    (
        "Ancient Civilizations & Archaeology",
        "What was the significance of the Code of Hammurabi in ancient Babylon?",
        "Enacted around 1750 BCE by the sixth Amorite king of Babylon, Hammurabi, the Code of Hammurabi is one of the oldest deciphered writings of significant length in the world, comprising 282 laws inscribed in Akkadian cuneiform on a basalt stele covering commercial contracts, property rights, family law, and scaled judicial penalties.",
    ),
    (
        "Ancient Civilizations & Archaeology",
        "What was the Rosetta Stone and how did it unlock Ancient Egyptian hieroglyphs?",
        "The Rosetta Stone is a granodiorite stele inscribed in 196 BCE during the reign of King Ptolemy V Epiphanes with three versions of the same decree: Ancient Egyptian hieroglyphs, Demotic script, and Ancient Greek. In 1822, French philologist Jean-François Champollion used the parallel Greek text and royal cartouches to decipher the phonetic and ideographic system of Egyptian hieroglyphs.",
    ),
    (
        "Ancient Civilizations & Archaeology",
        "How did Athenian democracy function in Classical Greece during the 5th century BCE?",
        "Under reforms initiated by Solon, Cleisthenes, and Pericles, Athenian direct democracy operated through the Ekklesia (citizen assembly), the Boule (Council of 500 chosen by lot), and the Dikasteria (popular law courts), empowering adult male citizens to vote directly on legislation, treaties, and ostracism.",
    ),
    (
        "Ancient Civilizations & Archaeology",
        "What was the Laguna Copperplate Inscription (900 CE) in pre-colonial Philippine history?",
        "Dated to Saka year 822 (April 21, 900 CE), the Laguna Copperplate Inscription (LCI) is the earliest known dated written document found in the Philippines. Written in Old Malay using the Early Kawi script with Old Tagalog, Old Javanese, and Sanskrit loanwords, it records the official acquittal of a substantial gold debt held by a high-ranking dignitary named Namwaran across polities including Tondo (Tundun), Paila, Binwangan, and Pulilan.",
    ),
    (
        "Ancient Civilizations & Archaeology",
        "What characterized the Indus Valley (Harappan) Civilization?",
        "Flourishing from 3300 to 1300 BCE along the Indus River basin in South Asia, the Indus Valley Civilization (centered at Mohenjo-daro, Harappa, and Dholavira) engineered rectilinear grid urban planning, baked-brick drainage and sanitation networks, standardized weights and measures, and the Great Bath.",
    ),
    (
        "Ancient Civilizations & Archaeology",
        "What were the major engineering and mathematical achievements of the Maya civilization?",
        "The Maya civilization of Mesoamerica independently developed a positional vigesimal (base-20) numeral system featuring an explicit symbol for zero, a sophisticated logosyllabic hieroglyphic script, corbel-arch stone architecture, and highly accurate astronomical calendars (the Tzolk'in and Haab').",
    ),
    (
        "Ancient Civilizations & Archaeology",
        "What was the Pax Romana in the history of the Roman Empire?",
        "The Pax Romana (27 BCE to 180 CE), initiated by Emperor Augustus and concluding with Marcus Aurelius, was a two-century period of relative internal stability, legal codification, extensive stone road networks, aqueduct construction, and Mediterranean trade integration across the Roman Empire.",
    ),
    # 2. World History & Historical Timelines
    (
        "World History & Timelines",
        "How did Johannes Gutenberg's movable-type printing press (c. 1440) transform human history?",
        "Johannes Gutenberg's mechanical movable-type printing press dramatically reduced the cost of book production across Europe, accelerating literacy, enabling the rapid dissemination of empirical treaties during the Scientific Revolution, and catalyzing the Renaissance and Protestant Reformation.",
    ),
    (
        "World History & Timelines",
        "What drove the First and Second Industrial Revolutions?",
        "The First Industrial Revolution (c. 1760–1840) was driven by coal power, James Watt's steam engine, mechanized textile manufacturing, and iron metallurgy. The Second Industrial Revolution (c. 1870–1914) expanded into Bessemer steel production, electrical power grids, internal combustion engines, telegraph/telephone telecommunications, and industrial chemistry.",
    ),
    (
        "World History & Timelines",
        "What was the Manila-Acapulco Galleon Trade (1565–1815)?",
        "The Manila-Acapulco Galleon Trade linked the Philippines (Manila) and Mexico (Acapulco) across the Pacific Ocean for 250 years, inaugurating globalized maritime commerce by exchanging Asian silk, porcelain, and spices for New World silver.",
    ),
    # 3. Complex Mathematics & Formal Proofs
    (
        "Complex Mathematics & Formal Proofs",
        "State and explain the Fundamental Theorem of Calculus.",
        "The Fundamental Theorem of Calculus links differentiation and integration in two parts: (1) If f(t) is continuous on [a, b] and F(x) = integral from a to x of f(t) dt, then F'(x) = f(x); and (2) If F is any antiderivative of f on [a, b], then the definite integral from a to b of f(x) dx equals F(b) - F(a).",
    ),
    (
        "Complex Mathematics & Formal Proofs",
        "What is Singular Value Decomposition (SVD) in linear algebra?",
        "Singular Value Decomposition factorizes any real or complex m x n matrix A into A = U * Sigma * V^T, where U is an m x m unitary/orthogonal matrix of left-singular vectors, Sigma is an m x n rectangular diagonal matrix of non-negative singular values, and V^T is the conjugate transpose of an n x n unitary matrix of right-singular vectors.",
    ),
    (
        "Complex Mathematics & Formal Proofs",
        "What is Euler's Identity and why is it celebrated in complex analysis?",
        "Euler's Identity is the equation e^(i * pi) + 1 = 0, derived from Euler's formula e^(i*x) = cos(x) + i*sin(x) evaluated at x = pi. It unites the five fundamental constants of mathematics: 0 (additive identity), 1 (multiplicative identity), pi (ratio of a circle's circumference to diameter), e (base of natural logarithms), and i (the imaginary unit).",
    ),
    (
        "Complex Mathematics & Formal Proofs",
        "State Fermat's Little Theorem in number theory.",
        "Fermat's Little Theorem states that if p is a prime number and a is any integer not divisible by p, then a^(p - 1) is congruent to 1 modulo p (a^(p-1) ≡ 1 mod p).",
    ),
    (
        "Complex Mathematics & Formal Proofs",
        "What is the Kullback-Leibler (KL) divergence between two probability distributions P and Q?",
        "The Kullback-Leibler divergence D_KL(P || Q) = sum_x P(x) * log(P(x) / Q(x)) measures the relative entropy or information lost when distribution Q is used to approximate the true distribution P. It is non-negative (D_KL >= 0 by Gibbs' inequality) and asymmetric.",
    ),
    (
        "Complex Mathematics & Formal Proofs",
        "Solve the second-order homogeneous linear differential equation y'' - 5y' + 6y = 0.",
        "The characteristic equation is r^2 - 5r + 6 = 0, which factors into (r - 2)(r - 3) = 0 with distinct real roots r_1 = 2 and r_2 = 3. Therefore, the general solution is y(x) = C_1 * e^(2x) + C_2 * e^(3x).",
    ),
    # 4. Biology, Genetics & Life Sciences
    (
        "Biology, Genetics & Life Sciences",
        "How does the CRISPR-Cas9 gene-editing system achieve targeted DNA cleavage?",
        "CRISPR-Cas9 uses a synthetic single-guide RNA (sgRNA) containing a 20-nucleotide spacer sequence that hybridizes to a complementary target DNA locus adjacent to a Protospacer Adjacent Motif (5'-NGG-3' PAM). The Cas9 endonuclease then introduces a site-specific double-stranded DNA break, which the cell repairs via Non-Homologous End Joining (NHEJ) or Homology-Directed Repair (HDR).",
    ),
    (
        "Biology, Genetics & Life Sciences",
        "What is the Central Dogma of Molecular Biology?",
        "Formulated by Francis Crick, the Central Dogma describes the directional flow of genetic information within biological systems: DNA is replicated into DNA, transcribed into messenger RNA (mRNA) by RNA polymerase, and translated into functional polypeptide proteins by ribosomes and transfer RNA (tRNA).",
    ),
    (
        "Biology, Genetics & Life Sciences",
        "Explain the three primary stages of aerobic cellular respiration.",
        "Aerobic cellular respiration oxidizes glucose (C6H12O6) to generate ATP across three stages: (1) Glycolysis in the cytosol (splitting glucose into 2 pyruvate, net 2 ATP and 2 NADH), (2) the Citric Acid (Krebs) Cycle in the mitochondrial matrix (generating NADH, FADH2, and GTP), and (3) Oxidative Phosphorylation across the inner mitochondrial membrane via the electron transport chain and ATP synthase (~30–32 ATP per glucose).",
    ),
    # 5. Nature, Ecology & Earth Sciences
    (
        "Nature, Ecology & Earth Sciences",
        "What is the theory of Plate Tectonics and what drives lithospheric plate motion?",
        "Plate tectonics explains that Earth's outer rigid lithosphere is divided into tectonic plates floating atop the ductile asthenosphere. Plate motion is driven by mantle convection currents, ridge push at mid-ocean spreading centers, and slab pull at convergent subduction zones, generating earthquakes, volcanic arcs, and mountain building (orogeny).",
    ),
    (
        "Nature, Ecology & Earth Sciences",
        "What is a trophic cascade in ecosystem ecology?",
        "A trophic cascade is an ecological phenomenon triggered by the addition or removal of top apex predators, which propagates reciprocal changes in population density and biomass down through lower trophic levels (mesopredators, herbivores, and primary producers), such as the reintroduction of gray wolves in Yellowstone National Park.",
    ),
    (
        "Nature, Ecology & Earth Sciences",
        "How does the biogeochemical nitrogen cycle convert atmospheric N2 into biologically available forms?",
        "Diazotrophic bacteria and archaea perform nitrogen fixation (using the oxygen-sensitive nitrogenase enzyme complex) to reduce inert atmospheric N2 into ammonia (NH3/NH4+), followed by nitrification by Nitrosomonas and Nitrobacter into nitrites (NO2-) and nitrates (NO3-) assimilated by plants, and denitrification by anaerobic bacteria returning N2 to the atmosphere.",
    ),
    # 6. Places, World Geography & Landmarks
    (
        "Places, World Geography & Landmarks",
        "What is the deepest known oceanic trench on Earth and what is its maximum depth?",
        "The Mariana Trench in the western Pacific Ocean is the deepest oceanic trench on Earth, reaching a maximum known depth of approximately 10,935 to 10,984 meters (35,876–36,037 feet) at the Challenger Deep.",
    ),
    (
        "Places, World Geography & Landmarks",
        "What are the UNESCO World Heritage Sites located in the Philippines?",
        "The Philippines has six inscribed UNESCO World Heritage Sites: (1) Baroque Churches of the Philippines (Manila, Santa Maria, Paoay, Miagao), (2) Tubbataha Reefs Natural Park in the Sulu Sea, (3) Rice Terraces of the Philippine Cordilleras in Ifugao, (4) Historic City of Vigan in Ilocos Sur, (5) Puerto-Princesa Subterranean River National Park in Palawan, and (6) Mount Hamiguitan Range Wildlife Sanctuary in Davao Oriental.",
    ),
    (
        "Places, World Geography & Landmarks",
        "Which river has the largest water discharge volume on Earth, and which is the longest?",
        "The Amazon River in South America has the largest water discharge volume and drainage basin on Earth (accounting for ~20% of global riverine discharge), while the Nile River in northeastern Africa (~6,650 km) and the Amazon (~6,400–6,992 km) are the two longest river systems.",
    ),
    # 7. Coding, Algorithms & Software Engineering
    (
        "Coding, Algorithms & Software Engineering",
        "How does Dijkstra's shortest-path algorithm work and what is its time complexity with a binary min-heap?",
        "Dijkstra's algorithm finds shortest paths from a source vertex in a graph with non-negative edge weights by maintaining a priority queue of tentative distances and greedily relaxing outgoing edges of the unvisited vertex with the minimum distance. Using a binary min-heap on a graph with V vertices and E edges, its time complexity is O((V + E) log V).",
    ),
    (
        "Coding, Algorithms & Software Engineering",
        "Explain the difference between optimistic concurrency control (OCC) and Multi-Version Concurrency Control (MVCC) in databases.",
        "MVCC maintains multiple immutable timestamped versions of rows so readers never block writers and writers never block readers (used in PostgreSQL and SQLite WAL). Optimistic Concurrency Control (OCC) allows transactions to execute without acquiring locks and validates at commit time whether another transaction modified the read set, aborting and retrying on conflict.",
    ),
    # 8. Skills & Autonomous Agentic Workflows
    (
        "Skills & Autonomous Agentic Workflows",
        "How does the ReAct (Reasoning + Acting) paradigm improve autonomous AI agent execution?",
        "Introduced by Yao et al., ReAct interleaves explicit verbal reasoning traces ('Thought') with environment-grounded tool invocations ('Action') and sensory feedback ('Observation'). This closed-loop trajectory enables autonomous agents to decompose multi-step goals, inspect live external state, detect execution errors, and dynamically revise plans instead of hallucinating ungrounded steps.",
    ),
    (
        "Skills & Autonomous Agentic Workflows",
        "What is the difference between a single-turn tool call and a reusable Agentic Skill?",
        "A single-turn tool call executes an atomic function schema once, whereas an Agentic Skill packages domain invariants, prerequisite checks, multi-step workflow orchestration, verification gates, and fallback recovery policies into a modular procedural contract that an agent mounts on demand.",
    ),
    # 9. Model Context Protocol (MCP) & Tool Orchestration
    (
        "Model Context Protocol (MCP)",
        "How does JSON-RPC 2.0 message framing work in the Model Context Protocol (MCP)?",
        "In MCP, client-host and server processes communicate over stdio or HTTP/SSE using UTF-8 JSON-RPC 2.0 messages containing 'jsonrpc': '2.0', a unique request 'id', a 'method' (such as 'initialize', 'tools/list', 'tools/call', 'resources/list', or 'resources/read'), and structured 'params'. Responses return either a 'result' payload or a standardized 'error' object with code and message.",
    ),
    # 10. Technical Documentation & Specifications
    (
        "Technical Documentation & Specifications",
        "What is RFC 7807 (and RFC 9457) Problem Details for HTTP APIs?",
        "RFC 7807 (updated by RFC 9457) defines a standardized JSON/XML media type ('application/problem+json') for carrying machine-readable error details in HTTP responses, specifying canonical fields: 'type' (URI reference identifying the problem class), 'title' (short human-readable summary), 'status' (HTTP status code), 'detail' (occurrence-specific explanation), and 'instance' (URI reference to the specific request).",
    ),
    (
        "Technical Documentation & Specifications",
        "What is the difference between HTTP 401 Unauthorized and HTTP 403 Forbidden according to RFC 9110?",
        "Per RFC 9110, HTTP 401 Unauthorized indicates that the request lacks valid authentication credentials for the target resource (and must include a WWW-Authenticate header), whereas HTTP 403 Forbidden indicates that the server understood the request and authenticated the principal, but refuses to authorize access.",
    ),
    # 11. Scientific Research Methodology & Literature
    (
        "Scientific Research & Literature",
        "What is a double-blind randomized controlled trial (RCT) and why is it the gold standard in clinical research?",
        "In a double-blind randomized controlled trial, participants are randomly assigned to an experimental intervention group or a control/placebo group, and neither the participants nor the investigating clinicians know the group assignments until the trial concludes. This eliminates selection bias, confounding variables, and observer/placebo expectancy effects.",
    ),
    (
        "Scientific Research & Literature",
        "What is the difference between statistical significance (p-value) and effect size (Cohen's d) in scientific research?",
        "A p-value measures the probability of observing test results at least as extreme as the sample data assuming the null hypothesis is true, which depends heavily on sample size. Effect size (such as Cohen's d = (M1 - M2) / SD_pooled) quantifies the actual standardized magnitude of the phenomenon independently of sample size.",
    ),
]


def build_omni_domain_records() -> list[dict[str, Any]]:
    """Convert curated omni-domain tuples into standardized ChatML records."""
    records: list[dict[str, Any]] = []
    for idx, (domain, prompt, response) in enumerate(OMNI_DOMAIN_RECORDS, 1):
        records.append(
            {
                "id": f"OMNI-{idx:03d}",
                "domain": domain,
                "prompt": prompt,
                "response": response,
                "text": (
                    f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>"
                ),
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
            }
        )
    return records
