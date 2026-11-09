# HELIOS Dataset Catalog — Single Source of Truth Sync (177 External Datasets)

Last Synchronized: **2026-11-09 17:02:31 UTC** · Active Pretrain Category: **`all`** · Total External Datasets: **177** · Consolidated OneTouch Records: **120,616** (**90,376,089 tokens**)

## 1. Pretraining Categories & Record Breakdown (`/Volumes/One Touch/Helios/datasets/`)

| Category Key | Category Name | External Datasets | Consolidated Records | `helios chat` Command |
| :--- | :--- | ---: | ---: | :--- |
| `all` | **All Omni-Domain (177 Datasets)** | 177 datasets | 120,616 records | `/train all --category all --minutes 20` |
| `fineweb` | **FineWeb & Encyclopedic Web** | 32 datasets | 11,248 records | `/train all --category fineweb --minutes 20` |
| `math` | **Complex Mathematics & Proofs** | 12 datasets | 17,473 records | `/train all --category math --minutes 20` |
| `history` | **World History & Ancient Civilizations** | 22 datasets | 4,465 records | `/train all --category history --minutes 20` |
| `coding` | **Coding, Algorithms & Software Engineering** | 27 datasets | 53,628 records | `/train all --category coding --minutes 20` |
| `science` | **Science, Biology, Medicine & Ecology** | 32 datasets | 23,798 records | `/train all --category science --minutes 20` |
| `geography` | **Places, World Geography & Landmarks** | 10 datasets | 1,248 records | `/train all --category geography --minutes 20` |
| `law` | **Law, Governance, Philosophy & Ethics** | 10 datasets | 1,248 records | `/train all --category law --minutes 20` |
| `agents` | **Agentic Skills & Model Context Protocol (MCP)** | 22 datasets | 10,004 records | `/train all --category agents --minutes 20` |
| `tagalog` | **Bilingual Filipino / Tagalog & Multilingual** | 10 datasets | 1,248 records | `/train all --category tagalog --minutes 20` |

---

## 2. Complete Inventory of All 177 Pretraining Datasets

| # | Dataset Name | Hugging Face / Source ID | Domain Category | Source URL |
| ---: | :--- | :--- | :--- | :--- |
| 1 | **FineWeb-Edu (1.3T Tokens)** | `HuggingFaceFW/fineweb-edu` | 1. Foundational Web & Encyclopedic | [Link](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| 2 | **FineWeb (15T Tokens)** | `HuggingFaceFW/fineweb` | 1. Foundational Web & Encyclopedic | [Link](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| 3 | **DCLM-Baseline (4T Tokens)** | `mlfoundations/dCLM-baseline` | 1. Foundational Web & Encyclopedic | [Link](https://huggingface.co/datasets/mlfoundations/dCLM-baseline) |
| 4 | **Cosmopedia v2** | `HuggingFaceTB/cosmopedia-v2` | 1. Foundational Web & Encyclopedic | [Link](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2) |
| 5 | **Wikipedia Multi-Domain Dump** | `wikimedia/wikipedia` | 1. Foundational Web & Encyclopedic | [Link](https://huggingface.co/datasets/wikimedia/wikipedia) |
| 6 | **Wikidata Knowledge Graph Triples** | `wikidata/wikidata` | 1. Foundational Web & Encyclopedic | [Link](https://www.wikidata.org) |
| 7 | **RedPajama-Data-v2 (30T Tokens)** | `togethercomputer/RedPajama-Data-V2` | 1. Foundational Web & Encyclopedic | [Link](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) |
| 8 | **RefinedWeb (Falcon Pretraining Corpus)** | `tiiuae/falcon-refinedweb` | 1. Foundational Web & Encyclopedic | [Link](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) |
| 9 | **C4 (Colossal Clean Crawled Corpus)** | `allenai/c4` | 1. Foundational Web & Encyclopedic | [Link](https://huggingface.co/datasets/allenai/c4) |
| 10 | **The Pile (EleutherAI 825GB Open Corpus)** | `EleutherAI/the_pile` | 1. Foundational Web & Encyclopedic | [Link](https://huggingface.co/datasets/EleutherAI/the_pile) |
| 11 | **SmolLM Systems & CS Corpus** | `HuggingFaceTB/smollm-corpus` | 2. Technology, Systems & Cloud | [Link](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) |
| 12 | **SecQA Cybersecurity & Network Security** | `zefang-liu/secqa` | 2. Technology, Systems & Cloud | [Link](https://huggingface.co/datasets/zefang-liu/secqa) |
| 13 | **OpenHermes-2.5 Tech & Systems** | `teknium/OpenHermes-2.5` | 2. Technology, Systems & Cloud | [Link](https://huggingface.co/datasets/teknium/OpenHermes-2.5) |
| 14 | **IETF RFC Technical Standards Archive** | `ietf-tools/rfc-archive` | 2. Technology, Systems & Cloud | [Link](https://www.rfc-editor.org) |
| 15 | **SlimOrca Curated Reasoning & Technology** | `Open-Orca/SlimOrca` | 2. Technology, Systems & Cloud | [Link](https://huggingface.co/datasets/Open-Orca/SlimOrca) |
| 16 | **Infinity-Instruct (7M+ Instruction Mega-Corpus)** | `BAAI/Infinity-Instruct` | 2. Technology, Systems & Cloud | [Link](https://huggingface.co/datasets/BAAI/Infinity-Instruct) |
| 17 | **Cybersecurity Knowledge Dataset v1** | `AlicanKiraz0/Cybersecurity-Dataset-v1` | 2. Technology, Systems & Cloud | [Link](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-v1) |
| 18 | **Linux Kernel Documentation & Git History** | `torvalds/linux` | 2. Technology, Systems & Cloud | [Link](https://github.com/torvalds/linux) |
| 19 | **Kubernetes Architecture & Operational Guides** | `kubernetes/website` | 2. Technology, Systems & Cloud | [Link](https://kubernetes.io/docs) |
| 20 | **Cloud Native Computing Foundation (CNCF) Specs** | `cncf/landscape` | 2. Technology, Systems & Cloud | [Link](https://landscape.cncf.io) |
| 21 | **MMLU Computer Science & Engineering** | `cais/mmlu` | 2. Technology, Systems & Cloud | [Link](https://huggingface.co/datasets/cais/mmlu) |
| 22 | **UltraChat 200k Technical Dialogues** | `HuggingFaceH4/ultrachat_200k` | 2. Technology, Systems & Cloud | [Link](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |
| 23 | **Encyclopaedia Britannica Historical Editions** | `Vintage-LLM/Britannica` | 3. World History & Timelines | [Link](https://huggingface.co/datasets/Vintage-LLM/Britannica) |
| 24 | **VDR Academic History & Geography** | `racineai/VDR_History_Geography` | 3. World History & Timelines | [Link](https://huggingface.co/datasets/racineai/VDR_History_Geography) |
| 25 | **Project Gutenberg Historical Library** | `manu/project_gutenberg` | 3. World History & Timelines | [Link](https://huggingface.co/datasets/manu/project_gutenberg) |
| 26 | **BigScience Historical Texts** | `bigscience-historical-texts` | 3. World History & Timelines | [Link](https://huggingface.co/bigscience-historical-texts) |
| 27 | **MMLU World, European & US History** | `cais/mmlu` | 3. World History & Timelines | [Link](https://huggingface.co/datasets/cais/mmlu) |
| 28 | **Pleias Synthetic History Corpus** | `pleias/Synthetic-History-Corpus` | 3. World History & Timelines | [Link](https://huggingface.co/datasets/pleias/Synthetic-History-Corpus) |
| 29 | **Internet History Sourcebooks Project (Fordham)** | `fordham/ihsp` | 3. World History & Timelines | [Link](https://sourcebooks.fordham.edu) |
| 30 | **United States National Archives Declassified Papers** | `us-national-archives/historical` | 3. World History & Timelines | [Link](https://www.archives.gov) |
| 31 | **Diplomatic History & Foreign Relations of the US** | `state-dept/frus` | 3. World History & Timelines | [Link](https://history.state.gov) |
| 32 | **Harvard Library Historical Books Collection** | `Harvard-Library/historical-books` | 3. World History & Timelines | [Link](https://library.harvard.edu) |
| 33 | **Avalon Project Documents in Law, History and Diplomacy** | `yale-law/avalon` | 3. World History & Timelines | [Link](https://avalon.law.yale.edu) |
| 34 | **Imperial War Museums Historical Oral Archive** | `iwm/sound-archive` | 3. World History & Timelines | [Link](https://www.iwm.org.uk) |
| 35 | **CoT Reasoning: The Ancient Past** | `mattwesney/CoT_Reasoning_The_Ancient_Past` | 4. Ancient Civilizations & Archaeology | [Link](https://huggingface.co/datasets/mattwesney/CoT_Reasoning_The_Ancient_Past) |
| 36 | **Ancient Languages & Epigraphy** | `AncientLanguages` | 4. Ancient Civilizations & Archaeology | [Link](https://huggingface.co/AncientLanguages) |
| 37 | **CATMuS Classical & Medieval Manuscripts** | `CATMuS/medieval` | 4. Ancient Civilizations & Archaeology | [Link](https://huggingface.co/datasets/CATMuS/medieval) |
| 38 | **Perseus Digital Library (Classical Greece & Rome)** | `perseus-digital-library` | 4. Ancient Civilizations & Archaeology | [Link](https://www.perseus.tufts.edu) |
| 39 | **OpenPecha Ancient Buddhist & Himalayan Texts** | `openpecha/ancient-texts` | 4. Ancient Civilizations & Archaeology | [Link](https://huggingface.co/openpecha) |
| 40 | **Cuneiform Digital Library Initiative (CDLI)** | `cdli/inscriptions` | 4. Ancient Civilizations & Archaeology | [Link](https://cdli.mpiwg-berlin.mpg.de) |
| 41 | **Thesaurus Linguae Graecae (TLG)** | `uci/tlg` | 4. Ancient Civilizations & Archaeology | [Link](http://stephanus.tlg.uci.edu) |
| 42 | **Papyri.info Ancient Egyptian & Hellenistic Papyri** | `duke-columbia/papyri` | 4. Ancient Civilizations & Archaeology | [Link](https://papyri.info) |
| 43 | **Digital Epigraphy and Archaeology Project (DEA)** | `dea/inscriptions` | 4. Ancient Civilizations & Archaeology | [Link](https://www.digitalepigraphy.org) |
| 44 | **Pre-Colonial Philippine Inscriptions & Laguna Copperplate** | `Darknecrocities/Helios-Historical` | 4. Ancient Civilizations & Archaeology | [Link](https://huggingface.co/Darknecrocities/Helios) |
| 45 | **NuminaMath-CoT (860K Olympiad Math)** | `AI-MO/NuminaMath-CoT` | 5. Complex Mathematics & Formal Proofs | [Link](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| 46 | **NVIDIA OpenMathInstruct-2 (14M Math)** | `nvidia/OpenMathInstruct-2` | 5. Complex Mathematics & Formal Proofs | [Link](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) |
| 47 | **OpenWebMath (14.7B Tokens)** | `open-web-math/open-web-math` | 5. Complex Mathematics & Formal Proofs | [Link](https://huggingface.co/datasets/open-web-math/open-web-math) |
| 48 | **Proof-Pile-2 (Algebraic Stack & Formal Proofs)** | `EleutherAI/proof-pile-2` | 5. Complex Mathematics & Formal Proofs | [Link](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |
| 49 | **MathCode-Pile (Program-of-Thought Math)** | `MathGenie/MathCode-Pile` | 5. Complex Mathematics & Formal Proofs | [Link](https://huggingface.co/datasets/MathGenie/MathCode-Pile) |
| 50 | **GSM8K Multi-Step Quantitative Reasoning** | `openai/gsm8k` | 5. Complex Mathematics & Formal Proofs | [Link](https://huggingface.co/datasets/openai/gsm8k) |
| 51 | **Hendrycks Competition MATH** | `lighteval/MATH` | 5. Complex Mathematics & Formal Proofs | [Link](https://huggingface.co/datasets/lighteval/MATH) |
| 52 | **MetaMathQA (395K Augmented Math)** | `meta-math/MetaMathQA` | 5. Complex Mathematics & Formal Proofs | [Link](https://huggingface.co/datasets/meta-math/MetaMathQA) |
| 53 | **MathQA (Amini et al. Algorithmic Math)** | `allenai/math_qa` | 5. Complex Mathematics & Formal Proofs | [Link](https://huggingface.co/datasets/allenai/math_qa) |
| 54 | **Lean 4 Mathlib Formal Mathematical Proofs** | `leanprover-community/mathlib4` | 5. Complex Mathematics & Formal Proofs | [Link](https://github.com/leanprover-community/mathlib4) |
| 55 | **Isabelle/HOL Archive of Formal Proofs** | `afp/isabelle-afp` | 5. Complex Mathematics & Formal Proofs | [Link](https://www.isa-afp.org) |
| 56 | **Coq Proof Assistant Standard Library** | `coq/coq-stdlib` | 5. Complex Mathematics & Formal Proofs | [Link](https://coq.inria.fr) |
| 57 | **Structured Biology & Ecology Textbook** | `burgerbee/biology_textbook` | 6. Biology, Genetics & Life Sciences | [Link](https://huggingface.co/datasets/burgerbee/biology_textbook) |
| 58 | **MedMCQA (194K Biomedical & Clinical)** | `openlifescienceai/medmcqa` | 6. Biology, Genetics & Life Sciences | [Link](https://huggingface.co/datasets/openlifescienceai/medmcqa) |
| 59 | **PubMedQA Biomedical Literature** | `qiaojin/PubMedQA` | 6. Biology, Genetics & Life Sciences | [Link](https://huggingface.co/datasets/qiaojin/PubMedQA) |
| 60 | **BioASQ Biomedical Semantic QA** | `kroshan/BioASQ` | 6. Biology, Genetics & Life Sciences | [Link](https://huggingface.co/datasets/kroshan/BioASQ) |
| 61 | **UniProtKB Protein & Genomics Annotations** | `uniprot/uniprotkb` | 6. Biology, Genetics & Life Sciences | [Link](https://www.uniprot.org) |
| 62 | **MedQA (USMLE Clinical Question Bank)** | `GBaker/MedQA-USMLE-4-options` | 6. Biology, Genetics & Life Sciences | [Link](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) |
| 63 | **NCBI Gene & RefSeq Genomic Sequence DB** | `ncbi/refseq` | 6. Biology, Genetics & Life Sciences | [Link](https://www.ncbi.nlm.nih.gov/refseq) |
| 64 | **ChEMBL Bioactive Molecules & Target Assays** | `chembl/chembl_db` | 6. Biology, Genetics & Life Sciences | [Link](https://www.ebi.ac.uk/chembl) |
| 65 | **PDB (Protein Data Bank 3D Macromolecular Structures)** | `wwpdb/pdb` | 6. Biology, Genetics & Life Sciences | [Link](https://www.rcsb.org) |
| 66 | **ClinicalTrials.gov Protocol & Evidence DB** | `nih/clinicaltrials` | 6. Biology, Genetics & Life Sciences | [Link](https://clinicaltrials.gov) |
| 67 | **MMLU Clinical Medicine & Biology Suite** | `cais/mmlu` | 6. Biology, Genetics & Life Sciences | [Link](https://huggingface.co/datasets/cais/mmlu) |
| 68 | **Ensembl Genome Browser & Comparative Genomics** | `ensembl/genomes` | 6. Biology, Genetics & Life Sciences | [Link](https://www.ensembl.org) |
| 69 | **AllenAI SciQ (Physics, Chemistry, Biology, Ecology)** | `allenai/sciq` | 7. Nature, Ecology & Earth Sciences | [Link](https://huggingface.co/datasets/allenai/sciq) |
| 70 | **AI2 ARC (Natural & Earth Science Challenge)** | `allenai/ai2_arc` | 7. Nature, Ecology & Earth Sciences | [Link](https://huggingface.co/datasets/allenai/ai2_arc) |
| 71 | **OpenBookQA (Natural Laws & Elementary Science)** | `allenai/openbookqa` | 7. Nature, Ecology & Earth Sciences | [Link](https://huggingface.co/datasets/allenai/openbookqa) |
| 72 | **Climate & Earth Systems Corpus** | `climatebert/climate_sentiment` | 7. Nature, Ecology & Earth Sciences | [Link](https://huggingface.co/datasets/climatebert/climate_sentiment) |
| 73 | **GBIF Global Biodiversity Information Facility** | `gbif/species-occurrences` | 7. Nature, Ecology & Earth Sciences | [Link](https://www.gbif.org) |
| 74 | **NASA Earth Science Data Systems (ESDS)** | `nasa/earth-data` | 7. Nature, Ecology & Earth Sciences | [Link](https://earthdata.nasa.gov) |
| 75 | **USGS Geological Surveys & Seismic Data** | `usgs/geology-earthquake` | 7. Nature, Ecology & Earth Sciences | [Link](https://www.usgs.gov) |
| 76 | **IPCC Assessment Reports (Sixth Assessment AR6)** | `ipcc/ar6-reports` | 7. Nature, Ecology & Earth Sciences | [Link](https://www.ipcc.ch/assessment-report/ar6) |
| 77 | **NOAA Oceanic & Atmospheric Climatology DB** | `noaa/ncei` | 7. Nature, Ecology & Earth Sciences | [Link](https://www.ncei.noaa.gov) |
| 78 | **IUCN Red List of Threatened Species** | `iucn/redlist` | 7. Nature, Ecology & Earth Sciences | [Link](https://www.iucnredlist.org) |
| 79 | **Worldbuilding & Extreme Geography** | `archit11/worldbuilding` | 8. Places, World Geography & Landmarks | [Link](https://huggingface.co/datasets/archit11/worldbuilding) |
| 80 | **GeoNames Global Gazetteer (11M+ Places)** | `geonames/allCountries` | 8. Places, World Geography & Landmarks | [Link](https://www.geonames.org) |
| 81 | **UNESCO World Heritage Sites Catalog** | `unesco/world-heritage-list` | 8. Places, World Geography & Landmarks | [Link](https://whc.unesco.org) |
| 82 | **Philippine Geopolitical & Provincial Atlas** | `Darknecrocities/Helios` | 8. Places, World Geography & Landmarks | [Link](https://huggingface.co/Darknecrocities/Helios) |
| 83 | **OpenStreetMap Planetary Geodata Extracts** | `osm/planet` | 8. Places, World Geography & Landmarks | [Link](https://planet.openstreetmap.org) |
| 84 | **Natural Earth Public Domain Map Dataset** | `natural-earth/vector-data` | 8. Places, World Geography & Landmarks | [Link](https://www.naturalearthdata.com) |
| 85 | **CIA World Factbook Geopolitical Profiles** | `cia/world-factbook` | 8. Places, World Geography & Landmarks | [Link](https://www.cia.gov/the-world-factbook) |
| 86 | **UN Cartographic Section & Geographic Database** | `un/geospatial` | 8. Places, World Geography & Landmarks | [Link](https://www.un.org/geospatial) |
| 87 | **Global Administrative Areas (GADM) Boundaries** | `gadm/boundaries` | 8. Places, World Geography & Landmarks | [Link](https://gadm.org) |
| 88 | **World Protected Areas Database (WDPA)** | `unep-wcmc/wdpa` | 8. Places, World Geography & Landmarks | [Link](https://www.protectedplanet.net) |
| 89 | **The Stack v2 (67.5TB Source Code)** | `bigcode/the-stack-v2` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| 90 | **StarCoderData (86 Programming Languages)** | `bigcode/starcoderdata` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/bigcode/starcoderdata) |
| 91 | **CodeAlpaca-20k (Multi-Language Coding)** | `sahil2801/CodeAlpaca-20k` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) |
| 92 | **Python Code Instructions 18k Alpaca** | `iamtarun/python_code_instructions_18k_alpaca` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) |
| 93 | **Magicoder-OSS-Instruct-75K** | `ise-uiuc/Magicoder-OSS-Instruct-75K` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) |
| 94 | **Magicoder-Evol-Instruct-110K** | `ise-uiuc/Magicoder-Evol-Instruct-110K` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) |
| 95 | **Evol-Instruct-Code-80k-v1** | `nickrosh/Evol-Instruct-Code-80k-v1` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) |
| 96 | **Glaive Code Assistant v3** | `GlaiveAI/glaive-code-assistant-v3` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/GlaiveAI/glaive-code-assistant-v3) |
| 97 | **CodeFeedback-Filtered-Instruction (156K)** | `m-a-p/CodeFeedback-Filtered-Instruction` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) |
| 98 | **TigerBot Kaggle LeetCode Solutions (2K)** | `TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k) |
| 99 | **Hendrycks APPS (Advanced Programming Problem Solving)** | `hendrycks/apps` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/hendrycks/apps) |
| 100 | **DeepMind CodeContests Competitive Programming** | `deepmind/code_contests` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/deepmind/code_contests) |
| 101 | **HumanEvalPlus (Extended Python Coding Benchmarks)** | `evalplus/humanevalplus` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/evalplus/humanevalplus) |
| 102 | **MBPPPlus (Mostly Basic Python Problems Extended)** | `evalplus/mbppplus` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/evalplus/mbppplus) |
| 103 | **SmolLM Python-Edu Pretraining Subset** | `HuggingFaceTB/smollm-corpus` | 9. Coding, Algorithms & Competitive Programming | [Link](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) |
| 104 | **SWE-bench Verified Software Engineering** | `princeton-nlp/SWE-bench_Verified` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) |
| 105 | **SWE-bench Lite (Standard Bug Resolution Benchmark)** | `princeton-nlp/SWE-bench_Lite` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) |
| 106 | **SWE-bench Extra** | `nebius/SWE-bench-extra` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://huggingface.co/datasets/nebius/SWE-bench-extra) |
| 107 | **CommitPackFT (Verified Git Commits)** | `bigcode/commitpackft` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://huggingface.co/datasets/bigcode/commitpackft) |
| 108 | **SWE-gym Autonomous Software Engineering Environments** | `CMU-Perceptual-Computing-Lab/swe-gym` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://huggingface.co/datasets/CMU-Perceptual-Computing-Lab/swe-gym) |
| 109 | **GitHub Code Clean High-Star Repository Corpus** | `codeparrot/github-code-clean` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://huggingface.co/datasets/codeparrot/github-code-clean) |
| 110 | **SWE-smith Synthetic SWE Problem Generation** | `SWE-bench/SWE-smith` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://huggingface.co/datasets/SWE-bench/SWE-smith) |
| 111 | **CodeXGLUE Software Engineering Multi-Task Suite** | `microsoft/CodeXGLUE` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://github.com/microsoft/CodeXGLUE) |
| 112 | **Commit Chronicle Software Evolution History** | `JetBrains-Research/commit-chronicle` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://huggingface.co/datasets/JetBrains-Research/commit-chronicle) |
| 113 | **GitBug Real-World Bug-Fix Benchmark** | `gitbug/gitbug-benchmark` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://github.com/gitbug-benchmark/gitbug) |
| 114 | **Defects4J Real Java Faults Dataset** | `rjust/defects4j` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://github.com/rjust/defects4j) |
| 115 | **BugsInPy Real Python Faults Dataset** | `soarsmu/BugsInPy` | 10. Software Engineering (SWE), Architecture & GitHub PRs | [Link](https://github.com/soarsmu/BugsInPy) |
| 116 | **ToolMind (160K Graph-Correlated Agent Trajectories)** | `Nanbeige/ToolMind` | 11. Agentic Skills & Multi-Step Planning | [Link](https://huggingface.co/datasets/Nanbeige/ToolMind) |
| 117 | **Agentic Tool-Call Dataset (12K Deep Research)** | `pyromind/agentic-tool-call-dataset-12k` | 11. Agentic Skills & Multi-Step Planning | [Link](https://huggingface.co/datasets/pyromind/agentic-tool-call-dataset-12k) |
| 118 | **THUDM AgentInstruct (Multi-Turn Agent Skills)** | `THUDM/AgentInstruct` | 11. Agentic Skills & Multi-Step Planning | [Link](https://huggingface.co/datasets/THUDM/AgentInstruct) |
| 119 | **UltraInteract (Planning, Coding & Reflection)** | `openbmb/UltraInteract_sft` | 11. Agentic Skills & Multi-Step Planning | [Link](https://huggingface.co/datasets/openbmb/UltraInteract_sft) |
| 120 | **OpenThoughts-114k (Deep Reasoning & Reflection)** | `open-thoughts/OpenThoughts-114k` | 11. Agentic Skills & Multi-Step Planning | [Link](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| 121 | **Bespoke-Stratos-17k (Deep Reasoning Distillation)** | `bespokelabs/Bespoke-Stratos-17k` | 11. Agentic Skills & Multi-Step Planning | [Link](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) |
| 122 | **ServiceNow R1-Distill-SFT** | `ServiceNow-AI/R1-Distill-SFT` | 11. Agentic Skills & Multi-Step Planning | [Link](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT) |
| 123 | **Magpie-Reasoning-V2 (250K CoT)** | `Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B` | 11. Agentic Skills & Multi-Step Planning | [Link](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B) |
| 124 | **GAIA General AI Assistant Benchmark** | `gaia-benchmark/GAIA` | 11. Agentic Skills & Multi-Step Planning | [Link](https://huggingface.co/datasets/gaia-benchmark/GAIA) |
| 125 | **AgentBench Multi-Environment Agent Benchmark** | `THUDM/AgentBench` | 11. Agentic Skills & Multi-Step Planning | [Link](https://github.com/THUDM/AgentBench) |
| 126 | **WebArena Autonomous Web Navigation & Execution** | `web-arena/webarena` | 11. Agentic Skills & Multi-Step Planning | [Link](https://webarena.dev) |
| 127 | **OSWorld Multimodal Operating System Agent Benchmark** | `xlang-ai/OSWorld` | 11. Agentic Skills & Multi-Step Planning | [Link](https://os-world.github.io) |
| 128 | **Salesforce xLAM Function-Calling (60K)** | `Salesforce/xlam-function-calling-60k` | 12. Model Context Protocol (MCP), APIs & Function Calling | [Link](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) |
| 129 | **Glaive Function-Calling v2 (113K)** | `glaiveai/glaive-function-calling-v2` | 12. Model Context Protocol (MCP), APIs & Function Calling | [Link](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) |
| 130 | **Gorilla Berkeley Function-Calling Leaderboard (BFCL)** | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | 12. Model Context Protocol (MCP), APIs & Function Calling | [Link](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) |
| 131 | **Model Context Protocol (MCP) Official Specification** | `modelcontextprotocol/specification` | 12. Model Context Protocol (MCP), APIs & Function Calling | [Link](https://modelcontextprotocol.io) |
| 132 | **ToolACE Function Calling & Tool Interaction Benchmark** | `Team-ACE/ToolACE` | 12. Model Context Protocol (MCP), APIs & Function Calling | [Link](https://huggingface.co/datasets/Team-ACE/ToolACE) |
| 133 | **Hermes Function-Calling v1 Dataset** | `NousResearch/hermes-function-calling-v1` | 12. Model Context Protocol (MCP), APIs & Function Calling | [Link](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) |
| 134 | **ToolBench (16,000+ Real REST APIs)** | `OpenBMB/ToolBench` | 12. Model Context Protocol (MCP), APIs & Function Calling | [Link](https://github.com/OpenBMB/ToolBench) |
| 135 | **APIBench Executable API Invocations** | `gorilla-llm/APIBench` | 12. Model Context Protocol (MCP), APIs & Function Calling | [Link](https://github.com/ShishirPatil/gorilla) |
| 136 | **RestBench Real RESTful Web Service Interactions** | `Yifan-Song793/RestBench` | 12. Model Context Protocol (MCP), APIs & Function Calling | [Link](https://github.com/Yifan-Song793/RestBench) |
| 137 | **JSON-RPC 2.0 Architectural Specification Test Suite** | `jsonrpc/spec-suite` | 12. Model Context Protocol (MCP), APIs & Function Calling | [Link](https://www.jsonrpc.org/specification) |
| 138 | **The Stack v2 Documentation & ReadTheDocs** | `bigcode/the-stack-v2-docs` | 13. Technical Documentation, RFCs & Manuals | [Link](https://devdocs.io) |
| 139 | **MDN Web Docs & Developer Manuals** | `mdn/content` | 13. Technical Documentation, RFCs & Manuals | [Link](https://developer.mozilla.org) |
| 140 | **W3C Standards & Web Specifications Catalog** | `w3c/specifications` | 13. Technical Documentation, RFCs & Manuals | [Link](https://www.w3.org/TR) |
| 141 | **Python Official Documentation & PEP Archive** | `python/cpython-docs` | 13. Technical Documentation, RFCs & Manuals | [Link](https://docs.python.org/3) |
| 142 | **Rust Reference, Rustonomicon & Standard Library Docs** | `rust-lang/reference` | 13. Technical Documentation, RFCs & Manuals | [Link](https://doc.rust-lang.org/reference) |
| 143 | **POSIX IEEE 1003.1 Standard Specifications** | `opengroup/posix` | 13. Technical Documentation, RFCs & Manuals | [Link](https://pubs.opengroup.org/onlinepubs/9699919799) |
| 144 | **OpenAPI Specification 3.1 Standards** | `OAI/OpenAPI-Specification` | 13. Technical Documentation, RFCs & Manuals | [Link](https://spec.openapis.org/oas/v3.1.0) |
| 145 | **PostgreSQL Official Internals & Documentation** | `postgres/postgres-docs` | 13. Technical Documentation, RFCs & Manuals | [Link](https://www.postgresql.org/docs) |
| 146 | **Docker & OCI Container Runtime Specs** | `opencontainers/runtime-spec` | 13. Technical Documentation, RFCs & Manuals | [Link](https://github.com/opencontainers/runtime-spec) |
| 147 | **IEEE Xplore Standards Index** | `ieee/standards-index` | 13. Technical Documentation, RFCs & Manuals | [Link](https://standards.ieee.org) |
| 148 | **AllenAI PeS2o (40M+ Open Academic Papers)** | `allenai/peS2o` | 14. Scientific Research, Literature & Peer Review | [Link](https://huggingface.co/datasets/allenai/peS2o) |
| 149 | **AllenAI SciRIFF (Scientific Literature Instruction)** | `allenai/SciRIFF` | 14. Scientific Research, Literature & Peer Review | [Link](https://huggingface.co/datasets/allenai/SciRIFF) |
| 150 | **arXiv & PubMed Scientific Papers Corpus** | `armanc/scientific_papers` | 14. Scientific Research, Literature & Peer Review | [Link](https://huggingface.co/datasets/armanc/scientific_papers) |
| 151 | **OpenAlex Global Research Catalog (250M+ Works)** | `openalex/openalex-snapshot` | 14. Scientific Research, Literature & Peer Review | [Link](https://openalex.org) |
| 152 | **Semantic Scholar Open Research Corpus (S2ORC)** | `allenai/s2orc` | 14. Scientific Research, Literature & Peer Review | [Link](https://huggingface.co/datasets/allenai/s2orc) |
| 153 | **PLOS Open Access Complete Research Corpus** | `plos/open-access` | 14. Scientific Research, Literature & Peer Review | [Link](https://plos.org) |
| 154 | **DOAJ Directory of Open Access Journals** | `doaj/article-metadata` | 14. Scientific Research, Literature & Peer Review | [Link](https://doaj.org) |
| 155 | **BioRxiv & MedRxiv Preprint Archive** | `cshl/biorxiv-medrxiv` | 14. Scientific Research, Literature & Peer Review | [Link](https://www.biorxiv.org) |
| 156 | **Europe PMC Open Access Full Text Biomedical Articles** | `europepmc/oa-articles` | 14. Scientific Research, Literature & Peer Review | [Link](https://europepmc.org) |
| 157 | **CERN Zenodo Open Science Repository** | `cern/zenodo` | 14. Scientific Research, Literature & Peer Review | [Link](https://zenodo.org) |
| 158 | **Pile of Law (Legal Texts, Statutes & Court Opinions)** | `pile-of-law/pile-of-law` | 15. Law, Governance, Philosophy & Social Sciences | [Link](https://huggingface.co/datasets/pile-of-law/pile-of-law) |
| 159 | **Free Law Project CourtListener Archive** | `freelawproject/courtlistener` | 15. Law, Governance, Philosophy & Social Sciences | [Link](https://www.courtlistener.com) |
| 160 | **MMLU Law, Jurisprudence & Professional Ethics** | `cais/mmlu` | 15. Law, Governance, Philosophy & Social Sciences | [Link](https://huggingface.co/datasets/cais/mmlu) |
| 161 | **Stanford Encyclopedia of Philosophy (SEP) Archive** | `stanford/sep` | 15. Law, Governance, Philosophy & Social Sciences | [Link](https://plato.stanford.edu) |
| 162 | **World Bank Open Data & Governance Indicators** | `worldbank/indicators` | 15. Law, Governance, Philosophy & Social Sciences | [Link](https://data.worldbank.org) |
| 163 | **United Nations Official Document System (ODS)** | `un/ods-documents` | 15. Law, Governance, Philosophy & Social Sciences | [Link](https://documents.un.org) |
| 164 | **European Union EUR-Lex Legal Texts** | `eu/eur-lex` | 15. Law, Governance, Philosophy & Social Sciences | [Link](https://eur-lex.europa.eu) |
| 165 | **Philippine Supreme Court E-Library & Jurisprudence** | `sc-ph/jurisprudence` | 15. Law, Governance, Philosophy & Social Sciences | [Link](https://elibrary.judiciary.gov.ph) |
| 166 | **MMLU Macroeconomics, Microeconomics & Finance** | `cais/mmlu` | 15. Law, Governance, Philosophy & Social Sciences | [Link](https://huggingface.co/datasets/cais/mmlu) |
| 167 | **PhilPapers Directory of Philosophical Research** | `philpapers/archive` | 15. Law, Governance, Philosophy & Social Sciences | [Link](https://philpapers.org) |
| 168 | **CulturaX (Multilingual Tagalog Corpus)** | `uonlp/CulturaX` | 16. Bilingual Filipino / Tagalog & Multilingual Foundations | [Link](https://huggingface.co/datasets/uonlp/CulturaX) |
| 169 | **OSCAR-2301 & CC-100 (Tagalog Web Corpus)** | `oscar-corpus/OSCAR-2301` | 16. Bilingual Filipino / Tagalog & Multilingual Foundations | [Link](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) |
| 170 | **Alpaca-Cleaned General Instruction Corpus** | `yahma/alpaca-cleaned` | 16. Bilingual Filipino / Tagalog & Multilingual Foundations | [Link](https://huggingface.co/datasets/yahma/alpaca-cleaned) |
| 171 | **Aya Multi-Lingual Instruction Collection (101 Languages)** | `CohereForAI/aya_collection` | 16. Bilingual Filipino / Tagalog & Multilingual Foundations | [Link](https://huggingface.co/datasets/CohereForAI/aya_collection) |
| 172 | **Capybara Multi-Turn High-Quality Dialogue Corpus** | `LDJnr/Capybara` | 16. Bilingual Filipino / Tagalog & Multilingual Foundations | [Link](https://huggingface.co/datasets/LDJnr/Capybara) |
| 173 | **Open-Platypus Curated Reasoning Corpus** | `garage-bAInd/Open-Platypus` | 16. Bilingual Filipino / Tagalog & Multilingual Foundations | [Link](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) |
| 174 | **NVIDIA HelpSteer2 Alignment Corpus** | `nvidia/HelpSteer2` | 16. Bilingual Filipino / Tagalog & Multilingual Foundations | [Link](https://huggingface.co/datasets/nvidia/HelpSteer2) |
| 175 | **Glot500 Multilingual Parallel Sentences** | `cis-lmu/Glot500` | 16. Bilingual Filipino / Tagalog & Multilingual Foundations | [Link](https://huggingface.co/datasets/cis-lmu/Glot500) |
| 176 | **Philippine National Historical Commission Digital Archives** | `nhcp/historical-texts` | 16. Bilingual Filipino / Tagalog & Multilingual Foundations | [Link](https://nhcp.gov.ph) |
| 177 | **Komisyon sa Wikang Filipino (KWF) Diksiyonaryo at Ortograpiya** | `kwf/diksiyonaryo` | 16. Bilingual Filipino / Tagalog & Multilingual Foundations | [Link](https://kwf.gov.ph) |

---

## 3. Continuous Ingestion & Pretraining Cluster Log

| Cluster ID | Stage | Source | Revision | Accepted Docs | Tokens | Content Hash | Completed At |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `onetouch_all_177ds` | `continuous_pretrain` | `sot/datasets_sot.json (177 datasets)` | `v0.3.0` | 120,616 | 90,376,089 | `verified_onetouch` | 2026-11-09 17:02:31 UTC |
