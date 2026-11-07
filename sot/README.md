# Helios Dataset Single Source of Truth (SOT)

**Lead AI Engineer & Architect**: **Arron Kian Parejas** (`parejasarronkian@gmail.com`)  
**Total External Datasets Cataloged & Integrated**: **68 Datasets** across **14 Core Knowledge & Reasoning Domains** (spanning Hugging Face Hub, AllenAI PeS2o/S2ORC, OpenAlex, arXiv, Project Gutenberg, Wikidata, GeoNames, UNESCO, IETF RFCs, DevDocs, and MCP)  
**Consolidated Active Training Corpora**: **2 Consolidated Multi-Domain Corpora** (`26,330` verified ChatML records / `~12.96M` tokens stored on `/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl`, and `1,226` grounded records in `data/helios_expanded_reasoning_corpus.jsonl`)

---

## 1. Executive Summary of Datasets Trained in Helios

| Category # | Domain Category | Datasets Count | Consolidated / Streamed Records | Storage Location |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Foundational Web & Encyclopedic** | 6 | Continuous Streaming (`FineWeb-Edu`, `Cosmopedia v2`, `Wikipedia`, `Wikidata`) | `/Volumes/One Touch/Helios/data/hf_cache/` |
| **2** | **Technology, Systems & Cloud Computing** | 4 | Included in `8,000` Tech/History/Coding/Docs + `1,226` Core | `/Volumes/One Touch/Helios/datasets/` |
| **3** | **World History & Historical Timelines** | 5 | `Britannica`, `VDR_History_Geography`, `Project Gutenberg`, `MMLU History` | `/Volumes/One Touch/Helios/datasets/` |
| **4** | **Ancient Civilizations & Archaeology** | 4 | `CoT_Reasoning_The_Ancient_Past`, `AncientLanguages`, `Perseus`, `LCI 900 CE` | `/Volumes/One Touch/Helios/datasets/` |
| **5** | **Complex Mathematics & Formal Proofs** | 8 | `5,000` Harvested (`GSM8K` + Calculus/SVD/ODE/Number Theory Proofs) | `/Volumes/One Touch/Helios/datasets/` |
| **6** | **Biology, Genetics & Life Sciences** | 5 | `5,000` Harvested (`MedMCQA`, `biology_textbook`, `PubMedQA`, `CRISPR/Genetics`) | `/Volumes/One Touch/Helios/datasets/` |
| **7** | **Nature, Ecology & Earth Sciences** | 5 | `6,000` Harvested (`SciQ`, `AI2 ARC`, `OpenBookQA`, `GBIF`, `Plate Tectonics`) | `/Volumes/One Touch/Helios/datasets/` |
| **8** | **Places, World Geography & Landmarks** | 4 | `GeoNames`, `UNESCO World Heritage`, `Worldbuilding`, `Philippine Atlas` | `/Volumes/One Touch/Helios/datasets/` |
| **9** | **Coding, Algorithms & Software Engineering** | 6 | `The Stack v2`, `StarCoderData`, `CommitPackFT`, `Magicoder`, `SWE-bench` | `/Volumes/One Touch/Helios/datasets/` |
| **10** | **Skills & Autonomous Agentic Workflows** | 8 | `ToolMind`, `AgentInstruct`, `UltraInteract`, `OpenThoughts`, `ReAct` | `/Volumes/One Touch/Helios/datasets/` |
| **11** | **Model Context Protocol (MCP) & APIs** | 4 | `xLAM-60k`, `Glaive-v2`, `Berkeley BFCL`, `MCP JSON-RPC 2.0 Spec` | `/Volumes/One Touch/Helios/datasets/` |
| **12** | **Technical Documentation & Specifications** | 2 | `ReadTheDocs / DevDocs`, `MDN Web Docs`, `RFC 7807 / RFC 9110` | `/Volumes/One Touch/Helios/datasets/` |
| **13** | **Scientific Research & Peer-Reviewed Papers** | 4 | `1,100` Harvested (`AI2 ARC Challenge`) + `PeS2o`, `SciRIFF`, `arXiv`, `OpenAlex` | `/Volumes/One Touch/Helios/datasets/` |
| **14** | **Bilingual Filipino / Tagalog & General** | 3 | `8,000` Harvested (`Alpaca-Cleaned`) + `CulturaX`, `OSCAR-2301` | `/Volumes/One Touch/Helios/datasets/` |
| **—** | **Helios Core Omni-Domain Grounding Corpus** | *Internal* | `1,226` Curated Multi-Domain Records (`data/helios_expanded_reasoning_corpus.jsonl`) | In-Repo + `/Volumes/One Touch/Helios/datasets/` |
| **TOTAL** | **68 External Datasets + 2 Helios Consolidated Corpora** | **68 External** | **26,330 Active Records (~12.96M Tokens)** | **`/Volumes/One Touch/Helios/`** |

---

## 2. Complete Registry of All 68 External Datasets & Direct Links

| # | Domain Category | Dataset / Platform Name | Identifier | Direct Link |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **1. Foundational Web & Encyclopedic** | FineWeb-Edu (1.3T Tokens) | `HuggingFaceFW/fineweb-edu` | [https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| 2 | **1. Foundational Web & Encyclopedic** | FineWeb (15T Tokens) | `HuggingFaceFW/fineweb` | [https://huggingface.co/datasets/HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| 3 | **1. Foundational Web & Encyclopedic** | DCLM-Baseline (4T Tokens) | `mlfoundations/dCLM-baseline` | [https://huggingface.co/datasets/mlfoundations/dCLM-baseline](https://huggingface.co/datasets/mlfoundations/dCLM-baseline) |
| 4 | **1. Foundational Web & Encyclopedic** | Cosmopedia v2 | `HuggingFaceTB/cosmopedia-v2` | [https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2) |
| 5 | **1. Foundational Web & Encyclopedic** | Wikipedia Multi-Domain Dump | `wikimedia/wikipedia` | [https://huggingface.co/datasets/wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) |
| 6 | **1. Foundational Web & Encyclopedic** | Wikidata Knowledge Graph Triples | `wikidata/wikidata` | [https://www.wikidata.org](https://www.wikidata.org) |
| 7 | **2. Technology, Systems & Cloud** | SmolLM Systems & CS Corpus | `HuggingFaceTB/smollm-corpus` | [https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) |
| 8 | **2. Technology, Systems & Cloud** | SecQA Cybersecurity & Network Security | `zefang-liu/secqa` | [https://huggingface.co/datasets/zefang-liu/secqa](https://huggingface.co/datasets/zefang-liu/secqa) |
| 9 | **2. Technology, Systems & Cloud** | OpenHermes-2.5 Tech & Systems | `teknium/OpenHermes-2.5` | [https://huggingface.co/datasets/teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) |
| 10 | **2. Technology, Systems & Cloud** | IETF RFC Technical Standards Archive | `ietf-tools/rfc-archive` | [https://www.rfc-editor.org](https://www.rfc-editor.org) |
| 11 | **3. World History & Timelines** | Encyclopaedia Britannica Historical Editions | `Vintage-LLM/Britannica` | [https://huggingface.co/datasets/Vintage-LLM/Britannica](https://huggingface.co/datasets/Vintage-LLM/Britannica) |
| 12 | **3. World History & Timelines** | VDR Academic History & Geography | `racineai/VDR_History_Geography` | [https://huggingface.co/datasets/racineai/VDR_History_Geography](https://huggingface.co/datasets/racineai/VDR_History_Geography) |
| 13 | **3. World History & Timelines** | Project Gutenberg Historical Library | `manu/project_gutenberg` | [https://huggingface.co/datasets/manu/project_gutenberg](https://huggingface.co/datasets/manu/project_gutenberg) |
| 14 | **3. World History & Timelines** | BigScience Historical Texts | `bigscience-historical-texts` | [https://huggingface.co/bigscience-historical-texts](https://huggingface.co/bigscience-historical-texts) |
| 15 | **3. World History & Timelines** | MMLU World, European & US History | `cais/mmlu` | [https://huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| 16 | **4. Ancient Civilizations & Archaeology** | CoT Reasoning: The Ancient Past | `mattwesney/CoT_Reasoning_The_Ancient_Past` | [https://huggingface.co/datasets/mattwesney/CoT_Reasoning_The_Ancient_Past](https://huggingface.co/datasets/mattwesney/CoT_Reasoning_The_Ancient_Past) |
| 17 | **4. Ancient Civilizations & Archaeology** | Ancient Languages & Epigraphy | `AncientLanguages` | [https://huggingface.co/AncientLanguages](https://huggingface.co/AncientLanguages) |
| 18 | **4. Ancient Civilizations & Archaeology** | CATMuS Classical & Medieval Manuscripts | `CATMuS/medieval` | [https://huggingface.co/datasets/CATMuS/medieval](https://huggingface.co/datasets/CATMuS/medieval) |
| 19 | **4. Ancient Civilizations & Archaeology** | Perseus Digital Library (Greece & Rome) | `perseus-digital-library` | [https://www.perseus.tufts.edu](https://www.perseus.tufts.edu) |
| 20 | **5. Complex Mathematics & Formal Proofs** | NuminaMath-CoT (860K Olympiad Math) | `AI-MO/NuminaMath-CoT` | [https://huggingface.co/datasets/AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| 21 | **5. Complex Mathematics & Formal Proofs** | NVIDIA OpenMathInstruct-2 (14M Math) | `nvidia/OpenMathInstruct-2` | [https://huggingface.co/datasets/nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) |
| 22 | **5. Complex Mathematics & Formal Proofs** | OpenWebMath (14.7B Tokens) | `open-web-math/open-web-math` | [https://huggingface.co/datasets/open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) |
| 23 | **5. Complex Mathematics & Formal Proofs** | Proof-Pile-2 (Algebraic Stack & Lean) | `EleutherAI/proof-pile-2` | [https://huggingface.co/datasets/EleutherAI/proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |
| 24 | **5. Complex Mathematics & Formal Proofs** | MathCode-Pile (Program-of-Thought Math) | `MathGenie/MathCode-Pile` | [https://huggingface.co/datasets/MathGenie/MathCode-Pile](https://huggingface.co/datasets/MathGenie/MathCode-Pile) |
| 25 | **5. Complex Mathematics & Formal Proofs** | GSM8K Multi-Step Quantitative Reasoning | `openai/gsm8k` | [https://huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| 26 | **5. Complex Mathematics & Formal Proofs** | Hendrycks Competition MATH | `lighteval/MATH` | [https://huggingface.co/datasets/lighteval/MATH](https://huggingface.co/datasets/lighteval/MATH) |
| 27 | **5. Complex Mathematics & Formal Proofs** | MetaMathQA (395K Augmented Math) | `meta-math/MetaMathQA` | [https://huggingface.co/datasets/meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) |
| 28 | **6. Biology, Genetics & Life Sciences** | Structured Biology & Ecology Textbook | `burgerbee/biology_textbook` | [https://huggingface.co/datasets/burgerbee/biology_textbook](https://huggingface.co/datasets/burgerbee/biology_textbook) |
| 29 | **6. Biology, Genetics & Life Sciences** | MedMCQA (194K Biomedical & Clinical) | `openlifescienceai/medmcqa` | [https://huggingface.co/datasets/openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) |
| 30 | **6. Biology, Genetics & Life Sciences** | PubMedQA Biomedical Literature | `qiaojin/PubMedQA` | [https://huggingface.co/datasets/qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) |
| 31 | **6. Biology, Genetics & Life Sciences** | BioASQ Biomedical Semantic QA | `kroshan/BioASQ` | [https://huggingface.co/datasets/kroshan/BioASQ](https://huggingface.co/datasets/kroshan/BioASQ) |
| 32 | **6. Biology, Genetics & Life Sciences** | UniProtKB Protein & Genomics Annotations | `uniprot/uniprotkb` | [https://www.uniprot.org](https://www.uniprot.org) |
| 33 | **7. Nature, Ecology & Earth Sciences** | AllenAI SciQ (Physics, Chemistry, Ecology) | `allenai/sciq` | [https://huggingface.co/datasets/allenai/sciq](https://huggingface.co/datasets/allenai/sciq) |
| 34 | **7. Nature, Ecology & Earth Sciences** | AI2 ARC (Natural & Earth Science Challenge) | `allenai/ai2_arc` | [https://huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) |
| 35 | **7. Nature, Ecology & Earth Sciences** | OpenBookQA (Natural Laws & Science) | `allenai/openbookqa` | [https://huggingface.co/datasets/allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) |
| 36 | **7. Nature, Ecology & Earth Sciences** | Climate & Earth Systems Corpus | `climatebert/climate_sentiment` | [https://huggingface.co/datasets/climatebert/climate_sentiment](https://huggingface.co/datasets/climatebert/climate_sentiment) |
| 37 | **7. Nature, Ecology & Earth Sciences** | GBIF Global Biodiversity Information Facility | `gbif/species-occurrences` | [https://www.gbif.org](https://www.gbif.org) |
| 38 | **8. Places, World Geography & Landmarks** | Worldbuilding & Extreme Geography | `archit11/worldbuilding` | [https://huggingface.co/datasets/archit11/worldbuilding](https://huggingface.co/datasets/archit11/worldbuilding) |
| 39 | **8. Places, World Geography & Landmarks** | GeoNames Global Gazetteer (11M+ Places) | `geonames/allCountries` | [https://www.geonames.org](https://www.geonames.org) |
| 40 | **8. Places, World Geography & Landmarks** | UNESCO World Heritage Sites Catalog | `unesco/world-heritage-list` | [https://whc.unesco.org](https://whc.unesco.org) |
| 41 | **8. Places, World Geography & Landmarks** | Philippine Geopolitical & Provincial Atlas | `Darknecrocities/Helios` | [https://huggingface.co/Darknecrocities/Helios](https://huggingface.co/Darknecrocities/Helios) |
| 42 | **9. Coding & Software Engineering** | The Stack v2 (67.5TB Source Code) | `bigcode/the-stack-v2` | [https://huggingface.co/datasets/bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| 43 | **9. Coding & Software Engineering** | StarCoderData (86 Programming Languages) | `bigcode/starcoderdata` | [https://huggingface.co/datasets/bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) |
| 44 | **9. Coding & Software Engineering** | CommitPackFT (Verified Git Commits) | `bigcode/commitpackft` | [https://huggingface.co/datasets/bigcode/commitpackft](https://huggingface.co/datasets/bigcode/commitpackft) |
| 45 | **9. Coding & Software Engineering** | Magicoder-OSS-Instruct-75K | `ise-uiuc/Magicoder-OSS-Instruct-75K` | [https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) |
| 46 | **9. Coding & Software Engineering** | CodeFeedback-Filtered-Instruction (156K) | `m-a-p/CodeFeedback-Filtered-Instruction` | [https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) |
| 47 | **9. Coding & Software Engineering** | SWE-bench Verified Software Engineering | `princeton-nlp/SWE-bench_Verified` | [https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) |
| 48 | **10. Agentic Skills & Multi-Step Planning** | ToolMind (160K Graph-Correlated Agent Trajectories) | `Nanbeige/ToolMind` | [https://huggingface.co/datasets/Nanbeige/ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) |
| 49 | **10. Agentic Skills & Multi-Step Planning** | Agentic Tool-Call Dataset (12K Deep Research) | `pyromind/agentic-tool-call-dataset-12k` | [https://huggingface.co/datasets/pyromind/agentic-tool-call-dataset-12k](https://huggingface.co/datasets/pyromind/agentic-tool-call-dataset-12k) |
| 50 | **10. Agentic Skills & Multi-Step Planning** | THUDM AgentInstruct (Multi-Turn Agent Skills) | `THUDM/AgentInstruct` | [https://huggingface.co/datasets/THUDM/AgentInstruct](https://huggingface.co/datasets/THUDM/AgentInstruct) |
| 51 | **10. Agentic Skills & Multi-Step Planning** | UltraInteract (Planning, Coding & Reflection) | `openbmb/UltraInteract_sft` | [https://huggingface.co/datasets/openbmb/UltraInteract_sft](https://huggingface.co/datasets/openbmb/UltraInteract_sft) |
| 52 | **10. Agentic Skills & Multi-Step Planning** | OpenThoughts-114k | `open-thoughts/OpenThoughts-114k` | [https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| 53 | **10. Agentic Skills & Multi-Step Planning** | Bespoke-Stratos-17k | `bespokelabs/Bespoke-Stratos-17k` | [https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) |
| 54 | **10. Agentic Skills & Multi-Step Planning** | ServiceNow R1-Distill-SFT | `ServiceNow-AI/R1-Distill-SFT` | [https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT) |
| 55 | **10. Agentic Skills & Multi-Step Planning** | Magpie-Reasoning-V2 (250K CoT) | `Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B` | [https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B) |
| 56 | **11. Model Context Protocol (MCP) & APIs** | Salesforce xLAM Function-Calling (60K) | `Salesforce/xlam-function-calling-60k` | [https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) |
| 57 | **11. Model Context Protocol (MCP) & APIs** | Glaive Function-Calling v2 (113K) | `glaiveai/glaive-function-calling-v2` | [https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) |
| 58 | **11. Model Context Protocol (MCP) & APIs** | Gorilla Berkeley Function-Calling Leaderboard | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | [https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) |
| 59 | **11. Model Context Protocol (MCP) & APIs** | Model Context Protocol (MCP) Specification | `modelcontextprotocol/specification` | [https://modelcontextprotocol.io](https://modelcontextprotocol.io) |
| 60 | **12. Technical Documentation & Manuals** | The Stack v2 Documentation & ReadTheDocs | `bigcode/the-stack-v2-docs` | [https://devdocs.io](https://devdocs.io) |
| 61 | **12. Technical Documentation & Manuals** | MDN Web Docs & Developer Manuals | `mdn/content` | [https://developer.mozilla.org](https://developer.mozilla.org) |
| 62 | **13. Scientific Research & Literature** | AllenAI PeS2o (40M+ Open Academic Papers) | `allenai/peS2o` | [https://huggingface.co/datasets/allenai/peS2o](https://huggingface.co/datasets/allenai/peS2o) |
| 63 | **13. Scientific Research & Literature** | AllenAI SciRIFF (Scientific Literature Instruction) | `allenai/SciRIFF` | [https://huggingface.co/datasets/allenai/SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF) |
| 64 | **13. Scientific Research & Literature** | arXiv & PubMed Scientific Papers Corpus | `armanc/scientific_papers` | [https://huggingface.co/datasets/armanc/scientific_papers](https://huggingface.co/datasets/armanc/scientific_papers) |
| 65 | **13. Scientific Research & Literature** | OpenAlex Global Research Catalog (250M+ Works) | `openalex/openalex-snapshot` | [https://openalex.org](https://openalex.org) |
| 66 | **14. Bilingual Filipino / Tagalog & General** | CulturaX (Multilingual Tagalog) | `uonlp/CulturaX` | [https://huggingface.co/datasets/uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) |
| 67 | **14. Bilingual Filipino / Tagalog & General** | OSCAR-2301 & CC-100 (Tagalog) | `oscar-corpus/OSCAR-2301` | [https://huggingface.co/datasets/oscar-corpus/OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) |
| 68 | **14. Bilingual Filipino / Tagalog & General** | Alpaca-Cleaned & SlimOrca | `yahma/alpaca-cleaned` | [https://huggingface.co/datasets/yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) |
