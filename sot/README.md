# Helios Dataset Single Source of Truth (SOT)

**Creator & Lead Architect**: **Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines** (`parejasarronkian@gmail.com`)  
**Total External Datasets Cataloged & Integrated**: **177 Datasets** across **16 Foundational Knowledge & Reasoning Domains**  
**Consolidated Active Training Corpora**: Massive Multi-Category Omni-Domain Pretraining Corpus stored on `/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl`, and seed grounding corpus in `data/helios_expanded_reasoning_corpus.jsonl`

---

## 1. Executive Summary of Knowledge Domains in Helios

| Category # | Domain Category | Datasets Count | Primary Focus & Provenance | Storage Location |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **1. Foundational Web & Encyclopedic** | 10 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **2** | **2. Technology, Systems & Cloud** | 12 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **3** | **3. World History & Timelines** | 12 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **4** | **4. Ancient Civilizations & Archaeology** | 10 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **5** | **5. Complex Mathematics & Formal Proofs** | 12 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **6** | **6. Biology, Genetics & Life Sciences** | 12 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **7** | **7. Nature, Ecology & Earth Sciences** | 10 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **8** | **8. Places, World Geography & Landmarks** | 10 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **9** | **9. Coding, Algorithms & Competitive Programming** | 15 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **10** | **10. Software Engineering (SWE), Architecture & GitHub PRs** | 12 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **11** | **11. Agentic Skills & Multi-Step Planning** | 12 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **12** | **12. Model Context Protocol (MCP), APIs & Function Calling** | 10 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **13** | **13. Technical Documentation, RFCs & Manuals** | 10 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **14** | **14. Scientific Research, Literature & Peer Review** | 10 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **15** | **15. Law, Governance, Philosophy & Social Sciences** | 10 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **16** | **16. Bilingual Filipino / Tagalog & Multilingual Foundations** | 10 | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |
| **TOTAL** | **16 Domains** | **177 Datasets** | **Massive Multi-Domain Harvest & Checkpoints** | **`/Volumes/One Touch/Helios/`** |

---

## 2. Complete Registry of All External Datasets & Direct Links

| # | Domain Category | Dataset / Platform Name | Identifier | Direct Link |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **1. Foundational Web & Encyclopedic** | FineWeb-Edu (1.3T Tokens) | `HuggingFaceFW/fineweb-edu` | [https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| 2 | **1. Foundational Web & Encyclopedic** | FineWeb (15T Tokens) | `HuggingFaceFW/fineweb` | [https://huggingface.co/datasets/HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| 3 | **1. Foundational Web & Encyclopedic** | DCLM-Baseline (4T Tokens) | `mlfoundations/dCLM-baseline` | [https://huggingface.co/datasets/mlfoundations/dCLM-baseline](https://huggingface.co/datasets/mlfoundations/dCLM-baseline) |
| 4 | **1. Foundational Web & Encyclopedic** | Cosmopedia v2 | `HuggingFaceTB/cosmopedia-v2` | [https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2) |
| 5 | **1. Foundational Web & Encyclopedic** | Wikipedia Multi-Domain Dump | `wikimedia/wikipedia` | [https://huggingface.co/datasets/wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) |
| 6 | **1. Foundational Web & Encyclopedic** | Wikidata Knowledge Graph Triples | `wikidata/wikidata` | [https://www.wikidata.org](https://www.wikidata.org) |
| 7 | **1. Foundational Web & Encyclopedic** | RedPajama-Data-v2 (30T Tokens) | `togethercomputer/RedPajama-Data-V2` | [https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) |
| 8 | **1. Foundational Web & Encyclopedic** | RefinedWeb (Falcon Pretraining Corpus) | `tiiuae/falcon-refinedweb` | [https://huggingface.co/datasets/tiiuae/falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) |
| 9 | **1. Foundational Web & Encyclopedic** | C4 (Colossal Clean Crawled Corpus) | `allenai/c4` | [https://huggingface.co/datasets/allenai/c4](https://huggingface.co/datasets/allenai/c4) |
| 10 | **1. Foundational Web & Encyclopedic** | The Pile (EleutherAI 825GB Open Corpus) | `EleutherAI/the_pile` | [https://huggingface.co/datasets/EleutherAI/the_pile](https://huggingface.co/datasets/EleutherAI/the_pile) |
| 11 | **2. Technology, Systems & Cloud** | SmolLM Systems & CS Corpus | `HuggingFaceTB/smollm-corpus` | [https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) |
| 12 | **2. Technology, Systems & Cloud** | SecQA Cybersecurity & Network Security | `zefang-liu/secqa` | [https://huggingface.co/datasets/zefang-liu/secqa](https://huggingface.co/datasets/zefang-liu/secqa) |
| 13 | **2. Technology, Systems & Cloud** | OpenHermes-2.5 Tech & Systems | `teknium/OpenHermes-2.5` | [https://huggingface.co/datasets/teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) |
| 14 | **2. Technology, Systems & Cloud** | IETF RFC Technical Standards Archive | `ietf-tools/rfc-archive` | [https://www.rfc-editor.org](https://www.rfc-editor.org) |
| 15 | **2. Technology, Systems & Cloud** | SlimOrca Curated Reasoning & Technology | `Open-Orca/SlimOrca` | [https://huggingface.co/datasets/Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) |
| 16 | **2. Technology, Systems & Cloud** | Infinity-Instruct (7M+ Instruction Mega-Corpus) | `BAAI/Infinity-Instruct` | [https://huggingface.co/datasets/BAAI/Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct) |
| 17 | **2. Technology, Systems & Cloud** | Cybersecurity Knowledge Dataset v1 | `AlicanKiraz0/Cybersecurity-Dataset-v1` | [https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-v1](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-v1) |
| 18 | **2. Technology, Systems & Cloud** | Linux Kernel Documentation & Git History | `torvalds/linux` | [https://github.com/torvalds/linux](https://github.com/torvalds/linux) |
| 19 | **2. Technology, Systems & Cloud** | Kubernetes Architecture & Operational Guides | `kubernetes/website` | [https://kubernetes.io/docs](https://kubernetes.io/docs) |
| 20 | **2. Technology, Systems & Cloud** | Cloud Native Computing Foundation (CNCF) Specs | `cncf/landscape` | [https://landscape.cncf.io](https://landscape.cncf.io) |
| 21 | **2. Technology, Systems & Cloud** | MMLU Computer Science & Engineering | `cais/mmlu` | [https://huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| 22 | **2. Technology, Systems & Cloud** | UltraChat 200k Technical Dialogues | `HuggingFaceH4/ultrachat_200k` | [https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |
| 23 | **3. World History & Timelines** | Encyclopaedia Britannica Historical Editions | `Vintage-LLM/Britannica` | [https://huggingface.co/datasets/Vintage-LLM/Britannica](https://huggingface.co/datasets/Vintage-LLM/Britannica) |
| 24 | **3. World History & Timelines** | VDR Academic History & Geography | `racineai/VDR_History_Geography` | [https://huggingface.co/datasets/racineai/VDR_History_Geography](https://huggingface.co/datasets/racineai/VDR_History_Geography) |
| 25 | **3. World History & Timelines** | Project Gutenberg Historical Library | `manu/project_gutenberg` | [https://huggingface.co/datasets/manu/project_gutenberg](https://huggingface.co/datasets/manu/project_gutenberg) |
| 26 | **3. World History & Timelines** | BigScience Historical Texts | `bigscience-historical-texts` | [https://huggingface.co/bigscience-historical-texts](https://huggingface.co/bigscience-historical-texts) |
| 27 | **3. World History & Timelines** | MMLU World, European & US History | `cais/mmlu` | [https://huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| 28 | **3. World History & Timelines** | Pleias Synthetic History Corpus | `pleias/Synthetic-History-Corpus` | [https://huggingface.co/datasets/pleias/Synthetic-History-Corpus](https://huggingface.co/datasets/pleias/Synthetic-History-Corpus) |
| 29 | **3. World History & Timelines** | Internet History Sourcebooks Project (Fordham) | `fordham/ihsp` | [https://sourcebooks.fordham.edu](https://sourcebooks.fordham.edu) |
| 30 | **3. World History & Timelines** | United States National Archives Declassified Papers | `us-national-archives/historical` | [https://www.archives.gov](https://www.archives.gov) |
| 31 | **3. World History & Timelines** | Diplomatic History & Foreign Relations of the US | `state-dept/frus` | [https://history.state.gov](https://history.state.gov) |
| 32 | **3. World History & Timelines** | Harvard Library Historical Books Collection | `Harvard-Library/historical-books` | [https://library.harvard.edu](https://library.harvard.edu) |
| 33 | **3. World History & Timelines** | Avalon Project Documents in Law, History and Diplomacy | `yale-law/avalon` | [https://avalon.law.yale.edu](https://avalon.law.yale.edu) |
| 34 | **3. World History & Timelines** | Imperial War Museums Historical Oral Archive | `iwm/sound-archive` | [https://www.iwm.org.uk](https://www.iwm.org.uk) |
| 35 | **4. Ancient Civilizations & Archaeology** | CoT Reasoning: The Ancient Past | `mattwesney/CoT_Reasoning_The_Ancient_Past` | [https://huggingface.co/datasets/mattwesney/CoT_Reasoning_The_Ancient_Past](https://huggingface.co/datasets/mattwesney/CoT_Reasoning_The_Ancient_Past) |
| 36 | **4. Ancient Civilizations & Archaeology** | Ancient Languages & Epigraphy | `AncientLanguages` | [https://huggingface.co/AncientLanguages](https://huggingface.co/AncientLanguages) |
| 37 | **4. Ancient Civilizations & Archaeology** | CATMuS Classical & Medieval Manuscripts | `CATMuS/medieval` | [https://huggingface.co/datasets/CATMuS/medieval](https://huggingface.co/datasets/CATMuS/medieval) |
| 38 | **4. Ancient Civilizations & Archaeology** | Perseus Digital Library (Classical Greece & Rome) | `perseus-digital-library` | [https://www.perseus.tufts.edu](https://www.perseus.tufts.edu) |
| 39 | **4. Ancient Civilizations & Archaeology** | OpenPecha Ancient Buddhist & Himalayan Texts | `openpecha/ancient-texts` | [https://huggingface.co/openpecha](https://huggingface.co/openpecha) |
| 40 | **4. Ancient Civilizations & Archaeology** | Cuneiform Digital Library Initiative (CDLI) | `cdli/inscriptions` | [https://cdli.mpiwg-berlin.mpg.de](https://cdli.mpiwg-berlin.mpg.de) |
| 41 | **4. Ancient Civilizations & Archaeology** | Thesaurus Linguae Graecae (TLG) | `uci/tlg` | [http://stephanus.tlg.uci.edu](http://stephanus.tlg.uci.edu) |
| 42 | **4. Ancient Civilizations & Archaeology** | Papyri.info Ancient Egyptian & Hellenistic Papyri | `duke-columbia/papyri` | [https://papyri.info](https://papyri.info) |
| 43 | **4. Ancient Civilizations & Archaeology** | Digital Epigraphy and Archaeology Project (DEA) | `dea/inscriptions` | [https://www.digitalepigraphy.org](https://www.digitalepigraphy.org) |
| 44 | **4. Ancient Civilizations & Archaeology** | Pre-Colonial Philippine Inscriptions & Laguna Copperplate | `Darknecrocities/Helios-Historical` | [https://huggingface.co/Darknecrocities/Helios](https://huggingface.co/Darknecrocities/Helios) |
| 45 | **5. Complex Mathematics & Formal Proofs** | NuminaMath-CoT (860K Olympiad Math) | `AI-MO/NuminaMath-CoT` | [https://huggingface.co/datasets/AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| 46 | **5. Complex Mathematics & Formal Proofs** | NVIDIA OpenMathInstruct-2 (14M Math) | `nvidia/OpenMathInstruct-2` | [https://huggingface.co/datasets/nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) |
| 47 | **5. Complex Mathematics & Formal Proofs** | OpenWebMath (14.7B Tokens) | `open-web-math/open-web-math` | [https://huggingface.co/datasets/open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) |
| 48 | **5. Complex Mathematics & Formal Proofs** | Proof-Pile-2 (Algebraic Stack & Formal Proofs) | `EleutherAI/proof-pile-2` | [https://huggingface.co/datasets/EleutherAI/proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |
| 49 | **5. Complex Mathematics & Formal Proofs** | MathCode-Pile (Program-of-Thought Math) | `MathGenie/MathCode-Pile` | [https://huggingface.co/datasets/MathGenie/MathCode-Pile](https://huggingface.co/datasets/MathGenie/MathCode-Pile) |
| 50 | **5. Complex Mathematics & Formal Proofs** | GSM8K Multi-Step Quantitative Reasoning | `openai/gsm8k` | [https://huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| 51 | **5. Complex Mathematics & Formal Proofs** | Hendrycks Competition MATH | `lighteval/MATH` | [https://huggingface.co/datasets/lighteval/MATH](https://huggingface.co/datasets/lighteval/MATH) |
| 52 | **5. Complex Mathematics & Formal Proofs** | MetaMathQA (395K Augmented Math) | `meta-math/MetaMathQA` | [https://huggingface.co/datasets/meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) |
| 53 | **5. Complex Mathematics & Formal Proofs** | MathQA (Amini et al. Algorithmic Math) | `allenai/math_qa` | [https://huggingface.co/datasets/allenai/math_qa](https://huggingface.co/datasets/allenai/math_qa) |
| 54 | **5. Complex Mathematics & Formal Proofs** | Lean 4 Mathlib Formal Mathematical Proofs | `leanprover-community/mathlib4` | [https://github.com/leanprover-community/mathlib4](https://github.com/leanprover-community/mathlib4) |
| 55 | **5. Complex Mathematics & Formal Proofs** | Isabelle/HOL Archive of Formal Proofs | `afp/isabelle-afp` | [https://www.isa-afp.org](https://www.isa-afp.org) |
| 56 | **5. Complex Mathematics & Formal Proofs** | Coq Proof Assistant Standard Library | `coq/coq-stdlib` | [https://coq.inria.fr](https://coq.inria.fr) |
| 57 | **6. Biology, Genetics & Life Sciences** | Structured Biology & Ecology Textbook | `burgerbee/biology_textbook` | [https://huggingface.co/datasets/burgerbee/biology_textbook](https://huggingface.co/datasets/burgerbee/biology_textbook) |
| 58 | **6. Biology, Genetics & Life Sciences** | MedMCQA (194K Biomedical & Clinical) | `openlifescienceai/medmcqa` | [https://huggingface.co/datasets/openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) |
| 59 | **6. Biology, Genetics & Life Sciences** | PubMedQA Biomedical Literature | `qiaojin/PubMedQA` | [https://huggingface.co/datasets/qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) |
| 60 | **6. Biology, Genetics & Life Sciences** | BioASQ Biomedical Semantic QA | `kroshan/BioASQ` | [https://huggingface.co/datasets/kroshan/BioASQ](https://huggingface.co/datasets/kroshan/BioASQ) |
| 61 | **6. Biology, Genetics & Life Sciences** | UniProtKB Protein & Genomics Annotations | `uniprot/uniprotkb` | [https://www.uniprot.org](https://www.uniprot.org) |
| 62 | **6. Biology, Genetics & Life Sciences** | MedQA (USMLE Clinical Question Bank) | `GBaker/MedQA-USMLE-4-options` | [https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) |
| 63 | **6. Biology, Genetics & Life Sciences** | NCBI Gene & RefSeq Genomic Sequence DB | `ncbi/refseq` | [https://www.ncbi.nlm.nih.gov/refseq](https://www.ncbi.nlm.nih.gov/refseq) |
| 64 | **6. Biology, Genetics & Life Sciences** | ChEMBL Bioactive Molecules & Target Assays | `chembl/chembl_db` | [https://www.ebi.ac.uk/chembl](https://www.ebi.ac.uk/chembl) |
| 65 | **6. Biology, Genetics & Life Sciences** | PDB (Protein Data Bank 3D Macromolecular Structures) | `wwpdb/pdb` | [https://www.rcsb.org](https://www.rcsb.org) |
| 66 | **6. Biology, Genetics & Life Sciences** | ClinicalTrials.gov Protocol & Evidence DB | `nih/clinicaltrials` | [https://clinicaltrials.gov](https://clinicaltrials.gov) |
| 67 | **6. Biology, Genetics & Life Sciences** | MMLU Clinical Medicine & Biology Suite | `cais/mmlu` | [https://huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| 68 | **6. Biology, Genetics & Life Sciences** | Ensembl Genome Browser & Comparative Genomics | `ensembl/genomes` | [https://www.ensembl.org](https://www.ensembl.org) |
| 69 | **7. Nature, Ecology & Earth Sciences** | AllenAI SciQ (Physics, Chemistry, Biology, Ecology) | `allenai/sciq` | [https://huggingface.co/datasets/allenai/sciq](https://huggingface.co/datasets/allenai/sciq) |
| 70 | **7. Nature, Ecology & Earth Sciences** | AI2 ARC (Natural & Earth Science Challenge) | `allenai/ai2_arc` | [https://huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) |
| 71 | **7. Nature, Ecology & Earth Sciences** | OpenBookQA (Natural Laws & Elementary Science) | `allenai/openbookqa` | [https://huggingface.co/datasets/allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) |
| 72 | **7. Nature, Ecology & Earth Sciences** | Climate & Earth Systems Corpus | `climatebert/climate_sentiment` | [https://huggingface.co/datasets/climatebert/climate_sentiment](https://huggingface.co/datasets/climatebert/climate_sentiment) |
| 73 | **7. Nature, Ecology & Earth Sciences** | GBIF Global Biodiversity Information Facility | `gbif/species-occurrences` | [https://www.gbif.org](https://www.gbif.org) |
| 74 | **7. Nature, Ecology & Earth Sciences** | NASA Earth Science Data Systems (ESDS) | `nasa/earth-data` | [https://earthdata.nasa.gov](https://earthdata.nasa.gov) |
| 75 | **7. Nature, Ecology & Earth Sciences** | USGS Geological Surveys & Seismic Data | `usgs/geology-earthquake` | [https://www.usgs.gov](https://www.usgs.gov) |
| 76 | **7. Nature, Ecology & Earth Sciences** | IPCC Assessment Reports (Sixth Assessment AR6) | `ipcc/ar6-reports` | [https://www.ipcc.ch/assessment-report/ar6](https://www.ipcc.ch/assessment-report/ar6) |
| 77 | **7. Nature, Ecology & Earth Sciences** | NOAA Oceanic & Atmospheric Climatology DB | `noaa/ncei` | [https://www.ncei.noaa.gov](https://www.ncei.noaa.gov) |
| 78 | **7. Nature, Ecology & Earth Sciences** | IUCN Red List of Threatened Species | `iucn/redlist` | [https://www.iucnredlist.org](https://www.iucnredlist.org) |
| 79 | **8. Places, World Geography & Landmarks** | Worldbuilding & Extreme Geography | `archit11/worldbuilding` | [https://huggingface.co/datasets/archit11/worldbuilding](https://huggingface.co/datasets/archit11/worldbuilding) |
| 80 | **8. Places, World Geography & Landmarks** | GeoNames Global Gazetteer (11M+ Places) | `geonames/allCountries` | [https://www.geonames.org](https://www.geonames.org) |
| 81 | **8. Places, World Geography & Landmarks** | UNESCO World Heritage Sites Catalog | `unesco/world-heritage-list` | [https://whc.unesco.org](https://whc.unesco.org) |
| 82 | **8. Places, World Geography & Landmarks** | Philippine Geopolitical & Provincial Atlas | `Darknecrocities/Helios` | [https://huggingface.co/Darknecrocities/Helios](https://huggingface.co/Darknecrocities/Helios) |
| 83 | **8. Places, World Geography & Landmarks** | OpenStreetMap Planetary Geodata Extracts | `osm/planet` | [https://planet.openstreetmap.org](https://planet.openstreetmap.org) |
| 84 | **8. Places, World Geography & Landmarks** | Natural Earth Public Domain Map Dataset | `natural-earth/vector-data` | [https://www.naturalearthdata.com](https://www.naturalearthdata.com) |
| 85 | **8. Places, World Geography & Landmarks** | CIA World Factbook Geopolitical Profiles | `cia/world-factbook` | [https://www.cia.gov/the-world-factbook](https://www.cia.gov/the-world-factbook) |
| 86 | **8. Places, World Geography & Landmarks** | UN Cartographic Section & Geographic Database | `un/geospatial` | [https://www.un.org/geospatial](https://www.un.org/geospatial) |
| 87 | **8. Places, World Geography & Landmarks** | Global Administrative Areas (GADM) Boundaries | `gadm/boundaries` | [https://gadm.org](https://gadm.org) |
| 88 | **8. Places, World Geography & Landmarks** | World Protected Areas Database (WDPA) | `unep-wcmc/wdpa` | [https://www.protectedplanet.net](https://www.protectedplanet.net) |
| 89 | **9. Coding, Algorithms & Competitive Programming** | The Stack v2 (67.5TB Source Code) | `bigcode/the-stack-v2` | [https://huggingface.co/datasets/bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| 90 | **9. Coding, Algorithms & Competitive Programming** | StarCoderData (86 Programming Languages) | `bigcode/starcoderdata` | [https://huggingface.co/datasets/bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) |
| 91 | **9. Coding, Algorithms & Competitive Programming** | CodeAlpaca-20k (Multi-Language Coding) | `sahil2801/CodeAlpaca-20k` | [https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) |
| 92 | **9. Coding, Algorithms & Competitive Programming** | Python Code Instructions 18k Alpaca | `iamtarun/python_code_instructions_18k_alpaca` | [https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) |
| 93 | **9. Coding, Algorithms & Competitive Programming** | Magicoder-OSS-Instruct-75K | `ise-uiuc/Magicoder-OSS-Instruct-75K` | [https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) |
| 94 | **9. Coding, Algorithms & Competitive Programming** | Magicoder-Evol-Instruct-110K | `ise-uiuc/Magicoder-Evol-Instruct-110K` | [https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) |
| 95 | **9. Coding, Algorithms & Competitive Programming** | Evol-Instruct-Code-80k-v1 | `nickrosh/Evol-Instruct-Code-80k-v1` | [https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) |
| 96 | **9. Coding, Algorithms & Competitive Programming** | Glaive Code Assistant v3 | `GlaiveAI/glaive-code-assistant-v3` | [https://huggingface.co/datasets/GlaiveAI/glaive-code-assistant-v3](https://huggingface.co/datasets/GlaiveAI/glaive-code-assistant-v3) |
| 97 | **9. Coding, Algorithms & Competitive Programming** | CodeFeedback-Filtered-Instruction (156K) | `m-a-p/CodeFeedback-Filtered-Instruction` | [https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) |
| 98 | **9. Coding, Algorithms & Competitive Programming** | TigerBot Kaggle LeetCode Solutions (2K) | `TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k` | [https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k](https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k) |
| 99 | **9. Coding, Algorithms & Competitive Programming** | Hendrycks APPS (Advanced Programming Problem Solving) | `hendrycks/apps` | [https://huggingface.co/datasets/hendrycks/apps](https://huggingface.co/datasets/hendrycks/apps) |
| 100 | **9. Coding, Algorithms & Competitive Programming** | DeepMind CodeContests Competitive Programming | `deepmind/code_contests` | [https://huggingface.co/datasets/deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests) |
| 101 | **9. Coding, Algorithms & Competitive Programming** | HumanEvalPlus (Extended Python Coding Benchmarks) | `evalplus/humanevalplus` | [https://huggingface.co/datasets/evalplus/humanevalplus](https://huggingface.co/datasets/evalplus/humanevalplus) |
| 102 | **9. Coding, Algorithms & Competitive Programming** | MBPPPlus (Mostly Basic Python Problems Extended) | `evalplus/mbppplus` | [https://huggingface.co/datasets/evalplus/mbppplus](https://huggingface.co/datasets/evalplus/mbppplus) |
| 103 | **9. Coding, Algorithms & Competitive Programming** | SmolLM Python-Edu Pretraining Subset | `HuggingFaceTB/smollm-corpus` | [https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) |
| 104 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | SWE-bench Verified Software Engineering | `princeton-nlp/SWE-bench_Verified` | [https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) |
| 105 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | SWE-bench Lite (Standard Bug Resolution Benchmark) | `princeton-nlp/SWE-bench_Lite` | [https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) |
| 106 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | SWE-bench Extra | `nebius/SWE-bench-extra` | [https://huggingface.co/datasets/nebius/SWE-bench-extra](https://huggingface.co/datasets/nebius/SWE-bench-extra) |
| 107 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | CommitPackFT (Verified Git Commits) | `bigcode/commitpackft` | [https://huggingface.co/datasets/bigcode/commitpackft](https://huggingface.co/datasets/bigcode/commitpackft) |
| 108 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | SWE-gym Autonomous Software Engineering Environments | `CMU-Perceptual-Computing-Lab/swe-gym` | [https://huggingface.co/datasets/CMU-Perceptual-Computing-Lab/swe-gym](https://huggingface.co/datasets/CMU-Perceptual-Computing-Lab/swe-gym) |
| 109 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | GitHub Code Clean High-Star Repository Corpus | `codeparrot/github-code-clean` | [https://huggingface.co/datasets/codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) |
| 110 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | SWE-smith Synthetic SWE Problem Generation | `SWE-bench/SWE-smith` | [https://huggingface.co/datasets/SWE-bench/SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith) |
| 111 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | CodeXGLUE Software Engineering Multi-Task Suite | `microsoft/CodeXGLUE` | [https://github.com/microsoft/CodeXGLUE](https://github.com/microsoft/CodeXGLUE) |
| 112 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | Commit Chronicle Software Evolution History | `JetBrains-Research/commit-chronicle` | [https://huggingface.co/datasets/JetBrains-Research/commit-chronicle](https://huggingface.co/datasets/JetBrains-Research/commit-chronicle) |
| 113 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | GitBug Real-World Bug-Fix Benchmark | `gitbug/gitbug-benchmark` | [https://github.com/gitbug-benchmark/gitbug](https://github.com/gitbug-benchmark/gitbug) |
| 114 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | Defects4J Real Java Faults Dataset | `rjust/defects4j` | [https://github.com/rjust/defects4j](https://github.com/rjust/defects4j) |
| 115 | **10. Software Engineering (SWE), Architecture & GitHub PRs** | BugsInPy Real Python Faults Dataset | `soarsmu/BugsInPy` | [https://github.com/soarsmu/BugsInPy](https://github.com/soarsmu/BugsInPy) |
| 116 | **11. Agentic Skills & Multi-Step Planning** | ToolMind (160K Graph-Correlated Agent Trajectories) | `Nanbeige/ToolMind` | [https://huggingface.co/datasets/Nanbeige/ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) |
| 117 | **11. Agentic Skills & Multi-Step Planning** | Agentic Tool-Call Dataset (12K Deep Research) | `pyromind/agentic-tool-call-dataset-12k` | [https://huggingface.co/datasets/pyromind/agentic-tool-call-dataset-12k](https://huggingface.co/datasets/pyromind/agentic-tool-call-dataset-12k) |
| 118 | **11. Agentic Skills & Multi-Step Planning** | THUDM AgentInstruct (Multi-Turn Agent Skills) | `THUDM/AgentInstruct` | [https://huggingface.co/datasets/THUDM/AgentInstruct](https://huggingface.co/datasets/THUDM/AgentInstruct) |
| 119 | **11. Agentic Skills & Multi-Step Planning** | UltraInteract (Planning, Coding & Reflection) | `openbmb/UltraInteract_sft` | [https://huggingface.co/datasets/openbmb/UltraInteract_sft](https://huggingface.co/datasets/openbmb/UltraInteract_sft) |
| 120 | **11. Agentic Skills & Multi-Step Planning** | OpenThoughts-114k (Deep Reasoning & Reflection) | `open-thoughts/OpenThoughts-114k` | [https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| 121 | **11. Agentic Skills & Multi-Step Planning** | Bespoke-Stratos-17k (Deep Reasoning Distillation) | `bespokelabs/Bespoke-Stratos-17k` | [https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) |
| 122 | **11. Agentic Skills & Multi-Step Planning** | ServiceNow R1-Distill-SFT | `ServiceNow-AI/R1-Distill-SFT` | [https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT) |
| 123 | **11. Agentic Skills & Multi-Step Planning** | Magpie-Reasoning-V2 (250K CoT) | `Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B` | [https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B) |
| 124 | **11. Agentic Skills & Multi-Step Planning** | GAIA General AI Assistant Benchmark | `gaia-benchmark/GAIA` | [https://huggingface.co/datasets/gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) |
| 125 | **11. Agentic Skills & Multi-Step Planning** | AgentBench Multi-Environment Agent Benchmark | `THUDM/AgentBench` | [https://github.com/THUDM/AgentBench](https://github.com/THUDM/AgentBench) |
| 126 | **11. Agentic Skills & Multi-Step Planning** | WebArena Autonomous Web Navigation & Execution | `web-arena/webarena` | [https://webarena.dev](https://webarena.dev) |
| 127 | **11. Agentic Skills & Multi-Step Planning** | OSWorld Multimodal Operating System Agent Benchmark | `xlang-ai/OSWorld` | [https://os-world.github.io](https://os-world.github.io) |
| 128 | **12. Model Context Protocol (MCP), APIs & Function Calling** | Salesforce xLAM Function-Calling (60K) | `Salesforce/xlam-function-calling-60k` | [https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) |
| 129 | **12. Model Context Protocol (MCP), APIs & Function Calling** | Glaive Function-Calling v2 (113K) | `glaiveai/glaive-function-calling-v2` | [https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) |
| 130 | **12. Model Context Protocol (MCP), APIs & Function Calling** | Gorilla Berkeley Function-Calling Leaderboard (BFCL) | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | [https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) |
| 131 | **12. Model Context Protocol (MCP), APIs & Function Calling** | Model Context Protocol (MCP) Official Specification | `modelcontextprotocol/specification` | [https://modelcontextprotocol.io](https://modelcontextprotocol.io) |
| 132 | **12. Model Context Protocol (MCP), APIs & Function Calling** | ToolACE Function Calling & Tool Interaction Benchmark | `Team-ACE/ToolACE` | [https://huggingface.co/datasets/Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) |
| 133 | **12. Model Context Protocol (MCP), APIs & Function Calling** | Hermes Function-Calling v1 Dataset | `NousResearch/hermes-function-calling-v1` | [https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) |
| 134 | **12. Model Context Protocol (MCP), APIs & Function Calling** | ToolBench (16,000+ Real REST APIs) | `OpenBMB/ToolBench` | [https://github.com/OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench) |
| 135 | **12. Model Context Protocol (MCP), APIs & Function Calling** | APIBench Executable API Invocations | `gorilla-llm/APIBench` | [https://github.com/ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) |
| 136 | **12. Model Context Protocol (MCP), APIs & Function Calling** | RestBench Real RESTful Web Service Interactions | `Yifan-Song793/RestBench` | [https://github.com/Yifan-Song793/RestBench](https://github.com/Yifan-Song793/RestBench) |
| 137 | **12. Model Context Protocol (MCP), APIs & Function Calling** | JSON-RPC 2.0 Architectural Specification Test Suite | `jsonrpc/spec-suite` | [https://www.jsonrpc.org/specification](https://www.jsonrpc.org/specification) |
| 138 | **13. Technical Documentation, RFCs & Manuals** | The Stack v2 Documentation & ReadTheDocs | `bigcode/the-stack-v2-docs` | [https://devdocs.io](https://devdocs.io) |
| 139 | **13. Technical Documentation, RFCs & Manuals** | MDN Web Docs & Developer Manuals | `mdn/content` | [https://developer.mozilla.org](https://developer.mozilla.org) |
| 140 | **13. Technical Documentation, RFCs & Manuals** | W3C Standards & Web Specifications Catalog | `w3c/specifications` | [https://www.w3.org/TR](https://www.w3.org/TR) |
| 141 | **13. Technical Documentation, RFCs & Manuals** | Python Official Documentation & PEP Archive | `python/cpython-docs` | [https://docs.python.org/3](https://docs.python.org/3) |
| 142 | **13. Technical Documentation, RFCs & Manuals** | Rust Reference, Rustonomicon & Standard Library Docs | `rust-lang/reference` | [https://doc.rust-lang.org/reference](https://doc.rust-lang.org/reference) |
| 143 | **13. Technical Documentation, RFCs & Manuals** | POSIX IEEE 1003.1 Standard Specifications | `opengroup/posix` | [https://pubs.opengroup.org/onlinepubs/9699919799](https://pubs.opengroup.org/onlinepubs/9699919799) |
| 144 | **13. Technical Documentation, RFCs & Manuals** | OpenAPI Specification 3.1 Standards | `OAI/OpenAPI-Specification` | [https://spec.openapis.org/oas/v3.1.0](https://spec.openapis.org/oas/v3.1.0) |
| 145 | **13. Technical Documentation, RFCs & Manuals** | PostgreSQL Official Internals & Documentation | `postgres/postgres-docs` | [https://www.postgresql.org/docs](https://www.postgresql.org/docs) |
| 146 | **13. Technical Documentation, RFCs & Manuals** | Docker & OCI Container Runtime Specs | `opencontainers/runtime-spec` | [https://github.com/opencontainers/runtime-spec](https://github.com/opencontainers/runtime-spec) |
| 147 | **13. Technical Documentation, RFCs & Manuals** | IEEE Xplore Standards Index | `ieee/standards-index` | [https://standards.ieee.org](https://standards.ieee.org) |
| 148 | **14. Scientific Research, Literature & Peer Review** | AllenAI PeS2o (40M+ Open Academic Papers) | `allenai/peS2o` | [https://huggingface.co/datasets/allenai/peS2o](https://huggingface.co/datasets/allenai/peS2o) |
| 149 | **14. Scientific Research, Literature & Peer Review** | AllenAI SciRIFF (Scientific Literature Instruction) | `allenai/SciRIFF` | [https://huggingface.co/datasets/allenai/SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF) |
| 150 | **14. Scientific Research, Literature & Peer Review** | arXiv & PubMed Scientific Papers Corpus | `armanc/scientific_papers` | [https://huggingface.co/datasets/armanc/scientific_papers](https://huggingface.co/datasets/armanc/scientific_papers) |
| 151 | **14. Scientific Research, Literature & Peer Review** | OpenAlex Global Research Catalog (250M+ Works) | `openalex/openalex-snapshot` | [https://openalex.org](https://openalex.org) |
| 152 | **14. Scientific Research, Literature & Peer Review** | Semantic Scholar Open Research Corpus (S2ORC) | `allenai/s2orc` | [https://huggingface.co/datasets/allenai/s2orc](https://huggingface.co/datasets/allenai/s2orc) |
| 153 | **14. Scientific Research, Literature & Peer Review** | PLOS Open Access Complete Research Corpus | `plos/open-access` | [https://plos.org](https://plos.org) |
| 154 | **14. Scientific Research, Literature & Peer Review** | DOAJ Directory of Open Access Journals | `doaj/article-metadata` | [https://doaj.org](https://doaj.org) |
| 155 | **14. Scientific Research, Literature & Peer Review** | BioRxiv & MedRxiv Preprint Archive | `cshl/biorxiv-medrxiv` | [https://www.biorxiv.org](https://www.biorxiv.org) |
| 156 | **14. Scientific Research, Literature & Peer Review** | Europe PMC Open Access Full Text Biomedical Articles | `europepmc/oa-articles` | [https://europepmc.org](https://europepmc.org) |
| 157 | **14. Scientific Research, Literature & Peer Review** | CERN Zenodo Open Science Repository | `cern/zenodo` | [https://zenodo.org](https://zenodo.org) |
| 158 | **15. Law, Governance, Philosophy & Social Sciences** | Pile of Law (Legal Texts, Statutes & Court Opinions) | `pile-of-law/pile-of-law` | [https://huggingface.co/datasets/pile-of-law/pile-of-law](https://huggingface.co/datasets/pile-of-law/pile-of-law) |
| 159 | **15. Law, Governance, Philosophy & Social Sciences** | Free Law Project CourtListener Archive | `freelawproject/courtlistener` | [https://www.courtlistener.com](https://www.courtlistener.com) |
| 160 | **15. Law, Governance, Philosophy & Social Sciences** | MMLU Law, Jurisprudence & Professional Ethics | `cais/mmlu` | [https://huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| 161 | **15. Law, Governance, Philosophy & Social Sciences** | Stanford Encyclopedia of Philosophy (SEP) Archive | `stanford/sep` | [https://plato.stanford.edu](https://plato.stanford.edu) |
| 162 | **15. Law, Governance, Philosophy & Social Sciences** | World Bank Open Data & Governance Indicators | `worldbank/indicators` | [https://data.worldbank.org](https://data.worldbank.org) |
| 163 | **15. Law, Governance, Philosophy & Social Sciences** | United Nations Official Document System (ODS) | `un/ods-documents` | [https://documents.un.org](https://documents.un.org) |
| 164 | **15. Law, Governance, Philosophy & Social Sciences** | European Union EUR-Lex Legal Texts | `eu/eur-lex` | [https://eur-lex.europa.eu](https://eur-lex.europa.eu) |
| 165 | **15. Law, Governance, Philosophy & Social Sciences** | Philippine Supreme Court E-Library & Jurisprudence | `sc-ph/jurisprudence` | [https://elibrary.judiciary.gov.ph](https://elibrary.judiciary.gov.ph) |
| 166 | **15. Law, Governance, Philosophy & Social Sciences** | MMLU Macroeconomics, Microeconomics & Finance | `cais/mmlu` | [https://huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| 167 | **15. Law, Governance, Philosophy & Social Sciences** | PhilPapers Directory of Philosophical Research | `philpapers/archive` | [https://philpapers.org](https://philpapers.org) |
| 168 | **16. Bilingual Filipino / Tagalog & Multilingual Foundations** | CulturaX (Multilingual Tagalog Corpus) | `uonlp/CulturaX` | [https://huggingface.co/datasets/uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) |
| 169 | **16. Bilingual Filipino / Tagalog & Multilingual Foundations** | OSCAR-2301 & CC-100 (Tagalog Web Corpus) | `oscar-corpus/OSCAR-2301` | [https://huggingface.co/datasets/oscar-corpus/OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) |
| 170 | **16. Bilingual Filipino / Tagalog & Multilingual Foundations** | Alpaca-Cleaned General Instruction Corpus | `yahma/alpaca-cleaned` | [https://huggingface.co/datasets/yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) |
| 171 | **16. Bilingual Filipino / Tagalog & Multilingual Foundations** | Aya Multi-Lingual Instruction Collection (101 Languages) | `CohereForAI/aya_collection` | [https://huggingface.co/datasets/CohereForAI/aya_collection](https://huggingface.co/datasets/CohereForAI/aya_collection) |
| 172 | **16. Bilingual Filipino / Tagalog & Multilingual Foundations** | Capybara Multi-Turn High-Quality Dialogue Corpus | `LDJnr/Capybara` | [https://huggingface.co/datasets/LDJnr/Capybara](https://huggingface.co/datasets/LDJnr/Capybara) |
| 173 | **16. Bilingual Filipino / Tagalog & Multilingual Foundations** | Open-Platypus Curated Reasoning Corpus | `garage-bAInd/Open-Platypus` | [https://huggingface.co/datasets/garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) |
| 174 | **16. Bilingual Filipino / Tagalog & Multilingual Foundations** | NVIDIA HelpSteer2 Alignment Corpus | `nvidia/HelpSteer2` | [https://huggingface.co/datasets/nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) |
| 175 | **16. Bilingual Filipino / Tagalog & Multilingual Foundations** | Glot500 Multilingual Parallel Sentences | `cis-lmu/Glot500` | [https://huggingface.co/datasets/cis-lmu/Glot500](https://huggingface.co/datasets/cis-lmu/Glot500) |
| 176 | **16. Bilingual Filipino / Tagalog & Multilingual Foundations** | Philippine National Historical Commission Digital Archives | `nhcp/historical-texts` | [https://nhcp.gov.ph](https://nhcp.gov.ph) |
| 177 | **16. Bilingual Filipino / Tagalog & Multilingual Foundations** | Komisyon sa Wikang Filipino (KWF) Diksiyonaryo at Ortograpiya | `kwf/diksiyonaryo` | [https://kwf.gov.ph](https://kwf.gov.ph) |
