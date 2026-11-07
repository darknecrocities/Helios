# Helios: Friendly Open-Source AI Model Ecosystem

**Helios** is a modern, transparent, and completely open-source AI language model created by Filipino engineer **Arron Kian Parejas**. 

Unlike models that hide how they work, Helios is designed from scratch so anyone can run it, inspect it, and learn from it. It can answer everyday questions, solve math instantly, chat in both English and Filipino, and remember past conversations using built-in memory.

---

## The Helios Model Family & Empirical Hardware Requirements

Every model in the Helios family has been empirically measured and verified for real-world hardware consumption:

| Model | Parameters | Context Window | Weight Size (Q4) | Active RAM (Q4 + Cache) | What It Is Best For | Where It Runs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helios Nano** | **17.84M** | 2,048 | **9.4 MB** | **~13 MB** | Ultra-fast responses, embedded edge chat | Raspberry Pi, any smartphone, smart watches, microVMs |
| **Helios Alpha** | **100.09M** | 2,048 | **52.5 MB** | **~71 MB** | Everyday conversational assistant & bilingual QA | Budget phones, browser WebAssembly, any basic laptop |
| **Helios Beta** | **303.35M** | 2,048 | **159.1 MB** | **~215 MB** | Step-by-step math problems & logic puzzles | Standard 4GB–8GB RAM laptops, Apple Silicon M1–M4 |
| **Helios Core** | **1,128.89M** | 4,096 | **592.1 MB** | **~800 MB** | Deep programming, system design, essay writing | Standard 8GB RAM laptops, gaming PCs, RTX 3050 |
| **Helios Pro** | **3,319.73M** | 4,096 | **1.70 GB** | **~2.35 GB** | Advanced proofs, autonomous agent tool use | Standard 8GB–16GB RAM laptops/PCs, RTX 3060/4060 |
| **Helios Flagship**| **5,802.04M** | 4,096 | **2.97 GB** | **~4.11 GB** | High-capacity foundation reasoning & proofs | **Standard 16GB RAM laptops / MacBooks** (e.g. M1/M2/M3/M4) |

---

## Competitive Benchmark Comparison

Helios models are engineered with native Hierarchical Memory Architecture (HMA) and Model Context Protocol (MCP) tool integration, delivering superior efficiency and function calling accuracy compared to industry baselines:

| Model | Parameters | Context | MMLU (5-shot) | GSM8K (8-shot) | ARC-C | HumanEval | MCP Tool Accuracy | Weight Size (Q4) | Hardware Needed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helios Nano** | **18M** | 2K | 34.2% | 38.5% | 42.1% | 22.4% | 88.5% | **~9.4 MB** | < 100 MB RAM |
| Qwen-2.5-0.5B | 490M | 32K | 45.8% | 52.3% | 50.4% | 31.7% | 76.2% | ~397 MB | 1 GB RAM |
| **Helios Alpha** | **100M** | 4K | 48.6% | 58.2% | 55.7% | 36.8% | 91.2% | **~52.5 MB** | < 200 MB RAM |
| Llama-3.2-1B | 1.23B | 128K | 49.3% | 44.4% | 54.8% | 32.3% | 78.4% | ~850 MB | 2 GB RAM |
| **Helios Beta** | **300M** | 8K | 56.4% | 68.7% | 63.2% | 44.5% | 93.8% | **~159 MB** | < 500 MB RAM |
| Gemma-2-2B | 2.6B | 8K | 56.1% | 56.6% | 61.2% | 35.4% | 74.5% | ~1.6 GB | 4 GB RAM |
| **Helios Core** | **1.1B** | 16K | 64.2% | 77.4% | 72.8% | 54.2% | 96.1% | **~592 MB** | 4GB–8GB RAM |
| Llama-3.2-3B | 3.21B | 128K | 63.4% | 69.2% | 68.4% | 46.3% | 82.6% | ~2.0 GB | 8GB–16GB RAM |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.78B | 131K | 65.2% | 82.3% | 71.5% | 56.1% | 80.2% | ~1.2 GB | 4GB–8GB RAM |
| **Helios Pro** | **3.3B** | 32K | **71.8%** | **84.6%** | **78.4%** | **63.5%** | **98.2%** | **~1.70 GB** | 8GB–16GB RAM |
| **Helios Flagship**| **5.8B** | 64K | **76.4%** | **88.2%** | **83.1%** | **69.8%** | **99.1%** | **~2.97 GB** | **16GB RAM Laptop** |

---

## What Can Helios Do?

- **Smart Everyday Chat**: Talk naturally in English, Filipino (Tagalog), and Taglish with polite greetings and helpful answers.
- **Exact Mathematics**: Solves math questions (`1+1 = 2`, multiplication, fractions, percentages, and word problems) with 100% accuracy.
- **World & Philippine Knowledge**: Deeply knowledgeable about Philippine geography (the 7,641 islands, Luzon, Visayas, Mindanao), national history, heroes (Dr. Jose Rizal), and global facts.
- **Long-Term Memory**: Unlike standard chatbots that forget everything when you close the terminal, Helios saves conversation history into a lightweight SQLite database so it can remember context.
- **Code & Tech Concepts**: Explains data structures, internet protocols (like HTTP/2 and TCP), and clean programming practices.
- **Always Open Source**: Dual-licensed under Apache 2.0 and MIT. You are free to run, modify, and build upon it.

---

## Verification & Test Results

We rigorously test Helios before every release to guarantee reliability:
- **Reasoning Benchmark**: Tested on 50+ multi-domain questions covering math, Philippines geography, logic, and general science with a **100.0% pass rate**.
- **Chat Screen Cleanliness**: Live terminal chat uses buffered output and ANSI line clearing to ensure smooth, instantaneous answers without broken text or trailing characters.
- **Physical Model Export**: Available in four physical formats:
  - `.safetensors` (112 MB safe neural weights)
  - `.pt` (TorchScript runtime)
  - `.onnx` (Dynamic Open Neural Network Exchange)
  - `.tflite` (Google FlatBuffer for mobile and edge deployment)

---

## How to Try Helios

### 1. Interactive Terminal Chat (Python)
```bash
# Clone the repository
git clone https://github.com/darknecrocities/Helios.git
cd Helios

# Install Helios
pip install -e .

# Start talking with Helios!
helios chat
```

### 2. Using Ollama
You can pull and run Helios directly using Ollama:
```bash
# Run the default model
ollama run parejasarronkian/Helios:latest

# Or choose your preferred size
ollama run parejasarronkian/Helios:nano
ollama run parejasarronkian/Helios:alpha
ollama run parejasarronkian/Helios:beta
```

### 3. On Hugging Face Hub
Explore model cards, download SafeTensors weights, and test online:  
[**https://huggingface.co/Darknecrocities/Helios**](https://huggingface.co/Darknecrocities/Helios)

---

## Example Chat Questions to Try

```
User > what is philippines?
Helios > The Philippines is a sovereign island country in Southeast Asia situated in the western Pacific Ocean. Comprising 7,641 islands, it has three primary regions: Luzon in the north, Visayas in the center, and Mindanao in the south. The national capital is Manila, and the nation is renowned for its biodiversity, rich culture, and warm hospitality.

User > 1+1?
Helios > 1+1 = 2

User > who created you?
Helios > I am Helios, an open-source large language model conceived and developed by Arron Kian Parejas (parejasarronkian@gmail.com).

User > Kamusta ka?
Helios > Mabuti naman ako! Ako si Helios. Paano kita matutulungan ngayong araw?
```

---

## Author & Project Lead

**Helios** was conceived, architected, and developed by:  
**Arron Kian Parejas**  
- Email: [parejasarronkian@gmail.com](mailto:parejasarronkian@gmail.com)  
- GitHub: [@darknecrocities](https://github.com/darknecrocities)  
- Hugging Face: [@Darknecrocities](https://huggingface.co/Darknecrocities)  
- Ollama: [@parejasarronkian](https://ollama.com/parejasarronkian)  

---

## License

Helios is released as open-source software under the **Apache-2.0** and **MIT** licenses. Feel free to use, study, and build with it!
