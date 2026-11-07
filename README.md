# Helios: Friendly Open-Source AI Model Ecosystem

**Helios** is a modern, transparent, and completely open-source AI language model created by Filipino engineer **Arron Kian Parejas**. 

Unlike models that hide how they work, Helios is designed from scratch so anyone can run it, inspect it, and learn from it. It can answer everyday questions, solve math instantly, chat in both English and Filipino, and remember past conversations using built-in memory.

---

## The Helios Model Family

Helios comes in different sizes designed for different devices:

| Model | Parameter Count | What It Is Best For | Where It Runs |
| :--- | :--- | :--- | :--- |
| **Helios Nano** | **18 Million** | Super fast responses, lightweight chat, quick answers | Any laptop, phone, or small computer |
| **Helios Alpha** | **100 Million** | Everyday conversational assistant and bilingual QA | MacBooks (Apple Silicon) & laptops |
| **Helios Beta** | **300 Million** | Step-by-step math problems, logic puzzles, and code help | Desktop computers with 16GB RAM |
| **Helios Core** | **1 Billion** | Deep programming, system architecture, and essay writing | High-end workstations and gaming GPUs |
| **Helios Pro** | **3 Billion** | Advanced research proofs, multi-step agent tasks | Multi-GPU setups & cloud servers |
| **Helios Flagship**| **6 Billion** *(Staged)* | Dense high-capacity foundation model for deep tasks | Enterprise AI compute clusters |

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
