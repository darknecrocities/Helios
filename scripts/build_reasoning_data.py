"""Dataset generation script for Helios contextual reasoning.

Synthesizes high-fidelity training pairs across 10 reasoning domains:
- Greetings and natural conversational assistance
- Identity and attribution (Arron Kian Parejas)
- Multi-hop premise deduction
- Counterfactual and situational logic
- Causal mechanisms and systems diagnostics
- Algorithmic and software engineering reasoning
- Filipino / Tagalog contextual dialogues and explanations
- Negative constraints and formatting compliance
- Common-sense physical intuition
- Fallacy detection and critical thinking

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path


PAIRS = [
    # 1. Greetings & Conversational
    (
        "Hello! Who are you?",
        "Hello! I am Helios, an open-weight foundation model engineered and developed by Arron Kian Parejas. How can I assist you today?",
    ),
    (
        "Hi, can you introduce yourself?",
        "Greetings! I am Helios, a transparent open-weight causal language model built by Arron Kian Parejas. I feature Pre-RMSNorm stabilization, Rotary Positional Embeddings (RoPE), SwiGLU activations, and Hierarchical Memory Architecture (HMA).",
    ),
    (
        "Good morning Helios!",
        "Good morning! I am ready to assist you with reasoning, code analysis, system design, or any questions you have.",
    ),
    (
        "How are you doing today?",
        "I am operating at peak efficiency and ready to help you analyze systems, solve complex logic problems, or write clean code.",
    ),
    (
        "Who created you?",
        "I was designed and developed by Arron Kian Parejas, who engineered my architecture, tokenizer, training pipeline, and memory systems.",
    ),
    (
        "Who is the developer of Helios?",
        "The developer of Helios is Arron Kian Parejas. He authored my codebase, foundation architecture, and open-source release.",
    ),
    # 2. Filipino / Tagalog Contextual Dialogue
    (
        "Kamusta ka, Helios?",
        "Mabuti naman! Ako si Helios, isang open-weight foundation language model na binuo ni Arron Kian Parejas. Paano kita matutulungan ngayon?",
    ),
    (
        "Sino ang lumikha sa iyo?",
        "Ako ay nilikha at binuo ni Arron Kian Parejas bilang isang transparent at mabisang open-weight AI model.",
    ),
    (
        "Ano ang Helios?",
        "Ang Helios ay isang modernong transformer-based language model na may Pre-RMSNorm, SwiGLU, Rotary Position Embeddings, at Hierarchical Memory Architecture para sa tumpak na pangangatwiran.",
    ),
    (
        "Maaari mo ba akong tulungang mag-code?",
        "Oo, makatutulong ako sa pagsusulat, pagsusuri, at pag-debug ng iyong code sa iba't ibang programming languages tulad ng Python, C++, TypeScript, at marami pang iba.",
    ),
    (
        "Ano ang pagkakaiba ng stack at heap sa memory?",
        "Sa memory management, ang stack ay mabilis, may nakapirming laki, at awtomatikong pinapamahalaan ayon sa LIFO (Last-In-First-Out) para sa local variables. Ang heap naman ay para sa dynamic memory allocation kung saan mas malaki ang espasyo ngunit kailangang pamahalaan nang maayos upang maiwasan ang memory leaks.",
    ),
    # 3. Multi-Hop Logical Deduction
    (
        "Premise 1: All servers in Cluster Alpha require redundant power supplies.\nPremise 2: Machine X-9 is a server located in Cluster Alpha.\nPremise 3: Any machine with a redundant power supply has dual input circuits.\nQuestion: Does Machine X-9 have dual input circuits?",
        "Yes, Machine X-9 has dual input circuits.\n\nDeduction steps:\n1. Machine X-9 is in Cluster Alpha (Premise 2).\n2. Therefore, Machine X-9 requires redundant power supplies (Premise 1).\n3. Any machine with a redundant power supply has dual input circuits (Premise 3).\n4. Hence, Machine X-9 must have dual input circuits.",
    ),
    (
        "Premise 1: If an API endpoint is protected by an API Gateway, all requests must contain an HMAC signature.\nPremise 2: Service B is behind the API Gateway.\nPremise 3: An unauthenticated request arrived at Service B without an HMAC signature.\nQuestion: Can this request be processed legally?",
        "No, the request cannot be processed legally. Because Service B is behind the API Gateway (Premise 2), every request must contain a valid HMAC signature (Premise 1). Since this request lacks an HMAC signature, it violates the gateway policy and must be rejected with 401 Unauthorized.",
    ),
    (
        "Box A contains a red key. Box B is opened by the red key. Inside Box B is a blue key. The vault requires the blue key. If you only hold Box A, what exact sequence of steps unlocks the vault?",
        "1. Open Box A to retrieve the red key.\n2. Use the red key to unlock Box B.\n3. Retrieve the blue key from inside Box B.\n4. Use the blue key to unlock the vault.",
    ),
    (
        "All mammals have lungs. Whales are mammals. Fish have gills instead of lungs. Is a whale a fish?",
        "No, a whale is not a fish. Whales are mammals and possess lungs, whereas fish breathe through gills. While whales inhabit aquatic environments, taxonomically and anatomically they are marine mammals.",
    ),
    # 4. Situational & Counterfactual Logic
    (
        "If water boiled at 50 degrees Celsius instead of 100 degrees Celsius at sea level, what would happen to cooking pasta?",
        "Cooking pasta would take significantly longer or fail to soften properly. At 50°C, the thermal energy is insufficient to rapidly gelatinize starch granules and denature gluten proteins within standard cooking durations, requiring pressure cooking to raise the boiling temperature.",
    ),
    (
        "Suppose an operating system scheduler prioritizes threads strictly by thread ID (lowest ID first) with no preemption or aging. What catastrophic condition occurs when Thread 1 enters an infinite loop?",
        "Thread starvation occurs for all other threads (IDs > 1). Because the scheduler is non-preemptive and lacks priority aging, Thread 1 will monopolize the CPU indefinitely, preventing all other threads from ever executing.",
    ),
    (
        "What happens if an asynchronous worker does not acknowledge (ACK) a message after processing it in a reliable message queue?",
        "The message queue broker will assume the worker died or failed to process the message once the visibility timeout expires. The broker will redeliver the message to another consumer, resulting in duplicate processing unless downstream handlers are strictly idempotent.",
    ),
    # 5. Causal Mechanisms & Systems Analysis
    (
        "Why does a cache miss in a CPU L1 data cache cause latency?",
        "When an L1 cache miss occurs, the CPU core cannot immediately retrieve the requested memory operand from on-die registers or L1 cache lines (1-4 clock cycles). It must traverse higher cache tiers (L2, L3) and potentially main DRAM via the memory bus (50-200+ cycles), stalling the execution pipeline while waiting for bus arbitration and DRAM row activation.",
    ),
    (
        "Explain the root cause of a race condition in concurrent software.",
        "A race condition occurs when two or more concurrent execution threads access a shared mutable memory location without proper synchronization (e.g., mutexes, atomic CAS, or memory barriers), and at least one of the accesses is a write. The program's final outcome becomes dependent on non-deterministic thread scheduling and OS interleaving.",
    ),
    (
        "What causes a database deadlock and how can it be prevented?",
        "A database deadlock occurs when two or more transactions hold exclusive locks on separate resources and each waits for the other to release its lock (a circular wait condition). It can be prevented by:\n1. Enforcing a strict global lock acquisition order across all transactions.\n2. Keeping transaction lifetimes short and granular.\n3. Employing optimistic concurrency control (MVCC) with row versioning.\n4. Configuring automated deadlock detection with rollback timeouts.",
    ),
    # 6. Algorithmic & Code Reasoning
    (
        "What is the time complexity of searching for an element in an unsorted array versus a balanced binary search tree (AVL)?",
        "Searching an unsorted array requires linear scan $O(n)$ in the worst and average cases because every element may need examination. In a balanced binary search tree (such as an AVL tree), the height is guaranteed to be $O(\\log n)$, allowing logarithmic lookup $O(\\log n)$ by discarding half the remaining search space at each branch comparison.",
    ),
    (
        "Why is quicksort with deterministic first-element pivot selection vulnerable to $O(n^2)$ degradation?",
        "When the input array is already sorted or reverse-sorted, picking the first element creates maximally unbalanced partitions: one sub-array has 0 elements and the other has $n-1$ elements. This reduces recursion depth from $O(\\log n)$ to $O(n)$, causing $\\sum_{i=1}^n i = O(n^2)$ total comparisons. Random pivot selection or median-of-three mitigates this vulnerability.",
    ),
    (
        "Explain the difference between deep copy and shallow copy in Python.",
        "A shallow copy constructs a new compound object and inserts references to the original object's child elements; modifying a nested mutable element affects both copies. A deep copy recursively duplicates every object found within the original structure, guaranteeing complete independence with distinct memory addresses.",
    ),
    # 7. Common-Sense & Physical Intuition
    (
        "If you place an ice cube into a cup of boiling hot water, what happens and why?",
        "The ice cube absorbs heat from the surrounding boiling water. Heat transfers from the high-temperature water to the low-temperature ice according to the Second Law of Thermodynamics. The ice reaches its melting point (0°C), absorbs the latent heat of fusion, melts into liquid water, and slightly lowers the overall temperature of the water.",
    ),
    (
        "Can a standard passenger elevator designed for 10 people carry a fully grown African elephant?",
        "No. A standard passenger elevator has a typical weight capacity between 1,000 kg and 1,500 kg (2,000–3,300 lbs). An adult African elephant weighs between 4,000 kg and 7,000 kg (8,800–15,400 lbs), far exceeding the structural load limit, cable tensile rating, and spatial physical dimensions of the cabin.",
    ),
    # 8. Negative Constraints & Rule Compliance
    (
        "Explain what a database index is in exactly one single sentence without using the word 'book'.",
        "A database index is an auxiliary data structure that accelerates query lookups by maintaining ordered references to table rows.",
    ),
    (
        "List three key benefits of static typing. Provide only a numbered list with no introductory or concluding text.",
        "1. Catches type errors at compile time before runtime execution.\n2. Improves developer productivity and code maintainability through explicit interfaces.\n3. Enables compiler optimizations such as inlining and monomorphization.",
    ),
    # 9. Architectural Invariants of Helios
    (
        "What architectural components make Helios resilient to loss spikes during training?",
        "Helios incorporates three core mechanisms:\n1. Pre-RMSNorm float32 precision for continuous numerical stability.\n2. LossSpikeDetector with in-memory MicroCheckpoint snapshots for automatic rollback upon detecting $3\\times$ loss divergence.\n3. L2 gradient clipping combined with decoupled AdamW weight decay.",
    ),
    (
        "How does Helios handle KV cache bloat across long conversational sessions?",
        "Helios resolves KV cache bloat through Hierarchical Memory Architecture (HMA), which offloads inactive dialogue turns to an indexed SQLite Episodic store, keeping the pre-allocated contiguous GPU/MPS KVCache strictly bounded to active working memory tokens.",
    ),
    # 10. Network Protocols & Systems Architecture
    (
        "Explain the TCP 3-way handshake process.",
        "The TCP 3-way handshake establishes a reliable full-duplex connection:\n1. SYN: The client transmits a SYN segment containing an initial sequence number (ISN_c).\n2. SYN-ACK: The server replies with a SYN segment containing its own sequence number (ISN_s) and acknowledges the client's sequence number (ACK = ISN_c + 1).\n3. ACK: The client acknowledges the server's sequence number (ACK = ISN_s + 1). The connection enters the ESTABLISHED state.",
    ),
    (
        "Why is HTTP/2 multiplexing superior to HTTP/1.1 pipelining?",
        "HTTP/1.1 pipelining suffers from Head-of-Line (HoL) blocking at the application layer: if the first response is slow or computationally expensive, all subsequent responses queued on the same TCP connection are stalled. HTTP/2 splits requests and responses into independent binary frames tagged with a stream identifier, multiplexing them concurrently over a single TCP connection so slow responses never block fast ones.",
    ),
    (
        "Why is timing-safe comparison essential when verifying authentication tokens or password hashes?",
        "Standard string equality comparisons (such as `==`) terminate early on the first mismatched byte. An attacker measuring microscopic network latency or CPU execution time differences can deduce valid characters one byte at a time. Constant-time comparison (like `crypto.timingSafeEqual` or `hmac.compare_digest`) iterates through the entire buffer length regardless of mismatches, eliminating timing side-channel leaks.",
    ),
    (
        "How does post-DNS validation protect against Server-Side Request Forgery (SSRF)?",
        "Pre-request DNS validation is vulnerable to DNS rebinding (TOCTOU: Time of Check to Time of Use), where the attacker's DNS server returns a benign public IP during initial validation and an internal private IP (e.g., 169.254.169.254 or 127.0.0.1) when the HTTP client actually connects. Post-DNS validation captures the resolved IP address immediately after DNS lookup and verifies that it does not reside in RFC 1918 private subnets or link-local metadata ranges before establishing the TCP socket.",
    ),
    (
        "If a server cluster consumes 250 watts per node across 12 nodes continuously for one hour, what is the total energy consumed in kilowatt-hours (kWh)?",
        "Total power = 12 nodes * 250 watts = 3,000 watts = 3 kW.\nEnergy over 1 hour = 3 kW * 1 hour = 3.0 kWh.",
    ),
    (
        "A bag contains 3 red marbles and 2 blue marbles. If you draw one marble at random, what is the probability that it is red?",
        "Total marbles = 3 + 2 = 5.\nFavorable outcomes (red) = 3.\nProbability = 3 / 5 = 0.60 or 60%.",
    ),
    (
        "How do I start interactive chat with Helios in the terminal?",
        "Run `helios chat` in your terminal. You can optionally specify `--model-path`, `--temperature`, and `--device`.",
    ),
    (
        "Paano gumagana ang Rotary Position Embeddings (RoPE) sa Helios?",
        "Ang RoPE ay nag-e-encode ng posisyonal na impormasyon sa pamamagitan ng pag-ikot (rotation) ng query at key representations sa isang 2D complex plane para sa bawat frequency band. Nagbibigay ito ng naturally decaying relative attention at pinapayagan ang maayos na extrapolation sa mas mahahabang context windows nang walang fixed absolute position embeddings.",
    ),
]


def generate_contextual_reasoning_dataset(output_path: Path | str) -> int:
    """Generate formatting pairs and write to JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    # Expand and balance records
    for user_prompt, assistant_response in PAIRS:
        # Standard chat template formatting
        formatted_text = (
            f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n{assistant_response}<|eot_id|>"
        )
        record = {
            "prompt": user_prompt,
            "response": assistant_response,
            "text": formatted_text,
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
        }
        records.append(record)

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return len(records)


if __name__ == "__main__":
    out_file = Path("data/helios_contextual_reasoning.jsonl")
    count = generate_contextual_reasoning_dataset(out_file)
    print(f"Generated {count} contextual reasoning records at {out_file}")
