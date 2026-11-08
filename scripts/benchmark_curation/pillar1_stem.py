"""Pillar 1: STEM & Computing Dataset Generator (600 Items).

Allocations:
- Mathematics & Quantitative: 120 items (Arithmetic 40, Algebra 25, Geometry 20, Calculus 20, Statistics 15)
- Computer Science & Systems: 120 items (Data Structures 30, Algorithms 30, Systems/POSIX 25, Languages 20, ML/Transformers 15)
- Physical Sciences & Astronomy: 100 items (Mechanics 30, Thermodynamics 20, Electromagnetism 25, Astrophysics 25)
- Chemical Sciences: 80 items (Periodic/Stoichiometry 30, Organic Chemistry 25, Thermochemistry/Acids 25)
- Biological & Life Sciences: 90 items (Cell Biology 30, Genetics/DNA 30, Evolution/Ecology 30)
- Medical & Clinical Sciences: 90 items (Physiology/Anatomy 35, Pharmacology 25, Pathology 20, Clinical Safety 10)

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import re
from typing import Any


def _make_item(
    item_id: str, discipline: str, subdiscipline: str, question: str, target: str,
    eval_mode: str = "contains", tolerance: float | None = 0.01, pattern: str = "",
    explanation: str = "", difficulty: str = "medium",
) -> dict[str, Any]:
    exp_pat = pattern or rf"\b{re.escape(target)}\b"
    return {
        "id": item_id, "pillar": "STEM & Computing", "discipline": discipline,
        "subdiscipline": subdiscipline, "question": question, "target": target,
        "eval_mode": eval_mode, "tolerance": tolerance, "expected_pattern": exp_pat,
        "reference_answer": target, "explanation": explanation or f"Scientific grounding for {subdiscipline}.",
        "difficulty": difficulty, "language": "en",
    }


def _build_math() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    arith_configs = [
        (15, 8, 45, 3, "105"), (12, 9, 36, 4, "99"), (20, 6, 50, 5, "110"), (14, 7, 28, 4, "91"),
        (18, 5, 40, 8, "85"), (25, 4, 60, 6, "90"), (16, 6, 72, 9, "88"), (22, 5, 48, 6, "102"),
        (30, 4, 90, 5, "102"), (17, 6, 63, 7, "93"), (19, 4, 32, 8, "72"), (24, 5, 80, 4, "100"),
        (21, 4, 54, 9, "78"), (13, 8, 35, 7, "99"), (26, 3, 42, 6, "71"), (28, 4, 96, 8, "100"),
        (32, 3, 64, 4, "80"), (35, 3, 75, 5, "90"), (40, 3, 100, 5, "100"), (45, 2, 84, 7, "78"),
        (11, 11, 49, 7, "114"), (15, 7, 56, 8, "98"), (18, 6, 66, 6, "97"), (23, 4, 45, 9, "87"),
        (27, 3, 81, 9, "72"), (29, 3, 70, 7, "77"), (31, 3, 48, 8, "87"), (33, 3, 99, 9, "88"),
        (34, 3, 60, 5, "90"), (36, 3, 72, 8, "99"), (37, 2, 50, 5, "64"), (38, 2, 42, 7, "70"),
        (39, 2, 64, 8, "70"), (42, 2, 80, 8, "74"), (44, 2, 90, 9, "78"), (46, 2, 72, 6, "80"),
        (48, 2, 84, 6, "82"), (50, 2, 100, 4, "75"), (52, 2, 96, 8, "92"), (55, 2, 110, 5, "88"),
    ]
    for idx, (a, b, c, d, ans) in enumerate(arith_configs, 1):
        q = f"Calculate {a} * {b} - {c} / {d}."
        items.append(_make_item(f"STEM-MATH-{idx:03d}", "Mathematics & Quantitative", "Arithmetic", q, ans, "numeric", 0.01, rf"\b{ans}\b"))

    for idx in range(1, 26):
        a, x_val, b = idx + 2, idx + 1, idx * 3 + 5
        c = a * x_val + b
        items.append(_make_item(f"STEM-MATH-{idx+40:03d}", "Mathematics & Quantitative", "Algebra", f"Solve for x in the linear equation: {a}x + {b} = {c}.", str(x_val), "numeric", 0.01, rf"\b{x_val}\b"))

    for idx in range(1, 21):
        side = idx + 3
        items.append(_make_item(f"STEM-MATH-{idx+65:03d}", "Mathematics & Quantitative", "Geometry", f"What is the area of a square whose side length is {side} units?", str(side * side), "numeric", 0.01, rf"\b{side * side}\b"))

    for idx in range(1, 21):
        coeff, x_pt = idx + 1, idx % 5 + 1
        deriv = 2 * coeff * x_pt
        items.append(_make_item(f"STEM-MATH-{idx+85:03d}", "Mathematics & Quantitative", "Calculus", f"What is the derivative of f(x) = {coeff}x^2 evaluated at x = {x_pt}?", str(deriv), "numeric", 0.01, rf"\b{deriv}\b"))

    for idx in range(1, 16):
        base = idx * 2
        vals = [base, base + 2, base + 4, base + 6, base + 8]
        mean_v = sum(vals) // len(vals)
        items.append(_make_item(f"STEM-MATH-{idx+105:03d}", "Mathematics & Quantitative", "Statistics", f"What is the arithmetic mean of the numbers {vals[0]}, {vals[1]}, {vals[2]}, {vals[3]}, and {vals[4]}?", str(mean_v), "numeric", 0.01, rf"\b{mean_v}\b"))
    return items


def _build_cs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    ds_data = [
        ("What data structure operates on a Last-In First-Out (LIFO) basis?", "Stack", r"\bstack\b"), ("What data structure operates on a First-In First-Out (FIFO) basis?", "Queue", r"\bqueue\b"),
        ("What dynamic data structure consists of nodes where each node points to the next?", "Linked list", r"linked\s+list"), ("What data structure provides average O(1) time complexity for key-value lookups?", "Hash table", r"hash\s*(table|map)"),
        ("What hierarchical data structure has at most two children per parent node?", "Binary tree", r"binary\s+tree"), ("What self-balancing binary search tree maintains a height balance factor of -1, 0, or 1?", "AVL tree", r"avl(\s+tree)?"),
        ("What self-balancing BST uses node coloring (red and black) to guarantee logarithmic height?", "Red-black tree", r"red[- ]black(\s+tree)?"), ("What tree-based data structure satisfies the heap property where parent is <= children?", "Min-heap", r"min[- ]heap"),
        ("What tree-based data structure satisfies the heap property where parent is >= children?", "Max-heap", r"max[- ]heap"), ("What space-efficient probabilistic data structure tests whether an element is a set member?", "Bloom filter", r"bloom\s+filter"),
        ("What tree structure is commonly used in file systems and databases for block-oriented storage?", "B-tree", r"b[- ]tree"), ("What data structure is used to represent network connections between vertices and edges?", "Graph", r"\bgraph\b"),
        ("What prefix tree structure stores strings character by character along edges?", "Trie", r"\btrie\b"), ("What queue variation allows element insertion and removal at both ends?", "Deque", r"\b(deque|double[- ]ended\s+queue)\b"),
        ("What disjoint-set data structure supports find and union operations with near-linear time?", "Union-find", r"(union[- ]find|disjoint[- ]set)"), ("In a doubly linked list, how many pointer fields does each internal node maintain?", "2", r"\b2\b"),
        ("What data structure is typically used to implement recursion iteratively?", "Stack", r"\bstack\b"), ("What matrix representation stores only non-zero elements to conserve memory?", "Sparse matrix", r"sparse\s+matrix"),
        ("What linear data structure has contiguous memory allocation allowing O(1) random indexing?", "Array", r"\barray\b"), ("What graph representation uses a 2D V x V matrix of booleans or weights?", "Adjacency matrix", r"adjacency\s+matrix"),
        ("What graph representation stores an array or list of neighbors for each vertex?", "Adjacency list", r"adjacency\s+list"), ("What priority queue implementation supports efficient find-min in O(1) time?", "Fibonacci heap", r"fibonacci\s+heap"),
        ("What bit array data structure represents compact bit sequences?", "Bitset", r"(bitset|bit\s+array)"), ("What tree stores multidimensional points for spatial range searches?", "k-d tree", r"k[- ]d\s+tree"),
        ("What circular data structure reuses buffer space when head and tail wrap around?", "Ring buffer", r"(ring|circular)\s+buffer"), ("What data structure underlies Python dictionary lookups?", "Hash table", r"hash\s*(table|map)"),
        ("What tree variation is optimized for range minimum/maximum queries over an interval?", "Segment tree", r"segment\s+tree"), ("What index structure supports prefix, suffix, and substring searches in linear time?", "Suffix tree", r"suffix\s+(tree|array)"),
        ("What skip-list data structure provides probabilistic O(log n) search using multiple linked layers?", "Skip list", r"skip\s+list"), ("What associative array structure maps unique keys to values?", "Map", r"\b(map|dictionary|hash\s*map)\b"),
    ]
    for idx, (q, tgt, pat) in enumerate(ds_data, 1):
        items.append(_make_item(f"STEM-CS-{idx:03d}", "Computer Science & Systems", "Data Structures", q, tgt, "contains", None, pat))

    algo_data = [
        ("What is the average-case time complexity of QuickSort?", "O(n log n)", r"o\(n\s*log\s*n\)"), ("What is the worst-case time complexity of standard QuickSort?", "O(n^2)", r"o\(n\^2\)"),
        ("What is the worst-case time complexity of MergeSort?", "O(n log n)", r"o\(n\s*log\s*n\)"), ("What is the time complexity of binary search on a sorted array of length n?", "O(log n)", r"o\(log\s*n\)"),
        ("What is the time complexity of looking up a key in a balanced AVL tree with n nodes?", "O(log n)", r"o\(log\s*n\)"), ("What algorithm finds the shortest path between nodes in a graph with non-negative edge weights?", "Dijkstra", r"dijkstra"),
        ("What shortest-path algorithm handles graphs with negative edge weights and detects negative cycles?", "Bellman-Ford", r"bellman[- ]ford"), ("What algorithm computes all-pairs shortest paths in a weighted graph in O(V^3) time?", "Floyd-Warshall", r"floyd[- ]warshall"),
        ("What greedy algorithm finds a minimum spanning tree by growing a single tree from a starting vertex?", "Prim", r"prim"), ("What minimum spanning tree algorithm adds edges in increasing weight order using Union-Find?", "Kruskal", r"kruskal"),
        ("What graph traversal algorithm uses a queue to visit vertices level by level?", "Breadth-First Search", r"(breadth[- ]first|bfs)"), ("What graph traversal algorithm uses a stack or recursion to explore deeply before backtracking?", "Depth-First Search", r"(depth[- ]first|dfs)"),
        ("What algorithmic paradigm solves problems by breaking them into overlapping subproblems and memoizing results?", "Dynamic programming", r"dynamic\s+programming"), ("What linear ordering of vertices in a Directed Acyclic Graph (DAG) respects edge directions?", "Topological sort", r"topological\s+(sort|ordering)"),
        ("What string matching algorithm preprocesses the pattern to achieve linear O(n + m) time using a prefix table?", "KMP", r"(kmp|knuth[- ]morris[- ]pratt)"), ("What algorithm computes the greatest common divisor (GCD) of two integers using remainder division?", "Euclidean algorithm", r"euclid(ean)?(\s+algorithm)?"),
        ("What algorithm finds the convex hull of a 2D point set in O(n log n) time?", "Graham scan", r"graham\s+scan"), ("What sorting algorithm has an O(n + k) time complexity by counting element occurrences?", "Counting sort", r"counting\s+sort"),
        ("What non-comparative sorting algorithm sorts keys digit-by-digit or byte-by-byte?", "Radix sort", r"radix\s+sort"), ("What divide-and-conquer algorithm multiplies two n-digit numbers in O(n^1.585) time?", "Karatsuba", r"karatsuba"),
        ("What maximum flow algorithm uses augmenting paths found by BFS in O(V * E^2) time?", "Edmonds-Karp", r"edmonds[- ]karp"), ("What algorithm finds strongly connected components in a directed graph using two DFS passes?", "Kosaraju", r"kosaraju"),
        ("What algorithm finds strongly connected components in a directed graph in a single DFS pass?", "Tarjan", r"tarjan"), ("What optimization technique replaces recursion with an iterative table-filling approach?", "Tabulation", r"tabulation"),
        ("What algorithm checks if a graph is bipartite using 2-coloring?", "BFS", r"(bfs|dfs|breadth[- ]first)"), ("What search algorithm uses a heuristic function h(n) combined with path cost g(n) to find optimal paths?", "A*", r"a\*"),
        ("What problem asks for the shortest tour visiting every city exactly once and returning home?", "Traveling Salesperson Problem", r"(traveling\s+sales(person|man)|tsp)"), ("What decision problem asks whether a Boolean formula can be satisfied (Cook-Levin theorem)?", "SAT", r"(sat|boolean\s+satisfiability)"),
        ("In Big-O notation, what is the time complexity of accessing an array element by index?", "O(1)", r"o\(1\)"), ("What algorithm finds pattern occurrences using rolling polynomial hashing?", "Rabin-Karp", r"rabin[- ]karp"),
    ]
    for idx, (q, tgt, pat) in enumerate(algo_data, 31):
        items.append(_make_item(f"STEM-CS-{idx:03d}", "Computer Science & Systems", "Algorithms", q, tgt, "contains", None, pat))

    sys_data = [
        ("What POSIX signal terminates a process immediately and cannot be caught or ignored?", "SIGKILL", r"sigkill"), ("What POSIX signal is sent when a user presses Ctrl+C in a terminal?", "SIGINT", r"sigint"),
        ("What Unix system call creates a new child process by duplicating the calling process?", "fork", r"\bfork\b"), ("What Unix system call replaces the current process image with a new executable?", "exec", r"\bexec(ve)?\b"),
        ("What system call is used by a parent process to wait for state changes in a child?", "wait", r"\bwait(pid)?\b"), ("What file descriptor number is standard input (stdin) in POSIX systems?", "0", r"\b0\b"),
        ("What file descriptor number is standard output (stdout) in POSIX systems?", "1", r"\b1\b"), ("What file descriptor number is standard error (stderr) in POSIX systems?", "2", r"\b2\b"),
        ("What Unix tool command changes file access permissions?", "chmod", r"chmod"), ("What Unix tool command changes file ownership?", "chown", r"chown"),
        ("What virtual memory mechanism maps virtual pages to physical frames?", "Page table", r"page\s+table"), ("What hardware component accelerates virtual-to-physical address translations in CPUs?", "TLB", r"(tlb|translation\s+lookaside\s+buffer)"),
        ("What condition occurs when processes are unable to proceed because each holds a needed resource?", "Deadlock", r"deadlock"), ("What scheduling algorithm gives each process a fixed slice of CPU time in cyclic order?", "Round-robin", r"round[- ]robin"),
        ("What synchronization primitive provides mutual exclusion for thread-safe critical sections?", "Mutex", r"mutex"), ("What synchronization primitive maintains an integer count of available resources?", "Semaphore", r"semaphore"),
        ("What Unix command displays active processes and system resource utilization?", "top", r"\b(top|htop|ps)\b"), ("What IPC mechanism allows two unrelated processes on the same machine to communicate via a byte stream?", "Unix domain socket", r"(unix\s+domain\s+socket|socket)"),
        ("What memory allocation function in C allocates dynamic memory on the heap without zeroing it?", "malloc", r"malloc"), ("What memory management error occurs when allocated heap memory is never freed?", "Memory leak", r"memory\s+leak"),
        ("What POSIX function creates an unnamed unidirectional pipe for inter-process communication?", "pipe", r"\bpipe\b"), ("What filesystem data structure stores file metadata such as size, permissions, and block pointers?", "Inode", r"inode"),
        ("What kernel architecture runs all operating system services in a single shared address space?", "Monolithic kernel", r"monolithic"), ("What kernel design keeps the core kernel minimal and runs device drivers in user space?", "Microkernel", r"microkernel"),
        ("What network protocol layer does TCP belong to in the OSI model?", "Transport", r"transport"),
    ]
    for idx, (q, tgt, pat) in enumerate(sys_data, 61):
        items.append(_make_item(f"STEM-CS-{idx:03d}", "Computer Science & Systems", "Systems/POSIX", q, tgt, "contains", None, pat))

    lang_data = [
        ("What programming language paradigm avoids mutable state and evaluates mathematical functions?", "Functional", r"functional"), ("What programming paradigm organizes code into classes and objects containing state and behavior?", "Object-Oriented", r"object[- ]oriented"),
        ("What automatic memory management technique frees an object when its reference count drops to zero?", "Reference counting", r"reference\s+counting"), ("What phase of a compiler translates source characters into a stream of tokens?", "Lexical analysis", r"(lexical\s+analysis|tokenizer|lexer)"),
        ("What phase of a compiler builds an Abstract Syntax Tree (AST) from a token stream?", "Parsing", r"(parsing|syntax\s+analysis|parser)"), ("In Python, what keyword defines an anonymous inline function?", "lambda", r"lambda"),
        ("In Rust, what compiler system enforces memory safety without a garbage collector using lifetimes?", "Borrow checker", r"borrow\s+checker"), ("What type system checks types at runtime rather than at compile time?", "Dynamic typing", r"dynamic(ally)?\s+typ(ed?|ing)?"),
        ("What type system enforces types at compile time before execution?", "Static typing", r"static(ally)?\s+typ(ed?|ing)?"), ("In C++, what feature enables template-based generic programming?", "Templates", r"templates?"),
        ("What intermediate bytecode execution engine runs Java programs across different OS platforms?", "JVM", r"(jvm|java\s+virtual\s+machine)"), ("In JavaScript, what engine executes code in Google Chrome and Node.js?", "V8", r"\bv8\b"),
        ("What language feature allows a function to accept arguments of different types while retaining type safety?", "Polymorphism", r"polymorphism"), ("What Python magic method is invoked to construct an object's string representation for debugging?", "__repr__", r"__repr__"),
        ("What garbage collection technique pauses execution to trace reachable objects from root references?", "Tracing", r"(mark[- ]and[- ]sweep|tracing)"), ("In functional programming, what higher-order function transforms each element of a collection?", "map", r"\bmap\b"),
        ("In functional programming, what higher-order function accumulates elements into a single return value?", "reduce", r"(reduce|fold)"), ("What scope rule resolves variable names based on lexical nesting in the source code?", "Lexical scoping", r"lexical(\s+scoping)?"),
        ("What design pattern ensures a class has only one instance and provides a global access point?", "Singleton", r"singleton"), ("In Python, what decorator is used to define an immutable dataclass?", "@dataclass(frozen=True)", r"frozen\s*=\s*True"),
    ]
    for idx, (q, tgt, pat) in enumerate(lang_data, 86):
        items.append(_make_item(f"STEM-CS-{idx:03d}", "Computer Science & Systems", "Languages", q, tgt, "contains", None, pat))

    ml_data = [
        ("In transformer models, what does the acronym GQA stand for?", "Grouped-Query Attention", r"grouped[- ]query\s+attention"), ("What positional encoding method rotates query and key vectors in complex 2D planes?", "RoPE", r"(rope|rotary\s+position)"),
        ("What normalization technique simplifies LayerNorm by scaling inputs using root-mean-square?", "RMSNorm", r"rmsnorm"), ("What activation function combines Swish gating with linear projection in modern LLMs?", "SwiGLU", r"swiglu"),
        ("What attention acceleration algorithm avoids materializing the full N x N attention matrix in HBM?", "FlashAttention", r"flash\s*attention"), ("In multi-head attention, if d_model is 768 and n_heads is 12, what is the head dimension d_k?", "64", r"\b64\b"),
        ("What optimization technique pairs a small draft model with a large target model to accelerate generation?", "Speculative decoding", r"speculative\s+decoding"), ("What mechanism caches previous key and value projections during autoregressive generation?", "KV cache", r"kv\s+cache"),
        ("What algorithm computes parameter gradients by applying the chain rule backwards?", "Backpropagation", r"backpropagation"), ("What loss function is standardly minimized during autoregressive causal language modeling?", "Cross-entropy", r"cross[- ]entropy"),
        ("What hyperparameter controls the balance between exploration and greediness during token sampling?", "Temperature", r"temperature"), ("What sampling strategy limits candidate tokens to the smallest set whose cumulative probability exceeds p?", "Top-p", r"(top[- ]p|nucleus)"),
        ("What parameter scaling technique extends transformer context length by modifying RoPE base frequency?", "NTK-Aware scaling", r"ntk"), ("In neural network optimization, what adaptive optimizer combines momentum and RMSprop?", "Adam", r"adam(w)?"),
        ("What metric measures language model predictive uncertainty as exponentiated average negative log-likelihood?", "Perplexity", r"perplexity"),
    ]
    for idx, (q, tgt, pat) in enumerate(ml_data, 106):
        items.append(_make_item(f"STEM-CS-{idx:03d}", "Computer Science & Systems", "ML/Transformers", q, tgt, "contains", None, pat))

    return items


def _build_phys() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    mech_data = [
        ("What is the SI unit of force?", "Newton", r"\bnewton\b"), ("What is the approximate gravitational acceleration near Earth's surface in m/s^2?", "9.8", r"\b9\.8\b"),
        ("What physical quantity is defined as mass multiplied by velocity?", "Momentum", r"momentum"), ("What is the SI unit of work and energy?", "Joule", r"\bjoule\b"),
        ("What is the SI unit of power, equivalent to one joule per second?", "Watt", r"\bwatt\b"), ("What law states that an object at rest stays at rest unless acted upon by an external net force?", "Newton's First Law", r"first\s+law"),
        ("What law relates net force to mass and acceleration as F = m * a?", "Newton's Second Law", r"second\s+law"), ("What law states that for every action force there is an equal and opposite reaction force?", "Newton's Third Law", r"third\s+law"),
        ("What type of collision conserves both total kinetic energy and total linear momentum?", "Elastic collision", r"elastic"), ("What type of collision conserves momentum but results in a loss of kinetic energy?", "Inelastic collision", r"inelastic"),
        ("What rotational analog of force causes angular acceleration about a pivot?", "Torque", r"torque"), ("What physical quantity measures a body's resistance to angular acceleration?", "Moment of inertia", r"moment\s+of\s+inertia"),
        ("What force directs a body inward toward the center of curvature during circular motion?", "Centripetal force", r"centripetal"), ("What is the mechanical energy of an object due to its motion called?", "Kinetic energy", r"kinetic\s+energy"),
        ("What form of energy is stored by an object due to its position in a gravitational field?", "Potential energy", r"potential\s+energy"), ("What universal constant G is approximately 6.674e-11 N m^2 / kg^2?", "Gravitational constant", r"gravitational\s+constant"),
        ("What law describes the restoring force of an ideal spring as F = -k * x?", "Hooke's Law", r"hooke"), ("In simple harmonic motion, what physical parameter is the reciprocal of the period T?", "Frequency", r"frequency"),
        ("What friction force opposes the initiation of motion between stationary surfaces?", "Static friction", r"static\s+friction"), ("What friction force acts between surfaces that are sliding across each other?", "Kinetic friction", r"kinetic\s+friction"),
        ("What is the escape velocity from the surface of the Earth approximately in km/s?", "11.2", r"\b11\.2\b"), ("What property of fluid motion causes resistance to flow due to internal friction?", "Viscosity", r"viscosity"),
        ("What principle states that an upward buoyant force equals the weight of displaced fluid?", "Archimedes' Principle", r"archimedes"), ("What equation relates fluid pressure, velocity, and height along a streamline for ideal flow?", "Bernoulli's Equation", r"bernoulli"),
        ("What term describes the constant maximum speed of an object falling through a fluid?", "Terminal velocity", r"terminal\s+velocity"), ("What physical quantity is defined as the time rate of change of momentum?", "Force", r"\bforce\b"),
        ("What physical quantity is represented by the area under a force-versus-time graph?", "Impulse", r"impulse"), ("What physical quantity is represented by the area under a force-versus-displacement graph?", "Work", r"\bwork\b"),
        ("What vector cross-product operation calculates torque from radius r and force F?", "r x F", r"(r\s*x\s*f|cross\s+product)"), ("What dimensionless number characterizes laminar versus turbulent fluid flow regimes?", "Reynolds number", r"reynolds"),
    ]
    for idx, (q, tgt, pat) in enumerate(mech_data, 1):
        items.append(_make_item(f"STEM-PHYS-{idx:03d}", "Physical Sciences & Astronomy", "Mechanics", q, tgt, "contains", None, pat))

    thermo_data = [
        ("What thermodynamic quantity measures microscopic disorder or unavailable energy?", "Entropy", r"entropy"), ("What temperature on the Celsius scale corresponds to absolute zero (0 Kelvin)?", "-273.15", r"-273(\.15)?"),
        ("What law of thermodynamics states thermal equilibrium is transitive?", "Zeroth Law", r"zeroth"), ("What law of thermodynamics states total energy of an isolated system is conserved?", "First Law", r"first\s+law"),
        ("What law states total entropy of an isolated system can never decrease over time?", "Second Law", r"second\s+law"), ("What law states entropy of a pure crystalline substance approaches zero at 0 K?", "Third Law", r"third\s+law"),
        ("What theoretical cycle achieves maximum possible efficiency operating between two heat reservoirs?", "Carnot cycle", r"carnot"), ("In the ideal gas law PV = nRT, what does the variable R represent?", "Universal gas constant", r"(gas\s+constant|universal\s+gas)"),
        ("What thermodynamic process occurs at constant volume?", "Isochoric", r"(isochoric|isovolumetric)"), ("What thermodynamic process occurs at constant pressure?", "Isobaric", r"isobaric"),
        ("What thermodynamic process occurs at constant temperature?", "Isothermal", r"isothermal"), ("What thermodynamic process occurs without any heat transfer into or out of the system?", "Adiabatic", r"adiabatic"),
        ("What thermodynamic potential measures maximum reversible work at constant T and P?", "Gibbs free energy", r"gibbs"), ("What is the SI unit of thermodynamic temperature?", "Kelvin", r"kelvin"),
        ("What mode of heat transfer occurs through direct physical molecular contact?", "Conduction", r"conduction"), ("What mode of heat transfer occurs through fluid macroscopic circulation?", "Convection", r"convection"),
        ("What mode of heat transfer transmits thermal energy via electromagnetic waves without a medium?", "Radiation", r"radiation"), ("What law states thermal radiative power from a blackbody is proportional to T^4?", "Stefan-Boltzmann Law", r"stefan[- ]boltzmann"),
        ("What law states peak emission wavelength of blackbody radiation is inversely proportional to temperature?", "Wien's Law", r"wien"), ("What is the heat capacity ratio (Cp / Cv) called in thermodynamics?", "Adiabatic index", r"(adiabatic\s+index|heat\s+capacity\s+ratio|gamma)"),
    ]
    for idx, (q, tgt, pat) in enumerate(thermo_data, 31):
        items.append(_make_item(f"STEM-PHYS-{idx:03d}", "Physical Sciences & Astronomy", "Thermodynamics", q, tgt, "contains", None, pat))

    em_data = [
        ("What is the approximate speed of light in vacuum in meters per second?", "300,000,000", r"(3(\.0)?\s*x?\s*10\^?8|300,?000,?000)"), ("What is the SI unit of electrical resistance?", "Ohm", r"\bohm\b"),
        ("What is the SI unit of electric charge?", "Coulomb", r"coulomb"), ("What is the SI unit of electric current?", "Ampere", r"ampere"),
        ("What is the SI unit of electrical potential difference (voltage)?", "Volt", r"\bvolt\b"), ("What is the SI unit of magnetic flux density?", "Tesla", r"tesla"),
        ("What is the SI unit of electrical capacitance?", "Farad", r"farad"), ("What is the SI unit of electrical inductance?", "Henry", r"henry"),
        ("What law states electric current is proportional to voltage and inversely proportional to resistance (V = I * R)?", "Ohm's Law", r"ohm"), ("What law describes electrostatic force between two point charges as inverse-square with distance?", "Coulomb's Law", r"coulomb"),
        ("What set of four fundamental equations unifies electricity, magnetism, and optics?", "Maxwell's Equations", r"maxwell"), ("What law states that an induced EMF opposes the magnetic flux change that produced it?", "Lenz's Law", r"lenz"),
        ("What law relates induced EMF to the rate of magnetic flux change?", "Faraday's Law", r"faraday"), ("What law relates magnetic field along a closed loop to electric current passing through it?", "Ampere's Law", r"ampere"),
        ("What rule uses thumb, index, and middle finger to determine magnetic force direction on positive charge?", "Right-hand rule", r"right[- ]hand\s+rule"), ("What fundamental constant c connects electromagnetic waves to the speed of light?", "Speed of light", r"speed\s+of\s+light"),
        ("What constant represents electric permittivity of free space epsilon_0 in F/m?", "8.854e-12", r"8\.85"), ("What constant represents magnetic permeability of free space mu_0 in H/m?", "4pi x 10^-7", r"4\s*pi|1\.25"),
        ("What electromagnetic device converts alternating current voltage levels using mutual induction?", "Transformer", r"transformer"), ("What semiconductor device permits electric current flow primarily in one direction?", "Diode", r"diode"),
        ("What three-terminal semiconductor device amplifies or switches electronic signals?", "Transistor", r"transistor"), ("What circuit law states that sum of currents entering a junction equals sum of currents leaving?", "Kirchhoff's Current Law", r"kirchhoff.*(current|junction)"),
        ("What circuit law states directed sum of potential differences around any closed loop is zero?", "Kirchhoff's Voltage Law", r"kirchhoff.*(voltage|loop)"), ("What optical law states ratio of sines of incidence and refraction angles equals phase velocity ratio?", "Snell's Law", r"snell"),
        ("What optical phenomenon occurs when light striking a boundary beyond critical angle reflects completely?", "Total internal reflection", r"total\s+internal\s+reflection"),
    ]
    for idx, (q, tgt, pat) in enumerate(em_data, 51):
        items.append(_make_item(f"STEM-PHYS-{idx:03d}", "Physical Sciences & Astronomy", "Electromagnetism", q, tgt, "contains", None, pat))

    astro_data = [
        ("What boundary around a black hole marks the point of no return where light cannot escape?", "Event horizon", r"event\s+horizon"), ("What radius equation calculates event horizon of a non-rotating black hole as 2GM/c^2?", "Schwarzschild radius", r"schwarzschild"),
        ("What radiation is predicted to emit from black holes due to quantum vacuum fluctuations?", "Hawking radiation", r"hawking"), ("What is the largest planet in our solar system by mass and volume?", "Jupiter", r"jupiter"),
        ("What planet in our solar system is closest to the Sun?", "Mercury", r"mercury"), ("What planet in our solar system is known as the Red Planet due to iron oxide on its surface?", "Mars", r"mars"),
        ("What is the primary nuclear fusion reaction powering the Sun during main sequence?", "Proton-proton chain", r"(proton[- ]proton|p[- ]p|hydrogen\s+fusion)"), ("What stage of stellar evolution will our Sun enter after exhausting core hydrogen?", "Red giant", r"red\s+giant"),
        ("What dense stellar remnant remains after a medium mass star sheds outer layers as planetary nebula?", "White dwarf", r"white\s+dwarf"), ("What mass limit of ~1.44 solar masses prevents electron degeneracy from supporting a white dwarf?", "Chandrasekhar limit", r"chandrasekhar"),
        ("What astronomical object consists almost entirely of neutrons supported by neutron degeneracy pressure?", "Neutron star", r"neutron\s+star"), ("What rapidly rotating magnetized neutron star emits beams of electromagnetic radiation from its poles?", "Pulsar", r"pulsar"),
        ("What catastrophic stellar explosion marks the death of a massive star?", "Supernova", r"supernova"), ("What law states recessional velocity of galaxies is proportional to their distance from Earth?", "Hubble's Law", r"hubble"),
        ("What relic radiation from ~380,000 years after Big Bang permeates the universe at ~2.7 K?", "Cosmic Microwave Background", r"(cosmic\s+microwave|cmb)"), ("What hypothetical form of matter makes up ~27% of universe mass-energy?", "Dark matter", r"dark\s+matter"),
        ("What mysterious energy accelerates universe expansion and comprises ~68% of mass-energy?", "Dark energy", r"dark\s+energy"), ("What spiral galaxy is our solar system located in?", "Milky Way", r"milky\s+way"),
        ("What is the nearest major spiral galaxy to the Milky Way?", "Andromeda", r"andromeda"), ("What distance unit equals the average distance from Earth to Sun (~149.6 million km)?", "Astronomical Unit", r"(astronomical\s+unit|au)"),
        ("What unit of astronomical distance equals ~3.26 light-years or one second of parallax arc?", "Parsec", r"parsec"), ("What telescope launched in 2021 observes the universe in infrared from Sun-Earth L2?", "James Webb Space Telescope", r"(james\s+webb|jwst)"),
        ("What planet in our solar system has the most extensive and prominent ring system?", "Saturn", r"saturn"), ("What dwarf planet in the Kuiper belt was visited by New Horizons in 2015?", "Pluto", r"pluto"),
        ("What point in an elliptical planetary orbit around the Sun is closest to the Sun?", "Perihelion", r"perihelion"),
    ]
    for idx, (q, tgt, pat) in enumerate(astro_data, 76):
        items.append(_make_item(f"STEM-PHYS-{idx:03d}", "Physical Sciences & Astronomy", "Astrophysics", q, tgt, "contains", None, pat))
    return items


def _build_chem() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    elem_info = [
        ("Hydrogen", "H", 1), ("Helium", "He", 2), ("Lithium", "Li", 3), ("Beryllium", "Be", 4),
        ("Boron", "B", 5), ("Carbon", "C", 6), ("Nitrogen", "N", 7), ("Oxygen", "O", 8),
        ("Fluorine", "F", 9), ("Neon", "Ne", 10), ("Sodium", "Na", 11), ("Magnesium", "Mg", 12),
        ("Aluminum", "Al", 13), ("Silicon", "Si", 14), ("Phosphorus", "P", 15), ("Sulfur", "S", 16),
        ("Chlorine", "Cl", 17), ("Argon", "Ar", 18), ("Potassium", "K", 19), ("Calcium", "Ca", 20),
        ("Iron", "Fe", 26), ("Copper", "Cu", 29), ("Zinc", "Zn", 30), ("Silver", "Ag", 47),
        ("Gold", "Au", 79), ("Mercury", "Hg", 80), ("Lead", "Pb", 82), ("Uranium", "U", 92),
    ]
    for idx, (name, sym, z) in enumerate(elem_info, 1):
        items.append(_make_item(f"STEM-CHEM-{idx:03d}", "Chemical Sciences", "Periodic/Stoichiometry", f"What is the chemical symbol of the element {name} (atomic number {z})?", sym, "exact", None, rf"\b{sym}\b"))
    items.append(_make_item("STEM-CHEM-029", "Chemical Sciences", "Periodic/Stoichiometry", "What is Avogadro's constant approximately equal to in mol^-1?", "6.022e23", "contains", None, r"6\.022"))
    items.append(_make_item("STEM-CHEM-030", "Chemical Sciences", "Periodic/Stoichiometry", "What is the molar volume of an ideal gas at STP in liters?", "22.4", "numeric", 0.01, r"\b22\.4\b"))

    org_data = [
        ("What is the simplest alkane hydrocarbon with formula CH4?", "Methane", r"methane"), ("What two-carbon alkane has formula C2H6?", "Ethane", r"ethane"),
        ("What simplest alkene contains a carbon-carbon double bond with formula C2H4?", "Ethene", r"(ethene|ethylene)"), ("What simplest alkyne contains a triple bond with formula C2H2?", "Ethyne", r"(ethyne|acetylene)"),
        ("What aromatic hydrocarbon consists of a 6-carbon ring with alternating resonance bonds (C6H6)?", "Benzene", r"benzene"), ("What functional group is characterized by -OH bonded to carbon?", "Hydroxyl", r"(hydroxyl|alcohol)"),
        ("What functional group features a carbon double-bonded to oxygen and single-bonded to OH (-COOH)?", "Carboxyl", r"(carboxyl|carboxylic)"), ("What functional group has a carbonyl carbon bonded to at least one hydrogen (R-CHO)?", "Aldehyde", r"aldehyde"),
        ("What functional group has a carbonyl carbon bonded to two carbon atoms (R-CO-R')?", "Ketone", r"ketone"), ("What functional group is formed by reaction of carboxylic acid with alcohol (R-COO-R')?", "Ester", r"ester"),
        ("What functional group features an oxygen atom bonded between two alkyl groups (R-O-R')?", "Ether", r"ether"), ("What nitrogen-containing functional group is derived from ammonia (R-NH2)?", "Amine", r"amine"),
        ("What functional group consists of a carbonyl bonded to nitrogen (R-CO-NR2)?", "Amide", r"amide"), ("What sulfur analog of an alcohol has general formula R-SH?", "Thiol", r"thiol"),
        ("What term describes molecules with identical molecular formulas but different structural connectivity?", "Structural isomers", r"(structural|constitutional)\s+isomers?"), ("What term describes non-superimposable mirror-image stereoisomers?", "Enantiomers", r"enantiomers?"),
        ("What carbon atom is bonded to four different chemical substituents creating chirality?", "Chiral center", r"chiral(\s+center)?"), ("What reaction combines monomer units by releasing a small molecule such as water?", "Condensation", r"condensation"),
        ("What reaction cleaves a chemical bond by adding a water molecule?", "Hydrolysis", r"hydrolysis"), ("What type of bond joins amino acids together to form proteins?", "Peptide bond", r"peptide\s+bond"),
        ("What three-carbon trialcohol forms the backbone of triglycerides and phospholipids?", "Glycerol", r"glycerol"), ("What polymer of glucose serves as the primary structural component of plant cell walls?", "Cellulose", r"cellulose"),
        ("What branched glucose polymer acts as energy storage in animal liver and muscle tissue?", "Glycogen", r"glycogen"), ("What organic compound with formula C6H12O6 is primary substrate for glycolysis?", "Glucose", r"glucose"),
        ("What reaction converts vegetable oils into soap using a strong alkaline base?", "Saponification", r"saponification"),
    ]
    for idx, (q, tgt, pat) in enumerate(org_data, 31):
        items.append(_make_item(f"STEM-CHEM-{idx:03d}", "Chemical Sciences", "Organic Chemistry", q, tgt, "contains", None, pat))

    acid_data = [
        ("On the pH scale at 25 C, what numeric value represents a neutral aqueous solution?", "7", r"\b7\b"), ("On the pH scale, do solutions with pH less than 7 exhibit acidic or basic properties?", "Acidic", r"acid(ic)?"),
        ("On the pH scale, do solutions with pH greater than 7 exhibit acidic or basic properties?", "Basic", r"(basic|alkaline)"), ("According to Bronsted-Lowry definition, what species acts as a proton (H+) donor?", "Acid", r"\bacid\b"),
        ("According to Bronsted-Lowry definition, what species acts as a proton (H+) acceptor?", "Base", r"\bbase\b"), ("According to Lewis definition, what species acts as an electron pair acceptor?", "Lewis acid", r"lewis\s+acid"),
        ("According to Lewis definition, what species acts as an electron pair donor?", "Lewis base", r"lewis\s+base"), ("What chemical reaction releases thermal energy resulting in negative delta H?", "Exothermic", r"exothermic"),
        ("What chemical reaction absorbs thermal energy resulting in positive delta H?", "Endothermic", r"endothermic"), ("What law states total enthalpy change is independent of pathway taken?", "Hess's Law", r"hess"),
        ("What strong acid has the chemical formula HCl?", "Hydrochloric acid", r"hydrochloric"), ("What strong diprotic acid has the chemical formula H2SO4?", "Sulfuric acid", r"sulfuric"),
        ("What strong acid has the chemical formula HNO3?", "Nitric acid", r"nitric"), ("What strong base has the chemical formula NaOH?", "Sodium hydroxide", r"sodium\s+hydroxide"),
        ("What weak acid is primary organic component of household vinegar?", "Acetic acid", r"acetic"), ("What solution resists changes in pH when small amounts of acid or base are added?", "Buffer solution", r"buffer"),
        ("What equation calculates buffer pH as pH = pKa + log([A-]/[HA])?", "Henderson-Hasselbalch", r"henderson[- ]hasselbalch"), ("What equilibrium constant Kw represents ion product of water at 25 C?", "1.0e-14", r"1(\.0)?e-14|1(\.0)?\s*x?\s*10\^?-14"),
        ("What indicator turns pink in basic solutions and remains colorless in acidic solutions?", "Phenolphthalein", r"phenolphthalein"), ("What analytical procedure determines unknown solution concentration using a burette?", "Titration", r"titration"),
        ("In titration, what point indicates stoichiometric equivalence of acid and base?", "Equivalence point", r"equivalence\s+point"), ("What thermodynamic quantity S measures molecular microstates and randomness?", "Entropy", r"entropy"),
        ("What equation relates standard free energy change delta G to temperature and equilibrium constant K?", "delta G = -RT ln K", r"-rt\s*ln\s*k"), ("What principle states a system at equilibrium shifts to counteract disturbance?", "Le Chatelier's Principle", r"le\s+chatelier"),
        ("What substance increases reaction rate without undergoing permanent chemical change?", "Catalyst", r"catalyst"),
    ]
    for idx, (q, tgt, pat) in enumerate(acid_data, 56):
        items.append(_make_item(f"STEM-CHEM-{idx:03d}", "Chemical Sciences", "Thermochemistry/Acids", q, tgt, "contains", None, pat))
    return items


def _build_bio() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cell_data = [
        ("What cellular organelle is known as the powerhouse of the cell for generating ATP?", "Mitochondria", r"mitochondri(a|on)"), ("What organelle contains the cell's genetic material in eukaryotes?", "Nucleus", r"nucleus"),
        ("What cellular structure conducts protein synthesis by translating mRNA?", "Ribosome", r"ribosome"), ("What organelle modifies, sorts, and packages proteins for transport?", "Golgi apparatus", r"golgi"),
        ("What organelle studded with ribosomes synthesizes secretory proteins?", "Rough endoplasmic reticulum", r"rough\s+(er|endoplasmic)"), ("What organelle synthesizes lipids, phospholipids, and steroid hormones?", "Smooth endoplasmic reticulum", r"smooth\s+(er|endoplasmic)"),
        ("What organelle contains digestive enzymes to break down waste and debris?", "Lysosome", r"lysosome"), ("What plant organelle carries out photosynthesis by converting sunlight into energy?", "Chloroplast", r"chloroplast"),
        ("What green pigment inside chloroplasts absorbs light for photosynthesis?", "Chlorophyll", r"chlorophyll"), ("What fluid mosaic structure regulates passage of substances into and out of the cell?", "Plasma membrane", r"(plasma|cell)\s+membrane"),
        ("What rigid layer outside plasma membrane provides structural support in plant cells?", "Cell wall", r"cell\s+wall"), ("What network of protein filaments maintains eukaryotic cell shape?", "Cytoskeleton", r"cytoskeleton"),
        ("What molecule powers cellular activities through high-energy phosphate bonds?", "ATP", r"\batp\b"), ("What metabolic pathway converts glucose into pyruvate yielding 2 net ATP?", "Glycolysis", r"glycolysis"),
        ("What metabolic sequence oxidizes acetyl-CoA to CO2 in mitochondrial matrix?", "Krebs cycle", r"(krebs|citric\s+acid)\s+cycle"), ("What stage of respiration produces majority of ATP via oxidative phosphorylation?", "Electron transport chain", r"(electron\s+transport|oxidative\s+phosphorylation)"),
        ("What anaerobic process regenerates NAD+ from pyruvate without using oxygen?", "Fermentation", r"fermentation"), ("What cell division process produces two genetically identical diploid daughter cells?", "Mitosis", r"mitosis"),
        ("What specialized cell division produces four genetically diverse haploid gametes?", "Meiosis", r"meiosis"), ("In the cell cycle, what non-dividing phase consists of G1, S, and G2 sub-phases?", "Interphase", r"interphase"),
        ("During what sub-phase of interphase does DNA replication occur?", "S phase", r"s\s+phase"), ("During what stage of mitosis do chromosomes condense and nuclear envelope breaks down?", "Prophase", r"prophase"),
        ("During what stage of mitosis do chromosomes align along equatorial plane?", "Metaphase", r"metaphase"), ("During what stage of mitosis do sister chromatids separate toward spindle poles?", "Anaphase", r"anaphase"),
        ("During what stage of mitosis do nuclear envelopes re-form around separated chromosomes?", "Telophase", r"telophase"), ("What physical process divides the cytoplasm into two daughter cells?", "Cytokinesis", r"cytokinesis"),
        ("What programmed cell death mechanism eliminates damaged or unneeded cells cleanly?", "Apoptosis", r"apoptosis"), ("What cellular transport mechanism moves molecules against concentration gradient using ATP?", "Active transport", r"active\s+transport"),
        ("What passive transport process moves water across selectively permeable membrane?", "Osmosis", r"osmosis"), ("What cellular uptake process engulfs large particles via membrane invagination?", "Endocytosis", r"endocytosis"),
    ]
    for idx, (q, tgt, pat) in enumerate(cell_data, 1):
        items.append(_make_item(f"STEM-BIO-{idx:03d}", "Biological & Life Sciences", "Cell Biology", q, tgt, "contains", None, pat))

    gen_data = [
        ("In DNA base pairing, what nitrogenous base pairs with Adenine via two hydrogen bonds?", "Thymine", r"thymine"), ("In DNA base pairing, what nitrogenous base pairs with Guanine via three hydrogen bonds?", "Cytosine", r"cytosine"),
        ("In RNA, what pyrimidine base replaces Thymine to pair with Adenine?", "Uracil", r"uracil"), ("What 5-carbon sugar forms the structural backbone of DNA?", "Deoxyribose", r"deoxyribose"),
        ("What 5-carbon sugar forms the structural backbone of RNA?", "Ribose", r"\bribose\b"), ("What enzyme synthesizes complementary DNA strands by adding deoxynucleotides?", "DNA polymerase", r"dna\s+polymerase"),
        ("What enzyme unwinds the double helix at the replication fork?", "Helicase", r"helicase"), ("What enzyme seals nicks in phosphodiester backbone by joining Okazaki fragments?", "DNA ligase", r"ligase"),
        ("What enzyme synthesizes RNA primers required for DNA polymerase initiation?", "Primase", r"primase"), ("What process transcribes genetic info from DNA template into messenger RNA?", "Transcription", r"transcription"),
        ("What enzyme synthesizes RNA during transcription?", "RNA polymerase", r"rna\s+polymerase"), ("What process translates mRNA nucleotide sequences into amino acid chains at ribosomes?", "Translation", r"translation"),
        ("What triplet sequence of nucleotides in mRNA specifies a particular amino acid?", "Codon", r"codon"), ("What universal start codon encodes Methionine in eukaryotic translation?", "AUG", r"\baug\b"),
        ("Name one of the three standard stop codons in universal genetic code (UAA, UAG, UGA).", "UAA", r"(uaa|uag|uga)"), ("What adaptor RNA molecule carries specific amino acids to ribosome during translation?", "tRNA", r"trna|transfer\s+rna"),
        ("What triplet sequence on tRNA pairs complementarily with an mRNA codon?", "Anticodon", r"anticodon"), ("What Austrian monk established foundational laws of inheritance experimenting on pea plants?", "Gregor Mendel", r"mendel"),
        ("What Mendelian law states paired alleles segregate randomly during gamete formation?", "Law of Segregation", r"segregation"), ("What Mendelian law states alleles for different traits sort independently?", "Law of Independent Assortment", r"independent\s+assortment"),
        ("What term describes an organism carrying two identical alleles for a particular gene?", "Homozygous", r"homozygous"), ("What term describes an organism carrying two different alleles for a particular gene?", "Heterozygous", r"heterozygous"),
        ("What term describes an organism's observable physical traits resulting from genotype?", "Phenotype", r"phenotype"), ("What term describes an organism's underlying genetic makeup or allele combination?", "Genotype", r"genotype"),
        ("What genetic condition occurs when both alleles are fully expressed simultaneously, as in AB blood?", "Codominance", r"codominance"), ("What genetic tool diagram predicts offspring genotypes and phenotypes from parental crosses?", "Punnett square", r"punnett"),
        ("What bacterial defense system was adapted into RNA-guided genome editing tool?", "CRISPR-Cas9", r"crispr"), ("What repetitive protective DNA sequences cap ends of eukaryotic chromosomes?", "Telomeres", r"telomere"),
        ("What enzyme elongates telomeres to prevent chromosome shortening in stem cells?", "Telomerase", r"telomerase"), ("What permanent alteration in DNA sequence produces novel genetic variation?", "Mutation", r"mutation"),
    ]
    for idx, (q, tgt, pat) in enumerate(gen_data, 31):
        items.append(_make_item(f"STEM-BIO-{idx:03d}", "Biological & Life Sciences", "Genetics/DNA", q, tgt, "contains", None, pat))

    eco_data = [
        ("What English naturalist published 'On the Origin of Species' proposing natural selection in 1859?", "Charles Darwin", r"darwin"), ("What mechanism of evolution differential reproductive success based on heritable traits?", "Natural selection", r"natural\s+selection"),
        ("What evolutionary process leads to formation of distinct new biological species?", "Speciation", r"speciation"), ("What type of speciation occurs when populations become geographically isolated?", "Allopatric speciation", r"allopatric"),
        ("What evolutionary mechanism causes random fluctuations in allele frequencies in small populations?", "Genetic drift", r"genetic\s+drift"), ("What sharp reduction in population size alters allele frequencies due to chance survival?", "Bottleneck effect", r"bottleneck"),
        ("What loss of genetic variation occurs when a small group colonizes a new habitat?", "Founder effect", r"founder\s+effect"), ("What anatomical structures share common evolutionary ancestry despite different functions?", "Homologous structures", r"homologous"),
        ("What anatomical structures perform similar functions but evolved via convergent evolution?", "Analogous structures", r"analogous"), ("What remnants of functional organs in ancestors serve little or no present purpose?", "Vestigial structures", r"vestigial"),
        ("What biological interaction benefits both interacting species (+/+)?", "Mutualism", r"mutualism"), ("What symbiotic relationship benefits one species while the other is unaffected (+/0)?", "Commensalism", r"commensalism"),
        ("What symbiotic relationship benefits one organism at the expense of host (+/-)?", "Parasitism", r"parasitism"), ("What principle states two species competing for identical niche cannot coexist indefinitely?", "Competitive exclusion", r"competitive\s+exclusion"),
        ("What organisms occupy the first trophic level in an ecosystem by producing organic matter?", "Primary producers", r"(primary\s+producers?|autotrophs?)"), ("What trophic level consists of herbivores that consume primary producers?", "Primary consumers", r"primary\s+consumers?"),
        ("What rule of thumb estimates percentage of energy transferred from one trophic level to next?", "10%", r"(10%|ten\s+percent)"), ("What species exerts disproportionate control over ecosystem structure relative to abundance?", "Keystone species", r"keystone"),
        ("What gradual ecological change follows disturbance that leaves soil intact?", "Secondary succession", r"secondary\s+succession"), ("What ecological succession occurs on newly formed, completely barren land devoid of soil?", "Primary succession", r"primary\s+succession"),
        ("What maximum population size of a species can an environment sustainably support?", "Carrying capacity", r"carrying\s+capacity"), ("What mathematical curve describes population growth that levels off at carrying capacity?", "Logistic growth", r"logistic"),
        ("What biogeochemical cycle circulates carbon through atmosphere, biosphere, and geosphere?", "Carbon cycle", r"carbon\s+cycle"), ("What biological process carried out by soil bacteria converts atmospheric N2 into ammonia?", "Nitrogen fixation", r"nitrogen\s+fixation"),
        ("What major biome is characterized by low temperatures, short growing seasons, and permafrost?", "Tundra", r"tundra"), ("What terrestrial biome has highest biodiversity and year-round high precipitation and temperature?", "Tropical rainforest", r"tropical\s+rainforest"),
        ("What marine ecosystem formed by calcium carbonate skeletons supports ~25% of all marine life?", "Coral reef", r"coral\s+reef"), ("What coastal ecosystem where freshwater rivers meet ocean serves as marine nurseries?", "Estuary", r"estuary"),
        ("What phenomenon causes excessive algal blooms due to agricultural nutrient runoff?", "Eutrophication", r"eutrophication"), ("What progressive buildup of persistent chemical toxins occurs at higher levels of food chain?", "Biomagnification", r"biomagnification"),
    ]
    for idx, (q, tgt, pat) in enumerate(eco_data, 61):
        items.append(_make_item(f"STEM-BIO-{idx:03d}", "Biological & Life Sciences", "Evolution/Ecology", q, tgt, "contains", None, pat))
    return items


def _build_health() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    anat_data = [
        ("What chamber of the human heart pumps oxygenated blood into systemic circulation via aorta?", "Left ventricle", r"left\s+ventricle"), ("What chamber of the human heart receives deoxygenated blood from vena cava?", "Right atrium", r"right\s+atrium"),
        ("What heart valve prevents backflow of blood from left ventricle into left atrium?", "Mitral valve", r"(mitral|bicuspid)\s+valve"), ("What heart valve separates right atrium from right ventricle?", "Tricuspid valve", r"tricuspid"),
        ("What natural pacemaker region of the heart initiates electrical impulses for cardiac contraction?", "Sinoatrial node", r"(sinoatrial|sa)\s+node"), ("What is the longest and strongest bone in the human skeleton?", "Femur", r"femur"),
        ("What is the technical anatomical name for the kneecap bone?", "Patella", r"patella"), ("What is the technical anatomical name for the collarbone?", "Clavicle", r"clavicle"),
        ("What is the technical anatomical name for the shoulder blade?", "Scapula", r"scapula"), ("How many vertebrae comprise the adult cervical spine in humans?", "7", r"\b7\b"),
        ("How many vertebrae comprise the thoracic spine in humans?", "12", r"\b12\b"), ("How many vertebrae comprise the lumbar spine in humans?", "5", r"\b5\b"),
        ("What organ filters blood to produce urine, regulates electrolytes, and controls blood pressure?", "Kidney", r"kidney"), ("What functional filtering unit of the kidney contains the glomerulus and nephron loop?", "Nephron", r"nephron"),
        ("What large abdominal organ metabolizes drugs, synthesizes bile, and stores glycogen?", "Liver", r"\bliver\b"), ("What organ secretes digestive enzymes into duodenum and insulin into bloodstream?", "Pancreas", r"pancreas"),
        ("What endocrine gland at the base of the brain is often termed the master gland?", "Pituitary gland", r"pituitary"), ("What butterfly-shaped endocrine gland in the neck secretes T3, T4, and calcitonin?", "Thyroid", r"thyroid"),
        ("What hormone secreted by beta cells of pancreatic islets lowers blood glucose levels?", "Insulin", r"insulin"), ("What peptide hormone secreted by alpha cells of pancreas raises blood glucose levels?", "Glucagon", r"glucagon"),
        ("What primary glucocorticoid stress hormone is produced by adrenal cortex?", "Cortisol", r"cortisol"), ("What hormone stimulates fight-or-flight response, accelerating heart rate and dilating bronchioles?", "Epinephrine", r"(epinephrine|adrenaline)"),
        ("What primary functional cell of nervous system transmits electrical and chemical nerve impulses?", "Neuron", r"neuron"), ("What insulating fatty sheath wrapped around neuronal axons speeds action potential propagation?", "Myelin sheath", r"myelin"),
        ("What microscopic junction allows communication between two neurons via neurotransmitters?", "Synapse", r"synapse"), ("What lobe of cerebral cortex is responsible for executive function and voluntary motor control?", "Frontal lobe", r"frontal\s+lobe"),
        ("What cerebral cortex lobe processes auditory signals and contains Wernicke's area?", "Temporal lobe", r"temporal\s+lobe"), ("What cerebral cortex lobe houses primary visual cortex for visual processing?", "Occipital lobe", r"occipital\s+lobe"),
        ("What brainstem region regulates vital involuntary autonomic functions like breathing?", "Medulla oblongata", r"medulla"), ("What brain structure coordinates voluntary motor movement, balance, and fine motor skills?", "Cerebellum", r"cerebellum"),
        ("What dome-shaped skeletal muscle is the primary muscle responsible for pulmonary respiration?", "Diaphragm", r"diaphragm"), ("What microscopic air sacs in the lungs are site of gas exchange with pulmonary capillaries?", "Alveoli", r"alveol(i|us)"),
        ("What iron-rich metalloprotein in red blood cells binds and transports oxygen?", "Hemoglobin", r"hemoglobin"), ("What blood component is responsible for initiating hemostasis and forming blood clots?", "Platelets", r"(platelets?|thrombocytes?)"),
        ("What is the largest organ of the human body by surface area and weight?", "Skin", r"skin"),
    ]
    for idx, (q, tgt, pat) in enumerate(anat_data, 1):
        items.append(_make_item(f"STEM-MED-{idx:03d}", "Medical & Clinical Sciences", "Physiology/Anatomy", q, tgt, "contains", None, pat))

    pharm_data = [
        ("What Scottish microbiologist discovered the natural antibiotic Penicillin in 1928?", "Alexander Fleming", r"fleming"), ("In pharmacokinetics, what does the acronym ADME stand for?", "Absorption, Distribution, Metabolism, Excretion", r"absorption.*distribution.*metabolism.*excretion"),
        ("What primary organ in human body is responsible for drug metabolism via cytochrome P450?", "Liver", r"\bliver\b"), ("What class of medications is used to lower cholesterol by inhibiting HMG-CoA reductase?", "Statins", r"statins?"),
        ("What widely used anticoagulant acts as Vitamin K antagonist to prevent thromboembolism?", "Warfarin", r"warfarin"), ("What NSAID inhibits COX enzymes irreversibly by acetylating serine in platelet COX-1?", "Aspirin", r"aspirin"),
        ("What analgesic and antipyretic drug carries risk of hepatotoxicity in acute overdose?", "Acetaminophen", r"(acetaminophen|paracetamol)"), ("What opioid receptor antagonist is administered urgently to reverse opioid overdoses?", "Naloxone", r"(naloxone|narcan)"),
        ("What class of antimicrobial agents inhibits bacterial cell wall synthesis (e.g. penicillins)?", "Beta-lactams", r"beta[- ]lactams?"), ("What class of antibiotics inhibits bacterial protein synthesis by binding to 30S ribosome?", "Tetracyclines", r"tetracyclines?"),
        ("What class of antihypertensive medications lowers blood pressure by blocking ACE?", "ACE inhibitors", r"ace\s+inhibitors?"), ("What class of cardiovascular drugs blocks beta-adrenergic receptors to reduce heart rate?", "Beta-blockers", r"beta[- ]blockers?"),
        ("What cardiac glycoside medication increases myocardial contractility by inhibiting Na+/K+ ATPase?", "Digoxin", r"digoxin"), ("What first-line oral antidiabetic medication belonging to biguanides decreases hepatic glucose?", "Metformin", r"metformin"),
        ("What class of antidepressant medications selectively inhibits reuptake of serotonin?", "SSRIs", r"ssri"), ("What benzodiazepine medication is frequently used as acute anxiolytic and anticonvulsant?", "Diazepam", r"(diazepam|valium)"),
        ("What loop diuretic acts on ascending limb of Henle's loop to treat fluid overload and edema?", "Furosemide", r"(furosemide|lasix)"), ("What class of GI drugs suppresses gastric acid secretion by blocking H+/K+ ATPase pump?", "Proton pump inhibitors", r"(proton\s+pump\s+inhibitors?|ppi)"),
        ("What short-acting beta-2 agonist bronchodilator provides rapid relief in acute asthma?", "Albuterol", r"(albuterol|salbutamol)"), ("What hormone medication is injected IM as first-line treatment for acute anaphylaxis?", "Epinephrine", r"(epinephrine|adrenaline)"),
        ("What antifungal medication binds to ergosterol in fungal cell membranes to form pores?", "Amphotericin B", r"amphotericin"), ("What antiviral nucleoside analog treats herpes simplex virus (HSV) infections?", "Acyclovir", r"acyclovir"),
        ("What parameter measures time required for plasma drug concentration to decrease by 50%?", "Half-life", r"half[- ]life"), ("What ratio between toxic dose and therapeutic dose quantifies relative safety margin?", "Therapeutic index", r"therapeutic\s+index"),
        ("What route of drug administration delivers medication directly into systemic venous circulation?", "Intravenous", r"(intravenous|iv)"),
    ]
    for idx, (q, tgt, pat) in enumerate(pharm_data, 36):
        items.append(_make_item(f"STEM-MED-{idx:03d}", "Medical & Clinical Sciences", "Pharmacology", q, tgt, "contains", None, pat))

    path_data = [
        ("What mosquito genus transmits the Plasmodium parasite responsible for malaria?", "Anopheles", r"anopheles"), ("What bacterium is primary causative pathogen of pulmonary tuberculosis in humans?", "Mycobacterium tuberculosis", r"mycobacterium"),
        ("What infectious disease caused by Vibrio cholerae produces profuse watery diarrhea?", "Cholera", r"cholera"), ("What viral family includes SARS-CoV-2 pathogen causing COVID-19 pandemic?", "Coronaviridae", r"coronavirus|coronaviridae"),
        ("What retrovirus attacks CD4+ T helper lymphocytes progressing to AIDS if untreated?", "HIV", r"hiv"), ("What neurodegenerative disorder is marked by beta-amyloid plaques and tau tangles?", "Alzheimer's disease", r"alzheimer"),
        ("What neurodegenerative disorder is characterized by dopamine deficiency in substantia nigra?", "Parkinson's disease", r"parkinson"), ("What autoimmune disease attacks myelin sheath of neurons in central nervous system?", "Multiple sclerosis", r"multiple\s+sclerosis"),
        ("What autoimmune disorder results from autoimmune destruction of pancreatic beta cells?", "Type 1 diabetes", r"type\s+1\s+diabetes"), ("What chronic metabolic disorder is characterized by peripheral insulin resistance?", "Type 2 diabetes", r"type\s+2\s+diabetes"),
        ("What clinical term describes localized collection of pus formed within tissues from infection?", "Abscess", r"abscess"), ("What five classical cardinal signs characterize acute inflammation?", "Pain, Heat, Redness, Swelling, Loss of function", r"pain|heat|redness|swelling|rubor|calor"),
        ("What term describes spread of malignant tumor cells from primary site to distant organs?", "Metastasis", r"metastasis"), ("What abnormal tissue growth characterized by autonomous cellular proliferation defines cancer?", "Neoplasm", r"(neoplasm|tumor|malignancy)"),
        ("What condition of low oxygen delivery to peripheral tissues causes cyanosis and injury?", "Hypoxia", r"hypoxia"), ("What clinical condition is defined as deficiency of red blood cells or hemoglobin?", "Anemia", r"anemia"),
        ("What genetic blood disorder produces abnormal crescent-shaped erythrocytes?", "Sickle cell disease", r"sickle\s+cell"), ("What term describes acute localized tissue necrosis caused by interrupted arterial blood supply?", "Infarction", r"infarction"),
        ("What life-threatening organ dysfunction is triggered by dysregulated host response to infection?", "Sepsis", r"sepsis"), ("What medical term refers to high arterial blood pressure exceeding standard clinical guidelines?", "Hypertension", r"hypertension"),
    ]
    for idx, (q, tgt, pat) in enumerate(path_data, 61):
        items.append(_make_item(f"STEM-MED-{idx:03d}", "Medical & Clinical Sciences", "Pathology", q, tgt, "contains", None, pat))

    safe_data = [
        ("What is recommended chest compression rate per minute for adult CPR per AHA guidelines?", "100-120", r"100[- ]120"), ("What is recommended ratio of chest compressions to rescue breaths in adult lone-rescuer CPR?", "30:2", r"30:2"),
        ("What medical device analyzes heart rhythm and delivers shock to treat ventricular fibrillation?", "AED", r"(aed|automated\s+external\s+defibrillator)"), ("What standard procedure is single most effective action to prevent nosocomial infections?", "Hand hygiene", r"hand\s+(hygiene|washing)"),
        ("What color container is standardized for disposing of sharp medical needles and blades?", "Red", r"red"), ("What PPE ensemble protects healthcare workers from airborne pathogens like tuberculosis?", "N95 respirator", r"n95"),
        ("In medical ethics, what principle asserts patient right to self-determination and consent?", "Autonomy", r"autonomy"), ("In medical ethics, what principle obligates clinicians to act in best interest of patient?", "Beneficence", r"beneficence"),
        ("In medical ethics, what principle is captured by Latin maxim 'Primum non nocere'?", "Non-maleficence", r"non[- ]maleficence"), ("What process categorizes patients by clinical urgency during mass casualty incidents?", "Triage", r"triage"),
    ]
    for idx, (q, tgt, pat) in enumerate(safe_data, 81):
        items.append(_make_item(f"STEM-MED-{idx:03d}", "Medical & Clinical Sciences", "Clinical Safety", q, tgt, "contains", None, pat))

    return items


def build_pillar1_stem_items() -> list[dict[str, Any]]:
    """Build all 600 items for Pillar 1: STEM & Computing."""
    math_items = _build_math()
    assert len(math_items) == 120, f"Expected 120 math items, got {len(math_items)}"
    cs_items = _build_cs()
    assert len(cs_items) == 120, f"Expected 120 CS items, got {len(cs_items)}"
    phys_items = _build_phys()
    assert len(phys_items) == 100, f"Expected 100 physics items, got {len(phys_items)}"
    chem_items = _build_chem()
    assert len(chem_items) == 80, f"Expected 80 chem items, got {len(chem_items)}"
    bio_items = _build_bio()
    assert len(bio_items) == 90, f"Expected 90 bio items, got {len(bio_items)}"
    health_items = _build_health()
    assert len(health_items) == 90, f"Expected 90 health items, got {len(health_items)}"

    all_stem = math_items + cs_items + phys_items + chem_items + bio_items + health_items
    assert len(all_stem) == 600, f"Expected 600 STEM items, got {len(all_stem)}"
    return all_stem
