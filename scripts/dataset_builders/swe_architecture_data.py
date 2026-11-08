"""Software Engineering & Architecture Seed Dataset for Helios Models.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from typing import Any

SWE_ARCH_RECORDS: list[tuple[str, str]] = [
    (
        "Explain the CAP Theorem and PACELC theorem in distributed systems.",
        "The CAP Theorem states that a distributed data store can simultaneously provide at most two of three guarantees: Consistency (every read receives the most recent write or an error), Availability (every non-failing node returns a response), and Partition Tolerance (the system continues to operate despite network partitions). Since network partitions are inevitable in physical networks, distributed databases must choose between CP (e.g., Google Spanner, HBase) and AP (e.g., Apache Cassandra, DynamoDB).\n\nThe PACELC theorem extends CAP: in case of Partition (P), choose between Availability (A) and Consistency (C); Else (E), choose between Latency (L) and Consistency (C). For example, MongoDB default is PC/EC (prefers consistency under partition, prefers consistency over latency under normal operations).",
    ),
    (
        "What is the difference between B-Trees and Log-Structured Merge (LSM) Trees in database storage engines?",
        "B-Trees (used in PostgreSQL, MySQL InnoDB, SQLite) are balanced tree structures organized into fixed-size disk blocks/pages. Updates perform in-place random writes, making B-Trees highly efficient for read-heavy workloads and point lookups (O(log N)), but susceptible to write amplification from random disk I/O.\n\nLSM Trees (used in RocksDB, LevelDB, Cassandra, ClickHouse) buffer writes in memory (MemTable) and append to an append-only Write-Ahead Log (WAL). When MemTable fills, it flushes sequentially to immutable disk SSTables. Compaction merges SSTables in the background. LSM trees excel at write-heavy workloads with near-zero write amplification on ingest, at the expense of read amplification requiring Bloom filters.",
    ),
    (
        "How does the Raft consensus algorithm achieve distributed agreement?",
        "Raft achieves consensus by decomposing the problem into three subproblems: Leader Election, Log Replication, and Safety. Nodes exist in one of three states: Follower, Candidate, or Leader. If followers don't receive heartbeats within a randomized election timeout, they transition to Candidate and request votes. A candidate receiving a majority of votes becomes Leader. The leader receives client requests, appends entries to its log, and broadcasts AppendEntries RPCs. Once an entry is safely replicated to a majority (quorum) of nodes, the leader commits it and applies it to its state machine.",
    ),
    (
        "What is Eventual Consistency and how do Conflict-Free Replicated Data Types (CRDTs) guarantee convergence?",
        "Eventual consistency guarantees that given no new updates, all replicas will eventually converge to identical values. CRDTs are data structures that can be replicated across multiple nodes and concurrently updated without central synchronization. They guarantee strong eventual consistency via mathematical properties (semilattices): operations are Commutative, Associative, and Idempotent. State-based CRDTs (CvRDTs) merge full states using a join semi-lattice, while Operation-based CRDTs (CmRDTs) broadcast commutative operations across reliable delivery channels.",
    ),
    (
        "What are the SOLID design principles in software engineering?",
        "SOLID represents five foundational object-oriented design principles:\n1. Single Responsibility Principle (SRP): A class or module should have one, and only one, reason to change.\n2. Open/Closed Principle (OCP): Software entities should be open for extension, but closed for modification.\n3. Liskov Substitution Principle (LSP): Subtypes must be substitutable for their base types without altering program correctness.\n4. Interface Segregation Principle (ISP): Clients should not be forced to depend on interfaces they do not use.\n5. Dependency Inversion Principle (DIP): High-level modules should not depend on low-level modules; both should depend on abstractions.",
    ),
    (
        "Explain Microservices versus Modular Monolith architectures.",
        "A Monolith packages the entire application into a single deployable unit sharing memory and database connections. A Modular Monolith enforces strict domain boundaries, clean module interfaces, and decoupled schemas within that single unit, avoiding distributed network latency and deployment complexity.\n\nMicroservices decompose an application into independently deployable, loosely coupled services communicating over network protocols (gRPC, REST, Kafka). Microservices enable independent team scaling, polyglot technology stacks, and fault isolation, but introduce operational complexity: distributed tracing, network latency, distributed transactions (Saga pattern), and eventual consistency.",
    ),
    (
        "What is the Twelve-Factor App methodology for cloud-native software?",
        "The Twelve-Factor App methodology outlines principles for building modern SaaS applications:\n1. Codebase (one codebase tracked in revision control, many deploys)\n2. Dependencies (explicitly declared and isolated)\n3. Config (stored in environment variables, never hardcoded)\n4. Backing Services (treated as attached resources via URLs/credentials)\n5. Build, Release, Run (strictly separated deployment stages)\n6. Processes (executed as stateless, share-nothing processes)\n7. Port Binding (export services via port binding)\n8. Concurrency (scale out via the process model)\n9. Disposability (fast startup and graceful shutdown via SIGTERM)\n10. Dev/Prod Parity (keep development, staging, and production as similar as possible)\n11. Logs (treated as unbuffered event streams emitted to stdout)\n12. Admin Processes (run administrative/migration tasks as one-off processes).",
    ),
]


def build_swe_architecture_records() -> list[dict[str, Any]]:
    """Return ChatML records for Software Engineering & Architecture."""
    records = []
    for prompt, response in SWE_ARCH_RECORDS:
        records.append({
            "domain": "software_engineering_and_architecture",
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
        })
    return records
