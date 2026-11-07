"""Computer Science, Systems Architecture, and Network Protocols Dataset.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

SYSTEMS_PAIRS: list[tuple[str, str]] = [
    (
        "What is the difference between HTTP/1.1 and HTTP/2?",
        "HTTP/2 introduces binary framing, multiplexing over a single TCP connection, HPACK header compression, and server push. Multiplexing eliminates the head-of-line blocking problem at the application layer that affected HTTP/1.1 pipelines.",
    ),
    (
        "What is head-of-line blocking in HTTP/1.1?",
        "In HTTP/1.1 pipelining, requests must be processed and responses delivered in the exact order they were sent. If an early request experiences latency or processing delay, all subsequent queued responses are blocked behind it.",
    ),
    (
        "What is the difference between TCP and UDP?",
        "TCP (Transmission Control Protocol) is connection-oriented, reliable, guarantees in-order delivery via acknowledgments and retransmissions, and enforces flow/congestion control. UDP (User Datagram Protocol) is connectionless, lightweight, has minimal header overhead, and offers low-latency best-effort delivery without delivery guarantees.",
    ),
    (
        "What are the ACID properties in database systems?",
        "ACID stands for:\n- Atomicity: All operations in a transaction succeed, or all are rolled back.\n- Consistency: Transactions transition the database from one valid state to another satisfying all schema constraints.\n- Isolation: Concurrent transactions execute without interfering with one another.\n- Durability: Once a transaction commits, its effects persist permanently even through power failure or system crashes.",
    ),
    (
        "What is Write-Ahead Logging (WAL)?",
        "Write-Ahead Logging is a database durability technique where all modifications are sequentially appended to a dedicated append-only log file before changes are applied to the database pages on disk. This enables crash recovery and concurrent read/write transactions.",
    ),
    (
        "What is the time complexity of binary search?",
        "Binary search has a time complexity of O(log n) because it divides the search space in half with each iteration, assuming the underlying array is already sorted.",
    ),
    (
        "What is the difference between a Stack and a Queue?",
        "A Stack follows Last-In, First-Out (LIFO) semantics where elements are pushed and popped from the top. A Queue follows First-In, First-Out (FIFO) semantics where elements are enqueued at the rear and dequeued from the front.",
    ),
    (
        "What is a Hash Table and how does it achieve O(1) average lookup?",
        "A Hash Table maps keys to bucket indices in an array using a hash function. By computing the key's hash in O(1) time, lookups, insertions, and deletions take O(1) average time, with collisions resolved via chaining or open addressing.",
    ),
    (
        "What is the difference between a Process and a Thread?",
        "A Process is an independent executing program with its own dedicated virtual address space, file descriptors, and security context. A Thread is a lightweight unit of execution within a process that shares memory and resources with other threads of that process.",
    ),
    (
        "What is a deadlock and what four Coffman conditions are required for it?",
        "A deadlock occurs when a set of concurrent processes are permanently blocked because each holds a resource while waiting for another resource held by another process. The four conditions are: 1) Mutual Exclusion, 2) Hold and Wait, 3) No Preemption, and 4) Circular Wait.",
    ),
]


def get_systems_pairs() -> list[tuple[str, str]]:
    """Return systems architecture and computer science pairs."""
    return list(SYSTEMS_PAIRS)
