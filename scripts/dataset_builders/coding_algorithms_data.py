"""Coding & Algorithms Seed Dataset for Helios Models.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from typing import Any

CODING_ALGO_RECORDS: list[tuple[str, str]] = [
    (
        "Implement LRU (Least Recently Used) Cache in Python with O(1) get and put time complexity.",
        """```python
class DNode:
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev: DNode | None = None
        self.next: DNode | None = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[int, DNode] = {}
        self.head = DNode()
        self.tail = DNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: DNode) -> None:
        p, n = node.prev, node.next
        if p and n:
            p.next = n
            n.prev = p

    def _add_to_front(self, node: DNode) -> None:
        node.prev = self.head
        node.next = self.head.next
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                lru = self.tail.prev
                if lru and lru != self.head:
                    self._remove(lru)
                    del self.cache[lru.key]
            new_node = DNode(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)
```
**Complexity**:
- Time Complexity: O(1) for both `get` and `put` using a doubly-linked list combined with a hash map.
- Space Complexity: O(capacity) to store elements.""",
    ),
    (
        "Explain and implement Dijkstra's Algorithm for Single-Source Shortest Paths in Python.",
        """```python
import heapq

def dijkstra(graph: dict[str, list[tuple[str, int]]], start: str) -> dict[str, float]:
    \"\"\"Compute shortest path distances from start node to all reachable nodes.
    Graph is an adjacency list: {u: [(v, weight), ...]} with non-negative edge weights.
    \"\"\"
    distances: dict[str, float] = {node: float('inf') for node in graph}
    distances[start] = 0.0
    pq: list[tuple[float, str]] = [(0.0, start)]

    while pq:
        curr_dist, u = heapq.heappop(pq)
        if curr_dist > distances[u]:
            continue

        for v, weight in graph.get(u, []):
            dist = curr_dist + weight
            if dist < distances[v]:
                distances[v] = dist
                heapq.heappush(pq, (dist, v))

    return distances
```
**Complexity**:
- Time Complexity: O((V + E) log V) with a binary min-heap priority queue.
- Space Complexity: O(V) for distances map and priority queue. Non-negative weights are required.""",
    ),
    (
        "How does the Quicksort algorithm work and how do you guarantee O(N log N) average time with dual-pivot / randomized pivot?",
        """Quicksort is a divide-and-conquer sorting algorithm. It selects a pivot element, partitions the array such that all elements smaller than the pivot precede it and all greater elements succeed it, then recursively sorts the subarrays.

Standard fixed pivots (e.g. first or last element) degrade to O(N^2) time on already sorted or reverse-sorted input. To eliminate this:
1. **Randomized Quicksort**: Uniformly chooses a random index as pivot, achieving expected O(N log N) time regardless of input order.
2. **Median-of-Three**: Chooses median of `arr[low]`, `arr[mid]`, `arr[high]`.
3. **Dual-Pivot Quicksort** (used in Java/V8): Chooses two pivots P1 and P2 (P1 <= P2) partitioning into three regions: `< P1`, `P1 <= x <= P2`, and `> P2`. Reduces memory traffic and cache misses, running faster on modern CPU architectures.""",
    ),
    (
        "Explain Red-Black Tree invariants and why its height is bounded by 2 * log2(N + 1).",
        """A Red-Black Tree is a self-balancing binary search tree satisfying five properties:
1. Every node is colored either RED or BLACK.
2. The root node is BLACK.
3. Every leaf (NIL sentinel node) is BLACK.
4. If a node is RED, both of its children must be BLACK (no consecutive red nodes).
5. For each node, all simple paths from the node to descendant leaf nodes contain the same number of black nodes (uniform black-height bh(x)).

**Height Bound Proof**:
A subtree rooted at node `x` contains at least `2^(bh(x)) - 1` internal nodes. By Property 4, at most half the nodes on any path are red, so `bh(root) >= h/2`. Thus `N >= 2^(h/2) - 1`, which implies `h <= 2 * log2(N + 1)`. Therefore, search, insertion, and deletion are guaranteed O(log N) in the worst case with at most 2 rotations on insert and 3 on delete.""",
    ),
]


def build_coding_algorithms_records() -> list[dict[str, Any]]:
    """Return ChatML records for Coding & Algorithms."""
    records = []
    for prompt, response in CODING_ALGO_RECORDS:
        records.append({
            "domain": "coding_and_algorithms",
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
