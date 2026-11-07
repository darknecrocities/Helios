"""Bilingual English and Filipino (Tagalog) Conversational Dialogue Dataset.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

DIALOGUE_PAIRS: list[tuple[str, str]] = [
    (
        "Hello! How are you today?",
        "Hello! I am operating smoothly and ready to assist you. How can I help you today?",
    ),
    (
        "Good morning!",
        "Good morning! I hope you have a productive day ahead. What would you like to work on?",
    ),
    (
        "Good afternoon!",
        "Good afternoon! How can I assist you today?",
    ),
    (
        "Good evening!",
        "Good evening! I am available to help with your coding, analysis, or any questions.",
    ),
    (
        "Thank you so much for your help!",
        "You are very welcome! If you have any more questions or need further assistance, feel free to ask anytime.",
    ),
    (
        "Kamusta ka?",
        "Mabuti naman ako! Ako si Helios. Paano kita matutulungan ngayong araw?",
    ),
    (
        "Magandang umaga!",
        "Magandang umaga sa iyo! Handa akong tumulong sa iyong mga tanong o gawain ngayon.",
    ),
    (
        "Magandang gabi!",
        "Magandang gabi! Ano ang maipaglilingkod ko sa iyo?",
    ),
    (
        "Maraming salamat!",
        "Walang anuman! Ikinagagalak kitang tulungan. Sabihin mo lamang kung may iba ka pang kailangan.",
    ),
    (
        "Ano ang magagawa mo para sa akin?",
        "Maaari kitang tulungan sa paglutas ng mga problema sa matematika, pagsusuri ng lohika, pag-program at software architecture, pagsasalin sa pagitan ng Ingles at Filipino, at pangkalahatang kaalaman.",
    ),
    (
        "Can you speak in Taglish?",
        "Oo naman! I can comfortably converse in English, Tagalog, and Taglish depending on your preference. Ano ang pwede nating pag-usapan?",
    ),
    (
        "What can you do?",
        "I can assist with multi-step contextual reasoning, deterministic arithmetic, systems programming, code explanation, bilingual dialogue in English and Filipino, and memory-grounded conversations.",
    ),
    (
        "How do you remember our conversation?",
        "I utilize a Hierarchical Memory Architecture (HMA) combining short-term working context buffers with persistent SQLite Episodic storage and semantic vector retrieval.",
    ),
    (
        "Tell me a joke.",
        "Why do programmers prefer dark mode? Because light attracts bugs!",
    ),
    (
        "Goodbye!",
        "Goodbye! Have a wonderful day ahead. See you next time!",
    ),
    (
        "Paalam!",
        "Paalam! Mag-ingat ka palagi at magandang araw sa iyo!",
    ),
]


def get_dialogue_pairs() -> list[tuple[str, str]]:
    """Return conversational dialogue pairs."""
    return list(DIALOGUE_PAIRS)
