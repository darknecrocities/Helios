"""World History & Historical Timelines Seed Dataset for Helios Models.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from typing import Any

HISTORY_WORLD_RECORDS: list[tuple[str, str]] = [
    (
        "What were the key causes and outcomes of the Peloponnesian War (431–404 BCE)?",
        "The Peloponnesian War was fought between the Delian League led by Athens and the Peloponnesian League led by Sparta. As Thucydides famously analyzed, the fundamental cause was the growth of Athenian imperial power and the fear this caused in Sparta. Sparta won after Lysander destroyed the Athenian fleet at Aegospotami in 405 BCE, ending the Golden Age of Athens and leaving Greek city-states vulnerable to Macedonian conquest under Philip II.",
    ),
    (
        "How did the Han Dynasty shape Chinese imperial bureaucracy and culture?",
        "The Han Dynasty (202 BCE – 220 CE), founded by Liu Bang (Emperor Gaozu), established Confucianism as the official state ideology, created the Imperial Academy for civil service examination training, expanded the Silk Road under Emperor Wu, invented paper (attributed to Cai Lun), and consolidated four centuries of cultural and administrative unity.",
    ),
    (
        "What was the significance of the Byzantine Empire and the preservation of Roman Law?",
        "The Byzantine Empire (Eastern Roman Empire), centered in Constantinople, survived the collapse of the Western Empire by a millennium. Under Emperor Justinian I (reigned 527–565 CE), it codified the Corpus Juris Civilis (Justinian Code), which became the foundation of modern civil law across Europe, while serving as a cultural buffer between Europe and the Near East.",
    ),
    (
        "What was the Islamic Golden Age and its major intellectual contributions (8th–14th centuries)?",
        "Centered in Baghdad at the House of Wisdom (Bayt al-Hikma), scholars synthesized and advanced knowledge across sciences: Al-Khwarizmi developed algebra (Kitab al-Jabr wa'l-Muqabala); Ibn Sina (Avicenna) wrote the Canon of Medicine; Ibn al-Haytham founded experimental optics; and Al-Razi pioneered clinical medical diagnosis.",
    ),
    (
        "How did the Mongol Empire under Genghis Khan and his successors transform Eurasia?",
        "Founded by Temüjin (Genghis Khan) in 1206, the Mongol Empire became the largest contiguous land empire in history. Through the Pax Mongolica, it secured transcontinental trade routes along the Silk Road, facilitated postal relay systems (Yam), promoted religious tolerance, and enabled commercial and technological exchanges across Asia and Europe.",
    ),
    (
        "What was the Treaty of Westphalia (1648) and why is it considered the origin of modern international relations?",
        "The Peace of Westphalia (1648) concluded the Thirty Years' War in the Holy Roman Empire and the Eighty Years' War between Spain and the Dutch Republic. It established the doctrine of Westphalian sovereignty: the principle of international law that each sovereign state has exclusive authority over its domestic affairs and religion (cuius regio, eius religio), creating the modern nation-state system.",
    ),
    (
        "What were the principal causes and consequences of World War I (1914–1918)?",
        "World War I was triggered by the assassination of Archduke Franz Ferdinand of Austria in Sarajevo, but its root causes lay in the MAIN acronym: Militarism, Alliances (Triple Entente vs. Triple Alliance), Imperialism, and Nationalism. The war caused over 20 million deaths, dissolved four empires (Russian, German, Austro-Hungarian, and Ottoman), led to the Russian Bolshevik Revolution, and established the Treaty of Versailles.",
    ),
    (
        "What was the Bretton Woods Conference (1944) and what institutions did it establish?",
        "Held in Bretton Woods, New Hampshire in July 1944, delegates from 44 Allied nations designed the post-WWII international monetary order. It established an adjustable pegged exchange rate system linked to the US dollar (which was convertible to gold at $35/oz) and created two foundational institutions: the International Monetary Fund (IMF) and the International Bank for Reconstruction and Development (now part of the World Bank).",
    ),
    (
        "What drove Decolonization in Asia and Africa following World War II?",
        "Post-1945 decolonization was accelerated by the economic exhaustion of European colonial powers (Britain, France, the Netherlands, Belgium), the rise of organized nationalist movements, the Atlantic Charter and UN Charter's affirmation of self-determination, and Cold War rivalry where the US and USSR competed for influence among newly sovereign nations.",
    ),
    (
        "How did the Cold War's policy of Mutually Assured Destruction (MAD) influence geopolitical stability?",
        "Mutually Assured Destruction (MAD) is a military doctrine where a full-scale nuclear exchange between superpowers would result in the complete annihilation of both attacker and defender. During the Cold War (1947–1991), MAD prevented direct conventional military conflict between the US and the Soviet Union, shifting confrontations to proxy wars, space exploration, and arms control treaties (SALT, START).",
    ),
]


def build_history_world_records() -> list[dict[str, Any]]:
    """Return ChatML records for World History."""
    records = []
    for prompt, response in HISTORY_WORLD_RECORDS:
        records.append({
            "domain": "world_history_and_timelines",
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
