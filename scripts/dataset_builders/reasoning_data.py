"""Contextual, Deductive, and Causal Reasoning Dataset.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

REASONING_PAIRS: list[tuple[str, str]] = [
    (
        "All humans are mortal. Socrates is human. What follows?",
        "By classical deductive logic (syllogism Modus Ponens), it definitively follows that Socrates is mortal.",
    ),
    (
        "If it rains, the grass gets wet. The grass is not wet. Did it rain?",
        "By Modus Tollens, if the conditional statement 'If it rains, then the grass is wet' holds, the negation of the consequent ('the grass is not wet') implies the negation of the antecedent. Therefore, it did not rain.",
    ),
    (
        "Anna is taller than Brian, and Brian is taller than Clara. Who is the shortest?",
        "By the transitive property of comparison: Anna > Brian > Clara. Therefore, Clara is the shortest.",
    ),
    (
        "A box contains 3 red balls and 2 green balls. If you draw one ball without looking, what is the probability of picking a green ball?",
        "Total balls = 3 + 2 = 5. The number of green balls is 2. The probability is 2 / 5 = 0.40 or 40%.",
    ),
    (
        "If a doctor gives you 3 pills and tells you to take one every half hour, how long will they last?",
        "They will last 1 hour. You take the 1st pill immediately (at 0 minutes), the 2nd pill at 30 minutes, and the 3rd pill at 60 minutes (1 hour).",
    ),
    (
        "Find the odd one out: Mercury, Venus, Mars, Sirius.",
        "Sirius is the odd one out. Mercury, Venus, and Mars are planets within our Solar System, whereas Sirius is a star (the brightest star in the night sky, located in Canis Major).",
    ),
    (
        "Identify the logical fallacy: 'Nobody has proven that ghosts don't exist, therefore they must exist.'",
        "This is an Appeal to Ignorance (Argumentum ad Ignorantiam) fallacy, which assumes a proposition is true simply because it has not yet been proven false.",
    ),
    (
        "Identify the logical fallacy: 'You cannot trust Senator Smith's tax reform proposal because he is wealthy.'",
        "This is an Ad Hominem fallacy, attacking the person making the argument rather than evaluating the merits of the proposal itself.",
    ),
    (
        "If a rooster lays an egg on top of a slanted barn roof, which way will it roll?",
        "Roosters do not lay eggs; hens do. Therefore, no egg will roll.",
    ),
    (
        "Some cats are black. All black animals absorb more heat in direct sunlight. Can we conclude that all cats absorb more heat in direct sunlight?",
        "No. Only the subcategory of cats that are black are guaranteed by the premise to absorb more heat in sunlight. We cannot generalize this trait to all cats.",
    ),
    (
        "If all squares are rectangles, and all rectangles are quadrilaterals, are all squares quadrilaterals?",
        "Yes. By transitivity of subsets: Squares ⊆ Rectangles ⊆ Quadrilaterals. Therefore, every square is a quadrilateral.",
    ),
    (
        "A car travels north for 10 km, then east for 10 km. What direction is it from its starting point?",
        "The car is northeast of its starting point.",
    ),
    (
        "If yesterday was Monday, what day will it be 3 days after tomorrow?",
        "If yesterday was Monday, today is Tuesday. Tomorrow will be Wednesday. Three days after Wednesday (Thursday, Friday, Saturday) will be Saturday.",
    ),
    (
        "What happens to water pressure as depth increases in an ocean?",
        "Water pressure increases linearly with depth due to hydrostatic pressure (P = ρ * g * h), where ρ is water density, g is gravitational acceleration, and h is depth.",
    ),
    (
        "Why does ice float on liquid water?",
        "Ice floats because water expands upon freezing, forming an open hexagonal crystal lattice held by hydrogen bonds. This makes solid ice less dense (approx. 0.92 g/cm³) than liquid water (approx. 1.00 g/cm³).",
    ),
]


def get_reasoning_pairs() -> list[tuple[str, str]]:
    """Return reasoning and logic QA pairs."""
    return list(REASONING_PAIRS)
