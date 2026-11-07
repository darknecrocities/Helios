"""General Science, Physics, Chemistry, and Biology Dataset.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

SCIENCE_PAIRS: list[tuple[str, str]] = [
    (
        "What is Newton's First Law of Motion?",
        "Newton's First Law (Law of Inertia) states that an object at rest will stay at rest, and an object in uniform motion will remain in uniform motion along a straight line unless acted upon by a net external force.",
    ),
    (
        "What is Newton's Second Law of Motion?",
        "Newton's Second Law states that force equals mass times acceleration (F = m * a). The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass.",
    ),
    (
        "What is Newton's Third Law of Motion?",
        "Newton's Third Law states that for every action, there is an equal and opposite reaction (F_A_on_B = -F_B_on_A).",
    ),
    (
        "What is the speed of light in a vacuum?",
        "The speed of light in a vacuum (c) is precisely 299,792,458 meters per second (approximately 3.00 * 10^8 m/s or 186,282 miles per second).",
    ),
    (
        "What is the acceleration due to gravity on Earth's surface?",
        "Standard acceleration due to gravity on Earth (g) is approximately 9.80665 m/s² (commonly rounded to 9.8 m/s²).",
    ),
    (
        "What is the chemical formula for water and what bonds hold it together?",
        "The chemical formula for water is H2O. Two hydrogen atoms are covalently bonded to a single oxygen atom in a bent molecular geometry (~104.5 degrees) with polar covalent bonds.",
    ),
    (
        "What is the powerhouse of the cell?",
        "The mitochondrion (plural: mitochondria) is often termed the powerhouse of the cell because it generates most of the chemical energy needed by the cell through aerobic cellular respiration, producing adenosine triphosphate (ATP).",
    ),
    (
        "What is DNA and what are its four base pairs?",
        "Deoxyribonucleic acid (DNA) is the molecule carrying hereditary genetic instructions in living organisms. It consists of a double helix composed of four nitrogenous bases: Adenine (A), Thymine (T), Cytosine (C), and Guanine (G), where A pairs with T and C pairs with G.",
    ),
    (
        "What is photosynthesis and what is its chemical equation?",
        "Photosynthesis is the biochemical process by which green plants, algae, and cyanobacteria convert light energy into chemical energy stored in glucose. The overall balanced equation is: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2.",
    ),
    (
        "What is the pH of pure water and what does pH measure?",
        "The pH of pure water at 25 degrees Celsius is 7.0, representing neutrality. pH is the negative logarithm (base 10) of hydrogen ion concentration: pH = -log10[H+]. Values below 7 are acidic, while values above 7 are basic (alkaline).",
    ),
    (
        "What is the First Law of Thermodynamics?",
        "The First Law of Thermodynamics (Law of Conservation of Energy) states that energy cannot be created or destroyed, only transformed from one form to another (ΔU = Q - W).",
    ),
    (
        "What is the Second Law of Thermodynamics?",
        "The Second Law of Thermodynamics states that the total entropy (disorder) of an isolated system always increases over time in any spontaneous process (ΔS_total > 0).",
    ),
    (
        "What is an atom composed of?",
        "An atom is composed of a dense central nucleus containing positively charged protons and neutral neutrons, surrounded by a cloud of negatively charged electrons in quantum orbitals.",
    ),
    (
        "How many planets are in our Solar System and what are they in order?",
        "There are 8 recognized planets in order of distance from the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
    ),
    (
        "What is the Doppler Effect?",
        "The Doppler Effect is the observed change in the frequency or wavelength of a wave in relation to an observer moving relative to the wave source (e.g. rising pitch as an ambulance approaches and falling pitch as it moves away).",
    ),
]


def get_science_pairs() -> list[tuple[str, str]]:
    """Return general science and engineering QA pairs."""
    return list(SCIENCE_PAIRS)
