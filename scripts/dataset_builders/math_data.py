"""Deterministic and Step-by-Step Mathematical Problem Solving Dataset.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

MATH_PAIRS: list[tuple[str, str]] = [
    ("What is 1+1?", "1 + 1 = 2."),
    ("1+1?", "1 + 1 = 2."),
    ("What is 2+2?", "2 + 2 = 4."),
    ("What is 5 + 7?", "5 + 7 = 12."),
    ("What is 15 + 27?", "15 + 27 = 42."),
    ("What is 99 + 1?", "99 + 1 = 100."),
    ("What is 100 - 37?", "100 - 37 = 63."),
    ("What is 25 * 4?", "25 * 4 = 100."),
    ("What is 12 * 12?", "12 * 12 = 144."),
    ("What is 144 / 12?", "144 / 12 = 12."),
    ("What is 1000 / 8?", "1000 / 8 = 125."),
    ("What is 7 * 8?", "7 * 8 = 56."),
    ("What is 9 * 6?", "9 * 6 = 54."),
    ("What is 2 to the power of 8?", "2^8 = 256."),
    ("What is the square root of 64?", "The square root of 64 is 8 (since 8 * 8 = 64)."),
    ("What is the square root of 144?", "The square root of 144 is 12 (since 12 * 12 = 144)."),
    (
        "If a train travels at 60 km/h for 2.5 hours, how far does it travel?",
        "Distance = Speed * Time. Here, Distance = 60 km/h * 2.5 h = 150 kilometers.",
    ),
    (
        "A store offers a 20% discount on a $50 shirt. What is the final price?",
        "Step 1: Calculate the discount amount: 20% of $50 = 0.20 * 50 = $10.\nStep 2: Subtract the discount from the original price: $50 - $10 = $40. The final price is $40.",
    ),
    (
        "What is 3/4 converted to a percentage?",
        "To convert a fraction to a percentage, divide the numerator by the denominator and multiply by 100: (3 / 4) * 100 = 0.75 * 100 = 75%.",
    ),
    (
        "Solve for x: 3x + 9 = 24",
        "Step 1: Subtract 9 from both sides: 3x = 24 - 9 => 3x = 15.\nStep 2: Divide both sides by 3: x = 15 / 3 => x = 5.",
    ),
    (
        "Solve for x: 2x - 5 = 15",
        "Step 1: Add 5 to both sides: 2x = 15 + 5 => 2x = 20.\nStep 2: Divide both sides by 2: x = 20 / 2 => x = 10.",
    ),
    (
        "What is the area of a circle with a radius of 7 meters?",
        "The formula for the area of a circle is A = π * r^2. Using π ≈ 22/7, Area ≈ (22/7) * 7^2 = 22 * 7 = 154 square meters.",
    ),
    (
        "What is the perimeter of a rectangle with length 12 cm and width 5 cm?",
        "Perimeter = 2 * (length + width) = 2 * (12 + 5) = 2 * 17 = 34 cm.",
    ),
    (
        "What is the Pythagorean theorem?",
        "In a right-angled triangle, the square of the hypotenuse (c) is equal to the sum of the squares of the other two sides (a and b): a^2 + b^2 = c^2.",
    ),
    (
        "If a right triangle has legs of length 3 and 4, what is the length of the hypotenuse?",
        "By the Pythagorean theorem: c^2 = 3^2 + 4^2 = 9 + 16 = 25. Therefore, c = √25 = 5.",
    ),
    (
        "What is 15% of 200?",
        "15% of 200 = 0.15 * 200 = 30.",
    ),
    (
        "Convert 0.375 into a simplified fraction.",
        "0.375 = 375 / 1000. Dividing both numerator and denominator by their greatest common divisor (125), we get 3 / 8.",
    ),
    (
        "If 5 apples cost $10, how much do 8 apples cost?",
        "Unit price = $10 / 5 = $2 per apple. For 8 apples: 8 * $2 = $16.",
    ),
    (
        "What is the average of 10, 20, 30, and 40?",
        "Average = (10 + 20 + 30 + 40) / 4 = 100 / 4 = 25.",
    ),
    (
        "What is 10 factorial (10!)?",
        "10! = 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 = 3,628,800.",
    ),
]


def get_math_pairs() -> list[tuple[str, str]]:
    """Return math QA pairs with dynamic variations."""
    pairs = list(MATH_PAIRS)
    # Generate systematic arithmetic pairs to ensure dense coverage
    for i in range(1, 13):
        for j in range(1, 13):
            pairs.append((f"What is {i} + {j}?", f"{i} + {j} = {i + j}."))
            pairs.append((f"What is {i} * {j}?", f"{i} * {j} = {i * j}."))
            prod = i * j
            pairs.append((f"What is {prod} / {i}?", f"{prod} / {i} = {j}."))
            if i >= j:
                pairs.append((f"What is {i} - {j}?", f"{i} - {j} = {i - j}."))
    return pairs
