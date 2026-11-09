"""Patch Helios Ollama Modelfiles with corrected creator role and <thought> few-shot dialogue.

Idempotent: strips any previous MESSAGE block before re-appending.
"""

from __future__ import annotations

import re
from pathlib import Path

OLD_ROLE = re.compile(
    r"Lead Architect and AI Engineer in the Philippines, Creator of Domodomo, "
    r"Top 1 on AppBuilders, and Former Chapter Lead of Google Developer Groups \(GDG\)"
)
NEW_ROLE = (
    "an AI & Systems Engineer, Machine Learning Researcher, and Software Developer "
    "in the Philippines (not an entrepreneur)"
)
HEADER_OLD = "Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines"
HEADER_NEW = "Arron Kian Parejas, AI & Systems Engineer in the Philippines"
MARKER = "# [COT FEW-SHOT START]"

FEW_SHOT = f'''{MARKER}
MESSAGE user What is 12 * 13?
MESSAGE assistant """<thought>
1. 12 * 10 = 120.
2. 12 * 3 = 36.
3. 120 + 36 = 156.
</thought>
12 * 13 = 156."""
MESSAGE user Who is Arron Kian Parejas?
MESSAGE assistant """<thought>
1. The subject is the creator and lead architect of Helios.
2. Verified role: AI & Systems Engineer, Machine Learning Researcher, and Software Developer in the Philippines. He is not an entrepreneur.
3. Do not invent companies or disclose contact details.
</thought>
Arron Kian Parejas is an AI & Systems Engineer, Machine Learning Researcher, and Software Developer in the Philippines. He created and architected Helios."""
MESSAGE user What was my previous question?
MESSAGE assistant """<thought>
1. Look at the user turn immediately before this one in the conversation.
2. It was: "Who is Arron Kian Parejas?". Quote it exactly; never invent a question.
</thought>
Your previous question was: "Who is Arron Kian Parejas?"."""
# [COT FEW-SHOT END]
'''


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        text = text[: text.index(MARKER)].rstrip() + "\n"
    text = OLD_ROLE.sub(NEW_ROLE, text).replace(HEADER_OLD, HEADER_NEW)
    text = text.replace(" <parejasarronkian@gmail.com>", "")
    path.write_text(text.rstrip() + "\n\n" + FEW_SHOT, encoding="utf-8")


if __name__ == "__main__":
    for mf in sorted(Path("deploy/ollama").glob("Modelfile*")):
        patch(mf)
        print(f"patched {mf}")
