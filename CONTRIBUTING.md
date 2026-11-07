# Contributing to Helios

Thank you for your interest in contributing to Helios. As an open-weight foundation model ecosystem, we hold ourselves and our contributors to strict engineering discipline, architectural invariants, and scientific rigor.

---

## 1. Development Environment Setup

Helios requires Python 3.11, 3.12, or 3.13. We use [uv](https://docs.astral.sh/uv/) for deterministic environment management and high-speed dependency resolution.

```bash
# Clone the repository
git clone https://github.com/darknecrocities/Helios.git
cd Helios

# Initialize virtual environment with Python 3.13
uv venv .venv --python 3.13
source .venv/bin/activate

# Install editable package with development and memory extras
uv pip install -e ".[dev,memory]"
```

---

## 2. Architectural Invariants & Engineering Standards

Every contribution must adhere to the following non-negotiable architectural standards:

### 2.1 Strict Physical File Ceiling (500-Line Limit)
- **Hard Ceiling**: No Python source file may exceed **500 physical lines** (including docstrings and whitespace).
- **Proactive Decomposition**: When a file reaches 350 lines, proactively decompose it into focused domain submodules. Target sweet spot: **150–250 lines**.
- The CI pipeline executes an automated line-ceiling audit on every pull request.

### 2.2 Cognitive Hygiene & Function Design
- **Function Length**: Maximum 40 physical lines per function or method.
- **Nesting Depth**: Maximum 3 levels of indentation. Utilize early returns and guard clauses to eliminate nested conditionals.
- **Cyclomatic Complexity**: Maintain cyclomatic complexity $\le 10$ across all routines.

### 2.3 Strict Type Annotations
- All functions, methods, class attributes, and module-level variables must have complete type hints.
- Type annotations must pass strict static analysis without untyped escapes (`Any` is only permitted for opaque serialization payloads).

### 2.4 Anti-AI-Slop & Anti-Mock Mandate
- **No Decorative Badges or Marketing Fluff**: Prohibit `shields.io` badge soup, neon borders, and synthetic AI marketing prose.
- **No Obvious Restatement Comments**: Do not write comments that merely restate the code (e.g., `# set step to zero`). Comments must explain *why*, invariant assumptions, or mathematical derivations.
- **Strict Anti-Mock Mandate**: Never introduce mock arrays, fake dictionaries, or simulated in-memory databases masquerading as persistence. Real persistence (SQLite with WAL, FAISS index files, atomic safetensors) is required for all stateful layers.
- **Zero Raw Emojis in Interactive Controls**: Interactive CLI output and terminal status messages must use structured text, semantic indicators, or ASCII symbols.

---

## 3. Testing Discipline

- **Colocation**: Tests reside in the `tests/` hierarchy (`tests/unit/`, `tests/integration/`, `tests/e2e/`).
- **Deterministic Seeding**: Always set deterministic random seeds in tests involving stochastic operations (`torch.manual_seed(42)`).
- **Comprehensive Coverage**: New features must include unit tests validating edge cases, boundary conditions, and invalid inputs.
- **Verification Commands**:
  ```bash
  # Run lint and format checking
  uv run ruff check .
  uv run ruff format --check .

  # Run full test suite
  uv run pytest -v --tb=short
  ```

---

## 4. Git Branching, Commits, and Pull Request Workflow

### 4.1 Branch Naming Taxonomy
| Prefix | Purpose | Example |
|---|---|---|
| `feat/` | New model capability or subsystem | `feat/m2-bpe-tokenizer` |
| `fix/` | Bug repair or regression fix | `fix/rope-scaling-factor` |
| `docs/` | Research document or guide | `docs/kv-cache-analysis` |
| `chore/` | Tooling, dependencies, or CI updates | `chore/update-uv-workflow` |
| `refactor/` | Restructuring with no behavioral change | `refactor/split-transformer-layer` |

### 4.2 Commit Architecture & 1,000-Line Budget
- **Single-Line Commits Are Prohibited**: Every commit must follow Conventional Commits with a structured description body:
  ```text
  <type>(<scope>): <summary line under 72 chars>

  - Context / Why: Motivation and architectural requirements.
  - Key Changes / What: Bulleted list of concrete modifications.
  - Verification / Impact: Test results, lint status, and line counts.
  ```
- **Commit Budget**: No single commit may exceed **1,000 lines of diff**. Decompose larger milestones into atomic, logically distinct commits.

### 4.3 Pull Request Checklist
Before opening a pull request, verify that:
1. `uv run ruff check .` reports zero violations.
2. `uv run ruff format --check .` reports zero formatting drift.
3. All source files satisfy the strict 500-line ceiling.
4. `uv run pytest tests/` passes 100%.
5. New functionality is covered by genuine, non-mocked tests.
6. Documentation is updated to reflect all public interface changes.
