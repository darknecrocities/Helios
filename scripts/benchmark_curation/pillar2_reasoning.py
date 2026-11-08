"""Pillar 2: Reasoning & Logic Dataset Generator (500 Items).

Allocations:
- Syllogistic Deduction (SYLL): 100 items (Categorical 35, Conditional 35, Multi-Premise 30)
- Causal & Counterfactual (CAUS): 100 items (Physical Causality 30, Counterfactual Scenarios 35, Confounding Variables 35)
- Fallacies & Critical Thinking (FALL): 100 items (Informal Fallacies 40, Formal Fallacies 30, Cognitive Biases 30)
- Spatial & Temporal Logic (SPAT): 100 items (3D/Compass Spatial 35, 12h/24h Time & Elapsed 35, Intervals/Scheduling 30)
- Algorithmic Tracing (ALGO): 100 items (State Machine Traces 35, Code Dry-Runs 35, Graph BFS/DFS Orders 30)

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import re
from typing import Any


def _make_item(
    item_id: str, discipline: str, subdiscipline: str, question: str, target: str,
    eval_mode: str = "contains", tolerance: float | None = None, pattern: str = "",
    explanation: str = "", difficulty: str = "medium",
) -> dict[str, Any]:
    exp_pat = pattern or rf"\b{re.escape(target)}\b"
    return {
        "id": item_id, "pillar": "Reasoning & Logic", "discipline": discipline,
        "subdiscipline": subdiscipline, "question": question, "target": target,
        "eval_mode": eval_mode, "tolerance": tolerance, "expected_pattern": exp_pat,
        "reference_answer": target, "explanation": explanation or f"Logical reasoning deduction in {subdiscipline}.",
        "difficulty": difficulty, "language": "en",
    }


def _build_syll() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    # 1. Categorical Syllogisms (35 items)
    for i in range(1, 36):
        m, p, s = f"group_{i}", f"property_{i}", f"subject_{i}"
        if i % 2 == 1:
            q = f"Premise 1: All {m} have {p}. Premise 2: All {s} are {m}. Is the conclusion 'All {s} have {p}' valid or invalid?"
            items.append(_make_item(f"REAS-SYLL-{i:03d}", "Syllogistic Deduction", "Categorical Syllogisms", q, "Valid", "exact", None, r"\bvalid\b"))
        else:
            q = f"Premise 1: Some {m} have {p}. Premise 2: Some {s} are {m}. Is the conclusion 'Some {s} have {p}' valid or invalid?"
            items.append(_make_item(f"REAS-SYLL-{i:03d}", "Syllogistic Deduction", "Categorical Syllogisms", q, "Invalid", "exact", None, r"\binvalid\b"))

    # 2. Conditional Reasoning (35 items)
    for i in range(1, 36):
        cond_id = i + 35
        p_name, q_name = f"condition_P_{i}", f"result_Q_{i}"
        if i % 3 == 1:
            q = f"Rule: If {p_name} is true, then {q_name} is true. Fact: {p_name} is true. What must be true about {q_name} (true or false)?"
            items.append(_make_item(f"REAS-SYLL-{cond_id:03d}", "Syllogistic Deduction", "Conditional Reasoning", q, "True", "exact", None, r"\btrue\b"))
        elif i % 3 == 2:
            q = f"Rule: If {p_name} is true, then {q_name} is true. Fact: {q_name} is false. By modus tollens, is {p_name} true or false?"
            items.append(_make_item(f"REAS-SYLL-{cond_id:03d}", "Syllogistic Deduction", "Conditional Reasoning", q, "False", "exact", None, r"\bfalse\b"))
        else:
            q = f"Rule: If {p_name} is true, then {q_name} is true. Fact: {q_name} is true. Does it necessarily follow that {p_name} is true (yes or no)?"
            items.append(_make_item(f"REAS-SYLL-{cond_id:03d}", "Syllogistic Deduction", "Conditional Reasoning", q, "No", "exact", None, r"\bno\b"))

    # 3. Multi-Premise Reasoning (30 items)
    for i in range(1, 31):
        multi_id = i + 70
        steps = i % 4 + 3  # 3 to 6 steps
        chain_names = [f"state_{chr(65+k)}_{i}" for k in range(steps)]
        chain_str = ". ".join([f"If {chain_names[k]} is active, then {chain_names[k+1]} is active" for k in range(steps - 1)])
        q = f"Given rule chain: {chain_str}. Fact: {chain_names[0]} is active. Which state is guaranteed to be active: {chain_names[-1]} or none?"
        items.append(_make_item(f"REAS-SYLL-{multi_id:03d}", "Syllogistic Deduction", "Multi-Premise", q, chain_names[-1], "contains", None, rf"\b{chain_names[-1]}\b"))

    return items


def _build_caus() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    # 1. Physical Causality (30 items)
    phys_caus = [
        ("If a metal rod is heated at one end, what physical process causes the other end to become warm?", "Conduction", r"conduction"),
        ("If an electrical circuit's current exceeds its fuse rating, what causes the circuit to open?", "Fuse melts", r"(fuse\s+(melts|blows)|breaker\s+trips)"),
        ("If atmospheric pressure drops significantly over an ocean region, what weather system tends to form?", "Low-pressure storm", r"(storm|cyclone|depression|low[- ]pressure)"),
        ("If water in an enclosed pipe freezes, what physical property causes the pipe to burst?", "Expansion", r"expansion"),
        ("If the temperature of an ideal gas in a rigid container increases, what happens to its pressure?", "Increases", r"increases?"),
        ("If an unbalanced net force acts continuously on a spacecraft in deep space, what happens to its velocity?", "Accelerates", r"(accelerates?|increases?)"),
        ("If sunlight is focused through a convex magnifying lens onto dry paper, what causes ignition?", "Concentrated heat", r"(heat|focal\s+point|thermal)"),
        ("If an open container of liquid nitrogen is exposed to room air, what causes rapid boiling?", "Heat absorption", r"(boiling|temperature|heat)"),
        ("If atmospheric humidity reaches 100% and temperature drops below dew point, what forms?", "Condensation", r"(condensation|dew|fog|rain)"),
        ("If two tectonic plates slide past each other horizontally along a transform fault, what event occurs?", "Earthquake", r"earthquake"),
        ("If sound waves travel from air into water, what causes their speed to increase?", "Higher density", r"(density|bulk\s+modulus|elasticity|faster)"),
        ("If light enters glass from air at an angle, what causes the beam to bend toward the normal?", "Refraction", r"refraction"),
        ("If an object is submerged in fluid and buoyant force exceeds its weight, what happens?", "Floats", r"(floats?|rises?)"),
        ("If electrical current flows through a resistive wire, what energy transformation occurs?", "Electrical to thermal", r"(thermal|heat)"),
        ("If a spinning ice skater pulls her arms inward, what causes her rotational speed to increase?", "Conservation of angular momentum", r"(angular\s+momentum|inertia)"),
        ("If an iron gate is exposed to oxygen and moisture without paint, what chemical compound forms?", "Rust", r"(rust|iron\s+oxide)"),
        ("If a pendulum's length is quadrupled, what factor multiplies its oscillation period?", "2", r"\b2\b"),
        ("If atmospheric carbon dioxide concentration increases, what causes ocean acidification?", "Carbonic acid formation", r"(carbonic\s+acid|acidification|lower\s+ph)"),
        ("If air accelerates over a curved airplane wing upper surface, what pressure difference creates lift?", "Bernoulli effect", r"(bernoulli|lower\s+pressure)"),
        ("If a sound source moves rapidly toward an observer, what phenomenon raises the observed pitch?", "Doppler effect", r"doppler"),
        ("If an electric generator's rotor spins inside a magnetic field, what phenomenon produces voltage?", "Electromagnetic induction", r"induction"),
        ("If salt is dissolved in water, what happens to the boiling point of the solution?", "Increases", r"(increases?|elevates?)"),
        ("If salt is dissolved in water, what happens to the freezing point of the solution?", "Decreases", r"(decreases?|depresses?)"),
        ("If oil and water are shaken together vigorously and left undisturbed, why do they separate?", "Density difference and polarity", r"(density|immiscible|polarity)"),
        ("If a bar magnet is cut in half, how many magnetic poles does each resulting piece possess?", "2", r"\b2\b"),
        ("If a gas expands adiabatically without heat exchange, what happens to its internal temperature?", "Decreases", r"(decreases?|cools?)"),
        ("If a guitar string's tension is tightened, what happens to its fundamental vibrational frequency?", "Increases", r"(increases?|higher)"),
        ("If a hot cup of coffee is left in a cooler room, heat flows spontaneously from coffee to room until what state?", "Thermal equilibrium", r"thermal\s+equilibrium"),
        ("If green light strikes a green leaf, why does the leaf appear green to the human eye?", "Reflects green light", r"reflects?"),
        ("If an electrical conductor has zero electrical resistance at low temperatures, what state is it in?", "Superconductivity", r"superconduct"),
    ]
    for idx, (q, tgt, pat) in enumerate(phys_caus, 1):
        items.append(_make_item(f"REAS-CAUS-{idx:03d}", "Causal & Counterfactual", "Physical Causality", q, tgt, "contains", None, pat))

    # 2. Counterfactual Scenarios (35 items)
    for i in range(1, 36):
        cf_id = i + 30
        if i % 3 == 1:
            q = f"Counterfactual #{i}: If Earth possessed zero atmosphere, would acoustic sound waves transmit across the air from a speaker to a listener (yes or no)?"
            items.append(_make_item(f"REAS-CAUS-{cf_id:03d}", "Causal & Counterfactual", "Counterfactual Scenarios", q, "No", "exact", None, r"\bno\b"))
        elif i % 3 == 2:
            q = f"Counterfactual #{i}: If friction were completely eliminated between tires and a flat icy road, could a car execute a standard curved turn via steering (yes or no)?"
            items.append(_make_item(f"REAS-CAUS-{cf_id:03d}", "Causal & Counterfactual", "Counterfactual Scenarios", q, "No", "exact", None, r"\bno\b"))
        else:
            q = f"Counterfactual #{i}: If Earth's rotational axis had zero axial tilt relative to its orbital plane, would distinct seasonal variations occur annually (yes or no)?"
            items.append(_make_item(f"REAS-CAUS-{cf_id:03d}", "Causal & Counterfactual", "Counterfactual Scenarios", q, "No", "exact", None, r"\bno\b"))

    # 3. Confounding Variables (35 items)
    for i in range(1, 36):
        conf_id = i + 65
        topic_a, topic_b, confounder = f"factor_A_{i}", f"outcome_B_{i}", f"lurking_variable_Z_{i}"
        q = f"Study #{i}: Researchers observe a positive correlation between {topic_a} and {topic_b}. If {confounder} influences both independently, does this correlation prove that {topic_a} directly causes {topic_b} (yes or no)?"
        items.append(_make_item(f"REAS-CAUS-{conf_id:03d}", "Causal & Counterfactual", "Confounding Variables", q, "No", "exact", None, r"\bno\b"))

    return items


def _build_fall() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    # 1. Informal Fallacies (40 items)
    informal_fallacies = [
        ("Ad hominem", "Attacking the opponent's character instead of their argument"),
        ("Straw man", "Misrepresenting or exaggerating an opponent's argument to make it easier to attack"),
        ("Slippery slope", "Asserting that a relatively small first step inevitably leads to a chain of negative events"),
        ("Appeal to authority", "Claiming something must be true solely because an authority figure asserted it"),
        ("False dilemma", "Presenting two alternative options as the only possibilities when more exist"),
        ("Red herring", "Introducing an irrelevant topic to divert attention from the original argument"),
        ("Post hoc ergo propter hoc", "Assuming that because event B occurred after event A, event A caused event B"),
        ("Begging the question", "An argument whose premises assume the truth of the conclusion being defended"),
        ("Appeal to emotion", "Manipulating an emotional response in place of a valid or compelling argument"),
        ("Bandwagon fallacy", "Arguing that an idea is correct simply because a large number of people believe it"),
    ]
    for idx in range(1, 41):
        fall_name, fall_desc = informal_fallacies[(idx - 1) % len(informal_fallacies)]
        var_id = (idx - 1) // len(informal_fallacies) + 1
        q = f"Scenario #{idx}: Speaker claims an argument is invalid solely by {fall_desc.lower()} (variation {var_id}). What informal fallacy is this?"
        items.append(_make_item(f"REAS-FALL-{idx:03d}", "Fallacies & Critical Thinking", "Informal Fallacies", q, fall_name, "contains", None, rf"\b{re.escape(fall_name)}\b"))

    # 2. Formal Fallacies (30 items)
    formal_fallacies = [
        ("Affirming the consequent", "Premise 1: If P then Q. Premise 2: Q is true. Conclusion: Therefore, P is true."),
        ("Denying the antecedent", "Premise 1: If P then Q. Premise 2: P is false. Conclusion: Therefore, Q is false."),
        ("Fallacy of the undistributed middle", "Premise 1: All A are B. Premise 2: All C are B. Conclusion: Therefore, all A are C."),
        ("Affirming a disjunct", "Premise 1: A or B. Premise 2: A is true. Conclusion: Therefore, B is false (in non-exclusive disjunction)."),
        ("Denying a conjunct", "Premise 1: Not both A and B. Premise 2: Not A. Conclusion: Therefore, B is true."),
    ]
    for idx in range(1, 31):
        fall_id = idx + 40
        f_name, f_struct = formal_fallacies[(idx - 1) % len(formal_fallacies)]
        rep = (idx - 1) // len(formal_fallacies) + 1
        q = f"Formal deduction pattern #{idx} (case {rep}): '{f_struct}'. What formal fallacy does this inference commit?"
        items.append(_make_item(f"REAS-FALL-{fall_id:03d}", "Fallacies & Critical Thinking", "Formal Fallacies", q, f_name, "contains", None, rf"\b{re.escape(f_name)}\b"))

    # 3. Cognitive Biases (30 items)
    biases = [
        ("Confirmation bias", "The tendency to search for, interpret, and recall information that confirms preexisting beliefs"),
        ("Anchoring bias", "The tendency to rely too heavily on the first piece of information encountered when making decisions"),
        ("Availability heuristic", "Overestimating the likelihood of events based on how easily examples come to mind"),
        ("Sunk cost fallacy", "Continuing an endeavor once an investment in money, effort, or time has been made"),
        ("Hindsight bias", "The tendency to perceive past events as having been more predictable than they were"),
        ("Dunning-Kruger effect", "A cognitive bias where people with low ability in a domain overestimate their competence"),
    ]
    for idx in range(1, 31):
        bias_id = idx + 70
        b_name, b_desc = biases[(idx - 1) % len(biases)]
        case_num = (idx - 1) // len(biases) + 1
        q = f"Cognitive pattern #{idx} (instance {case_num}): '{b_desc}'. What cognitive bias or psychological effect is this?"
        items.append(_make_item(f"REAS-FALL-{bias_id:03d}", "Fallacies & Critical Thinking", "Cognitive Biases", q, b_name, "contains", None, rf"\b{re.escape(b_name)}\b"))

    return items


def _build_spat() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    # 1. 3D/Compass Spatial (35 items)
    compass_dirs = ["North", "East", "South", "West"]
    for i in range(1, 36):
        start_dir = compass_dirs[(i - 1) % 4]
        turns_cw = (i * 90) % 360
        target_dir = compass_dirs[((i - 1) % 4 + turns_cw // 90) % 4]
        q = f"Spatial orientation #{i}: You start facing {start_dir}. You make a {turns_cw}-degree turn clockwise. Which cardinal direction are you now facing?"
        items.append(_make_item(f"REAS-SPAT-{i:03d}", "Spatial & Temporal Logic", "3D/Compass Spatial", q, target_dir, "exact", None, rf"\b{target_dir}\b"))

    # 2. 12h/24h Time & Elapsed (35 items)
    for i in range(1, 36):
        time_id = i + 35
        start_hr = (i % 12) + 1
        dur_hr = (i % 5) + 1
        end_hr = (start_hr + dur_hr)
        q = f"Temporal elapsed #{i}: An experiment starts at {start_hr:02d}:00 and runs continuously for {dur_hr} hours. At what hour does it finish?"
        items.append(_make_item(f"REAS-SPAT-{time_id:03d}", "Spatial & Temporal Logic", "12h/24h Time & Elapsed", q, str(end_hr), "numeric", 0.01, rf"\b{end_hr}\b"))

    # 3. Intervals/Scheduling (30 items)
    for i in range(1, 31):
        sched_id = i + 70
        t1, dur1 = i, 4
        t2, dur2 = i + 2, 5
        # [t1, t1+dur1] vs [t2, t2+dur2] -> [i, i+4] vs [i+2, i+7], overlap is [i+2, i+4] -> 2 units
        overlap = max(0, min(t1 + dur1, t2 + dur2) - max(t1, t2))
        q = f"Scheduling task #{i}: Task A runs from minute {t1} to {t1+dur1}. Task B runs from minute {t2} to {t2+dur2}. How many minutes do both tasks overlap?"
        items.append(_make_item(f"REAS-SPAT-{sched_id:03d}", "Spatial & Temporal Logic", "Intervals/Scheduling", q, str(overlap), "numeric", 0.01, rf"\b{overlap}\b"))

    return items


def _build_algo() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    # 1. State Machine Traces (35 items)
    for i in range(1, 36):
        states = ["S0", "S1", "S2"]
        # transition: S0 -> S1 on 1, S1 -> S2 on 1, S2 -> S0 on 1. All 0 stay.
        ones_count = i
        final_state = states[ones_count % 3]
        q = f"State Machine #{i}: A 3-state counter (states S0, S1, S2) transitions circularly (S0->S1->S2->S0) on input '1'. Starting at S0, what state is reached after {ones_count} '1' inputs?"
        items.append(_make_item(f"REAS-ALGO-{i:03d}", "Algorithmic Tracing", "State Machine Traces", q, final_state, "exact", None, rf"\b{final_state}\b"))

    # 2. Code Dry-Runs (35 items)
    for i in range(1, 36):
        code_id = i + 35
        # loop dry run: sum of first n integers: n*(n+1)//2
        n_val = (i % 10) + 3
        expected_sum = n_val * (n_val + 1) // 2
        q = f"Code Dry-Run #{i}: Consider the pseudocode: total = 0; for k in 1..{n_val}: total += k; return total. What value is returned?"
        items.append(_make_item(f"REAS-ALGO-{code_id:03d}", "Algorithmic Tracing", "Code Dry-Runs", q, str(expected_sum), "numeric", 0.01, rf"\b{expected_sum}\b"))

    # 3. Graph BFS/DFS Orders (30 items)
    for i in range(1, 31):
        graph_id = i + 70
        nodes_count = (i % 4) + 4  # 4 to 7
        q = f"Graph Traversal #{i}: In a linear path graph Node_1 -> Node_2 -> ... -> Node_{nodes_count}, what is the destination node reached after visiting all {nodes_count} nodes?"
        items.append(_make_item(f"REAS-ALGO-{graph_id:03d}", "Algorithmic Tracing", "Graph BFS/DFS Orders", q, f"Node_{nodes_count}", "contains", None, rf"\bNode_{nodes_count}\b"))

    return items


def build_pillar2_reasoning_items() -> list[dict[str, Any]]:
    """Build all 500 items for Pillar 2: Reasoning & Logic."""
    syll_items = _build_syll()
    assert len(syll_items) == 100, f"Expected 100 syll items, got {len(syll_items)}"
    caus_items = _build_caus()
    assert len(caus_items) == 100, f"Expected 100 caus items, got {len(caus_items)}"
    fall_items = _build_fall()
    assert len(fall_items) == 100, f"Expected 100 fall items, got {len(fall_items)}"
    spat_items = _build_spat()
    assert len(spat_items) == 100, f"Expected 100 spat items, got {len(spat_items)}"
    algo_items = _build_algo()
    assert len(algo_items) == 100, f"Expected 100 algo items, got {len(algo_items)}"

    all_reas = syll_items + caus_items + fall_items + spat_items + algo_items
    assert len(all_reas) == 500, f"Expected 500 reasoning items, got {len(all_reas)}"
    return all_reas
