"""Foundational Real-World Entities, Taxonomic Definitions, and Physical Concepts.

Curates authoritative knowledge for universal natural entities (dogs, humans, cats,
animals, ecology, celestial bodies, and computational fundamentals) to ensure Helios
reasons dynamically through first-principles taxonomy.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

FOUNDATIONAL_ENTITIES_PAIRS: list[tuple[str, str]] = [
    # Zoology & Animal Kingdom
    (
        "What is a dog?",
        "A dog (Canis lupus familiaris) is a domesticated carnivoran mammal belonging to the family Canidae. As a subspecies of the gray wolf, dogs were the first animal species to be domesticated by human hunter-gatherers over 15,000 to 30,000 years ago. Dogs possess highly developed olfactory and auditory senses, keen social intelligence, and remarkable phenotypic diversity spanning hundreds of breeds. They serve diverse societal roles including companionship, herding, hunting, search and rescue, guide assistance for individuals with disabilities, and scent detection.",
    ),
    (
        "What is dog?",
        "A dog (Canis lupus familiaris) is a domesticated canine mammal of the order Carnivora and family Canidae. Celebrated as 'man's best friend', dogs exhibit extraordinary loyalty, trainability, and social cognition. They are characterized by acute senses of smell and hearing, four-legged quadrupedal locomotion, and expressive vocal and body communication, ranging in size from tiny Chihuahuas to giant Great Danes.",
    ),
    (
        "Ano ang aso?",
        "Ang aso (Canis lupus familiaris) ay isang maamong hayop na kabilang sa pamilya Canidae at sub-uri ng lobo. Kilala bilang matapat na kaibigan ng tao, ang aso ang unang hayop na inalagaan ng tao libo-libong taon na ang nakalilipas. Mayroon silang matalas na pang-amoy at pandinig, at malawak ang naitutulong bilang bantay, gabay, at kasama sa tahanan.",
    ),
    (
        "What is a cat?",
        "A cat (Felis catus) is a small, typically carnivorous, domesticated mammal belonging to the family Felidae. Domesticated in the Near East around 7500 BCE, cats are known for their flexible bodies, retractable claws, acute hearing, nocturnal tapetum lucidum night vision, and solitary yet affectionate companionship, as well as their natural predatory prowess in hunting rodents and pests.",
    ),
    (
        "What is an animal?",
        "An animal is a multicellular, eukaryotic organism belonging to the biological kingdom Animalia. Animals are distinguished from plants and fungi by their heterotrophic nutrition (consuming other organic matter for energy), cellular structure lacking rigid cell walls, characteristic motility (ability to move spontaneously during at least one life stage), and embryonic development through blastula formation.",
    ),
    (
        "What is a mammal?",
        "A mammal is an endothermic (warm-blooded) vertebrate animal belonging to the class Mammalia, characterized by mammary glands that produce milk for nourishing young, hair or fur covering at least part of the body, three middle ear bones (malleus, incus, and stapes), and a neocortex region in the brain.",
    ),
    # Anthropology & Human Biology
    (
        "What is a human?",
        "A human (Homo sapiens) is a primate mammal belonging to the family Hominidae and the sole extant member of the genus Homo. Anatomically, humans are distinguished by habitual obligate bipedalism, an exceptionally high encephalization quotient (large, complex neocortex), dexterous opposable thumbs, and intricate vocal anatomy capable of articulated speech. Cognitively, humans are characterized by symbolic reasoning, recursive language, meta-cognition, episodic memory, self-awareness, moral agency, and the capacity to construct complex cumulative cultures, civilizations, technologies, and scientific knowledge.",
    ),
    (
        "What is human?",
        "A human (Homo sapiens) is a bipedal hominid mammal distinguished by advanced cognitive faculties, symbolic linguistic communication, reflective consciousness, and deep social structures. Humans create art, science, laws, ethics, and philosophy, utilizing tools and computation to transform environments and explore the cosmos.",
    ),
    (
        "Ano ang tao?",
        "Ang tao (Homo sapiens) ay isang uri ng mamalya na may mataas na antas ng talino, kamalayan sa sarili, at kakayahang magsalita, mag-isip nang malalim, at gumawa ng mga kagamitan. Ang mga tao ay may kakayahang bumuo ng lipunan, kultura, agham, at sining, at namumukod-tangi sa buong daigdig dahil sa kanilang kakayahang mangatwiran at magpasiya.",
    ),
    # Earth, Cosmos, and Computational Foundations
    (
        "What is the Earth?",
        "Earth is the third planet from the Sun and the only astronomical object known to harbor life. Formed approximately 4.54 billion years ago, Earth is a terrestrial rocky planet with a dynamic liquid iron outer core generating a protective geomagnetic field, an oxygen-nitrogen rich atmosphere, and an active hydrologic cycle where liquid water covers approximately 71% of its surface.",
    ),
    (
        "What is the Sun?",
        "The Sun is a G-type main-sequence star (spectral class G2V) at the center of the Solar System, comprising 99.86% of the system's total mass. Driven by nuclear fusion converting hydrogen into helium in its core at temperatures exceeding 15 million Kelvin, the Sun provides the fundamental electromagnetic radiation and heat that sustains weather, ocean currents, and biological life on Earth.",
    ),
    (
        "What is a computer?",
        "A computer is a programmable electronic device that accepts raw digital data as input, stores and processes it according to a sequence of stored instructions (software programs) using a central processing unit (CPU) and memory (RAM), and generates output information or actuates external hardware systems.",
    ),
    (
        "What is artificial intelligence?",
        "Artificial Intelligence (AI) is a multidisciplinary domain of computer science focused on developing computational systems and algorithms capable of performing tasks that traditionally require human cognitive intelligence. These capabilities include visual perception, speech recognition, natural language comprehension, causal reasoning, pattern synthesis, and decision-making under uncertainty.",
    ),
]
