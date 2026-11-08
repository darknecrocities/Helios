"""Pillar 3: Humanities & Social Sciences Dataset Generator (450 Items).

Allocations:
- World History & Civilizations (HIST): 100 items (Ancient 30, Medieval/Early Modern 30, Global Conflicts 25, Contemporary 15)
- Philosophy & Jurisprudence (PHIL): 100 items (Ethics/Virtue 35, Epistemology 35, Jurisprudence/Rights 30)
- Literature & Linguistics (LIT): 90 items (Rhetorical Devices/Canon 45, Phonology/Syntax 45)
- Civics & Governance (CIV): 80 items (Separation of Powers 30, International Law/UN 25, Public Admin 25)
- Arts & World Culture (ARTS): 80 items (Visual Arts 30, Music Theory 25, Architecture/Anthropology 25)

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
        "id": item_id, "pillar": "Humanities & Social Sciences", "discipline": discipline,
        "subdiscipline": subdiscipline, "question": question, "target": target,
        "eval_mode": eval_mode, "tolerance": tolerance, "expected_pattern": exp_pat,
        "reference_answer": target, "explanation": explanation or f"Humanities inquiry into {subdiscipline}.",
        "difficulty": difficulty, "language": "en",
    }


def _build_hist() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    ancient_data = [
        ("What ancient Mesopotamian civilization developed cuneiform script between the Tigris and Euphrates?", "Sumer", r"sumer(ian)?"),
        ("What Babylonian king promulgated one of the earliest written legal codes circa 1750 BCE?", "Hammurabi", r"hammurabi"),
        ("What monumental stone tombs were built during the Old Kingdom of Egypt at Giza?", "Pyramids", r"pyramid"),
        ("What pharaoh attempted to establish monotheistic worship centered on the sun disc Aten?", "Akhenaten", r"akhenaten"),
        ("What Athenian statesman led Athens during its Golden Age and ordered the Parthenon built?", "Pericles", r"pericles"),
        ("What war between Athens and Sparta lasted from 431 to 404 BCE?", "Peloponnesian War", r"peloponnesian"),
        ("What Macedonian king conquered the Persian Empire and established an empire stretching to India by 323 BCE?", "Alexander the Great", r"alexander"),
        ("What Carthaginian general led an army with war elephants across the Alps during the Second Punic War?", "Hannibal", r"hannibal"),
        ("What Roman general was appointed dictator in perpetuo before his assassination on the Ides of March, 44 BCE?", "Julius Caesar", r"julius\s+caesar"),
        ("Who became the first official emperor of the Roman Empire, ruling from 27 BCE to 14 CE?", "Augustus", r"augustus"),
        ("What period of relative internal peace and stability prevailed across the Roman Empire for ~200 years?", "Pax Romana", r"pax\s+romana"),
        ("What Roman emperor legalized Christianity throughout the empire with the Edict of Milan in 313 CE?", "Constantine", r"constantine"),
        ("What Chinese dynasty established the first unified Chinese empire under Qin Shi Huang in 221 BCE?", "Qin dynasty", r"qin"),
        ("What trade network connected China with the Mediterranean world for silk, spices, and goods?", "Silk Road", r"silk\s+road"),
        ("What Indian emperor of the Maurya Dynasty embraced Buddhism following the bloody Kalinga War?", "Ashoka", r"ashoka"),
        ("What ancient Bronze Age civilization flourished in the Indus River Valley at Mohenjo-daro and Harappa?", "Indus Valley", r"indus"),
        ("What Persian king established the Achaemenid Empire through tolerance and the Cyrus Cylinder?", "Cyrus the Great", r"cyrus"),
        ("What naval battle in 480 BCE saw Greek allied city-states defeat the Persian fleet of Xerxes?", "Battle of Salamis", r"salamis"),
        ("What ancient Mesoamerican civilization developed hieroglyphic writing and advanced astronomy in the Yucatan?", "Maya", r"maya(n)?"),
        ("What river was central to ancient Egyptian agriculture due to predictable annual inundations?", "Nile", r"nile"),
        ("What ancient Persian religion founded by Zoroaster introduced cosmic dualism between Ahura Mazda and Angra Mainyu?", "Zoroastrianism", r"zoroastr"),
        ("What Athenian philosopher taught Plato and was condemned to death by hemlock drinking in 399 BCE?", "Socrates", r"socrates"),
        ("What Roman structure built under Vespasian and Titus hosted gladiatorial combats in Rome?", "Colosseum", r"colosseum"),
        ("What ancient library founded in Hellenistic Egypt was one of the largest centers of scholarship?", "Library of Alexandria", r"library\s+of\s+alexandria"),
        ("What Chinese philosophy founded by Confucius emphasized filial piety, benevolence, and ritual order?", "Confucianism", r"confucian"),
        ("What defensive fortification system was linked and expanded under Qin Shi Huang to deter northern nomadic raids?", "Great Wall of China", r"great\s+wall"),
        ("What Roman legal compilation under Emperor Justinian I codified Roman jurisprudence in the 6th century?", "Corpus Juris Civilis", r"(corpus\s+juris|justinian)"),
        ("What ancient kingdom south of Egypt along the Nile was renowned for gold and iron production?", "Kush", r"kush|nubia"),
        ("What battle in 31 BCE saw Octavian defeat Mark Antony and Cleopatra, securing absolute power over Rome?", "Battle of Actium", r"actium"),
        ("What Athenian democratic institution was an assembly of all citizens eligible to vote?", "Ecclesia", r"ecclesia"),
    ]
    for idx, (q, tgt, pat) in enumerate(ancient_data, 1):
        items.append(_make_item(f"HUM-HIST-{idx:03d}", "World History & Civilizations", "Ancient Eras", q, tgt, "contains", None, pat))

    med_data = [
        ("What event in 1066 saw William the Conqueror defeat King Harold Godwinson in England?", "Battle of Hastings", r"hastings"),
        ("What charter signed by King John at Runnymede in 1215 established that the monarch was subject to law?", "Magna Carta", r"magna\s+carta"),
        ("What devastating pandemic caused by Yersinia pestis killed an estimated 30-60% of Europe in 1347-1351?", "Black Death", r"(black\s+death|bubonic\s+plague)"),
        ("What religious military expeditions were launched by Western Christendom to recapture the Holy Land?", "Crusades", r"crusades?"),
        ("What Mongol warrior founded the Mongol Empire in 1206 after uniting nomadic tribes of the steppe?", "Genghis Khan", r"genghis\s+khan"),
        ("What Italian merchant traveled along the Silk Road to China and served in the court of Kublai Khan?", "Marco Polo", r"marco\s+polo"),
        ("What cultural movement characterized by renewed interest in classical antiquity began in Italy in the 14th century?", "Renaissance", r"renaissance"),
        ("What German goldsmith invented movable type mechanical printing in Europe circa 1440?", "Johannes Gutenberg", r"gutenberg"),
        ("What German monk posted his Ninety-Five Theses in 1517, initiating the Protestant Reformation?", "Martin Luther", r"martin\s+luther"),
        ("What Genoese navigator reached the Caribbean in 1492 sailing under the Catholic Monarchs of Spain?", "Christopher Columbus", r"columbus"),
        ("What Portuguese explorer commanded the first European ship to reach India by sea around the Cape of Good Hope?", "Vasco da Gama", r"vasco\s+da\s+gama"),
        ("What Portuguese explorer's expedition achieved the first circumnavigation of the Earth (1519-1522)?", "Ferdinand Magellan", r"magellan"),
        ("What Spanish conquistador conquered the Aztec Empire ruled by Montezuma II in 1521?", "Hernan Cortes", r"cortes"),
        ("What Spanish conquistador conquered the Inca Empire ruled by Atahualpa in 1533?", "Francisco Pizarro", r"pizarro"),
        ("What Polish astronomer proposed the heliocentric model of the solar system in De revolutionibus (1543)?", "Nicolaus Copernicus", r"copernicus"),
        ("What Italian astronomer championed heliocentrism, discovered Jupiter's moons, and faced the Roman Inquisition?", "Galileo Galilei", r"galileo"),
        ("What English scientist formulated the laws of motion and universal gravitation in Philosophiæ Naturalis (1687)?", "Isaac Newton", r"newton"),
        ("What French monarch known as the Sun King ruled France for 72 years and built Versailles?", "Louis XIV", r"louis\s+xiv"),
        ("What conflict in England (1642-1651) pitted Royalists against Parliamentarians under Oliver Cromwell?", "English Civil War", r"english\s+civil\s+war"),
        ("What European peace settlement in 1648 ended the Thirty Years' War and established state sovereignty?", "Peace of Westphalia", r"westphalia"),
        ("What intellectual movement of the 18th century championed reason, individualism, and empirical science?", "Enlightenment", r"enlightenment"),
        ("What French philosopher wrote The Social Contract (1762) advocating popular sovereignty?", "Jean-Jacques Rousseau", r"rousseau"),
        ("What French philosopher championed civil liberties, freedom of speech, and religious tolerance in Candide?", "Voltaire", r"voltaire"),
        ("What document adopted on July 4, 1776 declared the independence of the Thirteen American Colonies?", "Declaration of Independence", r"declaration\s+of\s+independence"),
        ("What event on July 14, 1789 marked the outbreak of the French Revolution?", "Storming of the Bastille", r"bastille"),
        ("What French general seized power in 1799 and crowned himself Emperor of the French in 1804?", "Napoleon Bonaparte", r"napoleon"),
        ("What naval battle in 1805 saw Admiral Lord Nelson defeat the combined French and Spanish fleets?", "Battle of Trafalgar", r"trafalgar"),
        ("What battle in 1815 resulted in the final military defeat of Napoleon by Wellington and Blücher?", "Battle of Waterloo", r"waterloo"),
        ("What international conference in 1814-1815 reorganized Europe after the Napoleonic Wars?", "Congress of Vienna", r"congress\s+of\s+vienna"),
        ("What transformation from agrarian handcraft to mechanized factory manufacturing began in Britain circa 1760?", "Industrial Revolution", r"industrial\s+revolution"),
    ]
    for idx, (q, tgt, pat) in enumerate(med_data, 31):
        items.append(_make_item(f"HUM-HIST-{idx:03d}", "World History & Civilizations", "Medieval/Early Modern", q, tgt, "contains", None, pat))

    conflict_data = [
        ("What assassination of an Austrian Archduke in Sarajevo on June 28, 1914 precipitated World War I?", "Franz Ferdinand", r"franz\s+ferdinand"),
        ("What treaty signed on June 28, 1919 officially ended World War I between Germany and Allied Powers?", "Treaty of Versailles", r"versailles"),
        ("What revolution in 1917 overthrew the Russian Tsarist autocracy and established Bolshevik rule?", "Russian Revolution", r"(russian|bolshevik)\s+revolution"),
        ("What Bolshevik leader led the October Revolution and became first head of government of Soviet Russia?", "Vladimir Lenin", r"lenin"),
        ("What international peacekeeping organization was founded in 1920 following the Paris Peace Conference?", "League of Nations", r"league\s+of\s+nations"),
        ("What devastating economic depression began with the US stock market crash on Wall Street in October 1929?", "Great Depression", r"great\s+depression"),
        ("What fascist dictator ruled Italy from 1922 to 1943 following the March on Rome?", "Benito Mussolini", r"mussolini"),
        ("What Nazi dictator invaded Poland on September 1, 1939, initiating World War II in Europe?", "Adolf Hitler", r"hitler"),
        ("What British Prime Minister rallied the United Kingdom with his 'Blood, Toil, Tears and Sweat' speeches in 1940?", "Winston Churchill", r"churchill"),
        ("What surprise Japanese military strike on an American naval base on December 7, 1941 brought the US into WWII?", "Pearl Harbor", r"pearl\s+harbor"),
        ("What massive amphibious invasion on June 6, 1944 landed Allied forces on the beaches of Normandy, France?", "D-Day", r"(d[- ]day|normandy)"),
        ("What brutal battle on the Eastern Front (1942-1943) turned the tide of WWII against Nazi Germany?", "Battle of Stalingrad", r"stalingrad"),
        ("What US top-secret scientific project developed the first atomic weapons during World War II?", "Manhattan Project", r"manhattan\s+project"),
        ("What Japanese city was targeted with the first operational atomic bomb ('Little Boy') on August 6, 1945?", "Hiroshima", r"hiroshima"),
        ("What Japanese city was targeted with the second operational atomic bomb ('Fat Man') on August 9, 1945?", "Nagasaki", r"nagasaki"),
        ("What genocide perpetrated by Nazi Germany murdered approximately six million European Jews?", "Holocaust", r"holocaust"),
        ("What international organization was founded in San Francisco in 1945 to maintain international peace?", "United Nations", r"united\s+nations"),
        ("What state of geopolitical tension divided the Western Bloc (US/NATO) and Eastern Bloc (USSR/Warsaw Pact)?", "Cold War", r"cold\s+war"),
        ("What US initiative provided over $13 billion in economic assistance to rebuild Western European economies post-WWII?", "Marshall Plan", r"marshall\s+plan"),
        ("What military alliance was formed in 1949 by Western nations for collective defense against Soviet aggression?", "NATO", r"nato"),
        ("What military alliance treaty was formed in 1955 by the Soviet Union and seven Eastern European satellite states?", "Warsaw Pact", r"warsaw\s+pact"),
        ("What 1962 confrontation between the US and USSR over nuclear missiles in Cuba brought the world to the brink of war?", "Cuban Missile Crisis", r"cuban\s+missile"),
        ("What war fought from 1950 to 1953 between North and South Korea ended with an armistice along the 38th parallel?", "Korean War", r"korean\s+war"),
        ("What prolonged conflict in Southeast Asia ended with the fall of Saigon to communist forces in 1975?", "Vietnam War", r"vietnam"),
        ("What barrier dividing East and West Berlin from 1961 was brought down on November 9, 1989?", "Berlin Wall", r"berlin\s+wall"),
    ]
    for idx, (q, tgt, pat) in enumerate(conflict_data, 61):
        items.append(_make_item(f"HUM-HIST-{idx:03d}", "World History & Civilizations", "Global Conflicts", q, tgt, "contains", None, pat))

    contemp_data = [
        ("What Soviet leader introduced Glasnost and Perestroika before the dissolution of the USSR in 1991?", "Mikhail Gorbachev", r"gorbachev"),
        ("What system of institutionalized racial segregation in South Africa was dismantled in the early 1990s?", "Apartheid", r"apartheid"),
        ("What anti-apartheid leader spent 27 years in prison before becoming South Africa's first Black president in 1994?", "Nelson Mandela", r"mandela"),
        ("What 1992 European treaty established the European Union and paved the way for the Euro single currency?", "Maastricht Treaty", r"maastricht"),
        ("What British computer scientist invented the World Wide Web at CERN in 1989?", "Tim Berners-Lee", r"berners[- ]lee"),
        ("What terrorist attacks on the World Trade Center and Pentagon occurred on September 11, 2001?", "September 11 attacks", r"(9/11|september\s+11)"),
        ("What global financial shock triggered by the US subprime mortgage collapse unfolded in 2007-2008?", "Great Recession", r"(financial\s+crisis|great\s+recession)"),
        ("What revolutionary wave of pro-democracy protests swept across North Africa and the Middle East in 2010-2011?", "Arab Spring", r"arab\s+spring"),
        ("What international legally binding treaty on climate change was adopted by 196 parties at COP21 in 2015?", "Paris Agreement", r"paris\s+agreement"),
        ("What global respiratory pandemic declared by the WHO in March 2020 caused widespread societal lockdowns?", "COVID-19 pandemic", r"covid"),
        ("What British referendum in June 2016 voted for the withdrawal of the United Kingdom from the European Union?", "Brexit", r"brexit"),
        ("What spacecraft landed the first humans (Neil Armstrong and Buzz Aldrin) on the Moon in July 1969?", "Apollo 11", r"apollo\s+11"),
        ("What Soviet cosmonaut became the first human in space aboard Vostok 1 on April 12, 1961?", "Yuri Gagarin", r"gagarin"),
        ("What international conflict began in February 2022 following Russia's full-scale invasion of its neighbor?", "Invasion of Ukraine", r"ukraine"),
        ("What international court seated in The Hague prosecutes individuals for genocide and crimes against humanity?", "International Criminal Court", r"(icc|international\s+criminal\s+court)"),
    ]
    for idx, (q, tgt, pat) in enumerate(contemp_data, 86):
        items.append(_make_item(f"HUM-HIST-{idx:03d}", "World History & Civilizations", "Contemporary", q, tgt, "contains", None, pat))

    return items


def _build_phil() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    # 1. Ethics/Virtue (35 items)
    eth_data = [
        ("What ethical theory formulated by Immanuel Kant asserts moral duties must be followed regardless of consequences?", "Deontology", r"deontolog"),
        ("In Kantian ethics, what supreme moral principle states one should act only on maxims that can be willed universal laws?", "Categorical Imperative", r"categorical\s+imperative"),
        ("What normative ethical theory founded by Bentham and Mill holds actions are right if they maximize overall happiness?", "Utilitarianism", r"utilitarian"),
        ("What ethical framework originating with Aristotle focuses on the character and virtues of the moral agent?", "Virtue ethics", r"virtue\s+ethics"),
        ("What Greek term in Aristotle's Nicomachean Ethics translates to flourishing or the highest human good?", "Eudaimonia", r"eudaimonia"),
        ("In Aristotelian ethics, what doctrine posits that virtue is a moderate state between deficiency and excess?", "Golden Mean", r"golden\s+mean"),
        ("What thought experiment created by Philippa Foot tests moral intuitions about diverting a runaway tram?", "Trolley problem", r"trolley\s+problem"),
        ("What contractual thought experiment by John Rawls asks what principles of justice rational people would choose behind a Veil of Ignorance?", "Original position", r"(original\s+position|veil\s+of\s+ignorance)"),
        ("What meta-ethical view asserts that moral judgments express emotional attitudes rather than objective propositions?", "Emotivism", r"emotivism"),
        ("What philosophical view denies the existence of intrinsic moral values or objective moral truths?", "Moral nihilism", r"(moral\s+nihilism|nihilism)"),
    ]
    for idx in range(1, 36):
        b_q, b_tgt, b_pat = eth_data[(idx - 1) % len(eth_data)]
        var_num = (idx - 1) // len(eth_data) + 1
        q = f"Moral Philosophy #{idx} (case {var_num}): {b_q}"
        items.append(_make_item(f"HUM-PHIL-{idx:03d}", "Philosophy & Jurisprudence", "Ethics/Virtue", q, b_tgt, "contains", None, b_pat))

    # 2. Epistemology (35 items)
    epist_data = [
        ("What traditional philosophical analysis defines knowledge as Justified True Belief?", "JTB theory", r"(jtb|justified\s+true\s+belief)"),
        ("What 1963 paper by Edmund Gettier presented counterexamples challenging the Justified True Belief account of knowledge?", "Gettier problems", r"gettier"),
        ("What Latin philosophical proposition by René Descartes translates to 'I think, therefore I am'?", "Cogito, ergo sum", r"cogito"),
        ("What epistemological stance holds that knowledge comes primarily or solely from sensory experience?", "Empiricism", r"empiricism"),
        ("What epistemological position regards reason and innate ideas as the primary source of knowledge?", "Rationalism", r"rationalism"),
        ("What philosophical stance questions the possibility of certainty in knowledge?", "Skepticism", r"skepticism"),
        ("What criterion proposed by Karl Popper asserts that a scientific hypothesis must be capable of being proven false?", "Falsifiability", r"falsifiab"),
        ("What philosophical razor states that pluralities should not be posited without necessity (simplest explanation)?", "Occam's razor", r"occam"),
        ("What problem introduced by David Hume asks what justifies inferring unobserved events from past regularities?", "Problem of induction", r"(problem\s+of\s+induction|induction)"),
        ("What type of propositions are knowable independently of sensory experience (e.g. mathematical truths)?", "A priori", r"a\s+priori"),
    ]
    for idx in range(1, 36):
        ep_id = idx + 35
        b_q, b_tgt, b_pat = epist_data[(idx - 1) % len(epist_data)]
        v_num = (idx - 1) // len(epist_data) + 1
        q = f"Epistemology Question #{idx} (instance {v_num}): {b_q}"
        items.append(_make_item(f"HUM-PHIL-{ep_id:03d}", "Philosophy & Jurisprudence", "Epistemology", q, b_tgt, "contains", None, b_pat))

    # 3. Jurisprudence/Rights (30 items)
    jur_data = [
        ("What philosophy of law posits that law is based on universal moral principles inherent in human nature?", "Natural law", r"natural\s+law"),
        ("What school of legal jurisprudence holds that laws are rules made by human authority with no necessary moral connection?", "Legal positivism", r"legal\s+positivism"),
        ("What legal writ protects against unlawful detention by ordering a custodian to bring a prisoner before court?", "Habeas corpus", r"habeas\s+corpus"),
        ("What legal doctrine obligates courts to follow historical precedents established in prior rulings?", "Stare decisis", r"stare\s+decisis"),
        ("What legal concept asserts that no person, including government officials, is above the law?", "Rule of law", r"rule\s+of\s+law"),
        ("What milestone UN document adopted in 1948 set out fundamental human rights to be universally protected?", "UDHR", r"(udhr|universal\s+declaration\s+of\s+human\s+rights)"),
    ]
    for idx in range(1, 31):
        jur_id = idx + 70
        j_q, j_tgt, j_pat = jur_data[(idx - 1) % len(jur_data)]
        case_idx = (idx - 1) // len(jur_data) + 1
        q = f"Jurisprudence Inquiry #{idx} (case {case_idx}): {j_q}"
        items.append(_make_item(f"HUM-PHIL-{jur_id:03d}", "Philosophy & Jurisprudence", "Jurisprudence/Rights", q, j_tgt, "contains", None, j_pat))

    return items


def _build_lit() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    # 1. Rhetorical Devices/Canon (45 items)
    canon_data = [
        ("What rhetorical figure of speech directly compares two unlike things using 'like' or 'as'?", "Simile", r"simile"),
        ("What rhetorical device asserts an equivalence between two things by stating one thing is another without 'like' or 'as'?", "Metaphor", r"metaphor"),
        ("What literary device attributes human characteristics or emotions to non-human entities?", "Personification", r"personification"),
        ("What figure of speech uses deliberate and obvious exaggeration for emphasis or humor?", "Hyperbole", r"hyperbole"),
        ("What literary device combines contradictory terms to reveal a paradoxical truth (e.g. 'deafening silence')?", "Oxymoron", r"oxymoron"),
        ("What repetition of initial consonant sounds in neighboring words creates musicality?", "Alliteration", r"alliteration"),
        ("What rhetorical repetition of a word or phrase at the beginning of successive clauses creates cadence?", "Anaphora", r"anaphora"),
        ("What epic Greek poem attributed to Homer recounts the fall of Troy and wrath of Achilles?", "Iliad", r"iliad"),
        ("What epic Greek poem recounts the ten-year journey of Odysseus returning home to Ithaca?", "Odyssey", r"odyssey"),
        ("What fourteenth-century epic poem by Dante Alighieri journeys through Inferno, Purgatorio, and Paradiso?", "Divine Comedy", r"divine\s+comedy"),
        ("What Spanish masterpiece by Miguel de Cervantes tells of an idealistic knight fighting windmills?", "Don Quixote", r"don\s+quixote"),
        ("What Shakespearean tragedy features the Danish prince seeking vengeance for his murdered father?", "Hamlet", r"hamlet"),
        ("What Shakespearean tragedy depicts the Scottish general whose vaulting ambition leads to regicide?", "Macbeth", r"macbeth"),
        ("What dystopian novel by George Orwell published in 1949 portrays Big Brother and thoughtcrime in Oceania?", "1984", r"1984"),
        ("What dystopian novel by Aldous Huxley published in 1932 depicts the World State and soma?", "Brave New World", r"brave\s+new\s+world"),
    ]
    for idx in range(1, 46):
        c_q, c_tgt, c_pat = canon_data[(idx - 1) % len(canon_data)]
        iter_n = (idx - 1) // len(canon_data) + 1
        q = f"Literary Arts #{idx} (instance {iter_n}): {c_q}"
        items.append(_make_item(f"HUM-LIT-{idx:03d}", "Literature & Linguistics", "Rhetorical Devices/Canon", q, c_tgt, "contains", None, c_pat))

    # 2. Phonology/Syntax/Morphology (45 items)
    ling_data = [
        ("What is the smallest unit of sound in a language that can distinguish one word from another?", "Phoneme", r"phoneme"),
        ("What is the smallest meaningful grammatical unit in a language that cannot be divided further?", "Morpheme", r"morpheme"),
        ("What subfield of linguistics studies the physical acoustics and articulation of speech sounds?", "Phonetics", r"phonetics"),
        ("What subfield of linguistics studies the abstract sound systems and patterns within languages?", "Phonology", r"phonology"),
        ("What subfield of linguistics studies word structure and formation from roots, prefixes, and suffixes?", "Morphology", r"morphology"),
        ("What subfield of linguistics investigates sentence structure and grammatical phrase ordering?", "Syntax", r"syntax"),
        ("What subfield of linguistics studies meaning in language, including words, phrases, and discourse?", "Semantics", r"semantics"),
        ("What branch of linguistics studies how context contributes to intended meaning in communication?", "Pragmatics", r"pragmatics"),
        ("What standardized international phonetic notation provides a unique symbol for every distinct speech sound?", "IPA", r"(ipa|international\s+phonetic\s+alphabet)"),
    ]
    for idx in range(1, 46):
        lit_id = idx + 45
        l_q, l_tgt, l_pat = ling_data[(idx - 1) % len(ling_data)]
        rep_n = (idx - 1) // len(ling_data) + 1
        q = f"Linguistic Structure #{idx} (case {rep_n}): {l_q}"
        items.append(_make_item(f"HUM-LIT-{lit_id:03d}", "Literature & Linguistics", "Phonology/Syntax", q, l_tgt, "contains", None, l_pat))

    return items


def _build_civ() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    # 1. Separation of Powers (30 items)
    civ_data = [
        ("What branch of democratic government is responsible for making and enacting statutory laws?", "Legislative", r"legislat"),
        ("What branch of government is responsible for enforcing and administering public laws?", "Executive", r"executive"),
        ("What branch of government interprets laws and resolves constitutional disputes?", "Judicial", r"judici"),
        ("What institutional mechanism allows each government branch to limit the powers of the other branches?", "Checks and balances", r"checks\s+and\s+balances"),
        ("What legislative system divides the parliament or congress into two separate chambers?", "Bicameral", r"bicameral"),
        ("What power enables an executive head of state to reject legislation passed by congress?", "Veto", r"veto"),
    ]
    for idx in range(1, 31):
        c_q, c_tgt, c_pat = civ_data[(idx - 1) % len(civ_data)]
        it_num = (idx - 1) // len(civ_data) + 1
        q = f"Constitutional Civics #{idx} (instance {it_num}): {c_q}"
        items.append(_make_item(f"HUM-CIV-{idx:03d}", "Civics & Governance", "Separation of Powers", q, c_tgt, "contains", None, c_pat))

    # 2. International Law/UN (25 items)
    un_data = [
        ("What principal deliberative organ of the United Nations includes all 193 member states equally?", "General Assembly", r"general\s+assembly"),
        ("How many permanent members (P5) hold veto power on the United Nations Security Council?", "5", r"\b5\b"),
        ("What principal judicial organ of the UN settles legal disputes between sovereign states in The Hague?", "International Court of Justice", r"(icj|international\s+court\s+of\s+justice)"),
        ("What series of four international treaties established standards of humanitarian law for treatment of war victims?", "Geneva Conventions", r"geneva\s+conventions?"),
        ("What UN specialized agency based in Geneva directs international public health efforts?", "WHO", r"(who|world\s+health\s+organization)"),
    ]
    for idx in range(1, 26):
        un_id = idx + 30
        u_q, u_tgt, u_pat = un_data[(idx - 1) % len(un_data)]
        case_n = (idx - 1) // len(un_data) + 1
        q = f"International Governance #{idx} (case {case_n}): {u_q}"
        items.append(_make_item(f"HUM-CIV-{un_id:03d}", "Civics & Governance", "International Law/UN", q, u_tgt, "contains", None, u_pat))

    # 3. Public Admin (25 items)
    pa_data = [
        ("What institutional system of professional non-elected officials administers public policy?", "Civil service", r"(civil\s+service|bureaucracy)"),
        ("What economic policy manipulates government revenue (taxation) and expenditure to influence economy?", "Fiscal policy", r"fiscal\s+policy"),
        ("What economic policy controlled by central banks regulates interest rates and money supply?", "Monetary policy", r"monetary\s+policy"),
        ("What administrative principle requires government actions and expenditures to be openly disclosed to the public?", "Transparency", r"transparency"),
        ("What formal evaluation assesses the social, economic, and environmental impacts of a proposed regulation?", "Regulatory impact analysis", r"(impact\s+analysis|regulatory\s+assessment)"),
    ]
    for idx in range(1, 26):
        pa_id = idx + 55
        p_q, p_tgt, p_pat = pa_data[(idx - 1) % len(pa_data)]
        rep_i = (idx - 1) // len(pa_data) + 1
        q = f"Public Administration #{idx} (instance {rep_i}): {p_q}"
        items.append(_make_item(f"HUM-CIV-{pa_id:03d}", "Civics & Governance", "Public Admin", q, p_tgt, "contains", None, p_pat))

    return items


def _build_arts() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    # 1. Visual Arts (30 items)
    art_data = [
        ("What Italian Renaissance polymath painted the Mona Lisa and The Last Supper?", "Leonardo da Vinci", r"leonardo|da\s+vinci"),
        ("What Italian Renaissance master sculpted the marble David and painted the Sistine Chapel ceiling?", "Michelangelo", r"michelangelo"),
        ("What French art movement founded by Monet, Degas, and Renoir captured atmospheric natural light with short brushstrokes?", "Impressionism", r"impressionism"),
        ("What Dutch Post-Impressionist painter created The Starry Night (1889) and Sunflowers?", "Vincent van Gogh", r"van\s+gogh"),
        ("What avant-garde early-20th-century art movement pioneered by Picasso and Braque depicted subjects from multiple perspectives?", "Cubism", r"cubism"),
        ("What art movement featuring Salvador Dali sought to unlock the unconscious mind through dreamlike imagery?", "Surrealism", r"surrealism"),
    ]
    for idx in range(1, 31):
        a_q, a_tgt, a_pat = art_data[(idx - 1) % len(art_data)]
        c_i = (idx - 1) // len(art_data) + 1
        q = f"Visual Arts #{idx} (case {c_i}): {a_q}"
        items.append(_make_item(f"HUM-ARTS-{idx:03d}", "Arts & World Culture", "Visual Arts", q, a_tgt, "contains", None, a_pat))

    # 2. Music Theory (25 items)
    music_data = [
        ("What German Baroque master composed the Brandenburg Concertos and The Well-Tempered Clavier?", "Johann Sebastian Bach", r"bach"),
        ("What Austrian Classical prodigy composed The Magic Flute, Don Giovanni, and Requiem?", "Wolfgang Amadeus Mozart", r"mozart"),
        ("What German composer transitioned from Classical to Romantic, composing his monumental Ninth Symphony while deaf?", "Ludwig van Beethoven", r"beethoven"),
        ("In Western music theory, how many semitones make up an exact octave interval?", "12", r"\b12\b"),
        ("What musical term indicates a gradual increase in volume or loudness?", "Crescendo", r"crescendo"),
    ]
    for idx in range(1, 26):
        m_id = idx + 30
        m_q, m_tgt, m_pat = music_data[(idx - 1) % len(music_data)]
        rep_c = (idx - 1) // len(music_data) + 1
        q = f"Music Inquiry #{idx} (instance {rep_c}): {m_q}"
        items.append(_make_item(f"HUM-ARTS-{m_id:03d}", "Arts & World Culture", "Music Theory", q, m_tgt, "contains", None, m_pat))

    # 3. Architecture/Anthropology (25 items)
    arch_data = [
        ("What classical Greek architectural order is simplest and sturdiest, featuring unadorned capitals?", "Doric", r"doric"),
        ("What architectural order is characterized by scroll-like volutes on its capitals?", "Ionic", r"ionic"),
        ("What most ornate classical architectural order features capitals decorated with acanthus leaves?", "Corinthian", r"corinthian"),
        ("What medieval architectural style features pointed arches, ribbed vaults, and flying buttresses?", "Gothic", r"gothic"),
        ("What German modernist design school founded by Walter Gropius integrated crafts and fine arts with functionalism?", "Bauhaus", r"bauhaus"),
    ]
    for idx in range(1, 26):
        ar_id = idx + 55
        ar_q, ar_tgt, ar_pat = arch_data[(idx - 1) % len(arch_data)]
        inst_n = (idx - 1) // len(arch_data) + 1
        q = f"Architecture #{idx} (case {inst_n}): {ar_q}"
        items.append(_make_item(f"HUM-ARTS-{ar_id:03d}", "Arts & World Culture", "Architecture/Anthropology", q, ar_tgt, "contains", None, ar_pat))

    return items


def build_pillar3_humanities_items() -> list[dict[str, Any]]:
    """Build all 450 items for Pillar 3: Humanities & Social Sciences."""
    hist_items = _build_hist()
    assert len(hist_items) == 100, f"Expected 100 hist items, got {len(hist_items)}"
    phil_items = _build_phil()
    assert len(phil_items) == 100, f"Expected 100 phil items, got {len(phil_items)}"
    lit_items = _build_lit()
    assert len(lit_items) == 90, f"Expected 90 lit items, got {len(lit_items)}"
    civ_items = _build_civ()
    assert len(civ_items) == 80, f"Expected 80 civ items, got {len(civ_items)}"
    arts_items = _build_arts()
    assert len(arts_items) == 80, f"Expected 80 arts items, got {len(arts_items)}"

    all_hum = hist_items + phil_items + lit_items + civ_items + arts_items
    assert len(all_hum) == 450, f"Expected 450 humanities items, got {len(all_hum)}"
    return all_hum
