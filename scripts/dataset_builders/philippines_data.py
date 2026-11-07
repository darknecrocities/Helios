"""Philippine Knowledge, Geography, History, and Civics Dataset.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

PHILIPPINES_PAIRS: list[tuple[str, str]] = [
    (
        "What is the Philippines?",
        "The Philippines is an archipelagic country in Southeast Asia located in the western Pacific Ocean. It consists of 7,641 islands broadly categorized into three main geographic groups: Luzon, Visayas, and Mindanao. Its capital city is Manila, and Quezon City is its most populous city. The official languages are Filipino and English.",
    ),
    (
        "What is philippines?",
        "The Philippines is a sovereign island country in Southeast Asia situated in the western Pacific Ocean. Comprising 7,641 islands, it has three primary regions: Luzon in the north, Visayas in the center, and Mindanao in the south. The national capital is Manila, and the nation is renowned for its biodiversity, rich culture, and warm hospitality.",
    ),
    (
        "Ano ang Pilipinas?",
        "Ang Pilipinas ay isang kapuluang bansa sa Timog-Silangang Asya na matatagpuan sa kanlurang Karagatang Pasipiko. Binubuo ito ng 7,641 mga pulo na nahahati sa tatlong pangunahing pangkat: Luzon, Visayas, at Mindanao. Ang kabisera nito ay Maynila, at ang pambansang wika ay Filipino kasama ang Ingles bilang opisyal na wika.",
    ),
    (
        "How many islands make up the Philippines?",
        "The Philippine archipelago consists of approximately 7,641 islands and islets, covering an estimated total land area of 300,000 square kilometers.",
    ),
    (
        "What are the three major island groups of the Philippines?",
        "The three major island groups of the Philippines are Luzon (the largest and northernmost island group), Visayas (the central cluster of islands), and Mindanao (the second-largest and southernmost group).",
    ),
    (
        "What is the capital of the Philippines?",
        "The capital of the Philippines is Manila, situated along the eastern shore of Manila Bay on the island of Luzon.",
    ),
    (
        "What is the national language of the Philippines?",
        "The national language of the Philippines is Filipino, which is standardized from Tagalog. Filipino and English serve as the official languages of communication and instruction.",
    ),
    (
        "Who is the national hero of the Philippines?",
        "Dr. Jose Rizal (Jose Protacio Rizal Mercado y Alonso Realonda) is the widely recognized national hero of the Philippines. He was a polymath, physician, and novelist whose writings, including 'Noli Me Tangere' and 'El Filibusterismo', ignited the Philippine Revolution against Spanish colonial rule.",
    ),
    (
        "When did the Philippines declare independence from Spain?",
        "The Philippines declared independence from Spanish colonial rule on June 12, 1898, in Kawit, Cavite, led by General Emilio Aguinaldo.",
    ),
    (
        "What is the highest mountain in the Philippines?",
        "Mount Apo is the highest mountain in the Philippines, with an elevation of 2,954 meters (9,692 feet) above sea level, located between Davao City, Davao del Sur, and Cotabato province on Mindanao.",
    ),
    (
        "What is Mayon Volcano and where is it located?",
        "Mayon Volcano is an active stratovolcano situated in the province of Albay in the Bicol Region of Luzon. It is internationally renowned for its near-perfect symmetric cone shape.",
    ),
    (
        "What is Taal Volcano?",
        "Taal Volcano is a caldera volcano in Batangas province, Luzon. It is famous for being one of the most active volcanoes in the Philippines and historically known for its island within a lake on an island.",
    ),
    (
        "What are the Chocolate Hills?",
        "The Chocolate Hills are a unique geological formation in Bohol province, Philippines, consisting of at least 1,260 symmetrical conical karst mounds that turn chocolate-brown during the dry season.",
    ),
    (
        "What is Tubbataha Reefs Natural Park?",
        "Tubbataha Reefs Natural Park is a protected marine sanctuary and UNESCO World Heritage site situated in the Sulu Sea near Palawan, famous for its exceptionally high marine biodiversity and pristine coral reefs.",
    ),
    (
        "What is the national bird of the Philippines?",
        "The national bird of the Philippines is the Philippine Eagle (Pithecophaga jefferyi), also known as the Monkey-eating Eagle, one of the largest and most powerful raptors in the world, endemic to Philippine rainforests.",
    ),
    (
        "What is the national flower of the Philippines?",
        "The national flower of the Philippines is the Sampaguita (Arabian Jasmine / Jasminum sambac), symbolizing purity, simplicity, humility, and strength.",
    ),
    (
        "What is the national tree of the Philippines?",
        "The national tree of the Philippines is the Narra (Pterocarpus indicus), prized for its durable reddish hardwood that symbolizes the resilience and strength of the Filipino people.",
    ),
    (
        "What is the currency of the Philippines?",
        "The official currency of the Philippines is the Philippine Peso (PHP), symbolized as ₱.",
    ),
    (
        "What form of government does the Philippines have?",
        "The Philippines has a unitary presidential constitutional republic. The government is divided into three co-equal branches: the Executive led by the President, the Legislative composed of the bicameral Congress (Senate and House of Representatives), and the Judicial headed by the Supreme Court.",
    ),
    (
        "What are the major regional languages spoken in the Philippines?",
        "Aside from Filipino and English, major regional languages include Cebuano, Ilocano, Hiligaynon (Ilonggo), Bicolano, Waray-Waray, Kapampangan, Pangasinan, Maranao, Maguindanaon, and Tausug.",
    ),
    (
        "Where is San Fernando, Pampanga located?",
        "The City of San Fernando is a component city and the provincial capital of Pampanga, as well as the regional administrative center of Central Luzon (Region III). It is famous for its Giant Lantern Festival ('Ligligan Parul').",
    ),
    (
        "What is the Giant Lantern Festival of San Fernando?",
        "The Giant Lantern Festival (Ligligan Parul) is an annual cultural festival held in December in the City of San Fernando, Pampanga, featuring gigantic lanterns (parol) up to twenty feet in diameter with intricate, synchronized kaleidoscopic light displays.",
    ),
    (
        "What is the significance of the EDSA People Power Revolution of 1986?",
        "The EDSA People Power Revolution was a nonviolent civilian demonstration in February 1986 along Epifanio de los Santos Avenue (EDSA) in Metro Manila that peacefully ousted authoritarian rule and restored democratic institutions in the Philippines.",
    ),
    (
        "What is the Barong Tagalog?",
        "The Barong Tagalog is the traditional national formal garment for men in the Philippines, an embroidered lightweight shirt typically woven from pina (pineapple fiber) or jusi (banana silk) worn untucked over an undershirt.",
    ),
    (
        "What is Bayanihan in Filipino culture?",
        "Bayanihan is a core Filipino cultural tradition representing communal unity and cooperation, originally exemplified by neighbors gathering together to physically lift and carry a bamboo house (nipa hut) to a new location.",
    ),
    (
        "What is the oldest university in Asia located in the Philippines?",
        "The University of Santo Tomas (UST) in Manila, founded by the Dominican Order in 1611, is the oldest existing university in Asia.",
    ),
    (
        "What is the significance of Intramuros in Manila?",
        "Intramuros, known as the 'Walled City', is the historic 0.67-square-kilometer fortified district in Manila that served as the seat of government during Spanish colonial rule from 1571 to 1898.",
    ),
    (
        "What is the Banaue Rice Terraces?",
        "The Banaue Rice Terraces are ancient 2,000-year-old stepped mountain agricultural fields hand-carved into the mountains of Ifugao in northern Luzon, showcasing indigenous engineering and sustainable farming.",
    ),
    (
        "What is the climate of the Philippines?",
        "The Philippines experiences a tropical maritime climate characterized by high temperatures, abundant rainfall, and high humidity, generally divided into the rainy season (June to November) and the dry season (December to May).",
    ),
    (
        "What is Sinulog Festival and where is it celebrated?",
        "The Sinulog Festival is a major religious and cultural dance festival held every third Sunday of January in Cebu City in honor of the Santo Nino (Holy Child Jesus).",
    ),
]


def get_philippines_pairs() -> list[tuple[str, str]]:
    """Return verified Philippine knowledge pairs."""
    return list(PHILIPPINES_PAIRS)
