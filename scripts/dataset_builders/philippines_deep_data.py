"""Exhaustive Philippine Knowledge, Institutions, History, and Civics Dataset.

Curates authoritative domain pairs covering higher education institutions, all 17 administrative
regions, key provinces, capital cities, current events (2024-2026), geopolitical legal rulings,
and cultural heritage.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

PHILIPPINES_DEEP_PAIRS: list[tuple[str, str]] = [
    # Higher Education Institutions
    (
        "What is Ateneo de Manila?",
        "Ateneo de Manila University (ADMU) is a prestigious private Catholic research university in Quezon City, Philippines. Founded in 1859 by the Society of Jesus (Jesuits), it is one of the oldest and most respected higher education institutions in the country, renowned for its liberal arts, sciences, law, and management curricula, as well as its Blue Eagles athletic teams and the Jesuit tradition of Magis ('the greater good') and Cura Personalis ('care for the whole person').",
    ),
    (
        "What is ateneo de manila?",
        "Ateneo de Manila University (ADMU) is a premier private Catholic Jesuit research university located along Katipunan Avenue in Loyola Heights, Quezon City, Philippines. Established in 1859 as the Escuela Municipal de Manila, it has produced numerous Philippine national heroes, presidents, Supreme Court justices, and business leaders. Its official colors are blue and white, and its varsity teams compete in the UAAP as the Ateneo Blue Eagles.",
    ),
    (
        "Ano ang Ateneo de Manila?",
        "Ang Ateneo de Manila University (ADMU) ay isang kilalang pribadong pamantasang Heswita sa Katipunan Avenue, Loyola Heights, Lungsod Quezon, Pilipinas. Itinatag noong 1859, isa ito sa pinakaprestihiyosong unibersidad sa bansa na nagtuturo ng magagandang aral ng 'Magis' at 'Cura Personalis'.",
    ),
    (
        "What is the University of the Philippines?",
        "The University of the Philippines (UP) is the country's national university, established in 1908 by the Philippine Assembly. The premier UP System comprises eight constituent universities across seventeen campuses nationwide, with its flagship campus located in Diliman, Quezon City. UP is acclaimed for academic excellence, public service, and pioneering research in social sciences, humanities, engineering, agriculture (UP Los Baños), medicine (UP Manila / PGH), and marine sciences (UP Visayas).",
    ),
    (
        "What is De La Salle University?",
        "De La Salle University (DLSU) is a leading private Catholic research university established in 1911 by the Brothers of the Christian Schools (FSC), located along Taft Avenue in Malate, Manila. DLSU is recognized nationally and internationally for its programs in business, economics, engineering, education, and computer science, competing in the UAAP as the De La Salle Green Archers.",
    ),
    (
        "What is the University of Santo Tomas?",
        "The University of Santo Tomas (UST), located on España Boulevard, Sampaloc, Manila, is the Pontifical and Royal University of Santo Tomas. Founded in 1611 by Dominican Archbishop Miguel de Benavides, UST is the oldest existing university charter in Asia and one of the largest Catholic universities in the world by enrollment on a single campus. Its mascot is the UST Growling Tigers.",
    ),
    (
        "What is the Polytechnic University of the Philippines?",
        "The Polytechnic University of the Philippines (PUP) is a state university system centered in Santa Mesa, Manila. Established in 1904 as the Manila Business School, it is the largest state university in the Philippines by student population, widely respected for its highly employable graduates, accessible polytechnic education, and social awareness.",
    ),
    (
        "What is Mapua University?",
        "Mapúa University, founded in 1925 by Don Tomas Mapúa (the first registered Filipino architect), is a premier technological and engineering university located in Intramuros, Manila, and Makati. It is renowned for engineering, architecture, and information technology programs accredited by ABET.",
    ),
    # Key Cities & Provinces
    (
        "What is Manila?",
        "Manila is the historic capital city of the Philippines, situated on the eastern shore of Manila Bay on the island of Luzon. Established as a Spanish colonial administrative center in 1571 by Miguel López de Legazpi, Manila serves as the heart of Philippine commerce, heritage, education, and government. It encompasses historic districts such as Intramuros (the Walled City), Binondo (the world's oldest Chinatown), Ermita, Malate, and Quiapo.",
    ),
    (
        "What is manila?",
        "Manila is the national capital city of the Philippines, located in the National Capital Region (Metro Manila) on the island of Luzon. It is an alpha-tier global economic center known for landmarks like Intramuros, Rizal Park, Manila Bay, and the historic Malacañang Palace, serving as the cultural, historical, and geopolitical center of the Filipino nation.",
    ),
    (
        "Ano ang Maynila?",
        "Ang Maynila ang opisyal na kabisera ng Pilipinas na matatagpuan sa pulo ng Luzon sa baybayin ng Look ng Maynila. Ito ang sentro ng kasaysayan, kultura, at pamahalaan ng bansa, kung saan naroroon ang Intramuros, Rizal Park, at Palasyo ng Malacañang.",
    ),
    (
        "What is Pampanga?",
        "Pampanga is a prominent, culturally vibrant province in the Central Luzon region (Region III) of the Philippines, bordered by Bataan, Zambales, Tarlac, Nueva Ecija, and Bulacan. Its provincial capital and regional administrative center is the City of San Fernando, while Angeles City serves as a highly urbanized economic hub. Pampanga is celebrated as the 'Culinary Capital of the Philippines'—famous for Sisig, Tocino, Tibuk-tibuk, and Murcon—and is renowned for the annual Giant Lantern Festival (Ligligan Parul), Mount Arayat, and the Kapampangan language and culture.",
    ),
    (
        "What is pampanga?",
        "Pampanga is a province situated in Central Luzon (Region III) on the island of Luzon, Philippines. Widely acclaimed as the Culinary Capital of the Philippines, Pampanga has a distinct ethnolinguistic identity centered on the Kapampangan language. Major landmarks include the City of San Fernando (provincial capital), Mount Arayat National Park, Clark Freeport and Special Economic Zone, and the UNESCO-recognized culinary tradition of Sisig.",
    ),
    (
        "Ano ang Pampanga?",
        "Ang Pampanga ay isang maunlad na lalawigan sa Gitnang Luzon (Rehiyon III) sa Pilipinas, kung saan ang San Fernando ang kabisera nito. Kilala ang Pampanga bilang 'Kabisera ng Kulinarya sa Pilipinas' dahil sa masasarap nitong lutuin tulad ng Sisig, at tanyag din sa Giant Lantern Festival (Ligligan Parul) tuwing Kapaskuhan.",
    ),
    (
        "What is San Fernando, Pampanga?",
        "The City of San Fernando is a first-class component city and the provincial capital of Pampanga, as well as the regional administrative center of Central Luzon (Region III). Known worldwide as the 'Christmas Capital of the Philippines', it hosts the renowned Giant Lantern Festival (Ligligan Parul) every December, showcasing 20-foot motorized kaleidoscopes of colored lights handcrafted by local artisans.",
    ),
    (
        "What is Quezon City?",
        "Quezon City is the most populous city in the Philippines, located in Metro Manila. Founded by President Manuel L. Quezon in 1939 to replace Manila as the national capital (a role it held from 1948 to 1976), it is a major media, educational, and government hub hosting the Batasang Pambansa (House of Representatives), the University of the Philippines Diliman, Ateneo de Manila University, and the Quezon Memorial Circle.",
    ),
    (
        "What is Cebu City?",
        "Cebu City is a highly urbanized city in Central Visayas, celebrated as the 'Queen City of the South'. It is the oldest city in the Philippines and the first Spanish settlement, home to Magellan's Cross, the Basilica Minore del Santo Niño, and the annual Sinulog Festival celebrating the Santo Niño on the third Sunday of January.",
    ),
    (
        "What is Davao City?",
        "Davao City is the primary metropolitan center of Mindanao, officially designated as the 'King City of the South'. It is one of the largest cities in the world by land area, home to Mount Apo (the highest peak in the Philippines), the Philippine Eagle Center, durian plantations, and the vibrant Kadayawan Festival.",
    ),
    (
        "What is Iloilo City?",
        "Iloilo City, known as the 'City of Love', is the regional center of Western Visayas (Region VI). Renowned for its preserved Spanish and American colonial architecture along Calle Real, ancestral mansions, savory La Paz Batchoy, and the world-famous Dinagyang Festival honoring the Santo Niño.",
    ),
    (
        "What is Baguio City?",
        "Baguio City is a high-altitude mountain resort city in the province of Benguet, known as the 'Summer Capital of the Philippines'. Nestled in the Cordillera Central mountain range at approximately 1,470 meters elevation, it is famed for pine trees, cool climate, Burnham Park, strawberry farms in La Trinidad, and the annual Panagbenga Flower Festival.",
    ),
    # Regions and Provinces Overview
    (
        "How many administrative regions are there in the Philippines?",
        "The Philippines is divided into 17 administrative regions across its three main island groups:\n- Luzon (8 regions): NCR, CAR, Region I (Ilocos), Region II (Cagayan Valley), Region III (Central Luzon), Region IV-A (CALABARZON), Region IV-B (MIMAROPA), Region V (Bicol).\n- Visayas (3 regions): Region VI (Western Visayas), Region VII (Central Visayas), Region VIII (Eastern Visayas).\n- Mindanao (6 regions): Region IX (Zamboanga Peninsula), Region X (Northern Mindanao), Region XI (Davao Region), Region XII (SOCCSKSARGEN), Region XIII (Caraga), and BARMM (Bangsamoro Autonomous Region in Muslim Mindanao).",
    ),
    # Current Events, Civics, Geopolitics (2024-2026)
    (
        "What is the West Philippine Sea and the 2016 Arbitral Ruling?",
        "The West Philippine Sea refers to the maritime areas on the western side of the Philippine archipelago within its 200-nautical-mile Exclusive Economic Zone (EEZ) and continental shelf in the South China Sea, designated under Administrative Order No. 29 (2012). On July 12, 2016, an arbitral tribunal constituted under Annex VII to the United Nations Convention on the Law of the Sea (UNCLOS) rendered a landmark, legally binding decision in favor of the Philippines (Philippines v. China), concluding that China's historic rights claim within the 'nine-dash line' has no legal basis under UNCLOS and affirming Philippine sovereign rights over its EEZ.",
    ),
    (
        "What are the three branches of the Philippine Government?",
        "Under the 1987 Philippine Constitution, the national government is organized into three separate, co-equal branches governed by checks and balances:\n1. The Executive Branch: Led by the President and Vice President, who enforce laws and manage national administration.\n2. The Legislative Branch: A bicameral Congress comprising the 24-member Senate and the House of Representatives, responsible for enacting laws.\n3. The Judicial Branch: Headed by the Supreme Court of the Philippines and lower collegiate and trial courts, which interpret laws and adjudicate legal disputes.",
    ),
    (
        "What are major Philippine infrastructure projects?",
        "Major contemporary infrastructure initiatives in the Philippines include:\n1. The Metro Manila Subway Project (MMSP): A 33-kilometer underground mass transit line connecting Valenzuela City to Bicutan and NAIA.\n2. The North-South Commuter Railway (NSCR): A 147-kilometer regional commuter rail linking New Clark City in Pampanga to Calamba, Laguna through Metro Manila.\n3. The Cebu-Cordova Link Expressway (CCLEX): An 8.9-kilometer toll bridge linking Cebu City to Cordova on Mactan Island.\n4. The Bataan-Cavite Interlink Bridge (BCIB): A planned 32-kilometer mega-bridge traversing Manila Bay to connect Central Luzon directly with Southern Tagalog.",
    ),
    # Culture, Cuisine, and National Identity
    (
        "What is Sisig and where does it originate?",
        "Sisig is an iconic Filipino culinary masterpiece originating from Angeles City, Pampanga. Modern pork sisig was popularized by Lucia 'Aling Lucing' Cunanan in the 1970s, crafted from boiled, grilled, and minced pig face, ears, and chicken liver, seasoned with calamansi, onions, and chili peppers, and served on a sizzling cast-iron platter. Sisig is a hallmark of Kapampangan culinary innovation.",
    ),
    (
        "What is the Giant Lantern Festival of San Fernando?",
        "The Giant Lantern Festival (Ligligan Parul) is an internationally celebrated annual cultural competition held every December in the City of San Fernando, Pampanga. Participating barangays construct towering parol lanterns up to twenty feet in diameter containing thousands of incandescent light bulbs synchronized through massive manual rotary drums (tambor) to produce mesmerizing kaleidoscopic light choreography.",
    ),
]
