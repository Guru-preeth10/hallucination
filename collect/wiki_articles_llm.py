import wikipediaapi
import re
import pandas as pd
import ollama
import time
import requests

# -------------------------------
# 1. Setup
# -------------------------------
def generate_questions(context, n=2):
    prompt = f"""
    Context: {context}

    Task:
    Generate {n} clear, fact-based questions and their exact short answers.
    Each answer must compulsorily be a named entity (person, place, date, organization, or event).
    Avoid vague or generic questions. Your answer should not exceed more than 3 words.
    Each question should be specific and context-rich, referencing a unique fact from the context.

    Format:
    Q: <question>
    A: <answer>
    """

    resp = ollama.chat(model="gpt-oss:20b", messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]

def parse_output(output):
    qa_pairs = []
    lines = output.split("\n")
    q, a = None, None
    for line in lines:
        if line.strip().startswith("Q:"):
            q = line.replace("Q:", "").strip()
        elif line.strip().startswith("A:"):
            a = line.replace("A:", "").strip()
            if q and a:
                qa_pairs.append({"question": q, "answer": a})
                q, a = None, None
    return qa_pairs

# -------------------------------
# 2. Wikipedia Setup
# -------------------------------
wiki_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="WikiQABot/1.0 (https://github.com/kirtan-bhojani; kirtan.bhojani@gmail.com)"
)

topics = ['Architecture', 'Art history', 'Renaissance architecture', 'Modern architecture', 'Baroque architecture', 'Neoclassical architecture', 'Gothic architecture', 'Le Corbusier', 'Frank Lloyd Wright', 'Zaha Hadid', 'Bauhaus', 'Impressionism', 'Abstract expressionism', 'Cubism', 'Eiffel Tower', 'Post-Impressionism', 'Rococo', 'Art Deco', 'Pop Art', 'Renaissance', 'Neoclassicism', 'Expressionism', 'Dada', 'Surrealism', 'Romanticism', 'Film', 'Hollywood', 'Bollywood', 'Academy Awards', 'Alfred Hitchcock', 'Steven Spielberg', 'Star Wars', 'The Godfather', 'Animation', 'Cinematography', 'Charlie Chaplin', 'Martin Scorsese', 'Akira Kurosawa', 'Orson Welles', 'Stanley Kubrick', 'Federico Fellini', 'Pedro Almodóvar', 'Satyajit Ray', 'Andrei Tarkovsky', 'Wim Wenders', 'Yasujiro Ozu', 'Luis Buñuel', 'Jean-Luc Godard', 'Quentin Tarantino', 'Hayao Miyazaki', 'Economics', 'Capitalism', 'Stock market', 'Great Depression', 'International Monetary Fund', 'World Bank', 'Adam Smith', 'John Maynard Keynes', 'Supply and demand', 'Gross domestic product', 'Federal Reserve', 'Behavioral economics', 'Fiscal policy', 'Recession', 'Globalization', 'Inflation', 'Interest rate', 'Consumer price index', 'Supply chain', 'Microeconomics', 'Macroeconomics', 'Bubble (economics)', 'Foreign direct investment', 'Financial crisis', 'Monetary policy', 'Climate change', 'Global warming', 'Paris Agreement', 'Renewable energy', 'Deforestation', 'Biodiversity', 'Greenhouse gas', 'United Nations Environment Programme', 'Greta Thunberg', 'Ozone depletion', 'Carbon footprint', 'Recycling', 'Ocean acidification', 'Sustainable development', 'Climate change adaptation', 'Renewable energy transition', 'Deforestation in the Amazon', 'Biodiversity loss', 'Climate policy', 'Solar power', 'Wind power', 'Hydropower', 'Geothermal energy', 'Fossil fuel', 'Carbon cycle', 'Albert Einstein', 'Leonardo da Vinci', 'Marie Curie', 'Nelson Mandela', 'Martin Luther King Jr.', 'Mahatma Gandhi', 'Queen Elizabeth II', 'William Shakespeare', 'Isaac Newton', 'Cleopatra', 'Abraham Lincoln', 'Napoleon Bonaparte', 'Winston Churchill', 'Franklin D. Roosevelt', 'George Washington', 'Alexander the Great', 'Julius Caesar', 'Mother Teresa', 'Confucius', 'Aristotle', 'Galileo Galilei', 'Charles Darwin', 'Wolfgang Amadeus Mozart', 'Ludwig van Beethoven', 'Vincent van Gogh', 'Pablo Picasso', 'Michelangelo', 'Johann Sebastian Bach', 'Karl Marx', 'Adam Smith', 'Sigmund Freud', 'Plato', 'Joan of Arc', 'Genghis Khan', 'Queen Victoria', 'Buddha', 'Jesus', 'Muhammad', 'Elvis Presley', 'Michael Jackson', 'Stephen Hawking', 'Bill Gates', 'Steve Jobs', 'Walt Disney', 'Marco Polo', 'Christopher Columbus', 'Muhammad Ali', 'Jackie Robinson', 'Florence Nightingale', 'Indira Gandhi', 'Neil Armstrong', 'Diwali', 'Holi', 'Christmas', 'Oktoberfest', 'Carnival', 'Chinese New Year', 'Eid al-Fitr', 'Ramadan', 'Halloween', 'Easter', 'Mardi Gras', 'Carnival of Venice', 'La Tomatina', 'Burning Man', 'Songkran', 'Hanukkah', 'Kwanzaa', 'Passover', 'Juneteenth', 'Woodstock', 'Glastonbury Festival', 'Diá de Muertos', 'Thanksgiving', 'Fez Festival of World Sacred Music', 'Geography', 'Mount Everest', 'Amazon River', 'Sahara', 'Great Wall of China', 'Pacific Ocean', 'Nile', 'Antarctica', 'Arctic Circle', 'Equator', 'Mount Kilimanjaro', 'The Andes', 'The Gobi Desert', 'The Himalayas', 'Niagara Falls', 'The Serengeti', 'The Mariana Trench', 'Volcano', 'Coral reef', 'Desert', 'Ocean', 'Rainforest', 'Glacier', 'Lake Victoria', 'Mount Aconcagua', 'Indian classical music', 'Raga', 'Tala (music)', 'Hindustani classical music', 'Carnatic music', 'Bhairavi', 'Thaat', 'Sitar', 'Tabla', 'Thumri', 'Veena', 'Sarangi', 'Bansuri', 'Sarod', 'Dhrupad', 'Tanpura', 'Vocal music', 'Ravi Shankar', 'Ustad Amjad Ali Khan', 'Pandit Jasraj', 'Bhimsen Joshi', 'M. S. Subbulakshmi', 'Zakir Hussain', 'Folk music of India', 'Guru-shishya tradition', 'Timeline of scientific discoveries', 'Invention', 'Printing press', 'Wheel', 'Anesthesia', 'Light bulb', 'Telephone', 'Computer', 'Transistor', 'Internet', 'Telegraph', 'Automobile', 'Television', 'Jet engine', 'Radio', 'Steam engine', 'Vaccine', 'X-ray', 'Polio vaccine', 'DNA', 'Electromagnetism', 'Theory of relativity', 'Quantum mechanics', 'Telescope', 'Antibiotic', 'Literature', 'Jane Austen', 'Leo Tolstoy', 'George Orwell', 'Homer', 'To Kill a Mockingbird', '1984 (novel)', 'The Lord of the Rings', 'Pride and Prejudice', 'J. K. Rowling', 'Charles Dickens', 'Ernest Hemingway', 'Mark Twain', 'William Faulkner', 'Virginia Woolf', 'Gabriel García Márquez', 'J.D. Salinger', 'Herman Melville', 'Mary Shelley', 'Franz Kafka', 'T. S. Eliot', 'J. R. R. Tolkien', 'Fyodor Dostoevsky', 'Edgar Allan Poe', 'F. Scott Fitzgerald', 'Medicine', 'World Health Organization', 'DNA', 'Vaccine', 'Penicillin', 'Louis Pasteur', 'Florence Nightingale', 'Human anatomy', 'Cancer', 'Pandemic', 'Anatomy', 'Virology', 'Immunology', 'Genetics', 'Epidemiology', 'Neuroscience', 'Public health', 'Pharmacology', 'Microbiology', 'Cardiology', 'Oncology', 'Neurology', 'Psychiatry', 'Pathology', 'Anesthesia', 'Astronomy', 'NASA', 'Apollo 11', 'International Space Station', 'Hubble Space Telescope', 'Comet', 'Milky Way', 'Mars', 'Stephen Hawking', 'Carl Sagan', 'Jupiter', 'Saturn', 'Venus', 'Neptune', 'Pluto', 'James Webb Space Telescope', 'Proxima Centauri', 'The Sun', 'Moon landing', 'Big Bang', 'Galaxies', 'Solar System', 'Supernova', 'Black hole', 'Kuiper Belt', 'FIFA World Cup', 'Olympic Games', 'Association football', 'Cricket', 'Basketball', 'American football', 'Baseball', 'Tennis', 'Rugby', 'Sports', 'Michael Jordan', 'LeBron James', 'Lionel Messi', 'Serena Williams', 'Roger Federer', 'Tiger Woods', 'Usain Bolt', 'Babe Ruth', 'Pelé', 'Muhammad Ali', 'Michael Phelps', 'Jesse Owens', 'Jackie Robinson', 'Wayne Gretzky', 'Michael Schumacher', 'Internet', 'Cybernetics', 'Google', 'Social media', 'Artificial intelligence', 'Smartphone', 'Social networking service', 'Cybersecurity', 'Cloud computing', 'Cryptography', 'World Wide Web', 'Python (programming language)', 'Linux', 'JavaScript', 'Data science', 'Search engine optimization', 'Artificial neural network', 'Machine learning', 'E-commerce', 'Open-source software', 'Computer programming', 'Operating system', 'Mobile app', 'Computer virus', 'Internet of things', 'History of the world', 'Ancient Egypt', 'Roman Empire', 'World War I', 'World War II', 'Cold War', 'Industrial Revolution', 'Renaissance', 'Julius Caesar', 'Genghis Khan', 'Feudalism', 'French Revolution', 'Byzantine Empire', 'Ottoman Empire', 'Crusades', 'Fall of the Berlin Wall', 'Roman Republic', 'Ancient Greece', 'Stone Age', 'The Silk Road', 'Age of Discovery', 'Colonialism', 'Iron Age', 'Bronze Age', 'The Great Wall of China', 'Physics', 'Isaac Newton', 'Quantum mechanics', 'Albert Einstein', 'Stephen Hawking', 'Richard Feynman', 'Nikola Tesla', 'Galileo Galilei', 'Max Planck', 'J. Robert Oppenheimer', 'Niels Bohr', 'Erwin Schrödinger', 'Paul Dirac', 'Werner Heisenberg', 'James Clerk Maxwell', 'Marie Curie', 'Michael Faraday', 'Johannes Kepler', 'Thermodynamics', 'Electromagnetism', 'General relativity', 'Classical mechanics', 'Particle physics', 'String theory', 'Quantum field theory', 'Chemistry', 'Periodic table', 'Atom', 'Molecule', 'Chemical reaction', 'Organic chemistry', 'Inorganic chemistry', 'Nuclear chemistry', 'Thermodynamics', 'Chemical bond', 'Biochemistry', 'Polymer chemistry', 'Physical chemistry', 'Environmental chemistry', 'Materials science', 'Analytical chemistry', 'Organic compound', 'Biomolecule', 'Water (molecule)', 'Spectroscopy', 'Stoichiometry', 'Quantum chemistry', 'Catalysis', 'Distillation', 'Sub-discipline of chemistry', 'Biology', 'Cell (biology)', 'DNA', 'Evolution', 'Ecology', 'Genetics', 'Photosynthesis', 'Microbiology', 'Biotechnology', 'Organism', 'Human body', 'Cellular respiration', 'Metabolism', 'Homeostasis', 'Biome', 'Protein', 'Natural selection', 'Botany', 'Zoology', 'Ecosystem', 'Virology', 'Immunology', 'Anatomy', 'Physiology', 'Gene', 'Mathematics', 'Algebra', 'Geometry', 'Calculus', 'Fractal', 'Chaos theory', 'Logic', 'Combinatorics', 'Applied mathematics', 'Discrete mathematics', 'Linear algebra', 'Topology', 'Probability theory', 'Functional analysis', 'Number theory', 'Optimization', 'Mathematical physics', 'Cryptography', 'Mathematical logic', 'Set theory', 'Group theory', 'Complex analysis', 'Differential equations', 'Game theory', 'Statistics']  # your long list of topics

# -------------------------------
# 3. Load existing CSV to resume
# -------------------------------
try:
    existing_df = pd.read_csv("wiki_questions_ollama2.csv")
    processed_topics = set(existing_df["topic"].unique())
except FileNotFoundError:
    existing_df = pd.DataFrame()
    processed_topics = set()

data = existing_df.to_dict(orient="records")

# -------------------------------
# 4. Loop over topics with retry & timeout
# -------------------------------
for topic in topics:
    if topic in processed_topics:
        print(f"Skipping already processed topic: {topic}")
        continue

    success = False
    for attempt in range(3):  # retry up to 3 times
        try:
            page = wiki_wiki.page(topic)
            if not page.exists():
                print(f"Topic does not exist: {topic}")
                success = True
                break

            # Take first 5 paragraphs for richer facts
            text = " ".join(page.text.split("\n")[:5])
            text = re.sub(r"\[[0-9]+\]", "", text)  # remove citations

            output = generate_questions(text, n=2)
            qa_pairs = parse_output(output)

            for qa in qa_pairs:
                data.append({"topic": topic, "question": qa["question"], "answer": qa["answer"]})

            success = True
            print(f"✅ Processed topic: {topic}")
            break
        except (requests.exceptions.RequestException, TimeoutError) as e:
            print(f"Attempt {attempt+1} failed for topic: {topic} | Error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"Skipping topic: {topic} | Unexpected error: {e}")
            break

    if not success:
        print(f"⚠ Skipping topic after 3 failed attempts: {topic}")

# -------------------------------
# 5. Save dataset
# -------------------------------
df = pd.DataFrame(data)
df = df.drop_duplicates(subset=["question"])
df.to_csv("wiki_questions_ollama2.csv", index=False)
print("✅ Dataset saved as wiki_questions_ollama2.csv")

