from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
from collections import defaultdict
import spacy

# ------------------------
# 1. Setup
# ------------------------
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

# ------------------------
# 2. Templates
# ------------------------
TEMPLATES = {
    "human": {
        "date of birth": "When was {entity} born?",
        "place of birth": "Where was {entity} born?",
        "award received": "What awards did {entity} receive?",
        "occupation": "What is the occupation of {entity}?",
        "child": "Who are the children of {entity}?",
        "spouse": "Who is the spouse of {entity}?"
    },
    "city": {
        "country": "Which country is {entity} in?",
        "population": "What is the population of {entity}?",
        "inception": "When was {entity} founded?",
        "located in the administrative territorial entity": "Where is {entity} located?"
    },
    "organization": {
        "founder": "Who founded {entity}?",
        "inception": "When was {entity} founded?",
        "headquarters location": "Where is the headquarters of {entity}?",
        "industry": "What industry does {entity} belong to?"
    }
}

# ------------------------
# 3. Entities
# ------------------------
ENTITIES = [
    ("Q937", "Albert Einstein"),        # Human
    ("Q42", "Douglas Adams"),           # Human (writer)
    ("Q76", "Barack Obama"),            # Human (politician)
    ("Q90", "Paris"),                   # City
    ("Q64", "Berlin"),                  # City
    ("Q60", "New York City"),           # City
    ("Q95", "IBM"),                     # Organization
    ("Q312", "Microsoft"),              # Organization
    ("Q16917", "Google"),               # Organization
    ("Q49108", "World Health Organization") # Organization
]

# ------------------------
# 4. Get entity type
# ------------------------
def get_entity_type(qid):
    query = f"""
    SELECT ?typeLabel WHERE {{
      wd:{qid} wdt:P31 ?type .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results["results"]["bindings"]:
        return results["results"]["bindings"][0]["typeLabel"]["value"].lower()
    return "unknown"

# ------------------------
# 5. Get properties + values
# ------------------------
def get_properties(qid):
    query = f"""
    SELECT ?propertyLabel ?valueLabel
    WHERE {{
      wd:{qid} ?prop ?value .
      ?property wikibase:directClaim ?prop .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 100
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    props = defaultdict(list)
    for r in results["results"]["bindings"]:
        prop = r["propertyLabel"]["value"]
        val = r["valueLabel"]["value"]
        props[prop].append(val)
    return props

# ------------------------
# 6. Generate Q–A pairs
# ------------------------
def generate_qa(qid, name):
    etype = get_entity_type(qid)
    props = get_properties(qid)
    qa_data = []

    for prop, vals in props.items():
        val = ", ".join(vals)  # aggregate multi-values
        template_dict = TEMPLATES.get(etype, {})
        if prop.lower() in template_dict:
            q = template_dict[prop.lower()].format(entity=name)
        else:
            q = f"What is the {prop.lower()} of {name}?"
        qa_data.append({
            "entity": name,
            "entity_type": etype,
            "property": prop,
            "question": q,
            "answer": val
        })

    return qa_data

# ------------------------
# 7. Run for all entities
# ------------------------
all_data = []
for qid, name in ENTITIES:
    try:
        qa = generate_qa(qid, name)
        all_data.extend(qa)
        print(f"✅ Processed {name}")
    except Exception as e:
        print(f"❌ Failed {name}: {e}")

df = pd.DataFrame(all_data)

def is_clean_answer(text):
    # Drop URLs
    if text.startswith("http"):
        return False
    # Drop long numeric answers
    if text.isdigit():
        return False
    # Drop overly long weird strings
    if len(text) > 200:
        return False
    return True


# ------------------------
# 8. NER filter to remove junk answers
# ------------------------
nlp = spacy.load("en_core_web_sm")

def is_named_entity(text):
    doc = nlp(text)
    return any(ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "NORP"] for ent in doc.ents)

df = df[df["answer"].apply(lambda x: is_clean_answer(x) and is_named_entity(x))]


# ------------------------
# 9. Save
# ------------------------
df.to_csv("wikidata_generalized.csv", index=False)
print("✅ Saved all entities to wikidata_generalized.csv")
