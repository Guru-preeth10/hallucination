from SPARQLWrapper import SPARQLWrapper, JSON
import csv
import re
from datetime import datetime

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

queries = {
    "War": """
    SELECT ?event ?eventLabel ?date ?sitelinks WHERE {
      ?event wdt:P31 wd:Q198.
      ?event wdt:P580 ?date.
      ?event wikibase:sitelinks ?sitelinks.
      FILTER(?sitelinks > 20)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ORDER BY DESC(?sitelinks)
    LIMIT 50
    """,
    "Battle": """
    SELECT ?event ?eventLabel ?date ?sitelinks WHERE {
      ?event wdt:P31 wd:Q178561.
      ?event wdt:P585 ?date.
      ?event wikibase:sitelinks ?sitelinks.
      FILTER(?sitelinks > 20)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ORDER BY DESC(?sitelinks)
    LIMIT 50
    """
}

def normalize_date(date_str):
    # Handle BC dates (negative years from Wikidata)
    if date_str.startswith("-"):
        match = re.match(r"-0*(\d+)-(\d{2})-(\d{2})", date_str)
        if match:
            year, month, day = match.groups()
            year = int(year)
            # If month/day = 01/01 → only return "YEAR BC"
            if month == "01" and day == "01":
                return f"{year} BC"
            # Otherwise return full date
            try:
                month_name = datetime(2000, int(month), int(day)).strftime("%B")
                return f"{int(day)} {month_name} {year} BC"
            except:
                return f"{year} BC"
        return date_str
    
    # Handle AD dates normally
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", ""))
        year = dt.year
        if dt.month == 1 and dt.day == 1:
            return str(year)
        return dt.strftime("%d %B %Y")
    except Exception:
        return date_str


events = []
for category, query in queries.items():
    sparql.setQuery(query)
    results = sparql.query().convert()
    count = 0
    for result in results["results"]["bindings"]:
        label = result["eventLabel"]["value"]
        date_raw = result["date"]["value"]
        date = normalize_date(date_raw)
        events.append([category, label, date])
        count += 1
    print(f"[{category}] Fetched {count} events")

with open("famous_events.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Event", "Date"])
    writer.writerows(events)

print(f"✅ Saved {len(events)} events to famous_events.csv")
