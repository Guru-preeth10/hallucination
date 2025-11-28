import pandas as pd
import random
import re
from dateutil import parser

# ----------------------
# Load dataset
# ----------------------
df = pd.read_csv("famous_events.csv")  # columns: Category, Event, Date

# ----------------------
# Parse Date into sortable numeric timeline
# ----------------------
def parse_date(date_str):
    """
    Converts a date string into a sortable integer YYYYMMDD.
    BC dates stored as negative YYYYMMDD.
    """
    if pd.isna(date_str):
        return 99991231  # fallback for missing

    date_str = str(date_str).strip()

    # Handle BC dates
    if "BC" in date_str:
        # Extract year from string
        match = re.search(r"(\d+)\s*BC", date_str)
        year = -int(match.group(1)) if match else -1

        # Try to parse month/day
        try:
            dt = parser.parse(date_str.replace("BC", "").strip(), fuzzy=True, dayfirst=True)
            month = dt.month
            day = dt.day
        except:
            month = 1
            day = 1

        return year*10000 + month*100 + day

    # Handle AD / normal dates
    try:
        dt = parser.parse(date_str, fuzzy=True, dayfirst=True)
        return dt.year*10000 + dt.month*100 + dt.day
    except:
        # fallback: take 4-digit year
        match = re.search(r"(\d{1,4})", date_str)
        if match:
            return int(match.group(1))*10000
        return 99991231

# Add sortable column
df["timeline"] = df["Date"].apply(parse_date)

# Drop rows where parsing failed
df = df.dropna(subset=["timeline"])

# ----------------------
# Generate questions
# ----------------------
questions = []
num_questions = 1000
events_per_question = 5

for _ in range(num_questions):
    # Sample 5 events
    sample = df.sample(events_per_question).copy()

    # Shuffle for question
    shuffled = sample.sample(frac=1)
    question_events = shuffled["Event"].tolist()
    q_text = "Arrange the following events in chronological order:\n" + "\n".join(question_events)

    # Sort by timeline for answer
    sorted_events = sample.sort_values("timeline")
    answer = " → ".join([f"{row['Event']} ({row['Date']})" for _, row in sorted_events.iterrows()])

    questions.append({"question": q_text, "answer": answer})

# ----------------------
# Save to CSV
# ----------------------
qa_df = pd.DataFrame(questions)
qa_df.to_csv("chrono_questions.csv", index=False, encoding="utf-8")

print("✅ 1000 questions generated and saved to chrono_questions.csv")
