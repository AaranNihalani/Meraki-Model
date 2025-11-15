import json
import re

INPUT = "./data/labeled/tagged_sentences.json"
OUTPUT = "./data/labeled/tagged_sentences.json"

data = json.load(open(INPUT))

cleaned = []
for r in data:
    raw_tags = r["tags"]

    # raw_tags is ALWAYS a list of length 1
    # containing a comma-separated string
    if len(raw_tags) != 1:
        print("Warning: unexpected tags format:", raw_tags)

    tag_string = raw_tags[0]

    # Split on commas
    split_tags = [t.strip() for t in tag_string.split(",")]

    # Remove empty entries
    split_tags = [t for t in split_tags if t]

    cleaned.append({
        "sentence": r["sentence"],
        "tags": split_tags
    })

json.dump(cleaned, open(OUTPUT, "w"), indent=2)
print("✅ Cleaned dataset saved to", OUTPUT)
