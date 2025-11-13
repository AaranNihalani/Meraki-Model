"""
prepare_tagged_sentences.py
---------------------------
Reads a raw Excel file where each row has:
    Tag | Sentence
and merges duplicate sentences with multiple tags.

Outputs:
    data/labeled/unclustered_tagged_sentences.xlsx
with columns:
    sentence | tags  (comma-separated or list-style)
"""

import os
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

input_path = "./data/raw_unclustered_tagged_sentences.xlsx"     # <-- path to your input file
output_path = "./data/labeled/unclustered_tagged_sentences.xlsx"

# Create output folder if needed
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

print(f"📘 Loading file: {input_path}")
df = pd.read_excel(input_path, header=None)

# Try to detect columns automatically
if len(df.columns) < 2:
    raise ValueError("Expected at least 2 columns: [Tag, Sentence]")

df.columns = ["tag", "sentence"]

# Normalize text
df["tag"] = df["tag"].astype(str).str.strip().str.lower()
df["sentence"] = df["sentence"].astype(str).str.strip()

print(f"✅ Loaded {len(df)} rows.")

# ============================================================
# 2. GROUP BY SENTENCE AND COLLECT TAGS
# ============================================================

grouped = (
    df.groupby("sentence")["tag"]
    .apply(lambda tags: sorted(set(tags)))   # unique tags, sorted alphabetically
    .reset_index()
)

# Create comma-separated and list-like columns
grouped["tags_str"] = grouped["tag"].apply(lambda tlist: ", ".join(tlist))
grouped["tags_list"] = grouped["tag"].apply(lambda tlist: str(tlist))

# ============================================================
# 3. SAVE OUTPUT
# ============================================================

# Choose which form you prefer for training — string or list
output_df = grouped[["sentence", "tags_str"]].rename(columns={"tags_str": "tags"})

output_df.to_excel(output_path, index=False)
print(f"💾 Saved merged file: {output_path}")
print(f"📊 Sentences: {len(output_df)} (unique)")
print(f"Example rows:\n{output_df.head(5)}")
