import os, json, pandas as pd

def find_input():
    cands = [
        "./data/unclustered_tagged_sentences.xlsx",
        "./data/labeled/unclustered_tagged_sentences.xlsx",
        "./data/raw_tagged_sentences.xlsx",
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No input Excel found for unclustered tagged sentences")

def load_excel(path):
    df = pd.read_excel(path, header=None)
    if len(df.columns) < 2:
        raise ValueError("Expected at least 2 columns: [Tag, Sentence]")
    df = df.iloc[:, :2]
    df.columns = ["sentence", "tag"]
    df["tag"] = df["tag"].astype(str).str.strip().str.lower()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    return df

def to_records(df):
    grouped = (
        df.groupby("sentence")["tag"].apply(lambda s: sorted(set(t for t in s if t))).reset_index()
    )
    return [{"sentence": r["sentence"], "tags": r["tag"]} for _, r in grouped.iterrows()]

def main():
    in_path = find_input()
    out_path = "./data/labeled/unclustered_tagged_sentences.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = load_excel(in_path)
    recs = to_records(df)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(recs)} records to {out_path}")

if __name__ == "__main__":
    main()