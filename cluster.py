import os, json, numpy as np, torch, re
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cpu")
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
def load_unclustered(path="./data/labeled/unclustered_tagged_sentences.json"):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing unclustered JSON. Run convert_unclustered.py first.")
    return json.load(open(path))

def sentences_for_tag(records):
    by_tag = {}
    for r in records:
        for t in r.get("tags", []):
            by_tag.setdefault(t, []).append(r["sentence"])
    return by_tag

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-9)
    return s / d

def embed_sentences(texts, tokenizer, model, batch_size=64):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            pooled = mean_pool(out.last_hidden_state, inputs["attention_mask"]).cpu().numpy()
        vecs.append(pooled)
    if not vecs:
        return np.zeros((0, model.config.hidden_size))
    return np.vstack(vecs)

def build_tag_embeddings(by_tag, tokenizer, model, per_tag_limit=50):
    tags = sorted(by_tag.keys())
    embs = []
    for t in tags:
        sents = by_tag[t][:per_tag_limit]
        if not sents:
            embs.append(np.zeros(model.config.hidden_size))
            continue
        v = embed_sentences(sents, tokenizer, model)
        embs.append(v.mean(axis=0))
    return tags, np.vstack(embs)

def cluster_tags(tags, embeddings, k=50, seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(embeddings)
    return labels

def _make_readable(tag):
    # Convert kebab/snake/camelCase to spaces and title-case
    s = re.sub(r'[-_]', ' ', tag)
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return ' '.join(w.capitalize() for w in s.split())

def label_clusters(tags, labels, by_tag):
    inv = {}
    for t, y in zip(tags, labels):
        inv.setdefault(y, []).append(t)
    mapping = {}
    for y, members in inv.items():
        # Pick the member with the most sentences as canonical
        canonical = max(members, key=lambda m: len(by_tag.get(m, [])))
        readable = _make_readable(canonical)
        for m in members:
            mapping[m] = readable
    return mapping

def remap_sentences(records, mapping):
    out = []
    for r in records:
        new_tags = sorted(set(mapping.get(t, t) for t in r.get("tags", [])))
        out.append({"sentence": r["sentence"], "tags": new_tags})
    return out

def main():
    in_path = "./data/labeled/unclustered_tagged_sentences.json"
    out_map = "./data/labeled/tag_cluster_map.json"
    out_tagged = "./data/labeled/tagged_sentences.json"
    os.makedirs(os.path.dirname(out_map), exist_ok=True)
    recs = load_unclustered(in_path)
    by_tag = sentences_for_tag(recs)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL).to(DEVICE)
    tags, embs = build_tag_embeddings(by_tag, tokenizer, model)
    labels = cluster_tags(tags, embs, k=50)
    mapping = label_clusters(tags, labels, by_tag)
    json.dump(mapping, open(out_map, "w"), ensure_ascii=False, indent=2)
    remapped = remap_sentences(recs, mapping)
    json.dump(remapped, open(out_tagged, "w"), ensure_ascii=False, indent=2)
    print(f"Clusters: {len(set(labels))}, tags: {len(tags)}, sentences: {len(remapped)}")
    print(f"Saved {out_map} and {out_tagged}")

if __name__ == "__main__":
    main()