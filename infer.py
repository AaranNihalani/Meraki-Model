import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.training_args import TrainingArguments

# ---------------------------------------------------------
# 0. DEVICE SELECTION (Safe & Automatic)
# ---------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"🔥 Using device: {DEVICE}")


# ---------------------------------------------------------
# 1. Locate best checkpoint
# ---------------------------------------------------------
def get_best_checkpoint(model_dir="./models/meraki_sentence_tagger"):

    args_path = os.path.join(model_dir, "training_args.bin")
    torch.serialization.add_safe_globals([TrainingArguments])

    if os.path.exists(args_path):
        args = torch.load(args_path, map_location="cpu", weights_only=False)
        if getattr(args, "best_model_checkpoint", None):
            return args.best_model_checkpoint

    # fallback: highest checkpoint number
    ckpts = [
        os.path.join(model_dir, d)
        for d in os.listdir(model_dir)
        if d.startswith("checkpoint-")
    ]
    if not ckpts:
        raise FileNotFoundError("❌ No checkpoints found in model directory.")

    ckpts = sorted(ckpts, key=lambda x: int(x.split("checkpoint-")[-1]))
    return ckpts[-1]


BEST_CKPT = get_best_checkpoint()
print(f"\n🔥 Loading best checkpoint: {BEST_CKPT}\n")


# ---------------------------------------------------------
# 2. Load classifier + tokenizer
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BEST_CKPT)
model = AutoModelForSequenceClassification.from_pretrained(BEST_CKPT)
model.to(DEVICE)
model.eval()

# Load tag mapping
id2tag = json.load(open("./models/id2tag.json"))
num_labels = len(id2tag)


# ---------------------------------------------------------
# 3. Safe sentence splitter
# ---------------------------------------------------------
def split_sentences(text):
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        return nltk.sent_tokenize(text)
    except Exception:
        # fallback — simple split
        return [s.strip() for s in text.split(".") if s.strip()]


# ---------------------------------------------------------
# 4. Predict tags for one or many sentences (fast batching)
# ---------------------------------------------------------
def predict_batch(sentences, threshold=0.35, debug=False):
    enc = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()

    if debug:
        print("\n=== DEBUG: Raw Logits ===")
        print(logits.cpu())

    outputs = []
    for sent, vec in zip(sentences, probs):
        tags = [id2tag[str(i)] for i, p in enumerate(vec) if p > threshold]
        outputs.append((sent, tags))

    return outputs


# ---------------------------------------------------------
# 5. Tag a paragraph
# ---------------------------------------------------------
def tag_paragraph(text, threshold=0.35, debug=False):

    sentences = split_sentences(text)
    results = predict_batch(sentences, threshold, debug=debug)

    rows = []
    for sent, tags in results:
        print(f"• {sent[:80]} → {tags}")
        rows.append({"Sentence": sent, "Tags": ", ".join(tags)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 6. Example
# ---------------------------------------------------------
if __name__ == "__main__":

    example = """
    The team distributed food and water in the southern camps.
    Health clinics are overwhelmed by cholera cases.
    Many families still lack access to education for their children.
    I am really happy.
    The new infrastructure has been really useful.
    """

    df = tag_paragraph(example, threshold=0.35, debug=False)
    print("\nFinal table:\n", df)

    df.to_csv("./tagged_output.csv", index=False)
    print("\n✅ Saved to tagged_output.csv")
