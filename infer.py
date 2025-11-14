import os
import json
import torch
import numpy as np
import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.training_args import TrainingArguments

# ---------------------------------------------------------
# 1. Force CPU (avoid MPS inference issues)
# ---------------------------------------------------------
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# 2. Locate best checkpoint
# ---------------------------------------------------------
def get_best_checkpoint(model_dir="./models/meraki_sentence_tagger"):
    args_path = os.path.join(model_dir, "training_args.bin")
    torch.serialization.add_safe_globals([TrainingArguments])
    args = torch.load(args_path, map_location="cpu", weights_only=False)

    if getattr(args, "best_model_checkpoint", None):
        return args.best_model_checkpoint

    ckpts = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    ckpts = sorted(ckpts, key=lambda x: int(x.split("checkpoint-")[-1]))
    return ckpts[-1]


BEST_CKPT = get_best_checkpoint()
print(f"\n🔥 Loading best checkpoint: {BEST_CKPT}\n")


# ---------------------------------------------------------
# 3. Load model + tokenizer
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BEST_CKPT)
model = AutoModelForSequenceClassification.from_pretrained(BEST_CKPT).to(DEVICE)

id2tag = json.load(open("./models/id2tag.json"))


# ---------------------------------------------------------
# 4. Prediction
# ---------------------------------------------------------
def predict_tags(sentence, threshold=0.35):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return [id2tag[str(i)] for i, p in enumerate(probs) if p > threshold]


def tag_paragraph(text, threshold=0.35):
    sentences = nltk.sent_tokenize(text)
    results = []

    for sent in sentences:
        tags = predict_tags(sent, threshold)
        print(f"• {sent[:80]} → {tags}")
        results.append({"Sentence": sent, "Tags": ", ".join(tags)})

    return pd.DataFrame(results)


# ---------------------------------------------------------
# 5. Example usage
# ---------------------------------------------------------
example = """
The team distributed food and water in the southern camps.
Health clinics are overwhelmed by cholera cases.
Many families still lack access to education for their children.
I am really happy.
The new infrastructure has been really useful.
"""

df = tag_paragraph(example)
print("\nFinal table:\n", df)

df.to_csv("./tagged_output.csv", index=False)
print("\n✅ Saved to tagged_output.csv")
