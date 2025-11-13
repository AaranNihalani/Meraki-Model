import os, json, torch, numpy as np, pandas as pd, nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download NLTK sentence tokenizer
nltk.download("punkt")
nltk.download("punkt_tab")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_ROOT = "./models/meraki_sentence_tagger"

import torch
import os
from transformers.training_args import TrainingArguments

def get_best_checkpoint(model_dir="./models/meraki_sentence_tagger"):
    args_path = os.path.join(model_dir, "training_args.bin")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"training_args.bin not found in {model_dir}")

    # Allowlist TrainingArguments so torch.load can unpickle it
    torch.serialization.add_safe_globals([TrainingArguments])

    args = torch.load(args_path, map_location="cpu", weights_only=False)

    # New HF version
    if hasattr(args, "best_model_checkpoint") and args.best_model_checkpoint:
        return args.best_model_checkpoint

    # Older HF version
    if "best_model_checkpoint" in args.__dict__:
        return args.__dict__["best_model_checkpoint"]

    # Fallback: use last checkpoint
    print("⚠️ No best_model_checkpoint found, using last checkpoint…")

    ckpts = [
        os.path.join(model_dir, d)
        for d in os.listdir(model_dir)
        if d.startswith("checkpoint-")
    ]
    if not ckpts:
        raise ValueError("No checkpoint-* folders found inside the model directory.")

    ckpts_sorted = sorted(
        ckpts,
        key=lambda x: int(x.split("checkpoint-")[-1])
    )
    return ckpts_sorted[-1]


# ---------------------------------------------------------
# 2. Load id2tag
# ---------------------------------------------------------
id2tag = json.load(open("./models/id2tag.json"))
num_labels = len(id2tag)


# ---------------------------------------------------------
# 3. Load best checkpoint
# ---------------------------------------------------------
BEST_CKPT = get_best_checkpoint()
print(f"\n🔥 Loading best checkpoint: {BEST_CKPT}\n")

tokenizer = AutoTokenizer.from_pretrained(BEST_CKPT)
model_cls = AutoModelForSequenceClassification.from_pretrained(BEST_CKPT).to(DEVICE)

# Sanity check: real trained weights should have large values
print("Classifier weight sample:")
print(model_cls.classifier.weight[:5, :5])


# ---------------------------------------------------------
# 4. Prediction helpers
# ---------------------------------------------------------
def predict_tags(sentence, threshold=0.5):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True,
                       padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = model_cls(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return [id2tag[str(i)] for i, p in enumerate(probs) if p > threshold]


def tag_paragraph(text, threshold=0.5):
    sentences = nltk.sent_tokenize(text)
    results = []

    for sent in sentences:
        tags = predict_tags(sent, threshold)
        print(f"• {sent[:70]}... → {tags}")
        results.append({"Sentence": sent, "Tags": ", ".join(tags)})

    return pd.DataFrame(results)


# ---------------------------------------------------------
# 5. Example usage
# ---------------------------------------------------------
example_paragraph = """
The team distributed food and water in the southern camps.
Health clinics are overwhelmed by cholera cases.
Many families still lack access to education for their children.
"""

table = tag_paragraph(example_paragraph)
print("\nFinal table:\n", table)

table.to_csv("./tagged_output.csv", index=False)
print("\n✅ Tagged table saved to tagged_output.csv")
