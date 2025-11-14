import os
import json
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# =========================================================
# 1. FORCE CPU (disable MPS completely)
# =========================================================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

DEVICE = torch.device("cpu")
print("🔥 Training on:", DEVICE)


# =========================================================
# 2. Load Data
# =========================================================
records = json.load(open("./data/labeled/tagged_sentences.json"))

tags = sorted({t for r in records for t in r["tags"]})
tag2id = {t: i for i, t in enumerate(tags)}
id2tag = {i: t for t, i in tag2id.items()}

json.dump(id2tag, open("./models/id2tag.json", "w"), indent=2)

def encode_labels(tag_list):
    vec = np.zeros(len(tag2id), dtype=np.float32)
    for t in tag_list:
        if t in tag2id:
            vec[tag2id[t]] = 1.0
    return vec

dataset_list = [{
    "text": r["sentence"],
    "labels": encode_labels(r["tags"]).tolist()
} for r in records]

dataset = Dataset.from_list(dataset_list)
dataset = dataset.train_test_split(test_size=0.1, seed=42)


# =========================================================
# 3. Tokenizer
# =========================================================
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

def tokenize(batch):
    enc = tokenizer(batch["text"], max_length=128, truncation=True, padding="max_length")
    enc["labels"] = [np.array(lbl, dtype=np.float32) for lbl in batch["labels"]]
    return enc

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)


# =========================================================
# 4. Compute class weights for BCEWithLogits
# =========================================================
all_lbls = np.array([x["labels"] for x in dataset["train"]])
pos_freq = all_lbls.sum(axis=0)
pos_weight = (1.0 / (pos_freq + 1)).astype(np.float32)  # avoid div-by-zero


# =========================================================
# 5. Load Model
# =========================================================
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-base",
    num_labels=len(tag2id),
    problem_type="multi_label_classification"
)

# Add class weights so rare labels get learned
model.config.pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

model.to(DEVICE)


# =========================================================
# 6. Metrics
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.35).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


# =========================================================
# 7. TrainingArguments (CORRECTED)
# =========================================================
args = TrainingArguments(
    output_dir="./models/meraki_sentence_tagger",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    num_train_epochs=5,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",       # <── CORRECT
    save_strategy="epoch",             # <── CORRECT
    metric_for_best_model="f1_micro",
    load_best_model_at_end=True,       # <── now works
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("🚀 Training started…")
trainer.train()

trainer.save_model("./models/meraki_sentence_tagger")
tokenizer.save_pretrained("./models/meraki_sentence_tagger")

print("✅ Training complete!")
