import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pickle

DATA_FILE = "./data/labeled/tagged_sentences.json"
MODEL_NAME = "./models/domain_adapted"
OUTPUT_DIR = "./models/meraki_sentence_tagger"

# --------------------------
# Load dataset
# --------------------------
with open(DATA_FILE, "r") as f:
    data = json.load(f)

sentences = [x["sentence"] for x in data]

# Ensure all tags are lists
tag_lists = []
for x in data:
    if isinstance(x["tags"], str):
        tag_lists.append([x["tags"]])
    else:
        tag_lists.append(x["tags"])

print(f"Loaded {len(sentences)} sentences.")

# ---------------------------------
# MultiLabelBinarizer
# ---------------------------------
mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(tag_lists)
num_labels = len(mlb.classes_)

print(f"🔎 NUM LABELS = {num_labels}")
print(f"🔎 Example classes: {mlb.classes_[:10]}")

# --------------------------
# Tokenizer & Model
# --------------------------
if not os.path.exists(MODEL_NAME):
    MODEL_NAME = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Mistral models need a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# --------------------------
# Dataset class
# --------------------------
class MerakiDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=384
        )
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(self.labels[idx]).float()
        return encoding

# --------------------------
# Train/test split
# --------------------------
train_s, val_s, train_l, val_l = train_test_split(
    sentences, label_matrix, test_size=0.1, random_state=42
)

train_dataset = MerakiDataset(train_s, train_l)
val_dataset = MerakiDataset(val_s, val_l)

# --------------------------
# Evaluation
# --------------------------
def compute_metrics(pred):
    logits, labels = pred
    probs = 1 / (1 + np.exp(-logits))
    preds = probs > 0.5

    micro = f1_score(labels, preds, average="micro", zero_division=0)
    macro = f1_score(labels, preds, average="macro", zero_division=0)

    return {"f1_micro": micro, "f1_macro": macro}

# --------------------------
# Training Arguments
# --------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=8,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    logging_steps=20,
    warmup_ratio=0.1,
    weight_decay=0.01,
    gradient_checkpointing=True,
)

# --------------------------
# Trainer
# --------------------------
pos_counts = label_matrix.sum(axis=0).astype(np.float64)
neg_counts = (len(sentences) - pos_counts).astype(np.float64)
# avoid div by zero; if a label has no positives, set a moderate weight
safe_pos = np.where(pos_counts > 0, pos_counts, 1.0)
pos_weight_vec = torch.tensor(neg_counts / safe_pos, dtype=torch.float)

class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.bce(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    pos_weight=pos_weight_vec,
)

print("🚀 Starting fine-tuning...")
trainer.train()

# --------------------------
# Save model
# --------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save ML-Binarizer
with open(os.path.join(OUTPUT_DIR, "mlb.pkl"), "wb") as f:
    pickle.dump(mlb, f)

# Save id2label.json for infer.py
id2label = {i: label for i, label in enumerate(mlb.classes_)}
with open(os.path.join(OUTPUT_DIR, "id2label.json"), "w") as f:
    json.dump(id2label, f, indent=2)

# --------------------------
# Per-label threshold tuning
# --------------------------
pred = trainer.predict(val_dataset)
logits = pred.predictions
labels = pred.label_ids
probs = 1 / (1 + np.exp(-logits))

thresholds = {}
for i in range(num_labels):
    best_f1 = -1.0
    best_t = 0.5
    # search thresholds from 0.05 to 0.95
    for t in np.linspace(0.05, 0.95, 19):
        preds_i = (probs[:, i] > t).astype(int)
        f1_i = f1_score(labels[:, i], preds_i, average="binary", zero_division=0)
        if f1_i > best_f1:
            best_f1 = f1_i
            best_t = t
    thresholds[mlb.classes_[i]] = float(best_t)

with open(os.path.join(OUTPUT_DIR, "thresholds.json"), "w") as f:
    json.dump(thresholds, f, indent=2)

print("🎉 Training complete! Model saved.")
