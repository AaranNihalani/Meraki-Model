import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

DATA_FILE = "./data/labeled/tagged_sentences.json"
MODEL_NAME = "./models/domain_adapted"   # your domain-adapted base
OUTPUT_DIR = "./models/meraki_sentence_tagger"

# --------------------------
# Load dataset
# --------------------------
with open(DATA_FILE, "r") as f:
    data = json.load(f)

sentences = [x["sentence"] for x in data]
tag_lists = [x["tags"] for x in data]

# ---------------------------------
# Fit MultiLabelBinarizer
# ---------------------------------
mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(tag_lists)
num_labels = len(mlb.classes_)

print(f"🔎 NUM LABELS = {num_labels}")
print(f"🔎 Example classes: {mlb.classes_[:10]}")

# --------------------------
# Tokenizer & Model
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
            max_length=256
        )
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(self.labels[idx]).float()
        return encoding

# --------------------------
# Train/test split
# --------------------------
from sklearn.model_selection import train_test_split
train_s, val_s, train_l, val_l = train_test_split(
    sentences, label_matrix, test_size=0.1, random_state=42
)

train_dataset = MerakiDataset(train_s, train_l)
val_dataset = MerakiDataset(val_s, val_l)

# --------------------------
# Evaluation Metrics
# --------------------------
def compute_metrics(pred):
    logits, labels = pred
    preds = (1 / (1 + np.exp(-logits))) > 0.5  # sigmoid threshold
    micro = f1_score(labels, preds, average="micro", zero_division=0)
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "f1_micro": micro,
        "f1_macro": macro
    }

# --------------------------
# Training Arguments
# --------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
)

# --------------------------
# Trainer
# --------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("🚀 Starting fine-tuning...")
trainer.train()
trainer.save_model(OUTPUT_DIR)

# Save ML-Binarizer for inference
import pickle
with open(os.path.join(OUTPUT_DIR, "mlb.pkl"), "wb") as f:
    pickle.dump(mlb, f)

print("🎉 Training complete! Model saved.")
