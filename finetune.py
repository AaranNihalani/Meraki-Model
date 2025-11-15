import os
import json
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score

# ============================================================
# 0. DEVICE CONFIG
# ============================================================
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
DEVICE = torch.device("cpu")
print("🔥 Using device:", DEVICE)


# ============================================================
# 1. LOAD DATASET
# ============================================================
records = json.load(open("./data/labeled/tagged_sentences.json"))

tags = sorted({t for r in records for t in r["tags"]})
tag2id = {t: i for i, t in enumerate(tags)}
id2tag = {i: t for t, i in tag2id.items()}
json.dump(id2tag, open("./models/id2tag.json", "w"), indent=2)

print(f"\n🔎 NUM LABELS = {len(tag2id)}")
print("🔎 Example tags:", tags[:5])

def encode(lbls):
    v = np.zeros(len(tag2id), dtype=np.float32)
    for t in lbls:
        v[tag2id[t]] = 1.0
    return v

dataset = Dataset.from_list([
    {"text": r["sentence"], "labels": encode(r["tags"])}
    for r in records
])

# Debug print
print("\n🧪 RAW LABEL CHECK (first 5)")
for i in range(5):
    print(dataset[i]["labels"], " sum=", sum(dataset[i]["labels"]))

dataset = dataset.train_test_split(0.1, seed=42)


# ============================================================
# 2. TOKENIZER (CRITICAL: use BASE tokenizer)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

def tokenize(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    enc["labels"] = [np.array(lbl, dtype=np.float32) for lbl in batch["labels"]]
    return enc

tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names
)
tokenized.set_format("torch")

print("\n🧪 TOKENIZED LABEL CHECK (first 5)")
for i in range(5):
    print(tokenized["train"][i]["labels"].shape, tokenized["train"][i]["labels"].sum())


# ============================================================
# 3. CLASS WEIGHTS
# ============================================================
label_matrix = np.array([x["labels"] for x in dataset["train"]])
pos_freq = label_matrix.sum(axis=0)

print("\n📊 POS FREQ:", pos_freq[:10])
pos_weight = torch.tensor(1.0 / (pos_freq + 1.0), dtype=torch.float32)
print("📊 POS WEIGHT:", pos_weight[:10])


# ============================================================
# 4. LOAD MODEL (DOMAIN ADAPTED)
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    "./models/domain_adapted",
    num_labels=len(tag2id),
    problem_type="multi_label_classification"
).to(DEVICE)

loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ============================================================
# 5. DEBUG CAPABLE TRAINER
# ============================================================
class DebugTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Debug info every ~50 steps
        if self.state.global_step % 50 == 0:
            print("\n🔍 LOSS DEBUG:")
            print(" - logits mean:", logits.mean().item())
            print(" - labels sum:", labels.sum().item())
            print(" - logits std:", logits.std().item())

        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        out = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        return out


# ============================================================
# 6. METRICS
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.35).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


# ============================================================
# 7. TRAINING SETTINGS
# ============================================================
training_args = TrainingArguments(
    output_dir="./models/meraki_sentence_tagger",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    report_to="none",
)


# ============================================================
# 8. TRAIN
# ============================================================
trainer = DebugTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\n🚀 START TRAINING\n")
trainer.train()

trainer.save_model("./models/meraki_sentence_tagger")
tokenizer.save_pretrained("./models/meraki_sentence_tagger")

print("\n🎉 Fine-tuning complete!")
