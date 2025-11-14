import os
import json
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ============================================================
# 0. FORCE CPU (disable MPS)
# ============================================================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
DEVICE = torch.device("cpu")
print("🔥 Training on:", DEVICE)

# ============================================================
# 1. LOAD DATA
# ============================================================
records = json.load(open("./data/labeled/tagged_sentences.json"))

# All tags
tags = sorted({t for r in records for t in r["tags"]})
tag2id = {t: i for i, t in enumerate(tags)}
id2tag = {i: t for t, i in tag2id.items()}
json.dump(id2tag, open("./models/id2tag.json", "w"), indent=2)

def encode_tags(tag_list):
    v = np.zeros(len(tag2id), dtype=np.float32)
    for t in tag_list:
        v[tag2id[t]] = 1
    return v

# ------------------------------------------------------------
# 1.1 Instruction-formatted INPUT CONSTRUCTION
# ------------------------------------------------------------
# This improves learning by giving the model better context.
def format_input(sentence):
    return (
        "You are an expert humanitarian analyst.\n"
        "Your task is to assign ALL relevant tags from the label set.\n"
        "Classify the following sentence:\n\n"
        f"Sentence: {sentence}\n"
    )

dataset = Dataset.from_list([
    {
        "text": format_input(r["sentence"]),
        "labels": encode_tags(r["tags"])
    }
    for r in records
])

dataset = dataset.train_test_split(test_size=0.1, seed=42)


# ============================================================
# 2. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("./models/domain_adapted")

def tokenize(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    enc["labels"] = [np.array(lbl, dtype=np.float32) for lbl in batch["labels"]]
    return enc

tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names
)

tokenized.set_format("torch")
collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ============================================================
# 3. CLASS WEIGHTS
# ============================================================
label_matrix = np.array([x["labels"] for x in dataset["train"]])
pos_freq = label_matrix.sum(axis=0)
pos_weight = torch.tensor(1 / (pos_freq + 1), dtype=torch.float32)
loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ============================================================
# 4. LOAD MODEL
# ============================================================
# START FROM YOUR OWN domain_adapted checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    "./models/domain_adapted",
    num_labels=len(tag2id),
    problem_type="multi_label_classification"
).to(DEVICE)

# ============================================================
# 5. CUSTOM WEIGHTED TRAINER
# ============================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


# ============================================================
# 6. METRICS
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.30).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


# ============================================================
# 7. TRAINING ARGS
# ============================================================
args = TrainingArguments(
    output_dir="./models/meraki_sentence_tagger",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=6,
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
)

# ============================================================
# 8. TRAIN
# ============================================================
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

print("\n🚀 Starting training...\n")
trainer.train()

# Save final model + tokenizer
trainer.save_model("./models/meraki_sentence_tagger")
tokenizer.save_pretrained("./models/meraki_sentence_tagger")

print("\n🎉 Training complete!")
print("Saved to: ./models/meraki_sentence_tagger\n")
