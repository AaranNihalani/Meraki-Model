import os
import torch
import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from sklearn.metrics import f1_score

# =========================================================
# 0. Disable MPS/GPU on macOS — force CPU only
# =========================================================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

print("🔥 MPS available?", torch.backends.mps.is_available())

DEVICE = torch.device("cpu")
print("🔥 Training on:", DEVICE)

BASE_MODEL = "microsoft/deberta-base"
os.makedirs("./models", exist_ok=True)

# =========================================================
# 1. Load Dataset
# =========================================================
print("\n📘 Loading labeled JSON dataset...")
records = json.load(open("./data/labeled/tagged_sentences.json"))

# Build tag→id mappings
all_tags = sorted({t for r in records for t in r["tags"]})
tag2id = {t: i for i, t in enumerate(all_tags)}
id2tag = {i: t for t, i in tag2id.items()}
json.dump(id2tag, open("./models/id2tag.json", "w"), indent=2)

# Convert tag list → binary vector label
def encode_labels(tags):
    vec = np.zeros(len(tag2id))
    for t in tags:
        if t in tag2id:
            vec[tag2id[t]] = 1
    return vec.tolist()

texts = [{"text": r["sentence"], "labels": encode_labels(r["tags"]) } for r in records]
dataset = Dataset.from_list(texts)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"Training samples: {len(dataset['train'])}, Validation: {len(dataset['test'])}")

# =========================================================
# 2. Tokenizer
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    # IMPORTANT: keep labels!!
    enc["labels"] = examples["labels"]
    return enc

# remove __index_level_0__ silently carried by pandas
cols_to_remove = [c for c in dataset["train"].column_names if c not in ("labels", "text")]
tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=cols_to_remove)
tokenized_ds.set_format("torch")

val_labels = np.array([x["labels"] for x in tokenized_ds["test"]])
print("Validation label sums:", val_labels.sum(axis=0))
print("Total positives in validation:", val_labels.sum())


print("\n🔍 Sample tokenized item:", tokenized_ds["train"][0])

# =========================================================
# 3. Model
# =========================================================
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(tag2id),
    problem_type="multi_label_classification"
).to(DEVICE)

# =========================================================
# 4. Optional Callback for Gradient Checking
# =========================================================
class GradCheckCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        for name, param in kwargs["model"].named_parameters():
            if param.grad is not None:
                print("🟢 Grad norm:", param.grad.data.norm().item())
                break

# =========================================================
# 5. Training Arguments
# =========================================================
args = TrainingArguments(
    output_dir="./models/meraki_sentence_tagger",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=10,
    fp16=False,
    bf16=False,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    dataloader_pin_memory=False,
    logging_dir="./logs/sft",
    dataloader_num_workers=0,
)

# =========================================================
# 6. Metrics (correct multi-label F1)
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    return {
        "f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

# =========================================================
# 7. Trainer
# =========================================================
print("\n🚀 Starting supervised fine-tuning (multi-label)...")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[GradCheckCallback()],
)

trainer.train()

# =========================================================
# 8. Save model
# =========================================================
trainer.save_model("./models/meraki_sentence_tagger")
tokenizer.save_pretrained("./models/meraki_sentence_tagger")

print("✅ Fine-tuning complete!")
