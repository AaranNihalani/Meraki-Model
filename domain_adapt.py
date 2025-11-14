import os, re, glob, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# ============================================================
# CONFIG — FORCE CPU, CLEAN ENV
# ============================================================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

DEVICE = torch.device("cpu")
BASE_MODEL = "microsoft/deberta-base"

print("🔥 Training device:", DEVICE)

os.makedirs("./models", exist_ok=True)
os.makedirs("./logs/mlm", exist_ok=True)

# ============================================================
# 1. LOAD DOMAIN CORPUS
# ============================================================

def load_texts(path="./data/domain_docs_labeled/*.txt"):
    docs = []
    for fp in glob.glob(path):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()

            # Clean text
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                docs.append({"text": text})

        except Exception as e:
            print(f"⚠️ Error reading {fp}: {e}")

    return docs


if os.path.exists("./data/domain_docs_labeled"):
    print("\n📘 Loading domain corpus...")
    corpus = load_texts()
    print(f"Loaded {len(corpus)} documents.")
else:
    print("⚠️ No domain corpus found — skipping domain adaptation.")
    corpus = []

# ============================================================
# 2. TOKENIZER + CHUNKING (CORRECTED)
# ============================================================

if len(corpus) > 0:

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    dataset = Dataset.from_list(corpus)

    def chunk_text(batch):
        # Tokenize each doc into IDs
        ids = []
        for txt in batch["text"]:
            out = tokenizer(txt, add_special_tokens=False)
            ids.extend(out["input_ids"])

        # Chunk into blocks
        block_size = 256
        total = (len(ids) // block_size) * block_size
        chunks = [ids[i:i+block_size] for i in range(0, total, block_size)]

        return {"input_ids": chunks}

    tokenized = dataset.map(chunk_text, batched=True, remove_columns=["text"])

    print(f"📦 Total MLM training chunks: {len(tokenized)}")

    # Dynamic MLM masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )

    # ============================================================
    # 3. MODEL
    # ============================================================

    print("\n🔥 Initializing model...")
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(DEVICE)

    # ============================================================
    # 4. TRAINING ARGUMENTS
    # ============================================================

    args = TrainingArguments(
        output_dir="./models/domain_adapted",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        logging_steps=25,
        save_strategy="epoch",
        report_to="none",
        logging_dir="./logs/mlm",
        dataloader_pin_memory=False,
    )

    # ============================================================
    # 5. TRAIN
    # ============================================================

    print("\n🚀 Starting domain-adaptive pretraining (MLM)...")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # ============================================================
    # 6. SAFE SAVE (FIXES YOUR TOKENIZER ISSUE)
    # ============================================================

    print("\n💾 Saving clean domain-adapted model...")
    model.save_pretrained("./models/domain_adapted")

    # 🔥 Save *fresh* tokenizer from DeBERTa base — prevents corruption
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained("./models/domain_adapted")

    print("✅ Domain adaptation complete!")
    print("📦 Saved to ./models/domain_adapted")

else:
    print("⏩ Skipping MLM — no domain docs found.")
