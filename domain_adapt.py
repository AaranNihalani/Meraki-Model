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
# CONFIGURATION
# ============================================================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

print("🔥 MPS available?", torch.backends.mps.is_available())
DEVICE = torch.device("cpu")
BASE_MODEL = "microsoft/deberta-base"

os.makedirs("./models", exist_ok=True)
os.makedirs("./logs/mlm", exist_ok=True)

# ============================================================
# 1. LOAD TEXT CORPUS
# ============================================================

def load_text_corpus(path_pattern="./data/domain_docs_labeled/*.txt"):
    texts = []
    for file in glob.glob(path_pattern):
        try:
            with open(file, "r", encoding="utf-8") as f:
                text = re.sub(r"\s+", " ", f.read().strip())
            if text:
                texts.append({"text": text})
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")
    return texts


if os.path.exists("./data/domain_docs_labeled"):
    print("\n🧠 Loading domain corpus ...")
    corpus = load_text_corpus()
    print(f"Loaded {len(corpus)} documents.")
else:
    print("⚠️ No domain corpus found, skipping MLM stage.")
    corpus = []

# ============================================================
# 2. TOKENIZATION + CHUNKING
# ============================================================

if corpus:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    dataset = Dataset.from_list(corpus)

    def chunk_examples(examples):
        # Tokenize without truncation first
        tokenized = tokenizer(examples["text"])
        concatenated = []
        for ids in tokenized["input_ids"]:
            concatenated.extend(ids)

        block_size = 256
        total_length = (len(concatenated) // block_size) * block_size
        input_ids = [
            concatenated[i : i + block_size] for i in range(0, total_length, block_size)
        ]
        return {"input_ids": input_ids}

    tokenized_ds = dataset.map(chunk_examples, batched=True, remove_columns=["text"])
    print("Number of MLM samples:", len(tokenized_ds))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    # ============================================================
    # 3. MODEL + TRAINING SETUP
    # ============================================================

    model_mlm = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(DEVICE)

    # fp16 is only for CUDA; MPS does not support it
    fp16_flag = torch.cuda.is_available()

    args_mlm = TrainingArguments(
        output_dir="./models/domain_adapted",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        fp16=fp16_flag,
        learning_rate=5e-5,
        logging_steps=25,
        save_strategy="epoch",
        report_to="none",
        logging_dir="./logs/mlm",
        disable_tqdm=False,
        dataloader_pin_memory=False,
    )

    # ============================================================
    # 4. TRAIN
    # ============================================================

    print("\n🚀 Starting unsupervised domain adaptation (MLM)...")
    trainer_mlm = Trainer(
        model=model_mlm,
        args=args_mlm,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer_mlm.train()
    trainer_mlm.save_model("./models/domain_adapted")
    tokenizer.save_pretrained("./models/domain_adapted")

    print("✅ Domain adaptation complete.")
    print("📦 Model saved to ./models/domain_adapted")
else:
    print("⏩ Skipping MLM (no domain files provided).")
