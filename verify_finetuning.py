import os
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM

# ---------------------------------------------------------
# PATHS — adjust if needed
# ---------------------------------------------------------
BASE_MODEL = "microsoft/deberta-v3-large"
ADAPTED_DIR = "./models/domain_adapted"
FINETUNED_DIR = "./models/meraki_sentence_tagger"

print("\n🔍 VERIFYING FINETUNING WEIGHT UPDATES")
print("Base model:", BASE_MODEL)
print("Domain-adapted:", ADAPTED_DIR)
print("Fine-tuned:", FINETUNED_DIR)


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
print("\n📥 Loading base model...")
base = AutoModelForMaskedLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

print("📥 Loading domain-adapted model...")
adapted = AutoModelForMaskedLM.from_pretrained(ADAPTED_DIR, torch_dtype=torch.float32)

print("📥 Loading fine-tuned classifier (backbone only for comparison)...")
finetuned = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DIR, torch_dtype=torch.float32)

# We only compare encoder weights, not classifier head
finetuned_backbone = finetuned.deberta


# ---------------------------------------------------------
# FUNCTION TO COMPARE TWO MODELS
# ---------------------------------------------------------
def compare_models(model_a, model_b, title):
    print(f"\n===============================")
    print(f"🔎 {title}")
    print(f"===============================")

    total_params = 0
    changed_params = 0
    changed_layers = []

    # compare only overlapping layers (encoder)
    for (name_a, p_a), (name_b, p_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        if p_a.shape != p_b.shape:
            continue  # skip classifier head mismatch

        A = p_a.detach().cpu()
        B = p_b.detach().cpu()

        total_params += A.numel()
        diff = (A != B).sum().item()

        if diff > 0:
            changed_params += diff
            changed_layers.append((name_a, diff, A.numel()))

    pct = changed_params / total_params * 100
    print(f"Total parameters compared: {total_params:,}")
    print(f"Changed parameters:       {changed_params:,}")
    print(f"Percentage changed:        {pct:.4f}%")

    print("\nLayers with changes:")
    if not changed_layers:
        print("❌ No changed layers found.")
    else:
        for name, diff, total in changed_layers:
            layer_pct = diff / total * 100
            print(f"  • {name:<65}  {layer_pct:6.3f}% changed")


# ---------------------------------------------------------
# 1) Compare DOMAIN-ADAPTED vs FINE-TUNED
# ---------------------------------------------------------
compare_models(adapted, finetuned_backbone, "Domain-adapted  →  Fine-tuned backbone")


print("\n🎉 Verification complete!")
