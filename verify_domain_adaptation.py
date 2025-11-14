import os
import torch
from transformers import AutoModelForMaskedLM

BASE_MODEL = "microsoft/deberta-base"
ADAPTED_DIR = "./models/domain_adapted"

print("\n🔍 Checking domain-adaptation weight changes...")
print("Base model:", BASE_MODEL)
print("Adapted model:", ADAPTED_DIR)

# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------
print("\n📥 Loading base model...")
base = AutoModelForMaskedLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

print("📥 Loading domain-adapted model...")
adapted = AutoModelForMaskedLM.from_pretrained(ADAPTED_DIR, torch_dtype=torch.float32)

# ---------------------------------------------------------
# Compare weights
# ---------------------------------------------------------
total_params = 0
changed_params = 0
layer_reports = []

for (name_b, p_b), (name_a, p_a) in zip(base.named_parameters(), adapted.named_parameters()):
    assert name_b == name_a, f"Layer mismatch: {name_b} vs {name_a}"

    base_w = p_b.detach().cpu()
    adapt_w = p_a.detach().cpu()

    # Number of elements
    total_params += base_w.numel()

    # Is it identical?
    diff = (base_w != adapt_w).sum().item()

    if diff > 0:
        changed_params += diff
        layer_reports.append((name_b, diff, base_w.numel()))

# ---------------------------------------------------------
# Report summary
# ---------------------------------------------------------
print("\n===============================")
print("🔥 DOMAIN ADAPTATION REPORT")
print("===============================")

print(f"Total parameters: {total_params:,}")
print(f"Changed elements: {changed_params:,}")
print(f"Percentage changed: {changed_params / total_params * 100:.4f}%")

print("\nLayers with changes:")
for name, diff, total in layer_reports:
    pct = diff / total * 100
    print(f"  • {name:<60}  {pct:6.3f}% changed")

print("\n🎉 Done.")
