import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import importlib


ROOT = Path(__file__).resolve().parent


def _rm_path(p: Path):
    if p.is_dir():
        shutil.rmtree(p)
    elif p.is_file():
        p.unlink()


def _run_step(label: str, args: list[str]):
    print(f"\n=== {label} ===")
    subprocess.run(args, cwd=str(ROOT), check=True)

def _require_modules(modules: dict[str, str]):
    missing = []
    for mod, pkg in modules.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        missing = sorted(set(missing))
        cmd = f"{sys.executable} -m pip install " + " ".join(missing)
        raise SystemExit(f"Missing dependencies. Install with:\n{cmd}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    args = parser.parse_args()

    _require_modules(
        {
            "pandas": "pandas",
            "openpyxl": "openpyxl",
            "numpy": "numpy",
            "sklearn": "scikit-learn",
            "torch": "torch",
            "datasets": "datasets",
            "transformers": "transformers",
            "huggingface_hub": "huggingface-hub",
            "sentencepiece": "sentencepiece",
        }
    )

    if not args.no_clean:
        _rm_path(ROOT / "models" / "domain_adapted")
        _rm_path(ROOT / "models" / "meraki_sentence_tagger")
        _rm_path(ROOT / "data" / "labeled" / "unclustered_tagged_sentences.json")
        _rm_path(ROOT / "data" / "labeled" / "tag_cluster_map.json")
        _rm_path(ROOT / "data" / "labeled" / "tagged_sentences.json")

    _run_step("Convert Excel -> JSON", [sys.executable, str(ROOT / "convert_unclustered.py")])
    _run_step("Cluster tags + build training JSON", [sys.executable, str(ROOT / "cluster.py")])
    _run_step("Domain adaptation (MLM)", [sys.executable, str(ROOT / "domain_adapt.py")])
    _run_step("Fine-tune multi-label classifier", [sys.executable, str(ROOT / "finetune.py")])

    out_dir = ROOT / "models" / "meraki_sentence_tagger"
    thresholds_src = out_dir / "thresholds.json"
    id2label_src = out_dir / "id2label.json"
    if not thresholds_src.exists() or not id2label_src.exists():
        raise SystemExit("Missing thresholds.json or id2label.json in ./models/meraki_sentence_tagger")

    backend_dir = ROOT / "web_app" / "backend"
    backend_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(thresholds_src, backend_dir / "thresholds.json")
    shutil.copy2(id2label_src, backend_dir / "id2label.json")
    print("\nSynced thresholds/id2label to web_app/backend")

    if args.skip_push:
        print("\nSkipping Hugging Face push (--skip-push).")
        return

    if not os.getenv("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set. Set it to push the model to Hugging Face.")

    _run_step("Push model to Hugging Face Hub", [sys.executable, str(ROOT / "web_app" / "push_to_hub.py")])


if __name__ == "__main__":
    main()
