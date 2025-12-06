import os
from huggingface_hub import HfApi, create_repo

MODEL_PATH = "../models/meraki_sentence_tagger"
REPO_ID = "AaranNihalani/MerakiTagger"
TOKEN = os.getenv("HF_TOKEN")

def push_model():
    if not TOKEN:
        print("❌ Error: HF_TOKEN environment variable not found.")
        print("   Run: export HF_TOKEN=your_token_here")
        return

    print(f"🚀 Creating/Checking repository: {REPO_ID}...")
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True, token=TOKEN)
    except Exception as e:
        print(f"⚠️ Repo creation note: {e}")

    print(f"📤 Uploading model files from {MODEL_PATH}...")
    api = HfApi(token=TOKEN)
    
    # Upload folder
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=REPO_ID,
        repo_type="model",
        ignore_patterns=["checkpoint-*", "runs", "*.bin"]  # Prefer safetensors, skip checkpoints
    )
    
    # Also upload thresholds.json specifically if it wasn't in the ignore list (it isn't, but good to be explicit)
    # The upload_folder covers it, but let's confirm completion.
    print("✅ Model pushed successfully!")
    print(f"🔗 View at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    push_model()
