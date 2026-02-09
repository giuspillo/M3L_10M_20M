import os
import torch
from huggingface_hub import snapshot_download

# Defined local folder
LOCAL_MODEL_DIR = "models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

print(f"Downloading models to {os.path.abspath(LOCAL_MODEL_DIR)}...\n")

# --- 1. Download Hugging Face Models (Whisper & AST) ---
# We use snapshot_download to get config, weights, and preprocessor files.

hf_models = {
    "whisper": "openai/whisper-base",
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593"
}

for name, repo_id in hf_models.items():
    print(f"Downloading {name} ({repo_id})...")
    save_path = os.path.join(LOCAL_MODEL_DIR, name)
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=save_path,
        local_dir_use_symlinks=False  # Important: Actual files, not symlinks
    )
    print(f" -> Saved to {save_path}")

# --- 2. Download VGGish (Torch Hub) ---
# Torch Hub usually downloads to ~/.cache. We force it to local by setting TORCH_HOME.

print("\nDownloading VGGish...")
vggish_dir = os.path.join(LOCAL_MODEL_DIR, "torch_hub")
os.makedirs(vggish_dir, exist_ok=True)

# Temporarily overwrite environment variable to force download location
os.environ['TORCH_HOME'] = vggish_dir

# This triggers the download of the repo AND the checkpoint
torch.hub.load('harritaylor/torchvggish', 'vggish', verbose=True)

print(f" -> VGGish saved inside {vggish_dir}")
print("\nDone! All models are downloaded.")