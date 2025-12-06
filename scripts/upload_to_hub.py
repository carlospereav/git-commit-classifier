"""
Script to upload the trained model to HuggingFace Hub.

Usage:
    1. Login first: huggingface-cli login
    2. Run: python scripts/upload_to_hub.py
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path

# Configuration
MODEL_PATH = "./model"
REPO_NAME = "git-commit-classifier"  # Will create: {username}/git-commit-classifier

def upload_model() -> None:
    """Upload model to HuggingFace Hub."""
    api = HfApi()
    
    # Get username
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{REPO_NAME}"
    
    print(f"📦 Uploading model to: https://huggingface.co/{repo_id}")
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"⚠️ Repo creation note: {e}")
    
    # Upload all files from model directory
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload DistilBERT commit classifier model"
    )
    
    print(f"\n🎉 Success! Model uploaded to:")
    print(f"   https://huggingface.co/{repo_id}")
    print(f"\n📝 Use this in your code:")
    print(f'   model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")')
    print(f'   tokenizer = AutoTokenizer.from_pretrained("{repo_id}")')

if __name__ == "__main__":
    upload_model()

