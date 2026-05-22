"""
Déploiement de l'API sur Hugging Face Spaces.
Usage : python deploy_api.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

HF_USERNAME = "basmoket"
SPACE_NAME  = "scoring-credit-api"
REPO_ID     = f"{HF_USERNAME}/{SPACE_NAME}"
ROOT        = Path(__file__).parent

HF_TOKEN = os.environ.get("HF_TOKEN") or input("Token Hugging Face (write) : ").strip()
api = HfApi(token=HF_TOKEN)

# Création du Space si pas encore fait (exist_ok=True sinon)
create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker",
            token=HF_TOKEN, exist_ok=True, private=False)

# Uniquement les fichiers nécessaires à l'API — pas les notebooks ni outputs
fichiers = [
    (ROOT / "Dockerfile.api",                         "Dockerfile"),
    (ROOT / "requirements-api.txt",                   "requirements-api.txt"),
    (ROOT / "space_README_api.md",                    "README.md"),
    (ROOT / "api" / "__init__.py",                    "api/__init__.py"),
    (ROOT / "api" / "main.py",                        "api/main.py"),
    (ROOT / "api" / "model_loader.py",                "api/model_loader.py"),
    (ROOT / "api" / "schemas.py",                     "api/schemas.py"),
    (ROOT / "api" / "logger.py",                      "api/logger.py"),
    (ROOT / "models" / "model_reduit.pkl",            "models/model_reduit.pkl"),
    (ROOT / "models" / "features_selectionnees.json", "models/features_selectionnees.json"),
]

for local_path, remote_path in fichiers:
    print(f"  Upload {remote_path}...")
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=remote_path,
                    repo_id=REPO_ID, repo_type="space", token=HF_TOKEN)

print(f"\nDéployé : https://huggingface.co/spaces/{REPO_ID}")
print(f"Swagger  : https://{HF_USERNAME}-{SPACE_NAME}.hf.space/docs")
