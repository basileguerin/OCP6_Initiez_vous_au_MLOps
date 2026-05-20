"""
Déploiement de l'interface Gradio sur Hugging Face Spaces.
Usage : python deploy_gradio.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

HF_USERNAME  = "basmoket"
SPACE_NAME   = "scoring-credit-demo"
REPO_ID      = f"{HF_USERNAME}/{SPACE_NAME}"
API_SPACE_URL = f"https://{HF_USERNAME}-scoring-credit-api.hf.space"
ROOT         = Path(__file__).parent

HF_TOKEN = os.environ.get("HF_TOKEN") or input("Token Hugging Face (write) : ").strip()
api = HfApi(token=HF_TOKEN)

create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="gradio",
            token=HF_TOKEN, exist_ok=True, private=False)

# Variable d'environnement du Space : l'app lira API_URL au démarrage
api.add_space_variable(repo_id=REPO_ID, key="API_URL", value=API_SPACE_URL)
print(f"  Variable API_URL={API_SPACE_URL} configurée")

fichiers = [
    (ROOT / "space_README_gradio.md", "README.md"),
    # app.py doit être à la racine du Space (sdk: gradio le cherche là)
    (ROOT / "gradio_app" / "app.py",  "app.py"),
    # requirements.txt à la racine — HF l'installe automatiquement
    (ROOT / "requirements-gradio.txt", "requirements.txt"),
]

for local_path, remote_path in fichiers:
    print(f"  Upload {remote_path}...")
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=remote_path,
                    repo_id=REPO_ID, repo_type="space", token=HF_TOKEN)

print(f"\nDéployé : https://huggingface.co/spaces/{REPO_ID}")
print(f"Demo     : https://{HF_USERNAME}-{SPACE_NAME}.hf.space")
