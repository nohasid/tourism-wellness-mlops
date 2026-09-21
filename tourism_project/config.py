"""
Central configuration for the Tourism Wellness Package MLOps pipeline.

EDIT THE THREE VALUES BELOW ONCE. Every script in this project (data
registration, data prep, training, deployment, hosting) imports from here,
so you never have to hunt through multiple files to change a repo name.

HF_TOKEN is deliberately NOT stored here. It is read from the environment
variable HF_TOKEN at runtime (set locally via `export HF_TOKEN=...` or, in
CI, via the GitHub Actions secret of the same name). Never hardcode a token
in a script you intend to push to a public GitHub repo.
"""

import os

# ---- EDIT THESE THREE VALUES -------------------------------------------
HF_USERNAME = "Nohafx"          # your Hugging Face username/org
GITHUB_USERNAME = "Nohasid"  # your GitHub username
GITHUB_REPO_NAME = "tourism-wellness-mlops"  # the GitHub repo you create
# --------------------------------------------------------------------------

# Hugging Face Hub repo ID used for the raw/train/test DATA only.
# The trained MODEL is deliberately NOT registered on the HF Model Hub --
# per this project's presentation template, it is committed directly into
# this GitHub repo (tourism_project/deployment/best_model.joblib) by the CI
# pipeline's model-training job, and Streamlit Community Cloud (which
# deploys straight from this repo) loads it from there. See train.py,
# app.py and smoke_test.py.
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"

# SPACE_REPO_ID is only used by the OPTIONAL tourism_project/deployment/
# hosting.py script, for accounts that have Hugging Face PRO and want to
# additionally (or instead) host on an HF Space. Not used by default.
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-wellness-app"

# Filenames used inside the dataset repo on the Hub
RAW_DATA_FILENAME = "tourism.csv"
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"

# Model artifact filename. Lives at tourism_project/deployment/best_model.joblib
# once train.py runs -- see train.py's save path.
MODEL_ARTIFACT_FILENAME = "best_model.joblib"

# Target column
TARGET_COLUMN = "ProdTaken"

# Read at import time so a missing token fails fast with a clear message,
# rather than failing deep inside a huggingface_hub call.
HF_TOKEN = os.environ.get("HF_TOKEN")


def require_token():
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Set it locally with `export HF_TOKEN=<your_write_token>` "
            "or, in GitHub Actions, make sure the HF_TOKEN secret is "
            "configured and passed to this step's `env:` block."
        )
    return HF_TOKEN
