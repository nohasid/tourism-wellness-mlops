"""
Data Registration
------------------
Uploads the raw tourism.csv (already present locally under
tourism_project/data/) to a Hugging Face Dataset repo, creating the repo
if it does not already exist.

Run from the repository root:
    python tourism_project/model_building/register_dataset.py
"""

import os
import sys

from huggingface_hub import HfApi, create_repo

# allow `import config` when run as a script from the repo root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402


def main():
    token = config.require_token()
    api = HfApi(token=token)

    local_path = os.path.join(
        os.path.dirname(__file__), "..", "data", config.RAW_DATA_FILENAME
    )
    local_path = os.path.abspath(local_path)

    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Could not find {local_path}. Make sure tourism.csv has been "
            "placed in tourism_project/data/ before running this script."
        )

    print(f"Creating (or reusing) dataset repo: {config.DATASET_REPO_ID}")
    create_repo(
        repo_id=config.DATASET_REPO_ID,
        repo_type="dataset",
        token=token,
        exist_ok=True,
    )

    print(f"Uploading {local_path} -> {config.DATASET_REPO_ID}/{config.RAW_DATA_FILENAME}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=config.RAW_DATA_FILENAME,
        repo_id=config.DATASET_REPO_ID,
        repo_type="dataset",
    )

    print("Data registration complete.")
    print(f"View it at: https://huggingface.co/datasets/{config.DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
