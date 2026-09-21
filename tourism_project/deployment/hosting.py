"""
Hosting (OPTIONAL -- requires Hugging Face PRO)
-------------------------------------------------
Creates (if needed) a Hugging Face Space configured for the Docker SDK, and
pushes this deployment/ folder's contents (Dockerfile, app.py,
requirements.txt) to it. HF Spaces auto-builds and (re)starts the container
on every push, so this single upload is the entire "deploy" step.

NOTE: as of the current HF pricing, creating a Docker/Gradio Space on the
free "cpu-basic" hardware tier requires an HF PRO subscription -- a free
account gets HTTP 402 Payment Required from create_repo() below. This is a
Hugging Face platform policy, not a bug in this script.

This project's pipeline.yml does NOT call this script by default, for that
reason -- it runs tourism_project/deployment/smoke_test.py instead, and the
app is actually hosted on Streamlit Community Cloud (which deploys straight
from this GitHub repo, no push step needed; see the repo README).

Keep this script around for if/when the HF account has PRO: at that point
you can call it manually, or swap it back into pipeline.yml's
deploy-hosting job in place of smoke_test.py, to host on an HF Space
instead of / in addition to Streamlit Community Cloud.

Run from the repository root:
    python tourism_project/deployment/hosting.py
"""

import os
import sys

from huggingface_hub import HfApi, create_repo

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402


def main():
    token = config.require_token()
    api = HfApi(token=token)

    print(f"Creating (or reusing) Space: {config.SPACE_REPO_ID}")
    create_repo(
        repo_id=config.SPACE_REPO_ID,
        repo_type="space",
        space_sdk="docker",
        token=token,
        exist_ok=True,
    )

    deployment_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Uploading {deployment_dir} -> {config.SPACE_REPO_ID}")
    api.upload_folder(
        folder_path=deployment_dir,
        repo_id=config.SPACE_REPO_ID,
        repo_type="space",
        # app.py now loads the model from a local file (see app.py), so
        # that file must be pushed alongside it for the Space to work.
        allow_patterns=["Dockerfile", "app.py", "requirements.txt", config.MODEL_ARTIFACT_FILENAME],
    )

    print("Hosting complete.")
    print(f"View the app at: https://huggingface.co/spaces/{config.SPACE_REPO_ID}")


if __name__ == "__main__":
    main()
