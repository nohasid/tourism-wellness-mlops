## Where the model lives

The trained model is **not** registered on the Hugging Face Model Hub.
Following this project's presentation template ("commit the best model
back into the GitHub repository" / "load the model committed to the
GitHub repo"), `train.py` saves the best pipeline straight to
`tourism_project/deployment/best_model.joblib`, and the CI pipeline's
`model-training` job commits and pushes that file to `main` right after
training. `app.py` and `smoke_test.py` both load it from that local
path -- no Hugging Face Hub round-trip needed to serve predictions.

Hugging Face is still used for the **dataset** (raw CSV and the
train/test split), per the project's Data Preparation requirements.

## Where the app is hosted

Hosted on **Streamlit Community Cloud** (share.streamlit.io), which
deploys directly from this GitHub repo and auto-redeploys on every push
to `main`. Python version is pinned to **3.11** via the app's Advanced
settings on Streamlit Cloud (matching `pipeline.yml`'s
`python-version: "3.11"`), since Streamlit Cloud no longer respects a
`runtime.txt` file.

## What the `deploy-hosting` CI job does

Runs `tourism_project/deployment/smoke_test.py` against the model file
`model-training` just committed: it loads it and runs one real prediction
through it with the exact input schema `app.py` uses.

## One-time setup before running any of this

1. Edit the three values at the top of `tourism_project/config.py`
   (`HF_USERNAME`, `GITHUB_USERNAME`, `GITHUB_REPO_NAME`).
2. Create a Hugging Face **write-access** token and add it as a GitHub
   Actions secret named `HF_TOKEN` in this repo's Settings -> Secrets and
   variables -> Actions.
3. Connect this repo to Streamlit Community Cloud, selecting Python 3.11
   in Advanced settings before deploying.
