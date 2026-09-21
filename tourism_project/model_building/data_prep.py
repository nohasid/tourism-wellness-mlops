"""
Data Preparation
-----------------
1. Loads the raw dataset directly from the Hugging Face dataset repo
   (NOT from the local data/ folder -- rubric requires loading from the Hub).
2. Cleans it:
     - drops the junk index column ("Unnamed: 0") and CustomerID
       (a unique identifier carries no predictive signal and would leak
       row identity into the model).
     - fixes the "Fe Male" typo in Gender -> "Female".
     - merges the redundant "Unmarried" category in MaritalStatus into
       "Single" (both represent "not currently married"; keeping them
       separate only fragments the category for no analytical benefit).
3. Splits into train/test (80/20, stratified on the target because the
   target is imbalanced: ~19% positive class).
4. Saves train.csv/test.csv locally under tourism_project/data/.
5. Uploads train.csv and test.csv back to the same HF dataset repo.

Run from the repository root:
    python tourism_project/model_building/data_prep.py
"""

import os
import sys

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pure function, no I/O -- kept separate so it can be unit tested."""
    df = df.copy()

    # Drop columns with no predictive value / that would leak row identity
    for junk_col in ["Unnamed: 0", "CustomerID"]:
        if junk_col in df.columns:
            df = df.drop(columns=junk_col)

    # Fix known data-entry inconsistencies
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Drop exact duplicate rows, if any survived CustomerID removal
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} exact duplicate row(s).")

    return df


def main():
    token = config.require_token()

    print(f"Downloading raw data from {config.DATASET_REPO_ID}/{config.RAW_DATA_FILENAME}")
    raw_path = hf_hub_download(
        repo_id=config.DATASET_REPO_ID,
        repo_type="dataset",
        filename=config.RAW_DATA_FILENAME,
        token=token,
    )
    df = pd.read_csv(raw_path)
    print(f"Loaded raw data: {df.shape}")

    df_clean = clean_data(df)
    print(f"Cleaned data: {df_clean.shape}")

    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.2,
        random_state=42,
        stratify=df_clean[config.TARGET_COLUMN],
    )
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, config.TRAIN_FILENAME)
    test_path = os.path.join(data_dir, config.TEST_FILENAME)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved locally: {train_path}, {test_path}")

    api = HfApi(token=token)
    for local_path, repo_filename in [
        (train_path, config.TRAIN_FILENAME),
        (test_path, config.TEST_FILENAME),
    ]:
        print(f"Uploading {repo_filename} -> {config.DATASET_REPO_ID}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_filename,
            repo_id=config.DATASET_REPO_ID,
            repo_type="dataset",
        )

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
