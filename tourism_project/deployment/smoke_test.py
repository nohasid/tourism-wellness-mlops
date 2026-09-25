"""
Deployment Smoke Test
----------------------
Validates the exact thing app.py does at runtime: download
best_model.joblib from the Hugging Face Model Hub and run one real
prediction through it. This is what the CI pipeline's deploy-hosting job
runs as its deployment gate (see .github/workflows/pipeline.yml), catching
a missing or broken model artifact before Streamlit Community Cloud picks
up the same commit.

Run from the repository root:
    python tourism_project/deployment/smoke_test.py
"""

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "Nohafx/tourism-wellness-model"


def main():
    print(f"Downloading model from https://huggingface.co/{MODEL_REPO_ID}")
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="best_model.joblib")
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # A realistic, fully-populated row in the exact schema app.py builds
    # from the Streamlit form -- this is what actually exercises the
    # preprocessing pipeline (imputers, scaler, one-hot encoder) end to end.
    sample_input = pd.DataFrame([{
        "Age": 36,
        "TypeofContact": "Self Enquiry",
        "CityTier": 1,
        "DurationOfPitch": 15,
        "Occupation": "Salaried",
        "Gender": "Male",
        "NumberOfPersonVisiting": 2,
        "NumberOfFollowups": 4,
        "ProductPitched": "Basic",
        "PreferredPropertyStar": 3,
        "MaritalStatus": "Single",
        "NumberOfTrips": 3,
        "Passport": 1,
        "PitchSatisfactionScore": 3,
        "OwnCar": 1,
        "NumberOfChildrenVisiting": 0,
        "Designation": "Executive",
        "MonthlyIncome": 22000.0,
    }])

    prediction = model.predict(sample_input)[0]
    probability = model.predict_proba(sample_input)[0, 1]

    assert prediction in (0, 1), f"Unexpected prediction value: {prediction}"
    assert 0.0 <= probability <= 1.0, f"Unexpected probability: {probability}"

    print(f"Sample prediction: {prediction} (probability of purchase: {probability:.4f})")
    print("Deployment smoke test passed: the model downloaded from the "
          "Hugging Face Model Hub loads and predicts correctly with the "
          "exact input schema app.py uses.")


if __name__ == "__main__":
    main()
