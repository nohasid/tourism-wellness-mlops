"""
Deployment Smoke Test
----------------------
Validates the exact thing app.py does at runtime: load
tourism_project/deployment/best_model.joblib -- the file the pipeline's
model-training job committed into this GitHub repo -- and run one real
prediction through it. This is what the CI pipeline's deploy-hosting job
runs as its deployment gate (see .github/workflows/pipeline.yml), catching
a missing or broken model artifact before Streamlit Community Cloud picks
up the same commit.

No network access to Hugging Face or anywhere else is needed for this --
the model lives in the repo itself, matching the "commit the best model
back into the GitHub repository" / "load the model committed to the GitHub
repo" architecture from the project's presentation template.

Run from the repository root:
    python tourism_project/deployment/smoke_test.py
"""

import os

import joblib
import pandas as pd

MODEL_ARTIFACT_FILENAME = "best_model.joblib"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_ARTIFACT_FILENAME)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Make sure "
            "tourism_project/model_building/train.py has run and its "
            "output has been committed to this repo (the model-training "
            "job's 'Commit trained model to repository' step does this "
            "automatically in CI)."
        )

    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
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
    print("Deployment smoke test passed: the committed model loads and "
          "predicts correctly with the exact input schema app.py uses.")


if __name__ == "__main__":
    main()
