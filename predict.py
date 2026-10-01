"""Predict churn for one customer using the exported sklearn pipeline."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL = Path(__file__).parent / "artifacts" / "telco_churn_model.joblib"


def _load_payload(value: str | None) -> dict:
    if value is None:
        raw = sys.stdin.read()
    else:
        candidate = Path(value)
        raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Input must be one JSON object representing a customer.")
    return payload


def predict_customer(payload: dict, model) -> dict:
    customer = dict(payload)
    customer.pop("customerID", None)

    expected_features = list(model.feature_names_in_)
    missing_features = sorted(set(expected_features) - customer.keys())
    if missing_features:
        raise ValueError(f"Missing required customer fields: {', '.join(missing_features)}")

    customer["TotalCharges"] = pd.to_numeric(customer.get("TotalCharges"), errors="coerce")
    row = pd.DataFrame([customer]).reindex(columns=expected_features)
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "churn_label": "Yes" if prediction else "No",
        "churn_probability": probability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict churn for one Telco customer.")
    parser.add_argument(
        "input",
        nargs="?",
        help="JSON file path or a raw JSON object; reads stdin when omitted.",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Exported joblib pipeline path.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        parser.error(f"Model not found at {model_path}. Run the notebook export cell first.")

    try:
        payload = _load_payload(args.input)
        result = predict_customer(payload, joblib.load(model_path))
    except (json.JSONDecodeError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
