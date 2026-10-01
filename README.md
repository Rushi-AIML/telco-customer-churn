# Telco Customer Churn

End-to-end churn classification for the supplied IBM Telco Customer Churn CSV. The notebook performs EDA, leakage-safe preprocessing, stratified validation, model comparison, test-set evaluation, and exports the selected sklearn pipeline. `predict.py` loads that exported pipeline and returns a binary prediction plus churn probability for one customer.

## Setup

Run these commands from this directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Keep `WA_Fn-UseC_-Telco-Customer-Churn.csv` beside `telco_churn_analysis.ipynb`. Open the notebook in VS Code or launch Jupyter:

```powershell
jupyter notebook telco_churn_analysis.ipynb
```

Run the notebook cells in order. The final training/export cells write the deployable pipeline to `artifacts/telco_churn_model.joblib`.

## Inference

Save a single customer object using the CSV's feature names (the notebook includes the complete sample payload) as `customer.json`, then run:

```powershell
python predict.py customer.json
```

The script also accepts raw JSON as its argument or JSON from standard input. It emits JSON in this shape:

```json
{
  "prediction": 0,
  "churn_label": "No",
  "churn_probability": 0.12
}
```

`prediction` is `1` for churn and `0` otherwise. The default decision threshold is 0.5; tune it to business costs and retention capacity before operational use. `customerID` may be included but is ignored. Unknown categorical values are handled by the fitted encoder, and blank `TotalCharges` values are handled by the pipeline imputer.

## Model results

Run the notebook to populate the comparison table with held-out ROC-AUC, average precision (PR-AUC), precision, recall, and F1 for logistic regression and the tuned random forest. Hyperparameters are selected using stratified cross-validation on the training partition; the test partition is held out for final comparison. Accuracy is not sufficient by itself because the classes are imbalanced (about 26.5% churn): a majority-class predictor can have deceptively high accuracy while failing to identify churners.

| Iteration | Training CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test precision | Test recall | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression baseline | 0.8459 +/- 0.0124 | 0.8413 | 0.6326 | 0.5043 | 0.7834 | 0.6136 |
| Tuned random forest (10 candidates) | 0.8467 | 0.8423 | 0.6507 | 0.5339 | 0.7781 | 0.6333 |

The tuned random forest was selected by training-CV ROC-AUC and exported to `artifacts/telco_churn_model.joblib`. At threshold 0.5 its test accuracy was 0.7608; accuracy is included for context, not used as the main selection metric.

## Top improvements with two more days

1. Establish data and prediction contracts with schema validation, automated unit/integration tests, CI, and explicit model/data versioning.
2. Calibrate probabilities and select a decision threshold from retention economics; evaluate subgroup performance and fairness before deployment.
3. Add monitoring for input/data drift, prediction quality, latency, and failures, plus a documented retraining and rollback process.
