# test.py : loads the saved model and checks it on data it has never seen.

from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# Check that train.py was run first
if not (BASE / "model.pkl").exists() or not (BASE / "data" / "test_data.joblib").exists():
    raise SystemExit("Please run train.py first.")

# Load the saved model and the 20% test data
saved = joblib.load(BASE / "model.pkl")
test = joblib.load(BASE / "data" / "test_data.joblib")
model = saved["model"]
X_test = test["X_test"].reindex(columns=saved["columns"])
y_test = test["y_test"]

# Predict: probability first, then cut at the threshold
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= saved["threshold"]).astype(int)

print("=" * 50)
print("Model            :", saved["model_name"])
print("Test accidents   :", len(y_test))
print("Baseline accuracy:", round(1 - float(y_test.mean()), 4), "(if we always said 'low')")
print("-" * 50)
print("Accuracy         :", round(accuracy_score(y_test, pred), 4))
print("Macro F1         :", round(f1_score(y_test, pred, average="macro"), 4))
print("Precision (high) :", round(precision_score(y_test, pred, zero_division=0), 4))
print("Recall (high)    :", round(recall_score(y_test, pred), 4))
print("ROC-AUC          :", round(roc_auc_score(y_test, prob), 4))
print("=" * 50)
print(classification_report(y_test, pred, target_names=["Low (1-2)", "High (3-4)"]))

# Confusion matrix picture
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "High"], yticklabels=["Low", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix (test set)")
plt.savefig(OUT / "test_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved outputs/test_confusion_matrix.png")