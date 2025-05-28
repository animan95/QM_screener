# evaluate_and_plot.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import joblib
import sys

# Load saved model + test data
model = joblib.load(sys.argv[1])  # e.g., "model_rf.pkl"
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Predict
proba = model.predict_proba(X_test)[:, 1]
preds = (proba > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, proba)
print(f"\n📊 {sys.argv[1]} Results:")
print(f"Accuracy: {acc:.3f}")
print(f"ROC-AUC:  {auc:.3f}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, proba)
plt.plot(fpr, tpr, label=f"{sys.argv[1]} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {sys.argv[1]}")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{sys.argv[1]}_roc.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Candidate", "Drug-like"])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"{sys.argv[1]} - Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{sys.argv[1]}_confusion.png")
plt.close()

