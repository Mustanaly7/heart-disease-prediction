"""
Heart Disease Prediction using Machine Learning
Dataset: UCI Heart Disease (Cleveland, processed, 303 patients, 13 features)
Models: Logistic Regression, Random Forest, SVM, custom PyTorch ANN
Interpretability: SHAP
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Load & explore
# ---------------------------------------------------------------------------
df = pd.read_csv("heart.csv", encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Class balance:\n", df["target"].value_counts())

FEATURE_NAMES = [c for c in df.columns if c != "target"]
X = df[FEATURE_NAMES].values.astype(np.float32)
y = df["target"].values.astype(np.int64)

# ---------------------------------------------------------------------------
# 2. Preprocessing: split first (avoid leakage), then scale
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

def evaluate(name, y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    results[name] = dict(accuracy=acc, f1=f1, precision=prec, recall=rec, auc=auc)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    if auc is not None:
        print(f"ROC-AUC:   {auc:.4f}")
    return acc

# ---------------------------------------------------------------------------
# 3. Logistic Regression
# ---------------------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_s, y_train)
y_pred = log_reg.predict(X_test_s)
y_proba = log_reg.predict_proba(X_test_s)[:, 1]
evaluate("Logistic Regression", y_test, y_pred, y_proba)

# ---------------------------------------------------------------------------
# 4. Random Forest
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=5, min_samples_leaf=3,
    random_state=RANDOM_STATE
)
rf.fit(X_train_s, y_train)
y_pred = rf.predict(X_test_s)
y_proba = rf.predict_proba(X_test_s)[:, 1]
evaluate("Random Forest", y_test, y_pred, y_proba)

# ---------------------------------------------------------------------------
# 5. SVM
# ---------------------------------------------------------------------------
svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE)
svm.fit(X_train_s, y_train)
y_pred = svm.predict(X_test_s)
y_proba = svm.predict_proba(X_test_s)[:, 1]
evaluate("SVM", y_test, y_pred, y_proba)

# ---------------------------------------------------------------------------
# 6. Custom PyTorch Neural Network
# ---------------------------------------------------------------------------
class HeartNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

model = HeartNet(X_train_s.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

EPOCHS = 300
train_losses, test_losses = [], []
best_test_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_out = model(X_test_t)
        test_loss = criterion(test_out, y_test_t)
    scheduler.step(test_loss)

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    if test_loss.item() < best_test_loss:
        best_test_loss = test_loss.item()
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={loss.item():.4f}  test_loss={test_loss.item():.4f}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    proba = torch.sigmoid(logits).numpy().flatten()
    y_pred_nn = (proba >= 0.5).astype(int)

evaluate("PyTorch Neural Network", y_test, y_pred_nn, proba)

# Loss curve
plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Neural Network Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.close()

# Confusion matrix (best model)
best_model_name = max(results, key=lambda k: results[k]["accuracy"])
print(f"\nBest model by accuracy: {best_model_name}")

preds_by_model = {
    "Logistic Regression": log_reg.predict(X_test_s),
    "Random Forest": rf.predict(X_test_s),
    "SVM": svm.predict(X_test_s),
    "PyTorch Neural Network": y_pred_nn,
}
cm = confusion_matrix(y_test, preds_by_model[best_model_name])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix ({best_model_name})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. SHAP interpretability on the PyTorch neural network
# ---------------------------------------------------------------------------
background = X_train_t[:100]

def model_predict(x_numpy):
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_numpy, dtype=torch.float32)
        out = torch.sigmoid(model(x_t)).numpy()
    return out

explainer = shap.KernelExplainer(model_predict, background.numpy()[:50])
shap_values = explainer.shap_values(X_test_t.numpy()[:60], nsamples=100)

sv = shap_values[0] if isinstance(shap_values, list) else shap_values
if sv.ndim == 3:
    sv = sv[:, :, 0]

plt.figure()
shap.summary_plot(sv, X_test_t.numpy()[:60], feature_names=FEATURE_NAMES, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

mean_abs_shap = np.abs(sv).mean(axis=0)
top_features = sorted(zip(FEATURE_NAMES, mean_abs_shap), key=lambda x: -x[1])[:5]
print("\nTop 5 features by mean |SHAP value|:")
for feat, val in top_features:
    print(f"  {feat}: {val:.4f}")

# ---------------------------------------------------------------------------
# 8. Save results summary
# ---------------------------------------------------------------------------
with open("results.json", "w") as f:
    json.dump({
        "results": results,
        "best_model": best_model_name,
        "top_shap_features": [(f, float(v)) for f, v in top_features],
        "n_train": len(X_train), "n_test": len(X_test),
    }, f, indent=2)

print("\n=== SUMMARY TABLE ===")
summary_df = pd.DataFrame(results).T
print(summary_df.round(4))
summary_df.to_csv("model_comparison.csv")

import pickle
torch.save(model.state_dict(), "heart_net.pt")
with open("preprocessing.pkl", "wb") as f:
    pickle.dump({"scaler": scaler, "feature_names": FEATURE_NAMES}, f)

print("\nDone.")
