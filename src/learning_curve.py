# src/learning_curve.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/telecom_churn.csv")

# -----------------------------
# Data preprocessing
# -----------------------------
# حذف عمود الـ ID لأنه غير مفيد للتنبؤ
df = df.drop("customer_id", axis=1)

# فصل الـ features عن الـ target
X = df.drop("churned", axis=1)
y = df["churned"]

# تحويل الأعمدة النصية إلى أرقام
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# Model pipeline
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

# -----------------------------
# Stratified Cross Validation
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# Learning curve
# -----------------------------
train_sizes, train_scores, val_scores = learning_curve(
    model,
    X,
    y,
    cv=cv,
    scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

# -----------------------------
# Mean and standard deviation
# -----------------------------
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# -----------------------------
# Plot learning curve
# -----------------------------
plt.figure(figsize=(8, 6))

plt.plot(train_sizes, train_mean, marker="o", label="Training Score")
plt.plot(train_sizes, val_mean, marker="o", label="Validation Score")

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2
)

plt.fill_between(
    train_sizes,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.2
)

plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.title("Learning Curve - Logistic Regression")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("plots/learning_curve.png")
plt.show()