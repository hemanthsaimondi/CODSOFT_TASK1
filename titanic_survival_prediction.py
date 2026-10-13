"""
CODSOFT Data Science Internship - Task 1: Titanic Survival Prediction
Predict whether a passenger survived using passenger attributes.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv("Titanic-Dataset.csv")
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nOverall survival rate: %.1f%%" % (df.Survived.mean() * 100))

# ---------------------------------------------------------------- 2. EDA plots
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
sns.countplot(x="Survived", data=df, ax=ax[0, 0]); ax[0, 0].set_title("Survival count")
sns.barplot(x="Sex", y="Survived", data=df, ax=ax[0, 1]); ax[0, 1].set_title("Survival rate by sex")
sns.barplot(x="Pclass", y="Survived", data=df, ax=ax[0, 2]); ax[0, 2].set_title("Survival rate by class")
sns.histplot(data=df, x="Age", hue="Survived", kde=True, ax=ax[1, 0]); ax[1, 0].set_title("Age distribution")
sns.barplot(x="Embarked", y="Survived", data=df, ax=ax[1, 1]); ax[1, 1].set_title("Survival rate by port")
sns.boxplot(x="Survived", y="Fare", data=df, ax=ax[1, 2]); ax[1, 2].set_ylim(0, 150); ax[1, 2].set_title("Fare vs survival")
plt.tight_layout(); plt.savefig("eda.png", dpi=120); plt.close()

# ---------------------------------------------------------------- 3. Feature engineering
def engineer(d):
    d = d.copy()
    d["Title"] = d["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    d["Title"] = d["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common = ["Mr", "Mrs", "Miss", "Master"]
    d["Title"] = d["Title"].where(d["Title"].isin(common), "Rare")
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    d["HasCabin"] = d["Cabin"].notna().astype(int)
    return d

df = engineer(df)

num_features = ["Age", "Fare", "FamilySize"]
cat_features = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "HasCabin"]
X, y = df[num_features + cat_features], df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ---------------------------------------------------------------- 4. Preprocessing
preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_features),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

# ---------------------------------------------------------------- 5. Compare models (5-fold CV on train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("\n5-fold CV accuracy (train set):")
for name, m in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", m)])
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  {name:20s} {s.mean():.4f} (+/- {s.std():.4f})")

# ---------------------------------------------------------------- 6. Tune the best candidate
pipe = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))])
grid = GridSearchCV(pipe, {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [4, 6, 8, None],
    "clf__min_samples_leaf": [1, 3, 5],
}, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
print("\nBest params:", grid.best_params_)
print("Best CV accuracy: %.4f" % grid.best_score_)

# ---------------------------------------------------------------- 7. Final evaluation on held-out test set
best = grid.best_estimator_
pred = best.predict(X_test)
print("\n=== Test set results ===")
print("Accuracy : %.4f" % accuracy_score(y_test, pred))
print("Precision: %.4f" % precision_score(y_test, pred))
print("Recall   : %.4f" % recall_score(y_test, pred))
print("F1       : %.4f" % f1_score(y_test, pred))
print("\n", classification_report(y_test, pred, target_names=["Died", "Survived"]))

ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Died", "Survived"]).plot(cmap="Blues")
plt.title("Confusion matrix (test set)"); plt.savefig("confusion_matrix.png", dpi=120); plt.close()

# ---------------------------------------------------------------- 8. Feature importance
names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best.named_steps["clf"].feature_importances_, index=names).sort_values().tail(12)
imp.plot(kind="barh", figsize=(8, 5), title="Top feature importances"); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120); plt.close()
print("\nTop features:\n", imp.sort_values(ascending=False).head(6).round(3))
