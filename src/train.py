"""Train the baseline churn model and save it to models/model.pkl."""
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import CATEGORICAL, FEATURES, NUMERIC, load_frame

MODELS = Path(__file__).resolve().parent.parent / "models"


def main():
    df = load_frame()
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    pipe = Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=42))])
    pipe.fit(X_tr, y_tr)

    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    print(f"Test ROC-AUC: {auc:.3f}")

    importances = dict(zip(NUMERIC, pipe.named_steps["clf"].feature_importances_[: len(NUMERIC)]))
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "importances": importances,
                 "medians": df[NUMERIC].median().to_dict(),
                 "stds": df[NUMERIC].std().to_dict(),
                 "training_data_note": "Synthetic demo data; not validated for production decisions."},
                MODELS / "model.pkl")
    print("Saved models/model.pkl")


if __name__ == "__main__":
    main()
