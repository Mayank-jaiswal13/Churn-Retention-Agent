"""Load and merge raw tables into a model-ready frame."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

NUMERIC = ["tenure_months", "monthly_charges", "avg_logins_per_week",
           "feature_adoption", "support_tickets_30d", "late_payments_6m"]
CATEGORICAL = ["plan"]
FEATURES = NUMERIC + CATEGORICAL


def load_frame() -> pd.DataFrame:
    customers = pd.read_csv(DATA / "customers.csv")
    usage = pd.read_csv(DATA / "usage.csv")
    df = customers.merge(usage, on="customer_id", how="left")
    df[NUMERIC] = df[NUMERIC].fillna(df[NUMERIC].median())
    df["plan"] = df["plan"].fillna("basic")
    return df
