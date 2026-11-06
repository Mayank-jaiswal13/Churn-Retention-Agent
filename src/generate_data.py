"""Generate synthetic customers, usage and support tickets (stand-in for your real data)."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
N = 2000

plans = rng.choice(["basic", "standard", "premium"], N, p=[0.45, 0.4, 0.15])
price = {"basic": 20, "standard": 45, "premium": 90}
customers = pd.DataFrame({
    "customer_id": [f"CUST_{i:04d}" for i in range(N)],
    "tenure_months": rng.integers(1, 60, N),
    "plan": plans,
})
customers["monthly_charges"] = customers["plan"].map(price) + rng.normal(0, 3, N).round(2)

usage = pd.DataFrame({
    "customer_id": customers["customer_id"],
    "avg_logins_per_week": rng.gamma(3, 2, N).round(1),
    "feature_adoption": rng.beta(2, 3, N).round(2),
    "support_tickets_30d": rng.poisson(1.2, N),
    "late_payments_6m": rng.poisson(0.5, N),
})

logit = (
    -1.0
    - 0.04 * customers["tenure_months"]
    - 0.18 * usage["avg_logins_per_week"]
    - 1.5 * usage["feature_adoption"]
    + 0.45 * usage["support_tickets_30d"]
    + 0.5 * usage["late_payments_6m"]
    + 0.01 * customers["monthly_charges"]
    + 1.5
)
p = 1 / (1 + np.exp(-logit))
customers["churned"] = (rng.random(N) < p).astype(int)

churn_texts = [
    "The price went up again and I can't justify it. Thinking of switching to a competitor.",
    "Third outage this month. Dashboard keeps timing out and support is slow.",
    "I barely use the product anymore, features are too complicated for our team.",
    "Billing charged me twice and nobody has fixed it. Very frustrated.",
    "Competitor offers the same thing cheaper. Please explain why I should stay.",
]
ok_texts = [
    "Quick question about exporting reports, otherwise happy with the service.",
    "Love the new dashboard. Can I add two more team members?",
    "Thanks for the fast fix last week. Everything works well now.",
    "How do I enable the integration with our CRM?",
]
with open(DATA / "support_tickets.jsonl", "w") as f:
    for cid, churned, n in zip(customers["customer_id"], customers["churned"], usage["support_tickets_30d"]):
        for _ in range(min(int(n), 2)):
            pool = churn_texts if (churned and rng.random() < 0.8) else ok_texts
            f.write(json.dumps({"customer_id": cid, "text": str(rng.choice(pool))}) + "\n")

customers.to_csv(DATA / "customers.csv", index=False)
usage.to_csv(DATA / "usage.csv", index=False)
print(f"customers={len(customers)} churn_rate={customers['churned'].mean():.2%}")
