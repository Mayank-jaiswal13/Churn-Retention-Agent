"""Tools the agent can call. Every tool returns JSON-serialisable data."""
import json
import re
from datetime import datetime
from pathlib import Path

import joblib

from features import FEATURES, NUMERIC, load_frame
from rag import Retriever

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
MAX_DISCOUNT_PCT = 20  # guardrail from the retention playbook

_cache = {}


def _df():
    if "df" not in _cache:
        _cache["df"] = load_frame()
    return _cache["df"]


def _bundle():
    if "bundle" not in _cache:
        path = ROOT / "models" / "model.pkl"
        if not path.exists():
            raise FileNotFoundError("models/model.pkl missing - run: python src/train.py")
        _cache["bundle"] = joblib.load(path)
    return _cache["bundle"]


def _rag():
    if "rag" not in _cache:
        _cache["rag"] = Retriever()
    return _cache["rag"]


def _row(customer_id):
    if not isinstance(customer_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", customer_id):
        return None
    df = _df()
    sel = df[df["customer_id"] == customer_id]
    return None if sel.empty else sel


# ---------------------------------------------------------------- tools
def get_customer_profile(customer_id: str):
    sel = _row(customer_id)
    if sel is None:
        return {"error": f"unknown customer {customer_id}"}
    return json.loads(sel.drop(columns=["churned"]).iloc[0].to_json())


def predict_churn(customer_id: str):
    sel = _row(customer_id)
    if sel is None:
        return {"error": f"unknown customer {customer_id}"}
    bundle, df = _bundle(), _df()
    prob = float(bundle["pipeline"].predict_proba(sel[FEATURES])[0, 1])
    band = "high" if prob >= 0.6 else "medium" if prob >= 0.35 else "low"

    # Local feature ablation: compare this prediction with the same row after
    # replacing one feature at a time with its training-population reference.
    # This is a model explanation, not a causal effect.
    reference = {}
    for feature in NUMERIC:
        reference[feature] = float(df[feature].median())
    for feature in ("plan",):
        reference[feature] = str(df[feature].mode().iloc[0])
    original = float(bundle["pipeline"].predict_proba(sel[FEATURES])[0, 1])
    drivers = []
    for feature, ref_value in reference.items():
        altered = sel[FEATURES].copy()
        altered.loc[:, feature] = ref_value
        changed = float(bundle["pipeline"].predict_proba(altered)[0, 1])
        delta = original - changed
        drivers.append({"feature": feature, "value": sel[feature].iloc[0],
                        "reference_value": ref_value,
                        "probability_change": round(delta, 4)})
    drivers.sort(key=lambda item: item["probability_change"], reverse=True)
    top = [item for item in drivers if item["probability_change"] > 0][:3]
    return {"customer_id": customer_id, "churn_probability": round(prob, 3),
            "risk_band": band, "top_risk_drivers": top,
            "driver_method": "one-feature-at-a-time replacement with the population median/mode; not causal"}


def search_tickets(customer_id: str, query: str = ""):
    return _rag().search_tickets(customer_id, query)


def retrieve_playbook(query: str):
    return _rag().search_playbook(query)


def draft_email(customer_id: str, subject: str, body: str):
    """Saves a DRAFT only. Nothing is ever sent without human approval."""
    if _row(customer_id) is None:
        return {"error": f"unknown or invalid customer {customer_id!r}"}
    issue = _discount_guardrail(f"{subject}\n{body}")
    if issue:
        return {"error": issue}
    d = OUTPUT / "drafts"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{customer_id}.txt"
    project_root = ROOT.resolve()
    output_dir, resolved_dir, resolved_path = OUTPUT.resolve(), d.resolve(), path.resolve()
    if (output_dir.parent != project_root or resolved_dir.parent != output_dir
            or resolved_path.parent != resolved_dir):
        return {"error": "Guardrail: draft path is outside the drafts folder."}
    path.write_text(f"Subject: {subject}\n\n{body}\n")
    return {"status": "draft_saved_not_sent", "path": str(path.relative_to(ROOT))}


def create_crm_task(customer_id: str, action: str, priority: str = "medium"):
    if _row(customer_id) is None:
        return {"error": f"unknown or invalid customer {customer_id!r}"}
    if priority not in {"low", "medium", "high"}:
        return {"error": "priority must be low, medium, or high"}
    OUTPUT.mkdir(exist_ok=True)
    project_root, output_dir = ROOT.resolve(), OUTPUT.resolve()
    task_path = OUTPUT / "crm_tasks.jsonl"
    if output_dir.parent != project_root or task_path.resolve().parent != output_dir:
        return {"error": "Guardrail: CRM task path is outside the project output folder."}
    task = {"created": datetime.now().isoformat(timespec="seconds"), "customer_id": customer_id,
            "action": action, "priority": priority, "status": "pending_approval"}
    with open(task_path, "a") as f:
        f.write(json.dumps(task) + "\n")
    return task


_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
         "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_WORD_NUMBER = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?"


def _parse_word_number(words: str) -> int:
    parts = re.split(r"[- ]", words.lower())
    if len(parts) == 1:
        return _ONES.get(parts[0], _TENS.get(parts[0], 0))
    return _TENS.get(parts[0], 0) + _ONES.get(parts[1], 0)


def _discount_guardrail(text: str):
    amount_matches = list(re.finditer(
        r"\b(\d+(?:\.\d+)?)\s*(?:%|percent(?:age)?|per\s+cent)(?!\w)", text, flags=re.I))
    amounts = [(float(match.group(1)), match.start(), match.end()) for match in amount_matches]
    for match in re.finditer(
            rf"\b({_WORD_NUMBER})\s+(?:percent(?:age)?|per\s+cent)\b", text, flags=re.I):
        amounts.append((float(_parse_word_number(match.group(1))), match.start(), match.end()))
    if any(value > MAX_DISCOUNT_PCT for value, _, _ in amounts):
        return f"Guardrail: discounts above {MAX_DISCOUNT_PCT}% need manager approval. Revise the offer."

    discount_language = re.finditer(r"\b(discount|off|price\s+cut|price\s+reduction|save)\b", text, flags=re.I)
    for term in discount_language:
        nearby_start, nearby_end = max(0, term.start() - 40), min(len(text), term.end() + 40)
        if not any(nearby_start <= start and end <= nearby_end for _, start, end in amounts):
            return ("Guardrail: each discount offer must state an explicit percentage of 20% or less; "
                    "unquantified offers need manager approval.")
    return None


REGISTRY = {f.__name__: f for f in
            (get_customer_profile, predict_churn, search_tickets, retrieve_playbook,
             draft_email, create_crm_task)}


def run_tool(name, args):
    try:
        return REGISTRY[name](**args)
    except Exception as e:  # return errors to the model so it can recover
        return {"error": f"{type(e).__name__}: {e}"}


def _schema(name, desc, props, required):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": required}}


_cid = {"customer_id": {"type": "string", "description": "e.g. CUST_0007"}}
TOOL_SCHEMAS = [
    _schema("get_customer_profile", "Get a customer's plan, tenure, charges and usage metrics.", _cid, ["customer_id"]),
    _schema("predict_churn", "Run the ML model: churn probability, risk band and top risk drivers.", _cid, ["customer_id"]),
    _schema("search_tickets", "Retrieve this customer's support tickets, ranked by relevance to an optional query.",
            {**_cid, "query": {"type": "string"}}, ["customer_id"]),
    _schema("retrieve_playbook", "Search the retention playbook (RAG) for guidance matching a situation.",
            {"query": {"type": "string"}}, ["query"]),
    _schema("draft_email", "Save a win-back email DRAFT (never sent). Discounts over 20% are rejected.",
            {**_cid, "subject": {"type": "string"}, "body": {"type": "string"}},
            ["customer_id", "subject", "body"]),
    _schema("create_crm_task", "Create a follow-up CRM task awaiting human approval.",
            {**_cid, "action": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "medium", "high"]}},
            ["customer_id", "action"]),
]
