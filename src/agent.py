"""Retention agent.

With ANTHROPIC_API_KEY set, a Claude tool-use loop decides which tools to call.
Without a key, a deterministic offline agent runs the same tools in a fixed order,
so the whole project can be tested for free.

Usage:  python src/agent.py CUST_0007
"""
import os
import sys

from tools import REGISTRY, TOOL_SCHEMAS, run_tool

MODEL = os.getenv("CHURN_AGENT_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 10

SYSTEM = """You are a customer-retention agent. For the given customer:
1. Call predict_churn and get_customer_profile.
2. If risk is medium or high, call search_tickets and retrieve_playbook (use the risk drivers and
   ticket complaints as the query) to ground your advice in the playbook.
3. Draft a short, personalised win-back email with draft_email, then create_crm_task for the account owner.
Rules: never offer a discount above 20%; never promise features not in the playbook; drafts are never sent
automatically. Skip steps 2-3 for low-risk customers.
Finish with a brief report: risk, why, recommended action, and which playbook section you used."""

KEYWORDS = {
    "late_payments_6m": "billing late payment refund",
    "avg_logins_per_week": "low engagement onboarding training",
    "feature_adoption": "low engagement onboarding training",
    "support_tickets_30d": "service quality outage support escalation",
    "monthly_charges": "price increase competitor discount",
    "tenure_months": "long tenure loyal",
}


def run_llm_agent(customer_id: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": f"Assess churn risk for {customer_id} and prepare a retention plan."}]
    for _ in range(MAX_STEPS):
        resp = client.messages.create(model=MODEL, max_tokens=1500, system=SYSTEM,
                                      tools=TOOL_SCHEMAS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                print(f"  [tool] {b.name}({b.input})")
                import json
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(run_tool(b.name, b.input))})
        messages.append({"role": "user", "content": results})
    return "Stopped: step limit reached."


def run_offline_agent(customer_id: str) -> str:
    def call(name, **kw):
        print(f"  [tool] {name}({kw})")
        return run_tool(name, kw)

    profile = call("get_customer_profile", customer_id=customer_id)
    if "error" in profile:
        return profile["error"]
    pred = call("predict_churn", customer_id=customer_id)
    if "error" in pred:
        return f"Churn prediction unavailable: {pred['error']}"
    prob, band = pred["churn_probability"], pred["risk_band"]
    drivers = pred["top_risk_drivers"]
    lines = [f"Customer {customer_id} ({profile['plan']} plan, {profile['tenure_months']} months)",
             f"Churn probability: {prob:.0%} ({band} risk)"]
    if drivers:
        lines.append("Top drivers: " + "; ".join(
            f"{d['feature']}={d['value']} (reference {d['reference_value']}; "
            f"estimated risk change {d['probability_change']:+.1%})" for d in drivers))
        lines.append("Driver estimates are one-feature-at-a-time model comparisons, not causal effects.")
    if band == "low":
        lines.append("Recommendation: no action needed; keep on normal engagement cadence.")
        return "\n".join(lines)

    query = " ".join(KEYWORDS.get(d["feature"], "") for d in drivers)
    tickets = call("search_tickets", customer_id=customer_id, query=query)
    if isinstance(tickets, dict) and "error" in tickets:
        lines.append(f"Ticket search unavailable: {tickets['error']}")
        tickets = []
    if tickets:
        query += " " + " ".join(t["text"] for t in tickets[:2])
    guidance = call("retrieve_playbook", query=query)
    if isinstance(guidance, dict) and "error" in guidance:
        lines.append(f"Playbook search unavailable: {guidance['error']}")
        guidance = []
    best = guidance[0] if guidance else {"title": "General", "text": "Reach out personally."}

    lines.append("Recent tickets: " + (" | ".join(t["text"] for t in tickets[:2]) or "none"))
    lines.append(f"Playbook match: {best['title']} - {best['text'][:200]}")

    body = (f"Hi,\n\nWe noticed you may not be getting full value from your {profile['plan']} plan "
            f"and we want to put that right. {best['text'].split('.')[0]}.\n\n"
            "Can we book 15 minutes this week?\n\nBest,\nYour Customer Success Team")
    draft = call("draft_email", customer_id=customer_id, subject="We'd like to help", body=body)
    task = call("create_crm_task", customer_id=customer_id, action=f"Call customer: {best['title']}",
                priority="high" if band == "high" else "medium")
    if isinstance(draft, dict) and draft.get("status") == "draft_saved_not_sent":
        lines.append("Email draft saved and not sent.")
    else:
        lines.append(f"Email draft not saved: {draft.get('error', 'unexpected tool response')}")
    if isinstance(task, dict) and task.get("status") == "pending_approval":
        lines.append("CRM task created and pending approval.")
    else:
        lines.append(f"CRM task not created: {task.get('error', 'unexpected tool response')}")
    return "\n".join(lines)


def run(customer_id: str) -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return run_llm_agent(customer_id)
    print("(no ANTHROPIC_API_KEY found - running offline agent)")
    return run_offline_agent(customer_id)


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "CUST_0007"
    print(run(cid))
