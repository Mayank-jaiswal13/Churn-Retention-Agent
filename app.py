"""Streamlit dashboard for the churn retention demo."""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from features import load_frame  # noqa: E402
from tools import create_crm_task, draft_email, predict_churn, retrieve_playbook, search_tickets  # noqa: E402


st.set_page_config(page_title="Customer Retention", page_icon="↩️", layout="wide")
st.title("Customer Retention Dashboard")
st.caption("Explore churn risk, understand recent feedback, and prepare a follow-up for human review.")
st.info("Demo only: this app uses synthetic customer data. Risk estimates are not validated for real decisions.", icon="ℹ️")


@st.cache_data
def customers():
    return load_frame()


try:
    frame = customers()
except Exception as exc:
    st.error(f"Could not load the customer data: {exc}")
    st.stop()

customer_ids = frame["customer_id"].astype(str).tolist()
selected_id = st.sidebar.selectbox("Choose a customer", customer_ids, index=customer_ids.index("CUST_0007") if "CUST_0007" in customer_ids else 0)
row = frame.loc[frame["customer_id"] == selected_id].iloc[0]

prediction = predict_churn(selected_id)
if "error" in prediction:
    st.error(f"Could not predict churn: {prediction['error']}")
    st.stop()

probability = prediction["churn_probability"]
band = prediction["risk_band"].title()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Churn risk", band)
col2.metric("Estimated likelihood", f"{probability:.0%}")
col3.metric("Plan", str(row["plan"]).title())
col4.metric("Customer tenure", f"{int(row['tenure_months'])} months")

left, right = st.columns([1, 1])
with left:
    st.subheader("What may be driving the risk")
    drivers = prediction["top_risk_drivers"]
    if drivers:
        driver_frame = pd.DataFrame([
            {"Factor": d["feature"].replace("_", " ").title(),
             "Customer value": d["value"],
             "Reference": d["reference_value"],
             "Estimated risk change": f"{d['probability_change']:+.1%}"}
            for d in drivers
        ])
        st.dataframe(driver_frame, hide_index=True, use_container_width=True)
        st.caption("Approximate model comparisons, not proof of cause.")
    else:
        st.write("No strong positive risk drivers were identified by the demo model.")

with right:
    st.subheader("Recent support messages")
    query = " ".join(d["feature"].replace("_", " ") for d in drivers)
    tickets = search_tickets(selected_id, query=query)
    if isinstance(tickets, dict) and "error" in tickets:
        st.warning(tickets["error"])
        tickets = []
    if tickets:
        for ticket in tickets[:3]:
            st.markdown(f"- {ticket['text']}")
    else:
        st.write("No support messages found for this customer.")

st.subheader("Suggested guidance")
playbook_query = " ".join([query, *[ticket["text"] for ticket in tickets[:3]]])
guidance = retrieve_playbook(query=playbook_query or "customer retention")
if isinstance(guidance, dict) and "error" in guidance:
    st.warning(guidance["error"])
    guidance = []
suggestion = guidance[0] if guidance else {"title": "Personal outreach", "text": "Reach out personally and ask how you can help."}
st.markdown(f"**{suggestion['title']}**")
st.write(suggestion["text"])

if band == "Low":
    st.success("Suggested next step: continue the usual customer check-in schedule.")
else:
    st.subheader("Prepare a follow-up")
    email_body = (
        f"Hi,\n\nWe want to make sure you are getting good value from your {row['plan']} plan. "
        f"{suggestion['text'].split('.')[0]}.\n\nWould you be available for a 15-minute chat this week?\n\n"
        "Best,\nYour Customer Success Team"
    )
    subject = st.text_input("Email subject", "We'd like to help")
    email_body = st.text_area("Draft email (review before use)", email_body, height=180)
    task_action = st.text_input("CRM follow-up", f"Call customer: {suggestion['title']}")
    if st.button("Save draft and create review task", type="primary"):
        draft_result = draft_email(selected_id, subject, email_body)
        if "error" in draft_result:
            st.error(draft_result["error"])
        else:
            task_result = create_crm_task(selected_id, task_action, "high" if band == "High" else "medium")
            if "error" in task_result:
                st.error(f"Draft saved, but the CRM task was not created: {task_result['error']}")
            else:
                st.success("Email draft saved (not sent). CRM task created and awaiting approval.")

st.caption("The app never sends email. Saved drafts and CRM tasks are local demo records awaiting human review.")
