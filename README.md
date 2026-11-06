# Churn Retention Agent (ML + RAG + Agentic AI)

An agent that predicts customer churn, finds out why using support tickets and a retention
playbook (RAG), then drafts a win-back email and a CRM task for human approval.

## Structure
```
Churn Retention Agent/
├── data/                 customers.csv, usage.csv, support_tickets.jsonl (generated)
├── knowledge_base/       retention_playbook.md  <- indexed by the RAG layer
├── src/
│   ├── generate_data.py  synthetic data (replace with your real CSVs)
│   ├── features.py       load + merge tables
│   ├── train.py          gradient-boosting churn model -> models/model.pkl
│   ├── rag.py            TF-IDF retriever over playbook + tickets
│   ├── tools.py          6 agent tools + JSON schemas + guardrails
│   └── agent.py          Claude tool-use loop (offline fallback if no API key)
├── models/model.pkl
├── output/               drafts/, crm_tasks.jsonl
└── requirements.txt
```

## Run
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
python src/agent.py CUST_0007            # offline mode, no key needed
```

To enable the LLM agent, set the key in your shell first:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."    # PowerShell
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...       # macOS/Linux
```

Then run:

```bash
python src/agent.py CUST_0007
```

## How it works
1. `predict_churn` gives probability, risk band and top drivers.
2. `search_tickets` + `retrieve_playbook` (RAG) ground the advice in real context.
3. The agent drafts an email and CRM task. Drafts are never sent; tasks wait for approval.
4. Guardrail: any email offering a discount above 20% is rejected by the tool.

## Demo-data limitations

The checked-in customers, usage, tickets, and model are synthetic. The reported
probabilities and risk bands are for demonstrating the workflow; they are not
validated for real customer decisions. The default risk-band cutoffs (35% and
60%) are illustrative and should be calibrated against representative,
time-separated validation data before operational use. The displayed driver
estimates compare the model prediction with one feature replaced at a time by
the population median or mode. They explain model behavior approximately and
do not establish causal reasons for an individual customer's risk.

The email tool validates the customer ID against the loaded customer table,
confines draft filenames to `output/drafts`, and rejects percentage offers
above 20% in either the subject or body. Discount offers without an explicit
percentage are sent for manager review by being rejected.

## Ideas to extend
- Swap TF-IDF for sentence-transformers + FAISS/Chroma.
- Add an LLM sentiment score on tickets as a model feature.
- Add a Reviewer agent, evals (LLM-as-judge), and an A/B retention outcome loop.
- Use your own data: keep column names or edit `features.py`.

