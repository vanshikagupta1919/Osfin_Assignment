# Osfin Assignment

AI-Powered Dispute Assistant

This repository implements an AI assistant for payment dispute handling in fintech/banking workflows. It combines semantic analysis of dispute descriptions with transaction evidence to classify disputes, suggest next actions, and answer natural-language questions for support teams.

Scope

Classify each dispute into one of: DUPLICATE_CHARGE, FAILED_TRANSACTION, FRAUD, REFUND_PENDING, OTHERS

Suggest a next action: Auto-refund, Manual review, Escalate to bank, Mark as potential fraud, Ask for more info

Provide a prompt interface agents can use to query the latest disputes

What you need before running

Two CSVs in the project root:

disputes.csv with at least: dispute_id, description, txn_id, optional fields like created_at, status

transactions.csv with at least: txn_id, amount, status, optional fields like customer_id, created_at

Python 3.9+ and the packages listed below

An API key for Task 3 (prompt-based insights with LLM parsing). Set one of:

OPENAI_API_KEY for OpenAI-compatible models

or OPENROUTER_API_KEY for OpenRouter-compatible models
The script will check the environment; if no key is present, it will prompt you at runtime.

Installation

pip install -r requirements.txt


Typical requirements include: pandas, numpy, scikit-learn, transformers, torch (CPU), gradio (optional for web UI), rapidfuzz (optional if you enable fuzzy duplicate detection), matplotlib (optional for charts).

How it works (high level)

description understanding: zero-shot or lightweight ML converts the free-text description into category likelihoods

transaction evidence: rules verify and refine the label using actual data (status, duplicate patterns, refund flags)

hybrid decision: semantic signal is combined with transaction signals; confirmations from data can override text-only guesses

resolution mapping: the final category maps to the recommended next action

Duplicate detection logic

base normalization: transactions with a txn_id suffix like _DUP1 are normalized by removing the suffix so that TX123 and TX123_DUP1 resolve to the same primary transaction id

optional fuzzy detection (enable via USE_FUZZY=1):

character similarity on string fields (e.g., txn_id stripped of suffix noise, merchant, channel, status)

numeric proximity on amount (small absolute or percentage difference treated as similar)

temporal proximity on created_at (e.g., within a small window such as 5–10 minutes)

A composite score is computed from these components; pairs exceeding a threshold are marked duplicates and the smaller txn_id becomes the canonical id for both rows

Classification details

DUPLICATE_CHARGE: confirmed when description implies a duplicate and transaction evidence shows the same customer and the same amount posted twice within a short window, or a normalized duplicate ID is present

FAILED_TRANSACTION: description or status indicates failed/declined/timeout/reversed; if status shows captured/settled, it downgrades to Manual review later

FRAUD: description implies unauthorized or unknown; flagged even if status is successful

REFUND_PENDING: description mentions refund, and status or refund fields indicate pending or in-process

OTHERS: everything else that lacks strong semantic or transactional signals

Resolution mapping

DUPLICATE_CHARGE → Auto-refund if captured; otherwise void or allow auth to expire

FAILED_TRANSACTION → Ask for more info if no capture; Manual review if data shows capture despite “failed” description

FRAUD → Mark as potential fraud; freeze refunds; escalate for KYC/chargeback handling

REFUND_PENDING → Manual review by default; Escalate to bank for high-value cases or aging refunds

OTHERS → Manual review

Outputs produced after a standard run

classified_disputes.csv with columns: dispute_id, predicted_category, confidence, explanation

resolutions.csv with columns: dispute_id, suggested_action, justification

Running the pipeline

default end-to-end (classification + resolutions):

python osfin_ai_assignment.py


command-line query interface:

MODE=cli python osfin_ai_assignment.py


web interface for agents (Gradio):

MODE=gradio python osfin_ai_assignment.py


API key setup for Task 3 (required when using LLM parsing in CLI/Web modes):

export OPENAI_API_KEY=your_key_here
# or
export OPENROUTER_API_KEY=your_key_here


Prompt examples for agents

How many duplicate charges today?

List unresolved fraud disputes

Break down disputes by type
These work in both CLI and Gradio modes. With an API key configured, free-text prompts can be interpreted more flexibly by the LLM parser; without a key, the script falls back to rule-based parsing of a fixed set of phrases.

Assumptions about input data

disputes.csv contains one row per dispute with a stable dispute_id and description

transactions.csv contains one row per transaction; if duplicates exist they may appear as *_DUP1; these are normalized automatically

timestamps are either ISO strings or parseable by pandas.to_datetime

amounts are numeric and in the same currency

Troubleshooting

missing API key: set OPENAI_API_KEY or OPENROUTER_API_KEY; otherwise, the assistant runs, but LLM-based parsing in Task 3 will prompt or be disabled

empty outputs: ensure description and txn_id columns are present and non-empty in disputes.csv; ensure transactions.csv includes txn_id and amount/status

Poor duplicate detection: enable fuzzy detection with USE_FUZZY=1 and adjust the similarity threshold; confirm timestamps are present to leverage time proximity

Data handling and privacy

No personal data is required beyond the fields used for reconciliation signals

If you log prompts or responses in Task 3, review storage and retention policies appropriate for your environment
