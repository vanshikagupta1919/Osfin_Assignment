# Osfin Assignment

# AI-Powered Dispute Assistant

This repository implements an AI assistant for handling payment disputes in fintech/banking workflows. It combines semantic analysis of dispute descriptions with transaction evidence to classify disputes, suggest next actions, and answer natural-language questions for support teams.

# Scope

- Classify each dispute into one of: DUPLICATE_CHARGE, FAILED_TRANSACTION, FRAUD, REFUND_PENDING, OTHERS
- Suggest a next action: Auto-refund, Manual review, Escalate to bank, Mark as potential fraud, Ask for more info
- Provide a prompt interface that agents can use to query the latest disputes

# What you need before running

- Two CSVs in the project root: disputes.csv and transactions.csv 
- Python 3.9+ and the packages listed below
- An OpenRouter API key for Task 3 (prompt-based insights with LLM parsing). The code will prompt you at runtime to submit the key.
- Typical requirements include: pandas, numpy, scikit-learn, transformers, torch (CPU), gradio (optional for web UI), rapidfuzz (optional if you enable fuzzy duplicate detection), and matplotlib (optional for charts)

# How it works (high level)

- Zero-shot classification converts the free-text description into category likelihoods (using a pre-trained transformer model, facebook/bart-large-mnli from HuggingFace)
- Transaction evidence: rules verify and refine the label using actual data (status, duplicate patterns, refund flags)
- Hybrid decision: semantic signal is combined with transaction signals; confirmations from data can override text-only guesses
- Resolution mapping: the final category maps to the recommended next action
- For natural language queries, the system integrates with OpenRouter API to use LLM models (here using deepseek/deepseek-chat-v3.1:free) to parse and answer user prompts
- Generates charts showing dispute trends over time and case history transitions.
  
# Outputs produced after a standard run

- Classified disputes: classified_disputes.csv
- Resolution suggestions: resolutions.csv 
- Fuzzy duplicate detection results (classified_disputes_fuzzy.csv)
- It also provides both a command-line interface and a Gradio web interface for prompt-based insights
- Charts to visualize dispute trends and case histories


