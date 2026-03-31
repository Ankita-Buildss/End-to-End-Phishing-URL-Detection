# 🔍 Phishing URL Detection

Detects phishing websites in real-time using **XGBoost** and **ANN (MLP)** models — just paste a URL and get an instant prediction.

## How It Works

A URL goes through three layers of feature extraction — URL structure, DNS/SSL/domain checks, and HTML analysis — producing 14 binary features that feed into the ML model.

A rule-based override catches obvious cases (IP in URL, no SSL + no DNS + brand new domain) before the model even runs.

## Tech Stack

`FastAPI` `XGBoost` `scikit-learn` `MLflow` `BeautifulSoup` `dnspython` `python-whois` `uv`

## Quick Start

```bash
git clone https://github.com/Ankita-Buildss/End-to-End-Phishing-URL-Detection
cd End-to-End-Phishing-URL-Detection

uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

python run_pipeline.py        # train models
uvicorn app:app --reload      # start app → http://localhost:8000
```
