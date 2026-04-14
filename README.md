# FactorLens

Understand why your portfolio moved — powered by Fama-French factor analysis and Claude AI.

Submit any financial information (typed portfolio, plain text like "I lost 5% this week", or a screenshot of your brokerage app) and get a plain-English explanation of what happened, why, and what to do next.

## Setup

```bash
cd FactorLens
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Run

```bash
source venv/bin/activate
python3 app.py
```

Open [http://localhost:5000](http://localhost:5000)

## How it works

1. **Input** — Paste a portfolio (e.g. `AAPL 40%, MSFT 30%, NVDA 30%`), describe a loss/gain in plain text, or upload a screenshot
2. **Data** — Fetches live prices via yfinance, sector performance, VIX, and Fama-French 5-factor data from Ken French's library
3. **Analysis** — Runs OLS regression against the 5 factors (Market, Size, Value, Profitability, Investment) to compute factor loadings
4. **Explanation** — Sends everything to Claude which explains in plain English: what happened, why (factor attribution), and 3 actionable suggestions
