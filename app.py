import os
import json
import base64
import traceback
from datetime import datetime, timedelta
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from scipy import stats
import anthropic

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

CLAUDE_MODEL = "claude-sonnet-4-20250514"

MARKET_TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "TLT": "20Y Treasury",
    "GLD": "Gold",
    "DX-Y.NYB": "US Dollar",
}

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

COMMON_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
    "BRK-B", "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS",
    "BAC", "XOM", "PFE", "KO", "PEP", "CSCO", "INTC", "AMD", "NFLX",
    "CRM", "ORCL", "ADBE", "PYPL", "QCOM", "TXN", "AVGO", "COST",
    "T", "VZ", "NKE", "MCD", "SBUX", "BA", "CAT", "GS", "MS",
    "SPY", "QQQ", "IWM", "VOO", "VTI", "ARKK", "VGT", "SCHD",
]


def extract_tickers(text: str) -> list[str]:
    """Pull stock tickers out of free-form text."""
    if not text:
        return []
    import re
    words = re.findall(r'\b[A-Z]{1,5}(?:-[A-Z]{1,2})?\b', text.upper())
    noise = {
        "I", "A", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF",
        "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO",
        "TO", "UP", "US", "WE", "THE", "AND", "BUT", "FOR", "NOT", "YOU",
        "ALL", "ANY", "CAN", "HAD", "HER", "HIS", "HOW", "ITS", "MAY",
        "NEW", "NOW", "OLD", "OUR", "OUT", "OWN", "SAY", "SHE", "TOO",
        "USE", "HAS", "WAS", "WHO", "WHY", "YET", "ARE", "DID", "GET",
        "HIM", "LET", "PUT", "RAN", "SET", "THIS", "THAT", "WITH",
        "FROM", "HAVE", "JUST", "LIKE", "LOST", "MADE", "MUCH", "SOME",
        "THAN", "THEM", "THEN", "VERY", "WHAT", "WHEN", "WILL", "YOUR",
        "BEEN", "DOWN", "EACH", "OVER", "SUCH", "WEEK", "LAST", "YEAR",
        "P", "L", "PNL", "ETF", "USD", "EUR", "GBP", "ROI", "YTD",
        "MTD", "QTD", "BPS", "IPO", "CEO", "CFO", "COO", "CTO",
    }
    seen = set()
    tickers = []
    for w in words:
        if w not in noise and w not in seen and w in COMMON_TICKERS:
            seen.add(w)
            tickers.append(w)
    return tickers


def parse_portfolio(text: str) -> dict[str, float] | None:
    """Try to parse 'AAPL 40%, MSFT 30%, ...' style input into weights."""
    import re
    pattern = r'([A-Z]{1,5}(?:-[A-Z]{1,2})?)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%'
    matches = re.findall(pattern, text.upper())
    if len(matches) >= 2:
        portfolio = {}
        for ticker, weight in matches:
            portfolio[ticker] = float(weight) / 100.0
        return portfolio
    return None


def fetch_market_data() -> dict:
    """Get current market snapshot."""
    data = {}
    tickers_to_fetch = list(MARKET_TICKERS.keys()) + ["^VIX", "^TNX"]
    try:
        bulk = yf.download(tickers_to_fetch, period="5d", progress=False, threads=True)
        close = bulk["Close"] if "Close" in bulk.columns.get_level_values(0) else bulk
        for ticker in tickers_to_fetch:
            if ticker in close.columns:
                prices = close[ticker].dropna()
                if len(prices) >= 2:
                    current = float(prices.iloc[-1])
                    prev = float(prices.iloc[-2])
                    change_pct = ((current - prev) / prev) * 100
                    label = MARKET_TICKERS.get(ticker, ticker.replace("^", ""))
                    data[label] = {
                        "price": round(current, 2),
                        "change_pct": round(change_pct, 2),
                    }
    except Exception:
        pass
    return data


def fetch_ticker_data(tickers: list[str], period: str = "1mo") -> dict:
    """Fetch price data and compute returns for a list of tickers."""
    result = {}
    if not tickers:
        return result
    try:
        bulk = yf.download(tickers, period=period, progress=False, threads=True)
        close = bulk["Close"] if len(tickers) > 1 else bulk[["Close"]].rename(columns={"Close": tickers[0]})
        for t in tickers:
            col = t if t in close.columns else None
            if col is None:
                continue
            prices = close[col].dropna()
            if len(prices) >= 2:
                current = float(prices.iloc[-1])
                start = float(prices.iloc[0])
                period_return = ((current - start) / start) * 100
                daily_returns = prices.pct_change().dropna()
                result[t] = {
                    "current_price": round(current, 2),
                    "period_return_pct": round(period_return, 2),
                    "daily_returns": daily_returns.values.tolist(),
                    "dates": [d.strftime("%Y-%m-%d") for d in daily_returns.index],
                }
    except Exception:
        pass
    return result


def fetch_sector_performance() -> dict:
    """Get sector ETF performance."""
    sectors = {}
    try:
        tickers = list(SECTOR_ETFS.keys())
        bulk = yf.download(tickers, period="5d", progress=False, threads=True)
        close = bulk["Close"]
        for etf, name in SECTOR_ETFS.items():
            if etf in close.columns:
                prices = close[etf].dropna()
                if len(prices) >= 2:
                    change = ((float(prices.iloc[-1]) - float(prices.iloc[-2])) / float(prices.iloc[-2])) * 100
                    sectors[name] = round(change, 2)
    except Exception:
        pass
    return sectors


def fetch_fama_french_factors(days: int = 30) -> pd.DataFrame | None:
    """Fetch Fama-French 5-factor daily data."""
    try:
        from pandas_datareader.famafrench import get_available_datasets
        import pandas_datareader.data as web

        end = datetime.now()
        start = end - timedelta(days=days + 60)
        ff = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start, end)
        df = ff[0]
        df = df / 100.0  # convert from percentage to decimal
        return df.tail(days)
    except Exception:
        return None


def run_factor_regression(portfolio_returns: np.ndarray, ff_data: pd.DataFrame) -> dict | None:
    """OLS regression of portfolio returns against Fama-French 5 factors."""
    try:
        n = min(len(portfolio_returns), len(ff_data))
        if n < 10:
            return None
        pr = portfolio_returns[-n:]
        ff = ff_data.tail(n)

        excess_returns = pr - ff["RF"].values

        factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        X = ff[factor_names].values
        X = np.column_stack([np.ones(n), X])  # add intercept
        y = excess_returns

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        residuals = y - y_hat
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        loadings = {}
        loadings["alpha_annualized_pct"] = round(float(beta[0]) * 252 * 100, 2)
        for i, name in enumerate(factor_names):
            loadings[name] = round(float(beta[i + 1]), 3)
        loadings["R_squared"] = round(float(r_squared), 3)

        return loadings
    except Exception:
        return None


def build_context_packet(
    user_text: str,
    tickers: list[str],
    portfolio_weights: dict | None,
    ticker_data: dict,
    market_data: dict,
    sector_perf: dict,
    factor_loadings: dict | None,
) -> str:
    """Assemble everything into a context string for Claude."""
    parts = []

    parts.append("=== USER INPUT ===")
    parts.append(user_text or "(no text provided)")

    if tickers:
        parts.append("\n=== DETECTED TICKERS & PERFORMANCE (30-day) ===")
        for t in tickers:
            if t in ticker_data:
                d = ticker_data[t]
                parts.append(f"  {t}: ${d['current_price']} | 30-day return: {d['period_return_pct']}%")

    if portfolio_weights:
        parts.append("\n=== PORTFOLIO WEIGHTS ===")
        for t, w in portfolio_weights.items():
            parts.append(f"  {t}: {w*100:.1f}%")

    if factor_loadings:
        parts.append("\n=== FAMA-FRENCH 5-FACTOR REGRESSION ===")
        parts.append(f"  Alpha (annualized): {factor_loadings.get('alpha_annualized_pct', 'N/A')}%")
        parts.append(f"  Market Beta (Mkt-RF): {factor_loadings.get('Mkt-RF', 'N/A')}")
        parts.append(f"  Size (SMB): {factor_loadings.get('SMB', 'N/A')} (positive = tilted toward small-cap)")
        parts.append(f"  Value (HML): {factor_loadings.get('HML', 'N/A')} (positive = tilted toward value)")
        parts.append(f"  Profitability (RMW): {factor_loadings.get('RMW', 'N/A')} (positive = tilted toward quality)")
        parts.append(f"  Investment (CMA): {factor_loadings.get('CMA', 'N/A')} (positive = tilted toward conservative)")
        parts.append(f"  R-squared: {factor_loadings.get('R_squared', 'N/A')}")

    if market_data:
        parts.append("\n=== MARKET CONDITIONS TODAY ===")
        for name, d in market_data.items():
            direction = "+" if d["change_pct"] >= 0 else ""
            parts.append(f"  {name}: {d['price']} ({direction}{d['change_pct']}%)")

    if sector_perf:
        parts.append("\n=== SECTOR PERFORMANCE TODAY ===")
        sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
        for name, change in sorted_sectors:
            direction = "+" if change >= 0 else ""
            parts.append(f"  {name}: {direction}{change}%")

    return "\n".join(parts)


SYSTEM_PROMPT = """You are FactorLens, an expert financial analyst that explains portfolio performance in plain English.

You will receive a context packet containing:
- The user's input (portfolio description, loss/gain description, or question)
- Detected stock tickers and their recent performance
- Portfolio weights (if parsed)
- Fama-French 5-factor regression loadings (if available)
- Current market conditions
- Sector performance

If an image is provided, analyze it for any financial data (portfolio screenshots, P&L charts, brokerage app screenshots, etc.) and incorporate what you see.

Respond with a JSON object containing exactly three fields:
{
  "what_happened": "A clear, concise summary of what happened to the portfolio/investment. Reference specific numbers.",
  "why_it_happened": "Factor-based explanation of WHY. Reference the Fama-French loadings, sector moves, market conditions. Explain in plain English what each relevant factor means for their portfolio.",
  "what_to_do": "Exactly 3 specific, actionable suggestions. Each should name a specific ETF or strategy. Format as a numbered list."
}

Rules:
- Be specific. Use actual numbers from the context.
- Explain factors in plain English (e.g., "Your portfolio has a high market beta of 1.3, meaning it moves 30% more than the market — when the market drops 2%, you'd expect a ~2.6% loss").
- For suggestions, name specific tickers/ETFs and explain why.
- Keep each section to 2-4 sentences max. Be concise but insightful.
- If data is limited, work with what you have and note limitations.
- Always return valid JSON."""


def call_claude(context: str, image_b64: str | None = None) -> dict:
    """Send context + optional image to Claude and parse the response."""
    client = anthropic.Anthropic()

    content = []
    if image_b64:
        media_type = "image/png"
        if image_b64.startswith("/9j/"):
            media_type = "image/jpeg"
        elif image_b64.startswith("iVBOR"):
            media_type = "image/png"
        elif image_b64.startswith("R0lGOD"):
            media_type = "image/gif"
        elif image_b64.startswith("UklGR"):
            media_type = "image/webp"
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_b64,
            },
        })
    content.append({"type": "text", "text": context})

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

    return json.loads(raw)


def _analyze_text(user_text: str, image_b64: str | None = None,
                  market_data: dict | None = None,
                  sector_perf: dict | None = None,
                  ff_data=None) -> dict:
    tickers = extract_tickers(user_text)
    portfolio_weights = parse_portfolio(user_text)
    if portfolio_weights:
        tickers = list(set(tickers + list(portfolio_weights.keys())))

    ticker_data = fetch_ticker_data(tickers)
    if market_data is None:
        market_data = fetch_market_data()
    if sector_perf is None:
        sector_perf = fetch_sector_performance()

    needs_ff = (portfolio_weights and len(portfolio_weights) >= 1) or len(tickers) == 1
    if needs_ff and ff_data is None:
        ff_data = fetch_fama_french_factors(days=30)

    factor_loadings = None
    if portfolio_weights and ff_data is not None:
        weighted_returns = None
        for t, w in portfolio_weights.items():
            if t in ticker_data and "daily_returns" in ticker_data[t]:
                dr = np.array(ticker_data[t]["daily_returns"])
                if weighted_returns is None:
                    weighted_returns = np.zeros(len(dr))
                n = min(len(weighted_returns), len(dr))
                weighted_returns = weighted_returns[:n]
                weighted_returns += w * dr[:n]
        if weighted_returns is not None and len(weighted_returns) >= 10:
            factor_loadings = run_factor_regression(weighted_returns, ff_data)
    elif len(tickers) == 1 and tickers[0] in ticker_data and ff_data is not None:
        dr = np.array(ticker_data[tickers[0]]["daily_returns"])
        if len(dr) >= 10:
            factor_loadings = run_factor_regression(dr, ff_data)

    context = build_context_packet(
        user_text, tickers, portfolio_weights,
        ticker_data, market_data, sector_perf, factor_loadings,
    )
    analysis = call_claude(context, image_b64)

    return {
        "analysis": analysis,
        "meta": {
            "tickers_detected": tickers,
            "portfolio_weights": portfolio_weights,
            "factor_loadings": factor_loadings,
            "market_snapshot": market_data,
            "sectors": sector_perf,
        },
    }


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/market-pulse", methods=["GET"])
def market_pulse():
    """Return live market snapshot for the ticker strip."""
    try:
        data = fetch_market_data()
        return jsonify({"status": "ok", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint."""
    try:
        body = request.get_json(force=True)
        user_text = body.get("text", "").strip()
        image_b64 = body.get("image", None)

        if not user_text and not image_b64:
            return jsonify({"status": "error", "message": "Please provide some text or an image to analyze."}), 400

        result = _analyze_text(user_text, image_b64)
        return jsonify({"status": "ok", **result})

    except json.JSONDecodeError:
        return jsonify({"status": "error", "message": "Failed to parse analysis response. Please try again."}), 500
    except anthropic.APIError as e:
        return jsonify({"status": "error", "message": f"Claude API error: {str(e)}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/compare", methods=["POST"])
def compare():
    """Side-by-side portfolio comparison endpoint."""
    try:
        body = request.get_json(force=True)
        text_a = body.get("portfolio_a", "").strip()
        text_b = body.get("portfolio_b", "").strip()

        if not text_a or not text_b:
            return jsonify({"status": "error", "message": "Both portfolios are required."}), 400

        market_data = fetch_market_data()
        sector_perf = fetch_sector_performance()
        ff_data = fetch_fama_french_factors(days=30)

        result_a = _analyze_text(text_a, market_data=market_data, sector_perf=sector_perf, ff_data=ff_data)
        result_b = _analyze_text(text_b, market_data=market_data, sector_perf=sector_perf, ff_data=ff_data)

        return jsonify({"status": "ok", "results": {"a": result_a, "b": result_b}})

    except json.JSONDecodeError:
        return jsonify({"status": "error", "message": "Failed to parse analysis response. Please try again."}), 500
    except anthropic.APIError as e:
        return jsonify({"status": "error", "message": f"Claude API error: {str(e)}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Comparison failed: {str(e)}"}), 500


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠  ANTHROPIC_API_KEY not set.")
        print("   Create a .env file with: ANTHROPIC_API_KEY=sk-ant-...")
        print("   Or export it: export ANTHROPIC_API_KEY=sk-ant-...\n")
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
