"""
MF Screener - Indian Mutual Fund Analyzer
Single-file Flask app. Port: 8092
Metrics: Returns (35) + Risk (30) + Consistency (20) + Momentum (15) = 100
Verdict: INVEST >= 65 | WATCHLIST >= 42 | SKIP < 42
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
import requests
import requests_cache
import anthropic

load_dotenv('C:\\credentials\\.env')

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
RISK_FREE_RATE = 0.065  # 6.5% annual, India 10Y G-Sec proxy

requests_cache.install_cache(
    'mf_cache', backend='sqlite', expire_after=14400, allowable_codes=[200]
)

app = Flask(__name__)
logging.basicConfig(level=logging.ERROR)

# In-memory scheme list cache (refreshed every 24h)
_schemes_cache = {'data': None, 'ts': 0}


# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------

def get_all_schemes():
    now = time.time()
    if _schemes_cache['data'] and (now - _schemes_cache['ts']) < 86400:
        return _schemes_cache['data']
    resp = requests.get('https://api.mfapi.in/mf', timeout=30)
    resp.raise_for_status()
    _schemes_cache['data'] = resp.json()
    _schemes_cache['ts'] = now
    return _schemes_cache['data']


def search_funds(query):
    schemes = get_all_schemes()
    q = query.strip().lower()
    results = []
    for s in schemes:
        if q in s['schemeName'].lower():
            results.append({'name': s['schemeName'], 'code': str(s['schemeCode'])})
        if len(results) >= 15:
            break
    return results


def fetch_nav_history(scheme_code):
    url = f'https://api.mfapi.in/mf/{scheme_code}'
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    nav_list = payload.get('data', [])
    meta = payload.get('meta', {})

    df = pd.DataFrame(nav_list)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df = df.dropna().sort_values('date').set_index('date')
    return df, meta


# ---------------------------------------------------------------------------
# METRIC CALCULATION
# ---------------------------------------------------------------------------

def cagr(nav_df, years):
    end = nav_df.index[-1]
    start_target = end - timedelta(days=int(years * 365.25))
    sub = nav_df[nav_df.index >= start_target]
    if len(sub) < 5:
        return None
    actual_years = (sub.index[-1] - sub.index[0]).days / 365.25
    if actual_years < years * 0.75:
        return None
    start_nav = sub['nav'].iloc[0]
    end_nav = sub['nav'].iloc[-1]
    if start_nav <= 0:
        return None
    return ((end_nav / start_nav) ** (1 / actual_years) - 1) * 100


def calculate_metrics(nav_df):
    daily_ret = nav_df['nav'].pct_change().dropna()
    m = {}

    for label, yrs in [('1m', 1/12), ('3m', 3/12), ('6m', 6/12),
                        ('1y', 1), ('3y', 3), ('5y', 5), ('10y', 10)]:
        m[f'cagr_{label}'] = cagr(nav_df, yrs)

    ann_ret = daily_ret.mean() * 252
    ann_std = daily_ret.std() * np.sqrt(252)
    m['volatility'] = float(ann_std * 100) if not np.isnan(ann_std) else None

    m['sharpe'] = float((ann_ret - RISK_FREE_RATE) / ann_std) if ann_std > 0 else None

    neg = daily_ret[daily_ret < 0]
    dd_std = neg.std() * np.sqrt(252)
    m['sortino'] = float((ann_ret - RISK_FREE_RATE) / dd_std) if (len(neg) > 5 and dd_std > 0) else None

    cum = (1 + daily_ret).cumprod()
    roll_max = cum.expanding().max()
    drawdowns = (cum - roll_max) / roll_max
    m['max_drawdown'] = float(drawdowns.min() * 100)

    roll_1y = nav_df['nav'].pct_change(252).dropna()
    m['rolling_positive_pct'] = float((roll_1y > 0).mean() * 100) if len(roll_1y) > 0 else None

    m['current_nav'] = float(nav_df['nav'].iloc[-1])
    m['nav_date'] = nav_df.index[-1].strftime('%d %b %Y')
    m['inception_years'] = round((nav_df.index[-1] - nav_df.index[0]).days / 365.25, 1)

    return m


def get_benchmark_metrics(nav_df):
    try:
        import yfinance as yf
        end_date = nav_df.index[-1]
        start_date = end_date - timedelta(days=365 * 3)

        nifty = yf.download(
            '^NSEI',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True
        )
        if nifty.empty:
            return None, None

        nifty_ret = nifty['Close'].squeeze().pct_change().dropna()
        fund_ret = nav_df[nav_df.index >= start_date]['nav'].pct_change().dropna()

        aligned = pd.concat(
            [fund_ret.rename('fund'), nifty_ret.rename('nifty')], axis=1
        ).dropna()

        if len(aligned) < 50:
            return None, None

        cov_mat = np.cov(aligned['fund'].values, aligned['nifty'].values)
        beta = cov_mat[0, 1] / cov_mat[1, 1]

        fund_ann = aligned['fund'].mean() * 252
        nifty_ann = aligned['nifty'].mean() * 252
        alpha = (fund_ann - (RISK_FREE_RATE + beta * (nifty_ann - RISK_FREE_RATE))) * 100

        return float(beta), float(alpha)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# SCORING ENGINE
# ---------------------------------------------------------------------------

def score_fund(metrics, beta, alpha, expense_ratio=None):
    sc = {'returns': 0, 'risk': 0, 'consistency': 0, 'momentum': 0}
    d = {}

    # ── RETURNS (35 pts) ──────────────────────────────────────────────────
    c1y = metrics.get('cagr_1y')
    if c1y is not None:
        s = 10 if c1y > 25 else 8 if c1y > 18 else 6 if c1y > 12 else 3 if c1y > 6 else 1 if c1y > 0 else 0
        sc['returns'] += s; d['s_1y'] = s

    c3y = metrics.get('cagr_3y')
    if c3y is not None:
        s = 15 if c3y > 20 else 12 if c3y > 15 else 9 if c3y > 12 else 5 if c3y > 8 else 2 if c3y > 0 else 0
        sc['returns'] += s; d['s_3y'] = s

    c5y = metrics.get('cagr_5y')
    if c5y is not None:
        s = 10 if c5y > 18 else 8 if c5y > 14 else 6 if c5y > 10 else 3 if c5y > 6 else 1 if c5y > 0 else 0
        sc['returns'] += s; d['s_5y'] = s

    # ── RISK (30 pts) ─────────────────────────────────────────────────────
    sharpe = metrics.get('sharpe')
    if sharpe is not None:
        s = 12 if sharpe > 2 else 9 if sharpe > 1.5 else 6 if sharpe > 1 else 3 if sharpe > 0.5 else 0
        sc['risk'] += s; d['s_sharpe'] = s

    mdd = metrics.get('max_drawdown')
    if mdd is not None:
        s = 10 if mdd > -10 else 8 if mdd > -15 else 5 if mdd > -25 else 2 if mdd > -35 else 0
        sc['risk'] += s; d['s_mdd'] = s

    vol = metrics.get('volatility')
    if vol is not None:
        s = 8 if vol < 8 else 6 if vol < 12 else 4 if vol < 18 else 2 if vol < 25 else 0
        sc['risk'] += s; d['s_vol'] = s

    # ── CONSISTENCY (20 pts) ──────────────────────────────────────────────
    rpp = metrics.get('rolling_positive_pct')
    if rpp is not None:
        s = 10 if rpp > 90 else 8 if rpp > 80 else 6 if rpp > 70 else 4 if rpp > 60 else 2
        sc['consistency'] += s; d['s_rpp'] = s

    if alpha is not None:
        s = 10 if alpha > 5 else 7 if alpha > 3 else 4 if alpha > 1 else 2 if alpha > 0 else 0
        sc['consistency'] += s; d['s_alpha'] = s

    # ── MOMENTUM (15 pts) — recent trend: 1M + 3M + 6M returns ─────────────
    c1m = metrics.get('cagr_1m')
    if c1m is not None:
        s = 5 if c1m > 3 else 3 if c1m > 0 else 0
        sc['momentum'] += s; d['s_1m'] = s

    c3m = metrics.get('cagr_3m')
    if c3m is not None:
        s = 5 if c3m > 5 else 3 if c3m > 0 else 0
        sc['momentum'] += s; d['s_3m'] = s

    c6m = metrics.get('cagr_6m')
    if c6m is not None:
        s = 5 if c6m > 8 else 3 if c6m > 0 else 0
        sc['momentum'] += s; d['s_6m'] = s

    total = sum(sc.values())
    verdict = 'INVEST' if total >= 65 else 'WATCHLIST' if total >= 42 else 'SKIP'
    return total, verdict, sc, d


# ---------------------------------------------------------------------------
# AI NARRATIVE
# ---------------------------------------------------------------------------

def get_ai_narrative(fund_name, total, verdict, metrics, beta, alpha):
    if not ANTHROPIC_API_KEY:
        return None
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        c1y = metrics.get('cagr_1y')
        c3y = metrics.get('cagr_3y')
        sharpe = metrics.get('sharpe')
        mdd = metrics.get('max_drawdown')

        prompt = (
            f"Analyze this Indian mutual fund. Give exactly 3 bullet points, each under 15 words.\n"
            f"Fund: {fund_name}\n"
            f"Score: {total}/100 ({verdict})\n"
            f"1Y return: {f'{c1y:.1f}%' if c1y else 'N/A'}\n"
            f"3Y return: {f'{c3y:.1f}%' if c3y else 'N/A'}\n"
            f"Sharpe: {f'{sharpe:.2f}' if sharpe else 'N/A'}\n"
            f"Max Drawdown: {f'{mdd:.1f}%' if mdd else 'N/A'}\n"
            f"Beta vs Nifty: {f'{beta:.2f}' if beta else 'N/A'}\n"
            f"Alpha: {f'{alpha:.1f}%' if alpha else 'N/A'}\n\n"
            "Format: • [strength] • [risk] • [verdict]"
        )
        resp = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=200,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return resp.content[0].text.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/search')
def api_search():
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify([])
    try:
        return jsonify(search_funds(q))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    body = request.get_json(force=True) or {}
    scheme_code = str(body.get('code', '')).strip()
    if not scheme_code:
        return jsonify({'error': 'scheme code required'}), 400

    try:
        nav_df, meta = fetch_nav_history(scheme_code)
    except requests.HTTPError as e:
        return jsonify({'error': f'Fund not found: {e}'}), 404
    except Exception as e:
        return jsonify({'error': f'Failed to fetch NAV data: {e}'}), 503

    if nav_df.empty or len(nav_df) < 30:
        return jsonify({'error': 'Insufficient NAV history for this fund'}), 422

    metrics = calculate_metrics(nav_df)
    beta, alpha = get_benchmark_metrics(nav_df)

    expense_ratio = meta.get('scheme_type', None)  # mfapi meta rarely has ER; placeholder
    total, verdict, sc, d = score_fund(metrics, beta, alpha, expense_ratio=None)

    narrative = get_ai_narrative(
        meta.get('scheme_name', scheme_code), total, verdict, metrics, beta, alpha
    )

    def fmt(v, decimals=2, suffix=''):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'N/A'
        return f'{v:.{decimals}f}{suffix}'

    result = {
        'name':    meta.get('scheme_name', f'Fund #{scheme_code}'),
        'house':   meta.get('fund_house', ''),
        'category': meta.get('scheme_category', ''),
        'type':    meta.get('scheme_type', ''),
        'code':    scheme_code,
        'total':   total,
        'verdict': verdict,
        'scores':  sc,
        'detail':  d,
        'narrative': narrative,
        'data': {
            'nav':        fmt(metrics.get('current_nav'), 4),
            'nav_date':   metrics.get('nav_date', ''),
            'inception':  f"{metrics.get('inception_years', 'N/A')} yrs",
            'cagr_1m':    fmt(metrics.get('cagr_1m'), 1, '%'),
            'cagr_3m':    fmt(metrics.get('cagr_3m'), 1, '%'),
            'cagr_6m':    fmt(metrics.get('cagr_6m'), 1, '%'),
            'cagr_1y':    fmt(metrics.get('cagr_1y'), 1, '%'),
            'cagr_3y':    fmt(metrics.get('cagr_3y'), 1, '%'),
            'cagr_5y':    fmt(metrics.get('cagr_5y'), 1, '%'),
            'cagr_10y':   fmt(metrics.get('cagr_10y'), 1, '%'),
            'volatility': fmt(metrics.get('volatility'), 1, '%'),
            'sharpe':     fmt(metrics.get('sharpe'), 2),
            'sortino':    fmt(metrics.get('sortino'), 2),
            'max_dd':     fmt(metrics.get('max_drawdown'), 1, '%'),
            'roll_pos':   fmt(metrics.get('rolling_positive_pct'), 0, '%'),
            'beta':       fmt(beta, 2),
            'alpha':      fmt(alpha, 1, '%'),
        }
    }
    return jsonify(result)


# ---------------------------------------------------------------------------
# HTML TEMPLATE
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MF Screener — India</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --brand:        #D9008D;
    --brand-light:  #FFB2E4;
    --brand-softer: #FFE5F6;
    --secondary:    #443152;
    --bg:           #161021;
    --surface:      #352D42;
    --surface2:     #665E75;
    --border:       #665E75;
    --text:         #FAF9FC;
    --muted:        #A7A1B2;
    --subtle:       #E8E4F0;
    --positive:     #008043;
    --positive-bg:  #E1FAED;
    --negative:     #E01E00;
    --negative-bg:  #FFE9E5;
    --warning:      #FF8000;
    --warning-bg:   #FFE6CC;
    --info:         #6D17CE;
    --info-bg:      #F1E5FF;
    --gold:         #D5AF4A;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif;
         font-size: 14px; min-height: 100vh; }

  /* HEADER */
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 0 32px; height: 60px; display: flex; align-items: center;
           justify-content: space-between; position: sticky; top: 0; z-index: 100; }
  .logo { font-size: 18px; font-weight: 800; color: var(--text); display: flex; align-items: center; gap: 10px; }
  .logo span { color: var(--brand); }
  .logo-badge { background: var(--brand); color: #fff; font-size: 10px; font-weight: 700;
                padding: 2px 7px; border-radius: 99px; letter-spacing: .5px; }

  /* MAIN */
  main { max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px; }

  /* SEARCH */
  .hero { text-align: center; margin-bottom: 36px; }
  .hero h1 { font-size: 28px; font-weight: 800; margin-bottom: 8px; }
  .hero p  { color: var(--muted); font-size: 14px; }

  .search-wrap { position: relative; max-width: 580px; margin: 28px auto 0; }
  .search-input { width: 100%; background: var(--surface); border: 1.5px solid var(--border);
                  color: var(--text); font-size: 15px; font-family: inherit;
                  padding: 14px 52px 14px 18px; border-radius: 12px; outline: none;
                  transition: border-color .2s; }
  .search-input:focus { border-color: var(--brand); }
  .search-input::placeholder { color: var(--muted); }
  .search-btn { position: absolute; right: 10px; top: 50%; transform: translateY(-50%);
                background: var(--brand); border: none; color: #fff; width: 36px; height: 36px;
                border-radius: 8px; cursor: pointer; font-size: 17px; display: flex;
                align-items: center; justify-content: center; }
  .search-btn:hover { opacity: .88; }

  /* AUTOCOMPLETE DROPDOWN */
  .dropdown { position: absolute; top: calc(100% + 6px); left: 0; right: 0;
              background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
              box-shadow: 0 8px 24px rgba(0,0,0,.45); z-index: 200; overflow: hidden; display: none; }
  .dropdown.open { display: block; }
  .dd-item { padding: 11px 16px; cursor: pointer; border-bottom: 1px solid rgba(102,94,117,.3);
             font-size: 13px; line-height: 1.4; }
  .dd-item:last-child { border-bottom: none; }
  .dd-item:hover { background: rgba(217,0,141,.12); }
  .dd-item .dd-name { color: var(--text); font-weight: 500; }
  .dd-item .dd-code { color: var(--muted); font-size: 11px; margin-top: 2px; }

  /* SPINNER */
  .spinner { display: none; width: 28px; height: 28px; border: 3px solid var(--border);
             border-top-color: var(--brand); border-radius: 50%; animation: spin .7s linear infinite;
             margin: 48px auto; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ERROR */
  .error-box { background: rgba(224,30,0,.12); border: 1px solid var(--negative);
               color: #ff7b6b; border-radius: 12px; padding: 14px 18px; margin-top: 28px; font-size: 13px; }

  /* RESULT CARD */
  .result { display: none; margin-top: 32px; }

  /* FUND HEADER */
  .fund-header { background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
                 padding: 24px 28px; margin-bottom: 24px; display: flex;
                 align-items: flex-start; justify-content: space-between; gap: 20px; flex-wrap: wrap; }
  .fund-meta h2 { font-size: 20px; font-weight: 800; margin-bottom: 4px; }
  .fund-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
  .tag { background: rgba(102,94,117,.45); color: var(--muted); font-size: 11px;
         font-weight: 600; padding: 3px 9px; border-radius: 99px; text-transform: uppercase;
         letter-spacing: .4px; }
  .tag.house { background: rgba(217,0,141,.15); color: var(--brand-light); }

  .verdict-block { text-align: center; min-width: 120px; }
  .verdict-label { font-size: 11px; font-weight: 700; text-transform: uppercase;
                   letter-spacing: 1px; color: var(--muted); margin-bottom: 6px; }
  .verdict-badge { font-size: 22px; font-weight: 900; padding: 8px 24px; border-radius: 12px;
                   letter-spacing: .5px; }
  .verdict-badge.INVEST    { background: rgba(0,128,67,.18); color: #4dff9b; border: 1.5px solid var(--positive); }
  .verdict-badge.WATCHLIST { background: rgba(255,128,0,.15); color: #ffb84d; border: 1.5px solid var(--warning); }
  .verdict-badge.SKIP      { background: rgba(224,30,0,.15); color: #ff7b6b; border: 1.5px solid var(--negative); }
  .score-num { font-size: 32px; font-weight: 900; margin-top: 8px; }
  .score-denom { font-size: 14px; color: var(--muted); font-weight: 500; }

  /* SECTION TITLE */
  .section-title { font-size: 11px; font-weight: 700; text-transform: uppercase;
                   letter-spacing: 1px; color: var(--muted); margin-bottom: 14px; margin-top: 28px; }

  /* METRIC GRID */
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }
  .metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
                 padding: 16px 14px; }
  .mk { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px;
        color: var(--muted); margin-bottom: 6px; }
  .mv { font-size: 20px; font-weight: 800; }
  .mv.pos { color: #4dff9b; }
  .mv.neg { color: #ff7b6b; }
  .mv.warn { color: #ffb84d; }
  .mv.neutral { color: var(--text); }
  .mv.brand { color: var(--brand-light); }
  .mex { font-size: 11px; color: var(--muted); line-height: 1.5;
         margin-top: 7px; padding-top: 7px; border-top: 1px solid var(--border); }

  /* RETURNS ROW */
  .returns-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 10px; }
  .ret-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
              padding: 12px 14px; text-align: center; }
  .ret-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .4px;
               color: var(--muted); margin-bottom: 4px; }
  .ret-val { font-size: 18px; font-weight: 800; }

  /* BREAKDOWN */
  .breakdown-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  @media (max-width: 700px) { .breakdown-grid { grid-template-columns: repeat(2, 1fr); } }
  .breakdown-item { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
                    padding: 18px 16px; }
  .bd-icon { font-size: 22px; margin-bottom: 6px; }
  .bd-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .5px;
              color: var(--muted); margin-bottom: 4px; }
  .bd-score { font-size: 26px; font-weight: 900; }
  .bd-max { font-size: 12px; color: var(--muted); font-weight: 500; }
  .mini-track { background: rgba(255,255,255,.1); border-radius: 99px; height: 5px; margin-top: 10px; overflow: hidden; }
  .mini-fill { height: 100%; border-radius: 99px; background: var(--brand); transition: width .6s ease; }
  .bex { font-size: 11px; color: var(--muted); line-height: 1.5;
         margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); }

  /* AI NARRATIVE */
  .narrative-box { background: rgba(109,23,206,.1); border: 1px solid rgba(109,23,206,.35);
                   border-radius: 14px; padding: 18px 20px; margin-top: 20px; }
  .nar-header { display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
                font-size: 12px; font-weight: 700; text-transform: uppercase;
                letter-spacing: .6px; color: #b07fff; }
  .nar-body { color: var(--subtle); font-size: 13px; line-height: 1.7; white-space: pre-line; }

  /* NAV CHART */
  .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
                padding: 20px; margin-top: 0; }
  #nav-chart { width: 100%; height: 200px; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <span>MF</span>Scope
    <span class="logo-badge">IN</span>
  </div>
  <div style="font-size:12px;color:var(--muted);">Indian Mutual Fund Analyzer</div>
</header>

<main>
  <div class="hero">
    <h1>Analyze Any Mutual Fund</h1>
    <p>Type a fund name — get returns, risk, consistency, momentum & a verdict in seconds</p>

    <div class="search-wrap">
      <input id="q" class="search-input" type="text"
             placeholder="e.g. Mirae Asset Large Cap, Parag Parikh Flexi..."
             autocomplete="off" spellcheck="false">
      <button class="search-btn" onclick="triggerSearch()">&#8594;</button>
      <div id="dropdown" class="dropdown"></div>
    </div>
  </div>

  <div id="spinner" class="spinner"></div>
  <div id="error"  class="error-box" style="display:none;"></div>

  <div id="result" class="result">

    <!-- FUND HEADER -->
    <div class="fund-header">
      <div class="fund-meta">
        <h2 id="r-name">—</h2>
        <div style="color:var(--muted);font-size:13px;margin-top:4px;" id="r-category"></div>
        <div class="fund-tags" id="r-tags"></div>
      </div>
      <div class="verdict-block">
        <div class="verdict-label">Verdict</div>
        <div class="verdict-badge" id="r-verdict">—</div>
        <div class="score-num" id="r-score">—<span class="score-denom">/100</span></div>
      </div>
    </div>

    <!-- RETURNS -->
    <div class="section-title">Historical Returns (CAGR)</div>
    <div class="returns-grid" id="returns-grid"></div>

    <!-- NAV CHART placeholder removed for simplicity; add chart lib if needed -->

    <!-- RISK METRICS -->
    <div class="section-title">Risk Metrics</div>
    <div class="metric-grid" id="risk-grid"></div>

    <!-- BENCHMARK -->
    <div class="section-title">Benchmark (vs Nifty 50)</div>
    <div class="metric-grid" id="bench-grid"></div>

    <!-- BREAKDOWN -->
    <div class="section-title">Category Score Breakdown</div>
    <div class="breakdown-grid">
      <div class="breakdown-item">
        <div class="bd-icon">📈</div>
        <div class="bd-label">Returns</div>
        <div class="bd-score" id="bRet">—</div>
        <div class="bd-max">out of 35</div>
        <div class="mini-track"><div class="mini-fill" id="bRetBar" style="width:0%"></div></div>
        <div class="bex" id="bRetEx"></div>
      </div>
      <div class="breakdown-item">
        <div class="bd-icon">🛡️</div>
        <div class="bd-label">Risk</div>
        <div class="bd-score" id="bRisk">—</div>
        <div class="bd-max">out of 30</div>
        <div class="mini-track"><div class="mini-fill" id="bRiskBar" style="width:0%"></div></div>
        <div class="bex" id="bRiskEx"></div>
      </div>
      <div class="breakdown-item">
        <div class="bd-icon">🔄</div>
        <div class="bd-label">Consistency</div>
        <div class="bd-score" id="bCon">—</div>
        <div class="bd-max">out of 20</div>
        <div class="mini-track"><div class="mini-fill" id="bConBar" style="width:0%"></div></div>
        <div class="bex" id="bConEx"></div>
      </div>
      <div class="breakdown-item">
        <div class="bd-icon">⚡</div>
        <div class="bd-label">Momentum</div>
        <div class="bd-score" id="bMom">—</div>
        <div class="bd-max">out of 15</div>
        <div class="mini-track"><div class="mini-fill" id="bMomBar" style="width:0%"></div></div>
        <div class="bex" id="bMomEx"></div>
      </div>
    </div>

    <!-- AI NARRATIVE -->
    <div id="narrative-box" class="narrative-box" style="display:none;">
      <div class="nar-header">&#10024; AI Analysis</div>
      <div class="nar-body" id="nar-body"></div>
    </div>

  </div><!-- end #result -->
</main>

<script>
const $ = id => document.getElementById(id);
let debounceTimer = null;
let selectedCode = null;

// ── SEARCH & AUTOCOMPLETE ────────────────────────────────────────────────

$('q').addEventListener('input', () => {
  clearTimeout(debounceTimer);
  const q = $('q').value.trim();
  if (q.length < 2) { closeDropdown(); return; }
  debounceTimer = setTimeout(() => fetchSuggestions(q), 280);
});

$('q').addEventListener('keydown', e => {
  if (e.key === 'Enter') triggerSearch();
  if (e.key === 'Escape') closeDropdown();
});

document.addEventListener('click', e => {
  if (!e.target.closest('.search-wrap')) closeDropdown();
});

async function fetchSuggestions(q) {
  try {
    const r = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    const data = await r.json();
    renderDropdown(data);
  } catch(e) {}
}

function renderDropdown(items) {
  const dd = $('dropdown');
  if (!items.length) { closeDropdown(); return; }
  dd.innerHTML = items.map(i =>
    `<div class="dd-item" onclick="selectFund('${i.code}', \`${i.name.replace(/`/g,'&#96;')}\`)">
       <div class="dd-name">${i.name}</div>
       <div class="dd-code">Code: ${i.code}</div>
     </div>`
  ).join('');
  dd.classList.add('open');
}

function closeDropdown() {
  $('dropdown').classList.remove('open');
}

function selectFund(code, name) {
  selectedCode = code;
  $('q').value = name;
  closeDropdown();
  analyze(code);
}

function triggerSearch() {
  if (selectedCode) { analyze(selectedCode); return; }
  // If user typed without selecting, use first search result
  const q = $('q').value.trim();
  if (q.length < 2) return;
  fetch(`/api/search?q=${encodeURIComponent(q)}`)
    .then(r => r.json())
    .then(data => { if (data.length) selectFund(data[0].code, data[0].name); });
}

// ── ANALYZE ──────────────────────────────────────────────────────────────

async function analyze(code) {
  $('spinner').style.display = 'block';
  $('result').style.display  = 'none';
  $('error').style.display   = 'none';
  closeDropdown();

  try {
    const r = await fetch('/api/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({code})
    });
    const data = await r.json();
    if (!r.ok) { showError(data.error || 'Analysis failed'); return; }
    renderResult(data);
  } catch(e) {
    showError('Network error. Please try again.');
  } finally {
    $('spinner').style.display = 'none';
  }
}

function showError(msg) {
  $('error').textContent = msg;
  $('error').style.display = 'block';
}

// ── RENDER ───────────────────────────────────────────────────────────────

function n(v)  { return parseFloat(v); }
function na(v) { return v === 'N/A' || v == null; }

function colorClass(v, pos_thresh, warn_thresh) {
  if (na(v)) return 'neutral';
  const f = n(v);
  return f >= pos_thresh ? 'pos' : f >= warn_thresh ? 'warn' : 'neg';
}

function retColor(v) {
  if (na(v)) return 'neutral';
  return n(v) > 0 ? 'pos' : 'neg';
}

function metricCard(label, value, cls, explain) {
  cls = cls || 'neutral';
  const ex = explain ? `<div class="mex">${explain}</div>` : '';
  return `<div class="metric-card"><div class="mk">${label}</div><div class="mv ${cls}">${value}</div>${ex}</div>`;
}

function retCard(label, value) {
  const cls = na(value) ? 'neutral' : (n(value) > 0 ? 'pos' : 'neg');
  return `<div class="ret-card"><div class="ret-label">${label}</div><div class="ret-val ${cls}">${value}</div></div>`;
}

function interpret(key, val) {
  if (na(val)) return '';
  const v = n(val);
  switch(key) {
    case 'sharpe':
      if (v > 2)   return 'Excellent risk-adjusted returns — earns well above the risk taken.';
      if (v > 1.5) return 'Good risk-adjusted returns — solid reward per unit of risk.';
      if (v > 1)   return 'Decent risk-adjusted returns — reward slightly exceeds risk.';
      if (v > 0.5) return 'Marginal Sharpe — returns barely compensate the volatility.';
      return 'Poor risk-adjusted returns — volatility outpaces the reward earned.';
    case 'sortino':
      if (v > 2)   return 'Handles downside very well — negative periods are shallow.';
      if (v > 1)   return 'Reasonable downside protection relative to gains delivered.';
      return 'Downside losses are large relative to the upside generated.';
    case 'max_dd':
      if (v > -10) return 'Very resilient — deepest fall under 10% from peak.';
      if (v > -15) return 'Healthy drawdown — has not fallen more than 15% from peak.';
      if (v > -25) return 'Moderate drawdown — saw a 15–25% drop from its peak.';
      if (v > -35) return 'Significant drawdown — dropped 25–35% from peak at worst.';
      return 'Deep drawdown — fell over 35% from peak; high-risk fund.';
    case 'volatility':
      if (v < 8)   return 'Very low volatility — stable, suitable for conservative investors.';
      if (v < 12)  return 'Low volatility — relatively smooth NAV movement.';
      if (v < 18)  return 'Moderate volatility — typical for diversified equity funds.';
      if (v < 25)  return 'High volatility — significant NAV swings; not for the faint-hearted.';
      return 'Very high volatility — aggressive fund with large NAV swings.';
    case 'beta':
      if (v < 0.7) return 'Low beta — moves less than the market; more defensive.';
      if (v < 0.9) return 'Slightly defensive — somewhat less reactive than Nifty 50.';
      if (v < 1.1) return 'Market beta — mirrors Nifty 50 moves closely.';
      if (v < 1.3) return 'Slightly aggressive — amplifies market moves moderately.';
      return 'High beta — amplifies market swings; higher risk and potential reward.';
    case 'alpha':
      if (v > 5)   return 'Strong alpha — fund manager adds significant value over Nifty.';
      if (v > 2)   return 'Decent alpha — fund outperforms the index meaningfully.';
      if (v > 0)   return 'Marginal alpha — barely beats benchmark after adjusting for risk.';
      return 'Negative alpha — underperforms Nifty 50 on a risk-adjusted basis.';
    case 'roll_pos':
      if (v > 90)  return 'Extremely consistent — positive returns in over 90% of rolling 1Y periods.';
      if (v > 80)  return 'Very consistent — positive in most 12-month windows historically.';
      if (v > 70)  return 'Reasonably consistent — positive returns in most years.';
      return 'Inconsistent — significant number of negative 1-year periods historically.';
    default: return '';
  }
}

function categoryOneLiner(cat, score, d, data) {
  const na2 = v => v === 'N/A' || isNaN(parseFloat(v));
  const f = v => parseFloat(v);

  if (cat === 'returns') {
    const parts = [];
    if (!na2(data.cagr_1y)) parts.push(`1Y ${data.cagr_1y}`);
    if (!na2(data.cagr_3y)) parts.push(`3Y ${data.cagr_3y}`);
    if (!na2(data.cagr_5y)) parts.push(`5Y ${data.cagr_5y}`);
    const verdict = score >= 28 ? 'consistently strong returns' :
                    score >= 18 ? 'decent long-term returns' :
                    score >= 10 ? 'moderate returns' : 'below-average returns';
    return parts.length ? `${parts.join(' · ')} — ${verdict}.` : `Score of ${score}/35 reflects historical CAGR performance.`;
  }

  if (cat === 'risk') {
    const sh = na2(data.sharpe) ? '' : `Sharpe ${f(data.sharpe).toFixed(2)}`;
    const mdd = na2(data.max_dd) ? '' : `max drawdown ${data.max_dd}`;
    const vol = na2(data.volatility) ? '' : `volatility ${data.volatility}`;
    const parts = [sh, mdd, vol].filter(Boolean);
    const verdict = score >= 24 ? 'low-risk, well-managed volatility' :
                    score >= 16 ? 'moderate risk profile' :
                    score >= 8  ? 'above-average risk' : 'high-risk fund';
    return parts.length ? `${parts.join(' · ')} — ${verdict}.` : `Score of ${score}/30 reflects overall risk profile.`;
  }

  if (cat === 'consistency') {
    const rp = na2(data.roll_pos) ? '' : `positive 1Y returns ${data.roll_pos} of the time`;
    const al = na2(data.alpha) ? '' : `alpha ${data.alpha} vs Nifty`;
    const parts = [rp, al].filter(Boolean);
    const verdict = score >= 17 ? 'highly consistent outperformer' :
                    score >= 12 ? 'reasonably consistent' :
                    score >= 6  ? 'somewhat inconsistent' : 'inconsistent vs benchmark';
    return parts.length ? `${parts.join('; ')} — ${verdict}.` : `Score of ${score}/20 reflects consistency vs Nifty 50.`;
  }

  if (cat === 'momentum') {
    const c1m = na2(data.cagr_1m) ? '' : `1M ${data.cagr_1m}`;
    const c3m = na2(data.cagr_3m) ? '' : `3M ${data.cagr_3m}`;
    const c6m = na2(data.cagr_6m) ? '' : `6M ${data.cagr_6m}`;
    const parts = [c1m, c3m, c6m].filter(Boolean);
    const verdict = score >= 12 ? 'strong near-term momentum' :
                    score >= 8  ? 'moderate recent momentum' :
                    score >= 4  ? 'mixed short-term trend' : 'weak or negative momentum';
    return parts.length ? `${parts.join(' · ')} — ${verdict}.` : `Score of ${score}/15 reflects recent price momentum.`;
  }
  return '';
}

function renderResult(data) {
  const d = data.data;
  const sc = data.scores;

  // Header
  $('r-name').textContent = data.name;
  $('r-category').textContent = data.category || data.type || '';

  const tags = [];
  if (data.house)    tags.push(`<span class="tag house">${data.house}</span>`);
  if (data.type)     tags.push(`<span class="tag">${data.type}</span>`);
  if (d.inception)   tags.push(`<span class="tag">Since ${d.nav_date ? d.nav_date.split(' ').slice(1).join(' ') : ''} · ${d.inception}</span>`);
  $('r-tags').innerHTML = tags.join('');

  const vb = $('r-verdict');
  vb.textContent = data.verdict;
  vb.className = `verdict-badge ${data.verdict}`;
  $('r-score').innerHTML = `${data.total}<span class="score-denom">/100</span>`;

  // Returns grid
  $('returns-grid').innerHTML = [
    retCard('1 Month',  d.cagr_1m),
    retCard('3 Months', d.cagr_3m),
    retCard('6 Months', d.cagr_6m),
    retCard('1 Year',   d.cagr_1y),
    retCard('3 Years',  d.cagr_3y),
    retCard('5 Years',  d.cagr_5y),
    retCard('10 Years', d.cagr_10y),
  ].join('');

  // Risk metrics
  $('risk-grid').innerHTML = [
    metricCard('Current NAV',   d.nav,        'brand',   `As of ${d.nav_date}. NAV = fund's per-unit price.`),
    metricCard('Volatility',    d.volatility, colorClass(d.volatility, 0, 12) === 'pos' ? 'pos' : d.volatility === 'N/A' ? 'neutral' : n(d.volatility) < 12 ? 'pos' : n(d.volatility) < 20 ? 'warn' : 'neg',
                                              interpret('volatility', d.volatility)),
    metricCard('Sharpe Ratio',  d.sharpe,     colorClass(d.sharpe, 1.5, 0.8), interpret('sharpe', d.sharpe)),
    metricCard('Sortino Ratio', d.sortino,    colorClass(d.sortino, 1.5, 0.8), interpret('sortino', d.sortino)),
    metricCard('Max Drawdown',  d.max_dd,     na(d.max_dd) ? 'neutral' : n(d.max_dd) > -15 ? 'pos' : n(d.max_dd) > -25 ? 'warn' : 'neg',
                                              interpret('max_dd', d.max_dd)),
    metricCard('Rolling +ve 1Y', d.roll_pos,  colorClass(d.roll_pos, 80, 65), interpret('roll_pos', d.roll_pos)),
  ].join('');

  // Benchmark
  $('bench-grid').innerHTML = [
    metricCard('Beta vs Nifty', d.beta,  colorClass(d.beta, 0, 0.9) === 'pos' ? 'pos' : na(d.beta) ? 'neutral' : n(d.beta) < 1 ? 'pos' : n(d.beta) < 1.2 ? 'warn' : 'neg',
                                          interpret('beta', d.beta)),
    metricCard('Alpha (3Y ann.)', d.alpha, na(d.alpha) ? 'neutral' : n(d.alpha) > 0 ? 'pos' : 'neg',
                                           interpret('alpha', d.alpha)),
  ].join('');

  // Breakdown
  const cats = [
    { id: 'Ret',  score: sc.returns,     max: 35, cat: 'returns' },
    { id: 'Risk', score: sc.risk,        max: 30, cat: 'risk' },
    { id: 'Con',  score: sc.consistency, max: 20, cat: 'consistency' },
    { id: 'Mom',  score: sc.momentum,     max: 15, cat: 'momentum' },
  ];
  cats.forEach(c => {
    $(`b${c.id}`).textContent = c.score;
    $(`b${c.id}Bar`).style.width = `${(c.score / c.max) * 100}%`;
    $(`b${c.id}Ex`).textContent = categoryOneLiner(c.cat, c.score, data.detail, d);
  });

  // AI Narrative
  if (data.narrative) {
    $('nar-body').textContent = data.narrative;
    $('narrative-box').style.display = 'block';
  } else {
    $('narrative-box').style.display = 'none';
  }

  $('result').style.display = 'block';
  $('result').scrollIntoView({behavior: 'smooth', block: 'start'});
}
</script>
</body>
</html>"""


if __name__ == '__main__':
    print('MFScope IN -> http://localhost:8092')
    app.run(host='0.0.0.0', port=8092, debug=False)
