"""
ScreenerIN — Combined Indian Stock + Mutual Fund Analyzer
Run: python app.py
"""

import os
import sys
import webbrowser
import logging
import time
import numpy as np
import pandas as pd
import requests
import requests_cache
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

load_dotenv('C:\\credentials\\.env')

requests_cache.install_cache(
    'screener_cache',
    backend='sqlite',
    expire_after=14400,
    allowable_codes=[200],
)

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
RISK_FREE_RATE = 0.065

app = Flask(__name__)
logging.basicConfig(level=logging.ERROR)


# ═══════════════════════════════════════════════════════════════════════════
# STOCK SCREENER — BACKEND
# ═══════════════════════════════════════════════════════════════════════════

def safe(val, default=None):
    return val if val is not None and val == val else default

def fmt_num(val, decimals=2):
    v = safe(val)
    if v is None: return 'N/A'
    try: return f'{float(v):.{decimals}f}'
    except: return 'N/A'

def fmt_pct(val, decimals=1):
    v = safe(val)
    if v is None: return 'N/A'
    try: return f'{float(v) * 100:.{decimals}f}%'
    except: return 'N/A'

def fmt_mcap(val):
    v = safe(val)
    if v is None: return 'N/A'
    cr = v / 1e7
    if cr >= 100000: return f'₹{cr/100000:.1f}L Cr'
    if cr >= 1000: return f'₹{cr/1000:.1f}K Cr'
    return f'₹{cr:.0f} Cr'


def score_stock(info):
    reasons, risks = [], []
    breakdown = {'valuation': 0, 'profitability': 0, 'safety': 0, 'momentum': 0}

    price = safe(info.get('currentPrice'), safe(info.get('regularMarketPrice'), 0))
    if not price:
        price = 0

    v = 0
    pe  = safe(info.get('trailingPE'))
    pb  = safe(info.get('priceToBook'))
    target = safe(info.get('targetMeanPrice'))

    if pe and pe > 0:
        if pe < 12:   v += 10; reasons.append(f'Attractive P/E: {pe:.1f}x')
        elif pe < 20: v += 8
        elif pe < 30: v += 5
        elif pe < 50: v += 2
        else:         risks.append(f'Expensive P/E: {pe:.1f}x')

    if pb and pb > 0:
        if pb < 1:    v += 8; reasons.append(f'Below book value (P/B {pb:.2f})')
        elif pb < 2:  v += 6
        elif pb < 4:  v += 3
        elif pb < 8:  v += 1
        else:         risks.append(f'Pricey P/B: {pb:.1f}x')

    if target and price and price > 0:
        upside = (target - price) / price * 100
        if upside > 30:   v += 7; reasons.append(f'Analysts see {upside:.0f}% upside')
        elif upside > 15: v += 4
        elif upside > 0:  v += 2
        else:             risks.append(f'Analysts see {abs(upside):.0f}% downside')

    breakdown['valuation'] = min(v, 25)

    p = 0
    roe   = safe(info.get('returnOnEquity'))
    opm   = safe(info.get('operatingMargins'))
    rev_g = safe(info.get('revenueGrowth'))

    if roe is not None:
        r = roe * 100
        if r > 25:   p += 10; reasons.append(f'Excellent ROE: {r:.1f}%')
        elif r > 18: p += 8
        elif r > 12: p += 5
        elif r > 6:  p += 2
        else:        risks.append(f'Weak ROE: {r:.1f}%')

    if opm is not None:
        o = opm * 100
        if o > 25:   p += 8; reasons.append(f'Strong margins (OPM {o:.1f}%)')
        elif o > 18: p += 6
        elif o > 12: p += 4
        elif o > 5:  p += 2
        else:        risks.append(f'Thin margins (OPM {o:.1f}%)')

    if rev_g is not None:
        g = rev_g * 100
        if g > 25:   p += 7; reasons.append(f'Revenue growing {g:.1f}%')
        elif g > 15: p += 5
        elif g > 5:  p += 3
        elif g > 0:  p += 1
        else:        risks.append(f'Revenue declining {g:.1f}%')

    breakdown['profitability'] = min(p, 25)

    s = 0
    de   = safe(info.get('debtToEquity'))
    cr   = safe(info.get('currentRatio'))
    beta = safe(info.get('beta'))

    if de is not None:
        de_r = de / 100
        if de_r == 0:    s += 10; reasons.append('Debt-free company')
        elif de_r < 0.3: s += 9;  reasons.append(f'Very low debt (D/E {de_r:.2f})')
        elif de_r < 0.7: s += 7
        elif de_r < 1.5: s += 4
        elif de_r < 3:   s += 1
        else:            risks.append(f'High debt load (D/E {de_r:.1f}x)')

    if cr is not None:
        if cr > 2.5:   s += 8
        elif cr > 1.5: s += 6
        elif cr > 1.0: s += 3
        else:          risks.append(f'Liquidity risk (CR {cr:.1f})')

    if beta is not None:
        if 0.5 <= beta <= 1.0:   s += 7
        elif beta < 0.5:         s += 4
        elif beta <= 1.5:        s += 5
        elif beta <= 2.0:        s += 2
        else:                    risks.append(f'Very volatile (β {beta:.2f})')

    breakdown['safety'] = min(s, 25)

    m = 0
    high_52w = safe(info.get('fiftyTwoWeekHigh'))
    eg       = safe(info.get('earningsGrowth'))
    rec      = safe(info.get('recommendationMean'))

    if high_52w and price and price > 0:
        pct_below = (high_52w - price) / high_52w * 100
        if pct_below < 5:    m += 8; reasons.append('Near 52-week high — strong trend')
        elif pct_below < 15: m += 6
        elif pct_below < 30: m += 4
        elif pct_below < 50: m += 2
        else:                risks.append(f'{pct_below:.0f}% below 52W high')

    if eg is not None:
        e = eg * 100
        if e > 30:   m += 10; reasons.append(f'Earnings surging {e:.0f}%')
        elif e > 15: m += 7
        elif e > 5:  m += 4
        elif e > 0:  m += 2
        else:        risks.append(f'Earnings declining {e:.0f}%')

    if rec is not None:
        if rec <= 1.5:   m += 7; reasons.append('Analysts strongly recommend Buy')
        elif rec <= 2.5: m += 5
        elif rec <= 3.0: m += 2
        else:            risks.append(f'Analysts lean bearish ({rec:.1f}/5)')

    breakdown['momentum'] = min(m, 25)

    total = sum(breakdown.values())
    verdict = 'BUY' if total >= 68 else 'WATCHLIST' if total >= 42 else 'SKIP'

    return {'score': total, 'verdict': verdict, 'breakdown': breakdown,
            'reasons': reasons[:5], 'risks': risks[:4]}


def get_ai_narrative_stock(name, ticker, score_data):
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        bd = score_data['breakdown']
        prompt = (
            f"You are an expert Indian stock analyst writing for everyday retail investors.\n\n"
            f"Stock: {name} ({ticker})\n"
            f"Verdict: {score_data['verdict']} (Score: {score_data['score']}/100)\n"
            f"Valuation {bd['valuation']}/25 | Profitability {bd['profitability']}/25 | "
            f"Safety {bd['safety']}/25 | Momentum {bd['momentum']}/25\n"
            f"Positives: {'; '.join(score_data['reasons']) or 'none notable'}\n"
            f"Risks: {'; '.join(score_data['risks']) or 'none notable'}\n\n"
            f"Write exactly 3 bullet points explaining this verdict. Rules:\n"
            f"- Each = 1 sentence, max 18 words\n"
            f"- Plain English, zero jargon\n"
            f"- Start each with a relevant emoji (not •)\n"
            f"- Be specific with numbers"
        )
        resp = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=200,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return resp.content[0].text.strip()
    except Exception:
        return None


def fetch_stock_info(symbol):
    import yfinance as yf
    for attempt in range(3):
        try:
            t = yf.Ticker(symbol)
            info = t.info or {}
            return t, info
        except Exception as e:
            err_str = str(e).lower()
            if 'rate' in err_str or 'too many' in err_str or '429' in err_str:
                if attempt < 2:
                    time.sleep(3 + attempt * 2)
                    continue
                raise RuntimeError('Yahoo Finance is rate-limiting right now. Please wait 30 seconds and try again.')
            raise
    return None, {}


@app.route('/api/stock/analyze', methods=['POST'])
def stock_analyze():
    data  = request.json or {}
    query = (data.get('ticker') or '').strip().upper().replace(' ', '-')

    if not query:
        return jsonify({'error': 'Enter a stock ticker (e.g. RELIANCE, INFY, TCS)'}), 400

    clean = query.replace('.NS', '').replace('.BO', '')
    ticker_obj, info, used_symbol = None, {}, clean + '.NS'

    try:
        for suffix in ['.NS', '.BO']:
            sym = clean + suffix
            t, inf = fetch_stock_info(sym)
            price = safe(inf.get('currentPrice'), safe(inf.get('regularMarketPrice')))
            if price:
                ticker_obj, info, used_symbol = t, inf, sym
                break
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Data fetch failed: {str(e)}'}), 500

    if ticker_obj is None or not info:
        return jsonify({'error': (
            f'Could not find "{query}" on NSE or BSE. '
            'Try the exact NSE symbol (e.g. RELIANCE, HDFCBANK, INFY).'
        )}), 404

    info = ticker_obj.info or {}
    score_data = score_stock(info)

    price      = safe(info.get('currentPrice'), safe(info.get('regularMarketPrice'), 0))
    change     = safe(info.get('regularMarketChange'), 0)
    change_pct = safe(info.get('regularMarketChangePercent'), 0)
    de = safe(info.get('debtToEquity'))

    metrics = {
        'pe':         fmt_num(info.get('trailingPE'), 1),
        'fpe':        fmt_num(info.get('forwardPE'), 1),
        'pb':         fmt_num(info.get('priceToBook'), 2),
        'peg':        fmt_num(info.get('pegRatio'), 2),
        'evebitda':   fmt_num(info.get('enterpriseToEbitda'), 1),
        'roe':        fmt_pct(info.get('returnOnEquity')),
        'opm':        fmt_pct(info.get('operatingMargins')),
        'npm':        fmt_pct(info.get('profitMargins')),
        'rev_growth': fmt_pct(info.get('revenueGrowth')),
        'eg_growth':  fmt_pct(info.get('earningsGrowth')),
        'de':         fmt_num(de / 100, 2) if de is not None else 'N/A',
        'cr':         fmt_num(info.get('currentRatio'), 2),
        'beta':       fmt_num(info.get('beta'), 2),
        'div_yield':  fmt_pct(info.get('dividendYield')),
        'mcap':       fmt_mcap(info.get('marketCap')),
        'high_52w':   fmt_num(info.get('fiftyTwoWeekHigh'), 2),
        'low_52w':    fmt_num(info.get('fiftyTwoWeekLow'), 2),
        'target':     fmt_num(info.get('targetMeanPrice'), 2),
        'analysts':   str(safe(info.get('numberOfAnalystOpinions'), 'N/A')),
        'rec':        (safe(info.get('recommendationKey'), 'N/A') or 'N/A').replace('_', ' ').title(),
        'sector':     safe(info.get('sector'), 'N/A'),
        'industry':   safe(info.get('industry'), 'N/A'),
    }

    ai_text = get_ai_narrative_stock(info.get('longName', used_symbol), used_symbol, score_data)

    return jsonify({
        'ticker':       used_symbol,
        'name':         info.get('longName', used_symbol),
        'price':        round(float(price), 2),
        'change':       round(float(change), 2),
        'change_pct':   round(float(change_pct), 2),
        'score':        score_data,
        'metrics':      metrics,
        'ai_narrative': ai_text,
    })


# ═══════════════════════════════════════════════════════════════════════════
# MF SCREENER — BACKEND
# ═══════════════════════════════════════════════════════════════════════════

_schemes_cache = {'data': None, 'ts': 0}


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
        nifty = yf.download('^NSEI', start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
        if nifty.empty:
            return None, None
        nifty_ret = nifty['Close'].squeeze().pct_change().dropna()
        fund_ret = nav_df[nav_df.index >= start_date]['nav'].pct_change().dropna()
        aligned = pd.concat([fund_ret.rename('fund'), nifty_ret.rename('nifty')], axis=1).dropna()
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


def score_fund(metrics, beta, alpha, expense_ratio=None):
    sc = {'returns': 0, 'risk': 0, 'consistency': 0, 'momentum': 0}
    d = {}

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

    rpp = metrics.get('rolling_positive_pct')
    if rpp is not None:
        s = 10 if rpp > 90 else 8 if rpp > 80 else 6 if rpp > 70 else 4 if rpp > 60 else 2
        sc['consistency'] += s; d['s_rpp'] = s

    if alpha is not None:
        s = 10 if alpha > 5 else 7 if alpha > 3 else 4 if alpha > 1 else 2 if alpha > 0 else 0
        sc['consistency'] += s; d['s_alpha'] = s

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


def get_ai_narrative_mf(fund_name, total, verdict, metrics, beta, alpha):
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
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


@app.route('/api/mf/search')
def mf_search():
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify([])
    try:
        return jsonify(search_funds(q))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mf/analyze', methods=['POST'])
def mf_analyze():
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
    total, verdict, sc, d = score_fund(metrics, beta, alpha)
    narrative = get_ai_narrative_mf(meta.get('scheme_name', scheme_code), total, verdict, metrics, beta, alpha)

    def fmt(v, decimals=2, suffix=''):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'N/A'
        return f'{v:.{decimals}f}{suffix}'

    return jsonify({
        'name':      meta.get('scheme_name', f'Fund #{scheme_code}'),
        'house':     meta.get('fund_house', ''),
        'category':  meta.get('scheme_category', ''),
        'type':      meta.get('scheme_type', ''),
        'code':      scheme_code,
        'total':     total,
        'verdict':   verdict,
        'scores':    sc,
        'detail':    d,
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
    })


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED HTML
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template_string(COMBINED_HTML)


COMBINED_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ScreenerIN — Stocks & Mutual Funds</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300..700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
/* ── WIOM DESIGN TOKENS ── */
:root {
  --brand:        #D9008D;
  --brand-light:  #FFB2E4;
  --brand-softer: #FFE5F6;
  --secondary:    #443152;
  --bg:           #161021;
  --surface:      #352D42;
  --surface2:     #1e1730;
  --border:       #4a4060;
  --text:         #FAF9FC;
  --muted:        #A7A1B2;
  --subtle:       #E8E4F0;
  --positive:     #008043;
  --positive-bg:  #0d2a1c;
  --positive-fg:  #4ade80;
  --negative:     #E01E00;
  --negative-bg:  #2d0f0a;
  --negative-fg:  #f87171;
  --warning:      #FF8000;
  --warning-bg:   #2d1800;
  --warning-fg:   #fb923c;
  --info:         #6D17CE;
  --info-bg:      #1a0d2e;
  --gold:         #D5AF4A;
  --radius-sm: 0.375rem;
  --radius-md: 0.625rem;
  --radius-lg: 1rem;
  --radius-xl: 1.5rem;
  --radius-full: 9999px;
  --font-body: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  --ease: 180ms cubic-bezier(0.16, 1, 0.3, 1);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { -webkit-font-smoothing: antialiased; }
body {
  min-height: 100dvh;
  font-family: var(--font-body);
  font-size: 14px;
  color: var(--text);
  background: var(--bg);
  line-height: 1.5;
}

/* ── UNIFIED TOPBAR ── */
.topbar {
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  padding: 0 28px;
  height: 56px;
  display: flex;
  align-items: center;
  gap: 20px;
  position: sticky;
  top: 0;
  z-index: 100;
}
.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
  font-size: 17px;
  letter-spacing: -0.02em;
  color: var(--text);
  text-decoration: none;
  white-space: nowrap;
}
.logo-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--brand);
  box-shadow: 0 0 8px var(--brand);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{ opacity:1;transform:scale(1) } 50%{ opacity:0.7;transform:scale(0.9) } }

/* ── TAB NAV ── */
.tab-nav {
  display: flex;
  gap: 4px;
  background: color-mix(in srgb, var(--bg) 60%, transparent);
  padding: 4px;
  border-radius: var(--radius-lg);
  border: 1px solid var(--border);
}
.tab-btn {
  font-size: 13px;
  font-weight: 600;
  padding: 6px 16px;
  border-radius: calc(var(--radius-lg) - 2px);
  border: none;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  transition: all var(--ease);
  font-family: var(--font-body);
  white-space: nowrap;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active {
  background: var(--brand);
  color: #fff;
  box-shadow: 0 0 12px color-mix(in srgb, var(--brand) 40%, transparent);
}
.topbar-right { margin-left: auto; }
.badge {
  font-size: 11px; font-weight: 700;
  padding: 3px 10px;
  border-radius: var(--radius-full);
  background: color-mix(in srgb, var(--brand) 15%, transparent);
  color: var(--brand);
  border: 1px solid color-mix(in srgb, var(--brand) 30%, transparent);
}

/* ── COMMON SHARED ── */
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ══════════════════════════════════════════════
   STOCK SCREENER — scoped to .ss-wrap
   ══════════════════════════════════════════════ */

.ss-wrap .hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 64px 24px 40px;
  text-align: center;
}
.ss-wrap .hero h1 {
  font-size: clamp(1.75rem, 4vw, 3rem);
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1.15;
  margin-bottom: 12px;
}
.ss-wrap .hero h1 span { color: var(--brand); }
.ss-wrap .hero p {
  font-size: 15px;
  color: var(--muted);
  max-width: 48ch;
  margin-bottom: 36px;
}
.ss-wrap .search-wrap {
  width: 100%;
  max-width: 560px;
  display: flex;
  gap: 10px;
}
.ss-wrap .search-input {
  flex: 1;
  height: 52px;
  padding: 0 20px;
  font-size: 16px;
  font-family: var(--font-body);
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--radius-lg);
  color: var(--text);
  outline: none;
  transition: all var(--ease);
}
.ss-wrap .search-input::placeholder { color: var(--muted); }
.ss-wrap .search-input:focus {
  border-color: var(--brand);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--brand) 20%, transparent);
}
.ss-wrap .btn-analyze {
  height: 52px;
  padding: 0 28px;
  font-size: 15px;
  font-weight: 700;
  font-family: var(--font-body);
  color: #fff;
  background: var(--brand);
  border: none;
  border-radius: var(--radius-lg);
  cursor: pointer;
  white-space: nowrap;
  transition: all var(--ease);
}
.ss-wrap .btn-analyze:hover {
  background: #b8007a;
  box-shadow: 0 0 24px color-mix(in srgb, var(--brand) 40%, transparent);
}
.ss-wrap .btn-analyze:disabled { opacity: 0.5; cursor: not-allowed; }
.ss-wrap .quick-tickers {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 16px;
}
.ss-wrap .quick-btn {
  font-size: 12px;
  font-weight: 600;
  padding: 5px 12px;
  border-radius: var(--radius-full);
  border: 1px solid var(--border);
  background: var(--surface2);
  color: var(--muted);
  cursor: pointer;
  transition: all var(--ease);
  font-family: var(--font-mono);
}
.ss-wrap .quick-btn:hover {
  border-color: var(--brand);
  color: var(--brand);
  background: color-mix(in srgb, var(--brand) 10%, transparent);
}
.ss-wrap .loader-wrap {
  display: none;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 40px;
}
.ss-wrap .loader-wrap.active { display: flex; }
.ss-wrap .spinner {
  width: 44px; height: 44px;
  border: 3px solid var(--border);
  border-top-color: var(--brand);
  border-radius: 50%;
  animation: spin 0.75s linear infinite;
}
.ss-wrap .loader-wrap p { font-size: 14px; color: var(--muted); }
.ss-wrap .error-box {
  display: none;
  max-width: 560px;
  margin: 0 auto 24px;
  padding: 14px 18px;
  border-radius: var(--radius-md);
  background: var(--negative-bg);
  border: 1px solid color-mix(in srgb, var(--negative) 40%, transparent);
  color: var(--negative-fg);
  font-size: 14px;
}
.ss-wrap .error-box.show { display: block; }
.ss-wrap .result-section {
  display: none;
  max-width: 900px;
  margin: 0 auto;
  padding: 0 24px 64px;
  animation: fadeUp 0.4s var(--ease) both;
}
.ss-wrap .result-section.show { display: block; }
.ss-wrap .company-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 20px;
  flex-wrap: wrap;
  gap: 12px;
}
.ss-wrap .company-info h2 { font-size: 22px; font-weight: 800; letter-spacing: -0.02em; }
.ss-wrap .company-info .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
.ss-wrap .price-block { text-align: right; }
.ss-wrap .price-main { font-size: 26px; font-weight: 800; font-family: var(--font-mono); }
.ss-wrap .price-change { font-size: 13px; font-family: var(--font-mono); margin-top: 2px; }
.ss-wrap .up { color: var(--positive-fg); }
.ss-wrap .dn { color: var(--negative-fg); }
.ss-wrap .verdict-card {
  border-radius: var(--radius-xl);
  padding: 24px;
  margin-bottom: 20px;
  border: 2px solid;
  position: relative;
  overflow: hidden;
}
.ss-wrap .verdict-card::before {
  content: '';
  position: absolute;
  inset: 0;
  opacity: 0.04;
  background: radial-gradient(circle at 20% 50%, white, transparent 60%);
  pointer-events: none;
}
.ss-wrap .verdict-BUY      { background: var(--positive-bg); border-color: var(--positive); }
.ss-wrap .verdict-WATCHLIST { background: var(--warning-bg);  border-color: var(--warning); }
.ss-wrap .verdict-SKIP     { background: var(--negative-bg); border-color: var(--negative); }
.ss-wrap .verdict-row { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.ss-wrap .verdict-badge { font-size: 28px; font-weight: 900; letter-spacing: 0.08em; line-height: 1; }
.ss-wrap .verdict-BUY .verdict-badge      { color: var(--positive-fg); }
.ss-wrap .verdict-WATCHLIST .verdict-badge { color: var(--warning-fg); }
.ss-wrap .verdict-SKIP .verdict-badge     { color: var(--negative-fg); }
.ss-wrap .verdict-score {
  display: flex; flex-direction: column; gap: 4px; flex: 1; min-width: 200px;
}
.ss-wrap .score-label {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
}
.ss-wrap .score-num { font-size: 13px; color: var(--muted); font-family: var(--font-mono); }
.ss-wrap .score-track {
  height: 8px;
  background: color-mix(in srgb, var(--muted) 20%, transparent);
  border-radius: var(--radius-full);
  overflow: hidden;
}
.ss-wrap .score-fill {
  height: 100%;
  border-radius: var(--radius-full);
  transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
.ss-wrap .verdict-BUY .score-fill      { background: var(--positive); }
.ss-wrap .verdict-WATCHLIST .score-fill { background: var(--warning); }
.ss-wrap .verdict-SKIP .score-fill     { background: var(--negative); }
.ss-wrap .breakdown-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}
.ss-wrap .breakdown-item {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 14px;
}
.ss-wrap .breakdown-icon { font-size: 18px; margin-bottom: 6px; }
.ss-wrap .breakdown-label {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted);
}
.ss-wrap .breakdown-score { font-size: 22px; font-weight: 900; font-family: var(--font-mono); margin: 4px 0; }
.ss-wrap .breakdown-max  { font-size: 11px; color: var(--muted); }
.ss-wrap .bex {
  font-size: 11px; color: var(--muted); line-height: 1.5;
  margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border);
}
.ss-wrap .mini-track {
  height: 4px; background: var(--border); border-radius: var(--radius-full); overflow: hidden; margin-top: 8px;
}
.ss-wrap .mini-fill {
  height: 100%; border-radius: var(--radius-full); background: var(--brand);
  transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
.ss-wrap .ai-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 18px;
  margin-bottom: 20px;
}
.ss-wrap .ai-label {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--brand); margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
}
.ss-wrap .ai-text { font-size: 14px; line-height: 1.8; color: var(--subtle); }
.ss-wrap .ai-text p { margin: 4px 0; }
.ss-wrap .tags-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.ss-wrap .tag {
  font-size: 12px; font-weight: 600; padding: 4px 12px;
  border-radius: var(--radius-full);
  display: flex; align-items: center; gap: 4px;
}
.ss-wrap .tag-bull {
  background: color-mix(in srgb, var(--positive) 15%, transparent);
  color: var(--positive-fg);
  border: 1px solid color-mix(in srgb, var(--positive) 30%, transparent);
}
.ss-wrap .tag-bear {
  background: color-mix(in srgb, var(--negative) 15%, transparent);
  color: var(--negative-fg);
  border: 1px solid color-mix(in srgb, var(--negative) 30%, transparent);
}
.ss-wrap .section-label {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin-bottom: 12px;
}
.ss-wrap .metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 8px;
  margin-bottom: 20px;
}
.ss-wrap .metric-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px 14px;
}
.ss-wrap .metric-card .mk { font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
.ss-wrap .metric-card .mv { font-size: 16px; font-weight: 800; font-family: var(--font-mono); margin-top: 4px; }
.ss-wrap .mv.good    { color: var(--positive-fg); }
.ss-wrap .mv.bad     { color: var(--negative-fg); }
.ss-wrap .mv.warn    { color: var(--warning-fg); }
.ss-wrap .mv.neutral { color: var(--text); }
.ss-wrap .metric-card .mex {
  font-size: 11px; color: var(--muted); line-height: 1.5;
  margin-top: 7px; padding-top: 7px; border-top: 1px solid var(--border);
}
.ss-wrap .divider { height: 1px; background: var(--border); margin: 20px 0; }
@media (max-width: 700px) {
  .ss-wrap .breakdown-grid { grid-template-columns: repeat(2, 1fr); }
  .ss-wrap .search-wrap    { flex-direction: column; }
  .ss-wrap .btn-analyze    { width: 100%; }
  .ss-wrap .hero           { padding: 40px 20px 32px; }
}

/* ══════════════════════════════════════════════
   MF SCREENER — scoped to .mf-wrap
   ══════════════════════════════════════════════ */

.mf-wrap main { max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px; }
.mf-wrap .hero { text-align: center; margin-bottom: 36px; }
.mf-wrap .hero h1 { font-size: 28px; font-weight: 800; margin-bottom: 8px; }
.mf-wrap .hero p  { color: var(--muted); font-size: 14px; }
.mf-wrap .search-wrap { position: relative; max-width: 580px; margin: 28px auto 0; }
.mf-wrap .search-input {
  width: 100%; background: var(--surface); border: 1.5px solid var(--border);
  color: var(--text); font-size: 15px; font-family: inherit;
  padding: 14px 52px 14px 18px; border-radius: 12px; outline: none; transition: border-color .2s;
}
.mf-wrap .search-input:focus { border-color: var(--brand); }
.mf-wrap .search-input::placeholder { color: var(--muted); }
.mf-wrap .search-btn {
  position: absolute; right: 10px; top: 50%; transform: translateY(-50%);
  background: var(--brand); border: none; color: #fff; width: 36px; height: 36px;
  border-radius: 8px; cursor: pointer; font-size: 17px; display: flex;
  align-items: center; justify-content: center;
}
.mf-wrap .search-btn:hover { opacity: .88; }
.mf-wrap .dropdown {
  position: absolute; top: calc(100% + 6px); left: 0; right: 0;
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0,0,0,.45); z-index: 200; overflow: hidden; display: none;
}
.mf-wrap .dropdown.open { display: block; }
.mf-wrap .dd-item {
  padding: 11px 16px; cursor: pointer; border-bottom: 1px solid rgba(102,94,117,.3); font-size: 13px; line-height: 1.4;
}
.mf-wrap .dd-item:last-child { border-bottom: none; }
.mf-wrap .dd-item:hover { background: rgba(217,0,141,.12); }
.mf-wrap .dd-item .dd-name { color: var(--text); font-weight: 500; }
.mf-wrap .dd-item .dd-code { color: var(--muted); font-size: 11px; margin-top: 2px; }
.mf-wrap .spinner {
  display: none; width: 28px; height: 28px; border: 3px solid var(--border);
  border-top-color: var(--brand); border-radius: 50%; animation: spin .7s linear infinite;
  margin: 48px auto;
}
.mf-wrap .error-box {
  background: rgba(224,30,0,.12); border: 1px solid var(--negative);
  color: #ff7b6b; border-radius: 12px; padding: 14px 18px; margin-top: 28px; font-size: 13px;
}
.mf-wrap .result { display: none; margin-top: 32px; }
.mf-wrap .fund-header {
  background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
  padding: 24px 28px; margin-bottom: 24px; display: flex;
  align-items: flex-start; justify-content: space-between; gap: 20px; flex-wrap: wrap;
}
.mf-wrap .fund-meta h2 { font-size: 20px; font-weight: 800; margin-bottom: 4px; }
.mf-wrap .fund-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.mf-wrap .tag {
  background: rgba(102,94,117,.45); color: var(--muted); font-size: 11px;
  font-weight: 600; padding: 3px 9px; border-radius: 99px; text-transform: uppercase; letter-spacing: .4px;
}
.mf-wrap .tag.house { background: rgba(217,0,141,.15); color: var(--brand-light); }
.mf-wrap .verdict-block { text-align: center; min-width: 120px; }
.mf-wrap .verdict-label {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 6px;
}
.mf-wrap .verdict-badge {
  font-size: 22px; font-weight: 900; padding: 8px 24px; border-radius: 12px; letter-spacing: .5px;
}
.mf-wrap .verdict-badge.INVEST    { background: rgba(0,128,67,.18); color: #4dff9b; border: 1.5px solid var(--positive); }
.mf-wrap .verdict-badge.WATCHLIST { background: rgba(255,128,0,.15); color: #ffb84d; border: 1.5px solid var(--warning); }
.mf-wrap .verdict-badge.SKIP      { background: rgba(224,30,0,.15);  color: #ff7b6b; border: 1.5px solid var(--negative); }
.mf-wrap .score-num { font-size: 32px; font-weight: 900; margin-top: 8px; }
.mf-wrap .score-denom { font-size: 14px; color: var(--muted); font-weight: 500; }
.mf-wrap .section-title {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
  color: var(--muted); margin-bottom: 14px; margin-top: 28px;
}
.mf-wrap .metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }
.mf-wrap .metric-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px 14px;
}
.mf-wrap .mk { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); margin-bottom: 6px; }
.mf-wrap .mv { font-size: 20px; font-weight: 800; }
.mf-wrap .mv.pos     { color: #4dff9b; }
.mf-wrap .mv.neg     { color: #ff7b6b; }
.mf-wrap .mv.warn    { color: #ffb84d; }
.mf-wrap .mv.neutral { color: var(--text); }
.mf-wrap .mv.brand   { color: var(--brand-light); }
.mf-wrap .mex {
  font-size: 11px; color: var(--muted); line-height: 1.5;
  margin-top: 7px; padding-top: 7px; border-top: 1px solid var(--border);
}
.mf-wrap .returns-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 10px; }
.mf-wrap .ret-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; text-align: center;
}
.mf-wrap .ret-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .4px; color: var(--muted); margin-bottom: 4px; }
.mf-wrap .ret-val { font-size: 18px; font-weight: 800; }
.mf-wrap .breakdown-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.mf-wrap .breakdown-item {
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px 16px;
}
.mf-wrap .bd-icon { font-size: 22px; margin-bottom: 6px; }
.mf-wrap .bd-label {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); margin-bottom: 4px;
}
.mf-wrap .bd-score { font-size: 26px; font-weight: 900; }
.mf-wrap .bd-max { font-size: 12px; color: var(--muted); font-weight: 500; }
.mf-wrap .mini-track {
  background: rgba(255,255,255,.1); border-radius: 99px; height: 5px; margin-top: 10px; overflow: hidden;
}
.mf-wrap .mini-fill { height: 100%; border-radius: 99px; background: var(--brand); transition: width .6s ease; }
.mf-wrap .bex {
  font-size: 11px; color: var(--muted); line-height: 1.5; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border);
}
.mf-wrap .narrative-box {
  background: rgba(109,23,206,.1); border: 1px solid rgba(109,23,206,.35);
  border-radius: 14px; padding: 18px 20px; margin-top: 20px;
}
.mf-wrap .nar-header {
  display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
  font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .6px; color: #b07fff;
}
.mf-wrap .nar-body { color: var(--subtle); font-size: 13px; line-height: 1.7; white-space: pre-line; }
@media (max-width: 700px) {
  .mf-wrap .breakdown-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>

<!-- UNIFIED TOPBAR -->
<header class="topbar">
  <a class="logo" href="#">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>
    </svg>
    ScreenerIN
    <div class="logo-dot"></div>
  </a>

  <nav class="tab-nav">
    <button id="tab-stock-btn" class="tab-btn active" onclick="switchTab('stock')">📈 Stocks</button>
    <button id="tab-mf-btn"    class="tab-btn"        onclick="switchTab('mf')">🏦 Mutual Funds</button>
  </nav>

  <div class="topbar-right">
    <span class="badge">AI-powered</span>
  </div>
</header>

<!-- ══ STOCK TAB ══════════════════════════════════════════════════════════ -->
<div id="tab-stock" class="ss-wrap">

  <section class="hero">
    <h1>Should you <span>buy</span> this stock?</h1>
    <p>Enter any NSE ticker and get an instant research-backed verdict — valuation, profitability, safety, and momentum all analyzed for you.</p>
    <div class="search-wrap">
      <input class="search-input" id="tickerInput" type="text"
             placeholder="Enter ticker: RELIANCE, INFY, TCS…"
             autocomplete="off" autocapitalize="characters" spellcheck="false">
      <button class="btn-analyze" id="analyzeBtn" onclick="ssAnalyze()">Analyze →</button>
    </div>
    <div class="quick-tickers">
      <span style="font-size:12px;color:var(--muted);align-self:center">Try:</span>
      <button class="quick-btn" onclick="ssQuickSearch('RELIANCE')">RELIANCE</button>
      <button class="quick-btn" onclick="ssQuickSearch('INFY')">INFY</button>
      <button class="quick-btn" onclick="ssQuickSearch('HDFCBANK')">HDFCBANK</button>
      <button class="quick-btn" onclick="ssQuickSearch('TCS')">TCS</button>
      <button class="quick-btn" onclick="ssQuickSearch('TITAN')">TITAN</button>
      <button class="quick-btn" onclick="ssQuickSearch('BAJFINANCE')">BAJFINANCE</button>
      <button class="quick-btn" onclick="ssQuickSearch('WIPRO')">WIPRO</button>
      <button class="quick-btn" onclick="ssQuickSearch('NESTLEIND')">NESTLEIND</button>
    </div>
  </section>

  <div class="loader-wrap" id="loaderWrap">
    <div class="spinner"></div>
    <p id="loaderMsg">Fetching data from NSE…</p>
  </div>

  <div class="error-box" id="errorBox"></div>

  <section class="result-section" id="resultSection">
    <div class="company-header">
      <div class="company-info">
        <h2 id="rName">—</h2>
        <div class="sub" id="rMeta">—</div>
      </div>
      <div class="price-block">
        <div class="price-main" id="rPrice">—</div>
        <div class="price-change" id="rChange">—</div>
      </div>
    </div>

    <div class="verdict-card" id="verdictCard">
      <div class="verdict-row">
        <div>
          <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted);margin-bottom:6px;">Verdict</div>
          <div class="verdict-badge" id="rVerdict">—</div>
        </div>
        <div class="verdict-score">
          <span class="score-label">Composite Score</span>
          <span class="score-num" id="rScoreNum">—</span>
          <div class="score-track">
            <div class="score-fill" id="rScoreFill" style="width:0%"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="breakdown-grid" id="breakdownGrid">
      <div class="breakdown-item">
        <div class="breakdown-icon">💰</div>
        <div class="breakdown-label">Valuation</div>
        <div class="breakdown-score" id="bVal">—</div>
        <div class="breakdown-max">out of 25</div>
        <div class="mini-track"><div class="mini-fill" id="bValBar" style="width:0%"></div></div>
        <div class="bex" id="bValEx"></div>
      </div>
      <div class="breakdown-item">
        <div class="breakdown-icon">📈</div>
        <div class="breakdown-label">Profitability</div>
        <div class="breakdown-score" id="bProf">—</div>
        <div class="breakdown-max">out of 25</div>
        <div class="mini-track"><div class="mini-fill" id="bProfBar" style="width:0%"></div></div>
        <div class="bex" id="bProfEx"></div>
      </div>
      <div class="breakdown-item">
        <div class="breakdown-icon">🛡️</div>
        <div class="breakdown-label">Safety</div>
        <div class="breakdown-score" id="bSafe">—</div>
        <div class="breakdown-max">out of 25</div>
        <div class="mini-track"><div class="mini-fill" id="bSafeBar" style="width:0%"></div></div>
        <div class="bex" id="bSafeEx"></div>
      </div>
      <div class="breakdown-item">
        <div class="breakdown-icon">🚀</div>
        <div class="breakdown-label">Momentum</div>
        <div class="breakdown-score" id="ss-bMom">—</div>
        <div class="breakdown-max">out of 25</div>
        <div class="mini-track"><div class="mini-fill" id="ss-bMomBar" style="width:0%"></div></div>
        <div class="bex" id="ss-bMomEx"></div>
      </div>
    </div>

    <div class="ai-card" id="aiCard" style="display:none">
      <div class="ai-label">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
        AI Analysis
      </div>
      <div class="ai-text" id="aiText"></div>
    </div>

    <div class="tags-row" id="tagsRow"></div>

    <div class="section-label">Key Metrics</div>
    <div class="metrics-grid" id="metricsGrid"></div>

    <div class="section-label">Analyst Consensus</div>
    <div class="metrics-grid" id="analystGrid"></div>
  </section>

</div><!-- end #tab-stock -->

<!-- ══ MF TAB ════════════════════════════════════════════════════════════ -->
<div id="tab-mf" class="mf-wrap" style="display:none">

  <main>
    <div class="hero">
      <h1>Analyze Any Mutual Fund</h1>
      <p>Type a fund name — get returns, risk, consistency, momentum &amp; a verdict in seconds</p>
      <div class="search-wrap">
        <input id="mf-q" class="search-input" type="text"
               placeholder="e.g. Mirae Asset Large Cap, Parag Parikh Flexi..."
               autocomplete="off" spellcheck="false">
        <button class="search-btn" onclick="mfTriggerSearch()">&#8594;</button>
        <div id="mf-dropdown" class="dropdown"></div>
      </div>
    </div>

    <div id="mf-spinner" class="spinner"></div>
    <div id="mf-error"   class="error-box" style="display:none;"></div>

    <div id="mf-result" class="result">
      <div class="fund-header">
        <div class="fund-meta">
          <h2 id="mf-r-name">—</h2>
          <div style="color:var(--muted);font-size:13px;margin-top:4px;" id="mf-r-category"></div>
          <div class="fund-tags" id="mf-r-tags"></div>
        </div>
        <div class="verdict-block">
          <div class="verdict-label">Verdict</div>
          <div class="verdict-badge" id="mf-r-verdict">—</div>
          <div class="score-num" id="mf-r-score">—<span class="score-denom">/100</span></div>
        </div>
      </div>

      <div class="section-title">Historical Returns (CAGR)</div>
      <div class="returns-grid" id="mf-returns-grid"></div>

      <div class="section-title">Risk Metrics</div>
      <div class="metric-grid" id="mf-risk-grid"></div>

      <div class="section-title">Benchmark (vs Nifty 50)</div>
      <div class="metric-grid" id="mf-bench-grid"></div>

      <div class="section-title">Category Score Breakdown</div>
      <div class="breakdown-grid">
        <div class="breakdown-item">
          <div class="bd-icon">📈</div>
          <div class="bd-label">Returns</div>
          <div class="bd-score" id="mf-bRet">—</div>
          <div class="bd-max">out of 35</div>
          <div class="mini-track"><div class="mini-fill" id="mf-bRetBar" style="width:0%"></div></div>
          <div class="bex" id="mf-bRetEx"></div>
        </div>
        <div class="breakdown-item">
          <div class="bd-icon">🛡️</div>
          <div class="bd-label">Risk</div>
          <div class="bd-score" id="mf-bRisk">—</div>
          <div class="bd-max">out of 30</div>
          <div class="mini-track"><div class="mini-fill" id="mf-bRiskBar" style="width:0%"></div></div>
          <div class="bex" id="mf-bRiskEx"></div>
        </div>
        <div class="breakdown-item">
          <div class="bd-icon">🔄</div>
          <div class="bd-label">Consistency</div>
          <div class="bd-score" id="mf-bCon">—</div>
          <div class="bd-max">out of 20</div>
          <div class="mini-track"><div class="mini-fill" id="mf-bConBar" style="width:0%"></div></div>
          <div class="bex" id="mf-bConEx"></div>
        </div>
        <div class="breakdown-item">
          <div class="bd-icon">⚡</div>
          <div class="bd-label">Momentum</div>
          <div class="bd-score" id="mf-bMom">—</div>
          <div class="bd-max">out of 15</div>
          <div class="mini-track"><div class="mini-fill" id="mf-bMomBar" style="width:0%"></div></div>
          <div class="bex" id="mf-bMomEx"></div>
        </div>
      </div>

      <div id="mf-narrative-box" class="narrative-box" style="display:none;">
        <div class="nar-header">✨ AI Analysis</div>
        <div class="nar-body" id="mf-nar-body"></div>
      </div>
    </div><!-- end #mf-result -->
  </main>

</div><!-- end #tab-mf -->


<script>
const $ = id => document.getElementById(id);

// ── TAB SWITCHING ────────────────────────────────────────────────────────
function switchTab(tab) {
  $('tab-stock').style.display = tab === 'stock' ? '' : 'none';
  $('tab-mf').style.display    = tab === 'mf'    ? '' : 'none';
  $('tab-stock-btn').classList.toggle('active', tab === 'stock');
  $('tab-mf-btn').classList.toggle('active', tab === 'mf');
}

// ── MF HELPERS ────────────────────────────────────────────────────────────
function n(v)  { return parseFloat(v); }
function na(v) { return v === 'N/A' || v == null; }

// ════════════════════════════════════════════════════════════════════════════
// STOCK SCREENER — JS
// ════════════════════════════════════════════════════════════════════════════

function ssQuickSearch(t) {
  $('tickerInput').value = t;
  ssAnalyze();
}

$('tickerInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') ssAnalyze();
});

function ssShowLoader(msg) {
  $('loaderWrap').classList.add('active');
  $('loaderMsg').textContent = msg || 'Analyzing stock…';
  $('resultSection').classList.remove('show');
  $('errorBox').classList.remove('show');
}

function ssHideLoader() {
  $('loaderWrap').classList.remove('active');
}

function ssShowError(msg) {
  $('errorBox').textContent = '⚠️ ' + msg;
  $('errorBox').classList.add('show');
}

function ssColorClass(val, goodThreshold, badThreshold) {
  if (val === 'N/A') return 'neutral';
  const v = parseFloat(val);
  if (isNaN(v)) return 'neutral';
  if (goodThreshold !== null && v >= goodThreshold) return 'good';
  if (badThreshold !== null && v <= badThreshold) return 'bad';
  return 'warn';
}

function ssMetricCard(label, value, cls, explain) {
  cls = cls || 'neutral';
  const ex = explain ? `<div class="mex">${explain}</div>` : '';
  return `<div class="metric-card"><div class="mk">${label}</div><div class="mv ${cls}">${value}</div>${ex}</div>`;
}

function ssCategoryOneLiner(cat, score, m, d) {
  const naChk = v => v === 'N/A' || isNaN(parseFloat(v));
  const nv    = v => parseFloat(v);

  if (cat === 'valuation') {
    const parts = [];
    if (!naChk(m.pe))  parts.push(`P/E ${nv(m.pe).toFixed(0)}x`);
    if (!naChk(m.pb))  parts.push(`P/B ${nv(m.pb).toFixed(1)}x`);
    if (!naChk(m.target) && d.price) {
      const up = ((nv(m.target) - d.price) / d.price * 100).toFixed(0);
      parts.push(`analyst target implies ${up > 0 ? '+' : ''}${up}% move`);
    }
    const verdict = score >= 20 ? 'attractively valued' : score >= 12 ? 'fairly valued' : score >= 6 ? 'slightly expensive' : 'expensive';
    return parts.length ? `${parts.join(', ')} — ${verdict}.` : `Score of ${score}/25 reflects current valuation multiples.`;
  }
  if (cat === 'profitability') {
    const parts = [];
    if (!naChk(m.roe))        parts.push(`ROE ${m.roe}`);
    if (!naChk(m.opm))        parts.push(`margins ${m.opm}`);
    if (!naChk(m.rev_growth)) parts.push(`revenue growing ${m.rev_growth}`);
    const verdict = score >= 20 ? 'highly profitable' : score >= 12 ? 'decent profitability' : score >= 6 ? 'below-average earnings' : 'weak profitability';
    return parts.length ? `${parts.join(', ')} — ${verdict}.` : `Score of ${score}/25 reflects earnings quality.`;
  }
  if (cat === 'safety') {
    const parts = [];
    if (!naChk(m.de))   parts.push(nv(m.de) === 0 ? 'debt-free' : `D/E ${m.de}`);
    if (!naChk(m.cr))   parts.push(`current ratio ${m.cr}`);
    if (!naChk(m.beta)) parts.push(`beta ${m.beta}`);
    const verdict = score >= 20 ? 'very safe balance sheet' : score >= 12 ? 'reasonably safe' : score >= 6 ? 'some financial risk' : 'high financial risk';
    return parts.length ? `${parts.join(', ')} — ${verdict}.` : `Score of ${score}/25 reflects balance sheet health.`;
  }
  if (cat === 'momentum') {
    const parts = [];
    if (d.price && !naChk(m.high_52w)) {
      const pct = ((nv(m.high_52w) - d.price) / nv(m.high_52w) * 100).toFixed(0);
      parts.push(`${pct}% below 52W high`);
    }
    if (!naChk(m.eg_growth)) parts.push(`earnings ${m.eg_growth}`);
    if (m.rec && m.rec !== 'N/A') parts.push(`analysts say ${m.rec}`);
    const verdict = score >= 20 ? 'strong momentum' : score >= 12 ? 'moderate momentum' : score >= 6 ? 'weak momentum' : 'poor momentum';
    return parts.length ? `${parts.join(', ')} — ${verdict}.` : `Score of ${score}/25 reflects price and growth trend.`;
  }
  return '';
}

function ssInterpret(key, rawVal) {
  const v = parseFloat(rawVal);
  if (rawVal === 'N/A' || isNaN(v)) return 'Data not available for this stock.';
  switch (key) {
    case 'pe':
      if (v < 12)  return 'Very cheap — you pay just ₹' + v.toFixed(0) + ' for ₹1 of profit. Strong value opportunity.';
      if (v < 20)  return 'Fairly valued — ₹' + v.toFixed(0) + ' per ₹1 of profit. Reasonable for stable businesses.';
      if (v < 30)  return 'Slightly expensive at ₹' + v.toFixed(0) + '/₹1 profit. Acceptable if earnings are growing fast.';
      if (v < 50)  return 'High valuation — ₹' + v.toFixed(0) + ' per ₹1 profit. Only worth it with very strong growth.';
      return 'Very expensive at ₹' + v.toFixed(0) + '/₹1 profit. High risk if growth disappoints.';
    case 'fpe':
      if (v < 15)  return 'Forward earnings look cheap — analysts expect strong profit improvement ahead.';
      if (v < 25)  return 'Reasonable forward valuation — market expects moderate growth going forward.';
      return 'Market is pricing in high growth expectations — could disappoint if missed.';
    case 'pb':
      if (v < 1)   return 'Trading below its own book value — potentially very cheap vs. assets on paper.';
      if (v < 2)   return 'Low price-to-book. Good for asset-heavy businesses like banks and manufacturing.';
      if (v < 4)   return 'Moderate premium over assets — typical for quality businesses with good returns.';
      return 'High premium — market values brand/intangibles heavily. Justified only if ROE is strong.';
    case 'peg':
      if (v < 1)   return 'Growth at a bargain — PEG below 1 means you\'re not paying full price for growth.';
      if (v < 1.5) return 'Reasonable PEG — growth is priced in but not excessively so.';
      if (v < 2.5) return 'Growth is getting expensive. Ensure growth estimates are reliable.';
      return 'Expensive relative to growth. Little margin of safety if growth slows.';
    case 'evebitda':
      if (v < 10)  return 'Very cheap on enterprise value — strong acquisition target or deep value pick.';
      if (v < 18)  return 'Fair value — in line with market average for most Indian sectors.';
      if (v < 30)  return 'Somewhat expensive — premium for quality. Check if margins justify it.';
      return 'Expensive EV multiple — market pricing in a lot of future growth.';
    case 'roe':
      if (v > 25)  return 'Excellent — earns ₹' + v.toFixed(0) + ' for every ₹100 of shareholder money. Best-in-class.';
      if (v > 18)  return 'Strong ROE — well above the 15% threshold that signals quality management.';
      if (v > 12)  return 'Decent — meets minimum quality bar. Look for a consistent 3-year trend.';
      if (v > 6)   return 'Below average — low returns on equity. May signal inefficient capital use.';
      return 'Weak ROE — the company is not generating strong returns for its shareholders.';
    case 'opm':
      if (v > 30)  return 'Best-in-class margins at ' + v.toFixed(1) + '%. Exceptional pricing power and efficiency.';
      if (v > 20)  return 'Strong margins — ' + v.toFixed(1) + '% OPM signals solid competitive moat.';
      if (v > 12)  return 'Average margins — typical for most sectors. Watch for expansion or compression.';
      if (v > 5)   return 'Thin margins at ' + v.toFixed(1) + '%. Vulnerable to input cost spikes or pricing pressure.';
      return 'Very thin margins — even small cost increases can wipe out profitability.';
    case 'npm':
      if (v > 20)  return 'Excellent net margins — keeps ' + v.toFixed(1) + '% of revenue as profit after all costs.';
      if (v > 10)  return 'Healthy net margins. Company efficiently converts revenue to bottom-line profit.';
      if (v > 5)   return 'Moderate net margins. Acceptable but leaves little buffer for downturns.';
      return 'Thin net margins — high operating leverage means earnings are very sensitive to revenue.';
    case 'rev_growth':
      if (v > 25)  return 'Rapid revenue expansion at ' + v.toFixed(1) + '% — strong demand or market share gain.';
      if (v > 15)  return 'Healthy growth at ' + v.toFixed(1) + '% — ahead of India\'s nominal GDP growth rate.';
      if (v > 5)   return 'Moderate growth at ' + v.toFixed(1) + '%. Stable but not exceptional.';
      if (v > 0)   return 'Slow growth at ' + v.toFixed(1) + '%. May be a mature business or facing headwinds.';
      return 'Revenue is shrinking at ' + v.toFixed(1) + '% — red flag unless it\'s a one-time event.';
    case 'eg_growth':
      if (v > 30)  return 'Earnings surging ' + v.toFixed(0) + '% — strong business momentum or margin expansion.';
      if (v > 15)  return 'Solid earnings growth — outpacing market average. Positive signal.';
      if (v > 0)   return 'Modest earnings growth. Check if it\'s improving quarter-over-quarter.';
      return 'Earnings declining ' + Math.abs(v).toFixed(0) + '% — needs investigation before investing.';
    case 'de':
      if (v === 0) return 'Debt-free — maximum safety. No interest burden eating into profits.';
      if (v < 0.3) return 'Very low debt of ' + v.toFixed(2) + 'x equity. Conservative, financially strong.';
      if (v < 0.7) return 'Low debt at ' + v.toFixed(2) + 'x — healthy balance sheet with room to borrow if needed.';
      if (v < 1.5) return 'Moderate debt at ' + v.toFixed(1) + 'x — manageable, but watch interest coverage closely.';
      return 'High debt load at ' + v.toFixed(1) + 'x equity. Significant risk if earnings fall or rates rise.';
    case 'cr':
      if (v > 3)   return 'Very liquid — can pay all short-term bills ' + v.toFixed(0) + 'x over. Very safe.';
      if (v > 2)   return 'Healthy at ' + v.toFixed(1) + 'x — comfortably covers short-term liabilities.';
      if (v > 1.5) return 'Adequate liquidity. Can meet obligations but with limited buffer.';
      if (v > 1)   return 'Tight at ' + v.toFixed(1) + 'x — just barely covers short-term debt. Monitor closely.';
      return 'Current ratio below 1 — may struggle to pay near-term bills. Potential cash crunch.';
    case 'beta':
      if (v < 0.5) return 'Very stable — moves much less than the market. Good for conservative investors.';
      if (v < 0.8) return 'Low volatility — stock is more stable than Nifty 50. Defensive nature.';
      if (v < 1.2) return 'Market-like movement — rises and falls roughly in line with Nifty.';
      if (v < 1.8) return 'More volatile than market at ' + v.toFixed(2) + ' beta — higher risk, higher reward potential.';
      return 'High volatility (β ' + v.toFixed(2) + ') — big swings up and down. Only for risk-tolerant investors.';
    case 'div_yield':
      if (v === 0) return 'No dividend — company reinvests profits for growth. Common in growth stocks.';
      if (v < 1)   return 'Low yield of ' + v.toFixed(1) + '% — small income, but growth may compensate.';
      if (v < 2.5) return 'Decent yield of ' + v.toFixed(1) + '% — adds steady income on top of capital gains.';
      if (v < 5)   return 'Good yield at ' + v.toFixed(1) + '% — attractive income, comparable to FD rates.';
      return 'High yield of ' + v.toFixed(1) + '% — verify this is sustainable and not a sign of price distress.';
    default: return null;
  }
}

async function ssAnalyze() {
  const ticker = $('tickerInput').value.trim();
  if (!ticker) { ssShowError('Please enter a stock ticker.'); return; }
  $('analyzeBtn').disabled = true;
  ssShowLoader('Fetching data from NSE / BSE…');

  try {
    const res = await fetch('/api/stock/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ticker})
    });
    const d = await res.json();
    ssHideLoader();
    if (!res.ok || d.error) { ssShowError(d.error || 'Something went wrong.'); return; }
    ssRenderResult(d);
  } catch (e) {
    ssHideLoader();
    ssShowError('Network error — is the server running?');
  } finally {
    $('analyzeBtn').disabled = false;
  }
}

function ssRenderResult(d) {
  const s = d.score;
  const m = d.metrics;

  $('rName').textContent = d.name;
  $('rMeta').textContent = `${d.ticker}  ·  ${m.sector}  ·  ${m.industry}`;
  $('rPrice').textContent = `₹${d.price.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
  const up = d.change_pct >= 0;
  $('rChange').innerHTML = `<span class="${up ? 'up' : 'dn'}">${up ? '▲' : '▼'} ${Math.abs(d.change_pct).toFixed(2)}%  (${up ? '+' : ''}${d.change.toFixed(2)})</span>`;

  const vc = $('verdictCard');
  vc.className = 'verdict-card verdict-' + s.verdict;
  $('rVerdict').textContent = s.verdict;
  $('rScoreNum').textContent = `${s.score} / 100`;
  $('rScoreFill').style.width = s.score + '%';

  const bd = s.breakdown;
  $('bVal').textContent  = bd.valuation;
  $('bProf').textContent = bd.profitability;
  $('bSafe').textContent = bd.safety;
  $('ss-bMom').textContent  = bd.momentum;
  $('bValBar').style.width  = (bd.valuation  / 25 * 100) + '%';
  $('bProfBar').style.width = (bd.profitability / 25 * 100) + '%';
  $('bSafeBar').style.width = (bd.safety / 25 * 100) + '%';
  $('ss-bMomBar').style.width  = (bd.momentum / 25 * 100) + '%';
  $('bValEx').textContent  = ssCategoryOneLiner('valuation',     bd.valuation,     m, d);
  $('bProfEx').textContent = ssCategoryOneLiner('profitability', bd.profitability, m, d);
  $('bSafeEx').textContent = ssCategoryOneLiner('safety',        bd.safety,        m, d);
  $('ss-bMomEx').textContent  = ssCategoryOneLiner('momentum',  bd.momentum,      m, d);

  if (d.ai_narrative) {
    $('aiCard').style.display = '';
    const lines = d.ai_narrative.split('\n').filter(l => l.trim());
    $('aiText').innerHTML = lines.map(l => `<p>${l}</p>`).join('');
  } else {
    $('aiCard').style.display = 'none';
  }

  const tags = [
    ...s.reasons.map(r => `<span class="tag tag-bull">✓ ${r}</span>`),
    ...s.risks.map(r => `<span class="tag tag-bear">⚠ ${r}</span>`),
  ];
  $('tagsRow').innerHTML = tags.join('');

  const pe_v  = parseFloat(m.pe);
  const fpe_v = parseFloat(m.fpe);
  const pb_v  = parseFloat(m.pb);
  const peg_v = parseFloat(m.peg);
  const roe_v = parseFloat(m.roe);
  const opm_v = parseFloat(m.opm);
  const npm_v = parseFloat(m.npm);
  const rg_v  = parseFloat(m.rev_growth);
  const eg_v  = parseFloat(m.eg_growth);
  const de_v  = parseFloat(m.de);
  const cr_v  = parseFloat(m.cr);
  const bt_v  = parseFloat(m.beta);
  const dy_v  = parseFloat(m.div_yield);

  const hw = parseFloat(m.high_52w), lw = parseFloat(m.low_52w), cp = d.price;
  let w52ex = 'No 52-week data available.';
  if (!isNaN(hw) && !isNaN(cp)) {
    const pctBelow = ((hw - cp) / hw * 100).toFixed(1);
    const pctAbove = ((cp - lw) / lw * 100).toFixed(1);
    w52ex = `Stock is ${pctBelow}% below its 52-week high of ₹${m.high_52w} and ${pctAbove}% above its 52-week low of ₹${m.low_52w}.`;
  }
  let tgtEx = 'No analyst target data.';
  if (m.target !== 'N/A' && cp) {
    const upside = ((parseFloat(m.target) - cp) / cp * 100).toFixed(1);
    const dir = upside >= 0 ? 'upside' : 'downside';
    tgtEx = `${m.analysts} analysts average a target of ₹${m.target} — implying ${Math.abs(upside)}% ${dir} from current price.`;
  }

  $('metricsGrid').innerHTML = [
    ssMetricCard('Market Cap', m.mcap, 'neutral', 'Total market value of all shares. Large cap (>₹20K Cr) = established company; Small cap (<₹5K Cr) = higher risk and reward.'),
    ssMetricCard('P/E (TTM)', m.pe, !isNaN(pe_v) && pe_v < 25 ? 'good' : !isNaN(pe_v) && pe_v > 50 ? 'bad' : 'warn', ssInterpret('pe', m.pe)),
    ssMetricCard('Forward P/E', m.fpe, 'neutral', ssInterpret('fpe', m.fpe)),
    ssMetricCard('P/B Ratio', m.pb, !isNaN(pb_v) && pb_v < 3 ? 'good' : 'warn', ssInterpret('pb', m.pb)),
    ssMetricCard('PEG Ratio', m.peg, !isNaN(peg_v) && peg_v < 1.5 ? 'good' : 'warn', ssInterpret('peg', m.peg)),
    ssMetricCard('EV/EBITDA', m.evebitda, 'neutral', ssInterpret('evebitda', m.evebitda)),
    ssMetricCard('ROE', m.roe, !isNaN(roe_v) && roe_v > 15 ? 'good' : 'warn', ssInterpret('roe', m.roe)),
    ssMetricCard('Op. Margin', m.opm, !isNaN(opm_v) && opm_v > 15 ? 'good' : 'warn', ssInterpret('opm', m.opm)),
    ssMetricCard('Net Margin', m.npm, !isNaN(npm_v) && npm_v > 10 ? 'good' : 'warn', ssInterpret('npm', m.npm)),
    ssMetricCard('Rev. Growth', m.rev_growth, !isNaN(rg_v) && rg_v > 10 ? 'good' : rg_v < 0 ? 'bad' : 'warn', ssInterpret('rev_growth', m.rev_growth)),
    ssMetricCard('EPS Growth', m.eg_growth, !isNaN(eg_v) && eg_v > 10 ? 'good' : eg_v < 0 ? 'bad' : 'warn', ssInterpret('eg_growth', m.eg_growth)),
    ssMetricCard('Debt/Equity', m.de, !isNaN(de_v) && de_v < 0.5 ? 'good' : de_v > 1.5 ? 'bad' : 'warn', ssInterpret('de', m.de)),
    ssMetricCard('Current Ratio', m.cr, !isNaN(cr_v) && cr_v > 1.5 ? 'good' : cr_v < 1 ? 'bad' : 'warn', ssInterpret('cr', m.cr)),
    ssMetricCard('Beta', m.beta, 'neutral', ssInterpret('beta', m.beta)),
    ssMetricCard('Dividend Yield', m.div_yield, !isNaN(dy_v) && dy_v > 1.5 ? 'good' : 'neutral', ssInterpret('div_yield', m.div_yield)),
    ssMetricCard('52W High / Low', `₹${m.high_52w} / ₹${m.low_52w}`, 'neutral', w52ex),
    ssMetricCard('Analyst Target', `₹${m.target}`, 'neutral', tgtEx),
  ].join('');

  const recLow = (m.rec || '').toLowerCase();
  const recEx = recLow.includes('strong buy') ? 'Analysts strongly recommend buying this stock right now.'
    : recLow.includes('buy') ? 'Most analysts consider this a good buy at the current price.'
    : recLow.includes('hold') ? 'Analysts suggest holding — not a strong buy or sell signal.'
    : recLow.includes('sell') ? 'Analysts recommend selling or avoiding this stock.'
    : 'No clear consensus from analyst community.';
  $('analystGrid').innerHTML = [
    ssMetricCard('Consensus', m.rec, recLow.includes('buy') ? 'good' : recLow.includes('sell') ? 'bad' : 'warn', recEx),
    ssMetricCard('# Analysts', m.analysts, 'neutral', `${m.analysts} research analysts actively cover this stock. More coverage = more reliable consensus.`),
    ssMetricCard('Price Target', `₹${m.target}`, 'neutral', tgtEx),
  ].join('');

  $('resultSection').classList.add('show');
  $('resultSection').scrollIntoView({behavior: 'smooth', block: 'start'});
}


// ════════════════════════════════════════════════════════════════════════════
// MF SCREENER — JS
// ════════════════════════════════════════════════════════════════════════════

let mfDebounceTimer = null;
let mfSelectedCode  = null;

$('mf-q').addEventListener('input', () => {
  clearTimeout(mfDebounceTimer);
  const q = $('mf-q').value.trim();
  if (q.length < 2) { mfCloseDropdown(); return; }
  mfDebounceTimer = setTimeout(() => mfFetchSuggestions(q), 280);
});

$('mf-q').addEventListener('keydown', e => {
  if (e.key === 'Enter') mfTriggerSearch();
  if (e.key === 'Escape') mfCloseDropdown();
});

document.addEventListener('click', e => {
  if (!e.target.closest('.mf-wrap .search-wrap')) mfCloseDropdown();
});

async function mfFetchSuggestions(q) {
  try {
    const r = await fetch(`/api/mf/search?q=${encodeURIComponent(q)}`);
    const data = await r.json();
    mfRenderDropdown(data);
  } catch(e) {}
}

function mfRenderDropdown(items) {
  const dd = $('mf-dropdown');
  if (!items.length) { mfCloseDropdown(); return; }
  dd.innerHTML = items.map(i =>
    `<div class="dd-item" onclick="mfSelectFund('${i.code}', \`${i.name.replace(/`/g,'&#96;')}\`)">
       <div class="dd-name">${i.name}</div>
       <div class="dd-code">Code: ${i.code}</div>
     </div>`
  ).join('');
  dd.classList.add('open');
}

function mfCloseDropdown() {
  $('mf-dropdown').classList.remove('open');
}

function mfSelectFund(code, name) {
  mfSelectedCode = code;
  $('mf-q').value = name;
  mfCloseDropdown();
  mfAnalyze(code);
}

function mfTriggerSearch() {
  if (mfSelectedCode) { mfAnalyze(mfSelectedCode); return; }
  const q = $('mf-q').value.trim();
  if (q.length < 2) return;
  fetch(`/api/mf/search?q=${encodeURIComponent(q)}`)
    .then(r => r.json())
    .then(data => { if (data.length) mfSelectFund(data[0].code, data[0].name); });
}

async function mfAnalyze(code) {
  $('mf-spinner').style.display = 'block';
  $('mf-result').style.display  = 'none';
  $('mf-error').style.display   = 'none';
  mfCloseDropdown();

  try {
    const r = await fetch('/api/mf/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({code})
    });
    const data = await r.json();
    if (!r.ok) { mfShowError(data.error || 'Analysis failed'); return; }
    mfRenderResult(data);
  } catch(e) {
    mfShowError('Network error. Please try again.');
  } finally {
    $('mf-spinner').style.display = 'none';
  }
}

function mfShowError(msg) {
  $('mf-error').textContent = msg;
  $('mf-error').style.display = 'block';
}

function mfColorClass(v, pos_thresh, warn_thresh) {
  if (na(v)) return 'neutral';
  const f = n(v);
  return f >= pos_thresh ? 'pos' : f >= warn_thresh ? 'warn' : 'neg';
}

function mfRetColor(v) {
  if (na(v)) return 'neutral';
  return n(v) > 0 ? 'pos' : 'neg';
}

function mfMetricCard(label, value, cls, explain) {
  cls = cls || 'neutral';
  const ex = explain ? `<div class="mex">${explain}</div>` : '';
  return `<div class="metric-card"><div class="mk">${label}</div><div class="mv ${cls}">${value}</div>${ex}</div>`;
}

function mfRetCard(label, value) {
  const cls = na(value) ? 'neutral' : (n(value) > 0 ? 'pos' : 'neg');
  return `<div class="ret-card"><div class="ret-label">${label}</div><div class="ret-val ${cls}">${value}</div></div>`;
}

function mfInterpret(key, val) {
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

function mfCategoryOneLiner(cat, score, d, data) {
  const na2 = v => v === 'N/A' || isNaN(parseFloat(v));
  const f = v => parseFloat(v);

  if (cat === 'returns') {
    const parts = [];
    if (!na2(data.cagr_1y)) parts.push(`1Y ${data.cagr_1y}`);
    if (!na2(data.cagr_3y)) parts.push(`3Y ${data.cagr_3y}`);
    if (!na2(data.cagr_5y)) parts.push(`5Y ${data.cagr_5y}`);
    const verdict = score >= 28 ? 'consistently strong returns' : score >= 18 ? 'decent long-term returns' : score >= 10 ? 'moderate returns' : 'below-average returns';
    return parts.length ? `${parts.join(' · ')} — ${verdict}.` : `Score of ${score}/35 reflects historical CAGR performance.`;
  }
  if (cat === 'risk') {
    const sh  = na2(data.sharpe)     ? '' : `Sharpe ${f(data.sharpe).toFixed(2)}`;
    const mdd = na2(data.max_dd)     ? '' : `max drawdown ${data.max_dd}`;
    const vol = na2(data.volatility) ? '' : `volatility ${data.volatility}`;
    const parts = [sh, mdd, vol].filter(Boolean);
    const verdict = score >= 24 ? 'low-risk, well-managed volatility' : score >= 16 ? 'moderate risk profile' : score >= 8 ? 'above-average risk' : 'high-risk fund';
    return parts.length ? `${parts.join(' · ')} — ${verdict}.` : `Score of ${score}/30 reflects overall risk profile.`;
  }
  if (cat === 'consistency') {
    const rp = na2(data.roll_pos) ? '' : `positive 1Y returns ${data.roll_pos} of the time`;
    const al = na2(data.alpha)    ? '' : `alpha ${data.alpha} vs Nifty`;
    const parts = [rp, al].filter(Boolean);
    const verdict = score >= 17 ? 'highly consistent outperformer' : score >= 12 ? 'reasonably consistent' : score >= 6 ? 'somewhat inconsistent' : 'inconsistent vs benchmark';
    return parts.length ? `${parts.join('; ')} — ${verdict}.` : `Score of ${score}/20 reflects consistency vs Nifty 50.`;
  }
  if (cat === 'momentum') {
    const c1m = na2(data.cagr_1m) ? '' : `1M ${data.cagr_1m}`;
    const c3m = na2(data.cagr_3m) ? '' : `3M ${data.cagr_3m}`;
    const c6m = na2(data.cagr_6m) ? '' : `6M ${data.cagr_6m}`;
    const parts = [c1m, c3m, c6m].filter(Boolean);
    const verdict = score >= 12 ? 'strong near-term momentum' : score >= 8 ? 'moderate recent momentum' : score >= 4 ? 'mixed short-term trend' : 'weak or negative momentum';
    return parts.length ? `${parts.join(' · ')} — ${verdict}.` : `Score of ${score}/15 reflects recent price momentum.`;
  }
  return '';
}

function mfRenderResult(data) {
  const d = data.data;
  const sc = data.scores;

  $('mf-r-name').textContent = data.name;
  $('mf-r-category').textContent = data.category || data.type || '';

  const tags = [];
  if (data.house)  tags.push(`<span class="tag house">${data.house}</span>`);
  if (data.type)   tags.push(`<span class="tag">${data.type}</span>`);
  if (d.inception) tags.push(`<span class="tag">Since ${d.nav_date ? d.nav_date.split(' ').slice(1).join(' ') : ''} · ${d.inception}</span>`);
  $('mf-r-tags').innerHTML = tags.join('');

  const vb = $('mf-r-verdict');
  vb.textContent = data.verdict;
  vb.className = `verdict-badge ${data.verdict}`;
  $('mf-r-score').innerHTML = `${data.total}<span class="score-denom">/100</span>`;

  $('mf-returns-grid').innerHTML = [
    mfRetCard('1 Month',  d.cagr_1m),
    mfRetCard('3 Months', d.cagr_3m),
    mfRetCard('6 Months', d.cagr_6m),
    mfRetCard('1 Year',   d.cagr_1y),
    mfRetCard('3 Years',  d.cagr_3y),
    mfRetCard('5 Years',  d.cagr_5y),
    mfRetCard('10 Years', d.cagr_10y),
  ].join('');

  $('mf-risk-grid').innerHTML = [
    mfMetricCard('Current NAV',    d.nav,        'brand', `As of ${d.nav_date}. NAV = fund\'s per-unit price.`),
    mfMetricCard('Volatility',     d.volatility, na(d.volatility) ? 'neutral' : n(d.volatility) < 12 ? 'pos' : n(d.volatility) < 20 ? 'warn' : 'neg', mfInterpret('volatility', d.volatility)),
    mfMetricCard('Sharpe Ratio',   d.sharpe,     mfColorClass(d.sharpe, 1.5, 0.8),  mfInterpret('sharpe',   d.sharpe)),
    mfMetricCard('Sortino Ratio',  d.sortino,    mfColorClass(d.sortino, 1.5, 0.8), mfInterpret('sortino',  d.sortino)),
    mfMetricCard('Max Drawdown',   d.max_dd,     na(d.max_dd) ? 'neutral' : n(d.max_dd) > -15 ? 'pos' : n(d.max_dd) > -25 ? 'warn' : 'neg', mfInterpret('max_dd', d.max_dd)),
    mfMetricCard('Rolling +ve 1Y', d.roll_pos,   mfColorClass(d.roll_pos, 80, 65),  mfInterpret('roll_pos', d.roll_pos)),
  ].join('');

  $('mf-bench-grid').innerHTML = [
    mfMetricCard('Beta vs Nifty',  d.beta,  na(d.beta)  ? 'neutral' : n(d.beta)  < 1 ? 'pos' : n(d.beta)  < 1.2 ? 'warn' : 'neg', mfInterpret('beta',  d.beta)),
    mfMetricCard('Alpha (3Y ann.)', d.alpha, na(d.alpha) ? 'neutral' : n(d.alpha) > 0 ? 'pos' : 'neg',                              mfInterpret('alpha', d.alpha)),
  ].join('');

  const cats = [
    { id: 'Ret',  score: sc.returns,     max: 35, cat: 'returns' },
    { id: 'Risk', score: sc.risk,        max: 30, cat: 'risk' },
    { id: 'Con',  score: sc.consistency, max: 20, cat: 'consistency' },
    { id: 'Mom',  score: sc.momentum,    max: 15, cat: 'momentum' },
  ];
  cats.forEach(c => {
    $(`mf-b${c.id}`).textContent = c.score;
    $(`mf-b${c.id}Bar`).style.width = `${(c.score / c.max) * 100}%`;
    $(`mf-b${c.id}Ex`).textContent = mfCategoryOneLiner(c.cat, c.score, data.detail, d);
  });

  if (data.narrative) {
    $('mf-nar-body').textContent = data.narrative;
    $('mf-narrative-box').style.display = 'block';
  } else {
    $('mf-narrative-box').style.display = 'none';
  }

  $('mf-result').style.display = 'block';
  $('mf-result').scrollIntoView({behavior: 'smooth', block: 'start'});
}
</script>
</body>
</html>'''


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    if '--no-open' not in sys.argv:
        webbrowser.open(f'http://localhost:{port}')
    print(f'ScreenerIN -> http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
