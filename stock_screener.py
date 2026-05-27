"""
Indian Stock Analyzer
Simple input → BUY / WATCHLIST / SKIP verdict
Data: yfinance (NSE/BSE) | AI: Claude Haiku
Run: python stock_screener.py
"""

import os
import sys
import webbrowser
import requests_cache
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

# Cache Yahoo Finance requests for 4 hours to avoid rate limiting
requests_cache.install_cache(
    'yfinance_cache',
    backend='sqlite',
    expire_after=14400,
    allowable_codes=[200],
)

load_dotenv('C:\\credentials\\.env')

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

app = Flask(__name__)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def safe(val, default=None):
    return val if val is not None and val == val else default  # also catches NaN

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


# ─── SCORING ENGINE ───────────────────────────────────────────────────────────

def score_stock(info):
    reasons, risks = [], []
    breakdown = {'valuation': 0, 'profitability': 0, 'safety': 0, 'momentum': 0}

    price = safe(info.get('currentPrice'), safe(info.get('regularMarketPrice'), 0))
    if not price:
        price = 0

    # ── VALUATION (max 25) ──
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

    # ── PROFITABILITY (max 25) ──
    p = 0
    roe     = safe(info.get('returnOnEquity'))
    opm     = safe(info.get('operatingMargins'))
    rev_g   = safe(info.get('revenueGrowth'))

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

    # ── SAFETY (max 25) ──
    s = 0
    de   = safe(info.get('debtToEquity'))   # yfinance returns as %, divide by 100
    cr   = safe(info.get('currentRatio'))
    beta = safe(info.get('beta'))

    if de is not None:
        de_r = de / 100  # actual D/E ratio
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

    # ── MOMENTUM (max 25) ──
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
        # 1 = Strong Buy … 5 = Strong Sell
        if rec <= 1.5:   m += 7; reasons.append('Analysts strongly recommend Buy')
        elif rec <= 2.5: m += 5
        elif rec <= 3.0: m += 2
        else:            risks.append(f'Analysts lean bearish ({rec:.1f}/5)')

    breakdown['momentum'] = min(m, 25)

    total = sum(breakdown.values())
    if total >= 68:
        verdict = 'BUY'
    elif total >= 42:
        verdict = 'WATCHLIST'
    else:
        verdict = 'SKIP'

    return {
        'score': total,
        'verdict': verdict,
        'breakdown': breakdown,
        'reasons': reasons[:5],
        'risks': risks[:4],
    }


# ─── AI NARRATIVE ─────────────────────────────────────────────────────────────

def get_ai_narrative(name, ticker, score_data):
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


# ─── ROUTES ───────────────────────────────────────────────────────────────────

def fetch_info(symbol):
    """Fetch yfinance info with retry on rate limit."""
    import yfinance as yf
    import time

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


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data  = request.json or {}
    query = (data.get('ticker') or '').strip().upper().replace(' ', '-')

    if not query:
        return jsonify({'error': 'Enter a stock ticker (e.g. RELIANCE, INFY, TCS)'}), 400

    clean = query.replace('.NS', '').replace('.BO', '')

    ticker_obj, info, used_symbol = None, {}, clean + '.NS'

    try:
        for suffix in ['.NS', '.BO']:
            sym = clean + suffix
            t, inf = fetch_info(sym)
            price = safe(inf.get('currentPrice'), safe(inf.get('regularMarketPrice')))
            if price:
                ticker_obj, info, used_symbol = t, inf, sym
                break
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Data fetch failed: {str(e)}'}), 500

    if ticker_obj is None or not info:
        return jsonify({
            'error': (
                f'Could not find "{query}" on NSE or BSE. '
                'Try the exact NSE symbol (e.g. RELIANCE, HDFCBANK, INFY).'
            )
        }), 404

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

    ai_text = get_ai_narrative(
        info.get('longName', used_symbol),
        used_symbol,
        score_data
    )

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


# ─── HTML TEMPLATE ────────────────────────────────────────────────────────────

HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StockScope IN — Indian Stock Analyzer</title>
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
  --shadow: 0 8px 32px rgba(0,0,0,0.5);
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

/* ── TOPBAR ── */
.topbar {
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  padding: 14px 28px;
  display: flex;
  align-items: center;
  gap: 12px;
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
}
.logo-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--brand);
  box-shadow: 0 0 8px var(--brand);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{ opacity:1;transform:scale(1) } 50%{ opacity:0.7;transform:scale(0.9) } }
.topbar-sub { font-size: 12px; color: var(--muted); }
.topbar-right { margin-left: auto; }
.badge {
  font-size: 11px; font-weight: 700;
  padding: 3px 10px;
  border-radius: var(--radius-full);
  background: color-mix(in srgb, var(--brand) 15%, transparent);
  color: var(--brand);
  border: 1px solid color-mix(in srgb, var(--brand) 30%, transparent);
}

/* ── HERO SEARCH ── */
.hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 64px 24px 40px;
  text-align: center;
}
.hero h1 {
  font-size: clamp(1.75rem, 4vw, 3rem);
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1.15;
  margin-bottom: 12px;
}
.hero h1 span { color: var(--brand); }
.hero p {
  font-size: 15px;
  color: var(--muted);
  max-width: 48ch;
  margin-bottom: 36px;
}
.search-wrap {
  width: 100%;
  max-width: 560px;
  display: flex;
  gap: 10px;
}
.search-input {
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
.search-input::placeholder { color: var(--muted); }
.search-input:focus {
  border-color: var(--brand);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--brand) 20%, transparent);
}
.btn-analyze {
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
.btn-analyze:hover {
  background: #b8007a;
  box-shadow: 0 0 24px color-mix(in srgb, var(--brand) 40%, transparent);
}
.btn-analyze:disabled { opacity: 0.5; cursor: not-allowed; }

/* Popular tickers */
.quick-tickers {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 16px;
}
.quick-btn {
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
.quick-btn:hover {
  border-color: var(--brand);
  color: var(--brand);
  background: color-mix(in srgb, var(--brand) 10%, transparent);
}

/* ── LOADER ── */
.loader-wrap {
  display: none;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 40px;
}
.loader-wrap.active { display: flex; }
.spinner {
  width: 44px; height: 44px;
  border: 3px solid var(--border);
  border-top-color: var(--brand);
  border-radius: 50%;
  animation: spin 0.75s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loader-wrap p { font-size: 14px; color: var(--muted); }

/* ── ERROR ── */
.error-box {
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
.error-box.show { display: block; }

/* ── RESULT SECTION ── */
.result-section {
  display: none;
  max-width: 900px;
  margin: 0 auto;
  padding: 0 24px 64px;
  animation: fadeUp 0.4s var(--ease) both;
}
.result-section.show { display: block; }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Company header */
.company-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 20px;
  flex-wrap: wrap;
  gap: 12px;
}
.company-info h2 { font-size: 22px; font-weight: 800; letter-spacing: -0.02em; }
.company-info .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
.price-block { text-align: right; }
.price-main { font-size: 26px; font-weight: 800; font-family: var(--font-mono); }
.price-change {
  font-size: 13px;
  font-family: var(--font-mono);
  margin-top: 2px;
}
.up { color: var(--positive-fg); }
.dn { color: var(--negative-fg); }

/* ── VERDICT CARD ── */
.verdict-card {
  border-radius: var(--radius-xl);
  padding: 24px;
  margin-bottom: 20px;
  border: 2px solid;
  position: relative;
  overflow: hidden;
}
.verdict-card::before {
  content: '';
  position: absolute;
  inset: 0;
  opacity: 0.04;
  background: radial-gradient(circle at 20% 50%, white, transparent 60%);
  pointer-events: none;
}
.verdict-BUY { background: var(--positive-bg); border-color: var(--positive); }
.verdict-WATCHLIST { background: var(--warning-bg); border-color: var(--warning); }
.verdict-SKIP { background: var(--negative-bg); border-color: var(--negative); }

.verdict-row { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.verdict-badge {
  font-size: 28px;
  font-weight: 900;
  letter-spacing: 0.08em;
  line-height: 1;
}
.verdict-BUY .verdict-badge { color: var(--positive-fg); }
.verdict-WATCHLIST .verdict-badge { color: var(--warning-fg); }
.verdict-SKIP .verdict-badge { color: var(--negative-fg); }

.verdict-score {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  min-width: 200px;
}
.score-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.score-num { font-size: 13px; color: var(--muted); font-family: var(--font-mono); }
.score-track {
  height: 8px;
  background: color-mix(in srgb, var(--muted) 20%, transparent);
  border-radius: var(--radius-full);
  overflow: hidden;
}
.score-fill {
  height: 100%;
  border-radius: var(--radius-full);
  transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
.verdict-BUY .score-fill { background: var(--positive); }
.verdict-WATCHLIST .score-fill { background: var(--warning); }
.verdict-SKIP .score-fill { background: var(--negative); }

/* ── BREAKDOWN BARS ── */
.breakdown-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}
.breakdown-item {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 14px;
}
.breakdown-icon { font-size: 18px; margin-bottom: 6px; }
.breakdown-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); }
.breakdown-score {
  font-size: 22px;
  font-weight: 900;
  font-family: var(--font-mono);
  margin: 4px 0;
}
.breakdown-max { font-size: 11px; color: var(--muted); }
.bex {
  font-size: 11px;
  color: var(--muted);
  line-height: 1.5;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--border);
}
.mini-track {
  height: 4px;
  background: var(--border);
  border-radius: var(--radius-full);
  overflow: hidden;
  margin-top: 8px;
}
.mini-fill {
  height: 100%;
  border-radius: var(--radius-full);
  background: var(--brand);
  transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ── AI NARRATIVE ── */
.ai-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 18px;
  margin-bottom: 20px;
  position: relative;
}
.ai-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--brand);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.ai-text { font-size: 14px; line-height: 1.8; color: var(--subtle); }
.ai-text p { margin: 4px 0; }

/* ── REASONS & RISKS ── */
.tags-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.tag {
  font-size: 12px;
  font-weight: 600;
  padding: 4px 12px;
  border-radius: var(--radius-full);
  display: flex;
  align-items: center;
  gap: 4px;
}
.tag-bull { background: color-mix(in srgb, var(--positive) 15%, transparent); color: var(--positive-fg); border: 1px solid color-mix(in srgb, var(--positive) 30%, transparent); }
.tag-bear { background: color-mix(in srgb, var(--negative) 15%, transparent); color: var(--negative-fg); border: 1px solid color-mix(in srgb, var(--negative) 30%, transparent); }

/* ── METRICS GRID ── */
.section-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 12px;
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 8px;
  margin-bottom: 20px;
}
.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px 14px;
}
.metric-card .mk { font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-card .mv { font-size: 16px; font-weight: 800; font-family: var(--font-mono); margin-top: 4px; }
.mv.good { color: var(--positive-fg); }
.mv.bad  { color: var(--negative-fg); }
.mv.warn { color: var(--warning-fg); }
.mv.neutral { color: var(--text); }
.metric-card .mex {
  font-size: 11px;
  color: var(--muted);
  line-height: 1.5;
  margin-top: 7px;
  padding-top: 7px;
  border-top: 1px solid var(--border);
}

/* ── DIVIDER ── */
.divider { height: 1px; background: var(--border); margin: 20px 0; }

/* ── RESPONSIVE ── */
@media (max-width: 700px) {
  .breakdown-grid { grid-template-columns: repeat(2, 1fr); }
  .search-wrap { flex-direction: column; }
  .btn-analyze { width: 100%; }
  .hero { padding: 40px 20px 32px; }
}
</style>
</head>
<body>

<!-- TOPBAR -->
<header class="topbar">
  <a class="logo" href="#">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>
    </svg>
    StockScope<span style="color:var(--brand)">IN</span>
    <div class="logo-dot"></div>
  </a>
  <span class="topbar-sub">NSE · BSE · Indian Market Analyzer</span>
  <div class="topbar-right">
    <span class="badge" id="verifyBadge">AI-powered</span>
  </div>
</header>

<!-- HERO -->
<section class="hero">
  <h1>Should you <span>buy</span> this stock?</h1>
  <p>Enter any NSE ticker and get an instant research-backed verdict — valuation, profitability, safety, and momentum all analyzed for you.</p>

  <div class="search-wrap">
    <input
      class="search-input"
      id="tickerInput"
      type="text"
      placeholder="Enter ticker: RELIANCE, INFY, TCS…"
      autocomplete="off"
      autocapitalize="characters"
      spellcheck="false"
    >
    <button class="btn-analyze" id="analyzeBtn" onclick="analyze()">Analyze →</button>
  </div>

  <div class="quick-tickers">
    <span style="font-size:12px;color:var(--muted);align-self:center">Try:</span>
    <button class="quick-btn" onclick="quickSearch('RELIANCE')">RELIANCE</button>
    <button class="quick-btn" onclick="quickSearch('INFY')">INFY</button>
    <button class="quick-btn" onclick="quickSearch('HDFCBANK')">HDFCBANK</button>
    <button class="quick-btn" onclick="quickSearch('TCS')">TCS</button>
    <button class="quick-btn" onclick="quickSearch('TITAN')">TITAN</button>
    <button class="quick-btn" onclick="quickSearch('BAJFINANCE')">BAJFINANCE</button>
    <button class="quick-btn" onclick="quickSearch('WIPRO')">WIPRO</button>
    <button class="quick-btn" onclick="quickSearch('NESTLEIND')">NESTLEIND</button>
  </div>
</section>

<!-- LOADER -->
<div class="loader-wrap" id="loaderWrap">
  <div class="spinner"></div>
  <p id="loaderMsg">Fetching data from NSE…</p>
</div>

<!-- ERROR -->
<div class="error-box" id="errorBox"></div>

<!-- RESULT -->
<section class="result-section" id="resultSection">

  <!-- Company header -->
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

  <!-- Verdict card -->
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

  <!-- Category breakdown -->
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
      <div class="breakdown-score" id="bMom">—</div>
      <div class="breakdown-max">out of 25</div>
      <div class="mini-track"><div class="mini-fill" id="bMomBar" style="width:0%"></div></div>
      <div class="bex" id="bMomEx"></div>
    </div>
  </div>

  <!-- AI narrative -->
  <div class="ai-card" id="aiCard" style="display:none">
    <div class="ai-label">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
      AI Analysis
    </div>
    <div class="ai-text" id="aiText"></div>
  </div>

  <!-- Signals -->
  <div class="tags-row" id="tagsRow"></div>

  <!-- Key metrics -->
  <div class="section-label">Key Metrics</div>
  <div class="metrics-grid" id="metricsGrid"></div>

  <!-- Analyst data -->
  <div class="section-label">Analyst Consensus</div>
  <div class="metrics-grid" id="analystGrid"></div>

</section>

<script>
const $ = id => document.getElementById(id);

function quickSearch(t) {
  $('tickerInput').value = t;
  analyze();
}

$('tickerInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') analyze();
});

function showLoader(msg) {
  $('loaderWrap').classList.add('active');
  $('loaderMsg').textContent = msg || 'Analyzing stock…';
  $('resultSection').classList.remove('show');
  $('errorBox').classList.remove('show');
}

function hideLoader() {
  $('loaderWrap').classList.remove('active');
}

function showError(msg) {
  $('errorBox').textContent = '⚠️ ' + msg;
  $('errorBox').classList.add('show');
  $('errorBox').style.maxWidth = '560px';
  $('errorBox').style.margin = '0 auto 24px';
}

function colorClass(val, goodThreshold, badThreshold) {
  if (val === 'N/A') return 'neutral';
  const n = parseFloat(val);
  if (isNaN(n)) return 'neutral';
  if (goodThreshold !== null && n >= goodThreshold) return 'good';
  if (badThreshold !== null && n <= badThreshold) return 'bad';
  return 'warn';
}

function metricCard(label, value, cls, explain) {
  cls = cls || 'neutral';
  const ex = explain ? `<div class="mex">${explain}</div>` : '';
  return `<div class="metric-card"><div class="mk">${label}</div><div class="mv ${cls}">${value}</div>${ex}</div>`;
}

function categoryOneLiner(cat, score, m, d) {
  const na = v => v === 'N/A' || isNaN(parseFloat(v));
  const n  = v => parseFloat(v);

  if (cat === 'valuation') {
    const parts = [];
    if (!na(m.pe))  parts.push(`P/E ${n(m.pe).toFixed(0)}x`);
    if (!na(m.pb))  parts.push(`P/B ${n(m.pb).toFixed(1)}x`);
    if (!na(m.target) && d.price) {
      const up = ((n(m.target) - d.price) / d.price * 100).toFixed(0);
      parts.push(`analyst target implies ${up > 0 ? '+' : ''}${up}% move`);
    }
    const verdict = score >= 20 ? 'attractively valued' : score >= 12 ? 'fairly valued' : score >= 6 ? 'slightly expensive' : 'expensive';
    return parts.length ? `${parts.join(', ')} — ${verdict}.` : `Score of ${score}/25 reflects current valuation multiples.`;
  }

  if (cat === 'profitability') {
    const parts = [];
    if (!na(m.roe))        parts.push(`ROE ${m.roe}`);
    if (!na(m.opm))        parts.push(`margins ${m.opm}`);
    if (!na(m.rev_growth)) parts.push(`revenue growing ${m.rev_growth}`);
    const verdict = score >= 20 ? 'highly profitable' : score >= 12 ? 'decent profitability' : score >= 6 ? 'below-average earnings' : 'weak profitability';
    return parts.length ? `${parts.join(', ')} — ${verdict}.` : `Score of ${score}/25 reflects earnings quality.`;
  }

  if (cat === 'safety') {
    const parts = [];
    if (!na(m.de))   parts.push(n(m.de) === 0 ? 'debt-free' : `D/E ${m.de}`);
    if (!na(m.cr))   parts.push(`current ratio ${m.cr}`);
    if (!na(m.beta)) parts.push(`beta ${m.beta}`);
    const verdict = score >= 20 ? 'very safe balance sheet' : score >= 12 ? 'reasonably safe' : score >= 6 ? 'some financial risk' : 'high financial risk';
    return parts.length ? `${parts.join(', ')} — ${verdict}.` : `Score of ${score}/25 reflects balance sheet health.`;
  }

  if (cat === 'momentum') {
    const parts = [];
    if (d.price && !na(m.high_52w)) {
      const pct = ((n(m.high_52w) - d.price) / n(m.high_52w) * 100).toFixed(0);
      parts.push(`${pct}% below 52W high`);
    }
    if (!na(m.eg_growth)) parts.push(`earnings ${m.eg_growth}`);
    if (m.rec && m.rec !== 'N/A') parts.push(`analysts say ${m.rec}`);
    const verdict = score >= 20 ? 'strong momentum' : score >= 12 ? 'moderate momentum' : score >= 6 ? 'weak momentum' : 'poor momentum';
    return parts.length ? `${parts.join(', ')} — ${verdict}.` : `Score of ${score}/25 reflects price and growth trend.`;
  }

  return '';
}

function interpret(key, rawVal) {
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
    default:
      return null;
  }
}

async function analyze() {
  const ticker = $('tickerInput').value.trim();
  if (!ticker) {
    showError('Please enter a stock ticker.');
    return;
  }
  $('analyzeBtn').disabled = true;
  showLoader('Fetching data from NSE / BSE…');

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ticker})
    });
    const d = await res.json();
    hideLoader();

    if (!res.ok || d.error) {
      showError(d.error || 'Something went wrong.');
      return;
    }

    renderResult(d);
  } catch (e) {
    hideLoader();
    showError('Network error — is the server running?');
  } finally {
    $('analyzeBtn').disabled = false;
  }
}

function renderResult(d) {
  const s = d.score;
  const m = d.metrics;

  // Company header
  $('rName').textContent = d.name;
  $('rMeta').textContent = `${d.ticker}  ·  ${m.sector}  ·  ${m.industry}`;
  $('rPrice').textContent = `₹${d.price.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
  const up = d.change_pct >= 0;
  $('rChange').innerHTML = `<span class="${up ? 'up' : 'dn'}">${up ? '▲' : '▼'} ${Math.abs(d.change_pct).toFixed(2)}%  (${up ? '+' : ''}${d.change.toFixed(2)})</span>`;

  // Verdict
  const vc = $('verdictCard');
  vc.className = 'verdict-card verdict-' + s.verdict;
  $('rVerdict').textContent = s.verdict;
  $('rScoreNum').textContent = `${s.score} / 100`;
  $('rScoreFill').style.width = s.score + '%';

  // Breakdown
  const bd = s.breakdown;
  $('bVal').textContent  = bd.valuation;
  $('bProf').textContent = bd.profitability;
  $('bSafe').textContent = bd.safety;
  $('bMom').textContent  = bd.momentum;
  $('bValBar').style.width  = (bd.valuation  / 25 * 100) + '%';
  $('bProfBar').style.width = (bd.profitability / 25 * 100) + '%';
  $('bSafeBar').style.width = (bd.safety / 25 * 100) + '%';
  $('bMomBar').style.width  = (bd.momentum / 25 * 100) + '%';
  $('bValEx').textContent  = categoryOneLiner('valuation',     bd.valuation,     m, d);
  $('bProfEx').textContent = categoryOneLiner('profitability', bd.profitability, m, d);
  $('bSafeEx').textContent = categoryOneLiner('safety',        bd.safety,        m, d);
  $('bMomEx').textContent  = categoryOneLiner('momentum',      bd.momentum,      m, d);

  // AI narrative
  if (d.ai_narrative) {
    $('aiCard').style.display = '';
    const lines = d.ai_narrative.split('\n').filter(l => l.trim());
    $('aiText').innerHTML = lines.map(l => `<p>${l}</p>`).join('');
  } else {
    $('aiCard').style.display = 'none';
  }

  // Tags
  const tags = [
    ...s.reasons.map(r => `<span class="tag tag-bull">✓ ${r}</span>`),
    ...s.risks.map(r => `<span class="tag tag-bear">⚠ ${r}</span>`),
  ];
  $('tagsRow').innerHTML = tags.join('');

  const pe_v  = parseFloat(m.pe);
  const fpe_v = parseFloat(m.fpe);
  const pb_v  = parseFloat(m.pb);
  const peg_v = parseFloat(m.peg);
  const ev_v  = parseFloat(m.evebitda);
  const roe_v = parseFloat(m.roe);
  const opm_v = parseFloat(m.opm);
  const npm_v = parseFloat(m.npm);
  const rg_v  = parseFloat(m.rev_growth);
  const eg_v  = parseFloat(m.eg_growth);
  const de_v  = parseFloat(m.de);
  const cr_v  = parseFloat(m.cr);
  const bt_v  = parseFloat(m.beta);
  const dy_v  = parseFloat(m.div_yield);

  // 52W position explanation
  const hw = parseFloat(m.high_52w), lw = parseFloat(m.low_52w), cp = d.price;
  let w52ex = 'No 52-week data available.';
  if (!isNaN(hw) && !isNaN(cp)) {
    const pctBelow = ((hw - cp) / hw * 100).toFixed(1);
    const pctAbove = ((cp - lw) / lw * 100).toFixed(1);
    w52ex = `Stock is ${pctBelow}% below its 52-week high of ₹${m.high_52w} and ${pctAbove}% above its 52-week low of ₹${m.low_52w}.`;
  }

  // Target upside explanation
  let tgtEx = 'No analyst target data.';
  if (m.target !== 'N/A' && cp) {
    const upside = ((parseFloat(m.target) - cp) / cp * 100).toFixed(1);
    const dir = upside >= 0 ? 'upside' : 'downside';
    tgtEx = `${m.analysts} analysts average a target of ₹${m.target} — implying ${Math.abs(upside)}% ${dir} from current price.`;
  }

  $('metricsGrid').innerHTML = [
    metricCard('Market Cap', m.mcap, 'neutral', `Total market value of all shares. Large cap (>₹20K Cr) = established company; Small cap (<₹5K Cr) = higher risk and reward.`),
    metricCard('P/E (TTM)', m.pe, !isNaN(pe_v) && pe_v < 25 ? 'good' : !isNaN(pe_v) && pe_v > 50 ? 'bad' : 'warn', interpret('pe', m.pe)),
    metricCard('Forward P/E', m.fpe, 'neutral', interpret('fpe', m.fpe)),
    metricCard('P/B Ratio', m.pb, !isNaN(pb_v) && pb_v < 3 ? 'good' : 'warn', interpret('pb', m.pb)),
    metricCard('PEG Ratio', m.peg, !isNaN(peg_v) && peg_v < 1.5 ? 'good' : 'warn', interpret('peg', m.peg)),
    metricCard('EV/EBITDA', m.evebitda, 'neutral', interpret('evebitda', m.evebitda)),
    metricCard('ROE', m.roe, !isNaN(roe_v) && roe_v > 15 ? 'good' : 'warn', interpret('roe', m.roe)),
    metricCard('Op. Margin', m.opm, !isNaN(opm_v) && opm_v > 15 ? 'good' : 'warn', interpret('opm', m.opm)),
    metricCard('Net Margin', m.npm, !isNaN(npm_v) && npm_v > 10 ? 'good' : 'warn', interpret('npm', m.npm)),
    metricCard('Rev. Growth', m.rev_growth, !isNaN(rg_v) && rg_v > 10 ? 'good' : rg_v < 0 ? 'bad' : 'warn', interpret('rev_growth', m.rev_growth)),
    metricCard('EPS Growth', m.eg_growth, !isNaN(eg_v) && eg_v > 10 ? 'good' : eg_v < 0 ? 'bad' : 'warn', interpret('eg_growth', m.eg_growth)),
    metricCard('Debt/Equity', m.de, !isNaN(de_v) && de_v < 0.5 ? 'good' : de_v > 1.5 ? 'bad' : 'warn', interpret('de', m.de)),
    metricCard('Current Ratio', m.cr, !isNaN(cr_v) && cr_v > 1.5 ? 'good' : cr_v < 1 ? 'bad' : 'warn', interpret('cr', m.cr)),
    metricCard('Beta', m.beta, 'neutral', interpret('beta', m.beta)),
    metricCard('Dividend Yield', m.div_yield, !isNaN(dy_v) && dy_v > 1.5 ? 'good' : 'neutral', interpret('div_yield', m.div_yield)),
    metricCard('52W High / Low', `₹${m.high_52w} / ₹${m.low_52w}`, 'neutral', w52ex),
    metricCard('Analyst Target', `₹${m.target}`, 'neutral', tgtEx),
  ].join('');

  // Analyst grid
  const recLow = (m.rec || '').toLowerCase();
  const recEx = recLow.includes('strong buy') ? 'Analysts strongly recommend buying this stock right now.'
    : recLow.includes('buy') ? 'Most analysts consider this a good buy at the current price.'
    : recLow.includes('hold') ? 'Analysts suggest holding — not a strong buy or sell signal.'
    : recLow.includes('sell') ? 'Analysts recommend selling or avoiding this stock.'
    : 'No clear consensus from analyst community.';
  $('analystGrid').innerHTML = [
    metricCard('Consensus', m.rec, recLow.includes('buy') ? 'good' : recLow.includes('sell') ? 'bad' : 'warn', recEx),
    metricCard('# Analysts', m.analysts, 'neutral', `${m.analysts} Wall Street / research analysts actively cover this stock. More coverage = more reliable consensus.`),
    metricCard('Price Target', `₹${m.target}`, 'neutral', tgtEx),
  ].join('');

  $('resultSection').classList.add('show');
  $('resultSection').scrollIntoView({behavior: 'smooth', block: 'start'});
}
</script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8091))
    if '--no-open' not in sys.argv:
        webbrowser.open(f'http://localhost:{port}')
    print(f'StockScope IN -> http://localhost:{port}')
    app.run(port=port, debug=False)
