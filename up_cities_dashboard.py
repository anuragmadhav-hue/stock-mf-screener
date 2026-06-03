# -*- coding: utf-8 -*-
"""
Wiom UP Cities Dashboard — Flask Server
-----------------------------------------
4 sections mirroring the Bagda dashboard (card 1220):
  1. Customer Overview & Recharge
  2. Install Funnel (Assigned > Installed / Cancelled / Pending)
  3. Service Tickets (Total / Open / Resolved / TAT)
  4. Data Usage (Avg GB / High / Medium / Low)

Date filters: All Time | 1d | 3d | 7d | 14d | 30d

Port: 8081
Usage:
    python up_cities_dashboard.py          # start + auto-open browser
    python up_cities_dashboard.py --no-open
"""

import os
import sys
import threading
import webbrowser
import requests
from datetime import datetime
from flask import Flask, jsonify, request as freq
from dotenv import load_dotenv

load_dotenv(r'C:\credentials\.env')

METABASE_URL    = os.getenv('METABASE_URL', '').rstrip('/')
METABASE_API_KEY = os.getenv('METABASE_API_KEY')
DATABASE_ID     = 113
PORT            = 8081

if not METABASE_URL:
    print("ERROR: METABASE_URL not found in C:\\credentials\\.env"); sys.exit(1)
if not METABASE_API_KEY:
    print("ERROR: METABASE_API_KEY not found in C:\\credentials\\.env"); sys.exit(1)

HEADERS = {'x-api-key': METABASE_API_KEY, 'Content-Type': 'application/json'}

TARGET_CITIES = (
    'Agra', 'Aligarh', 'Bareilly', 'Bijnor', 'Budaun', 'Hapur', 'Hathras',
    'Mathura', 'Meerut_City', 'Moradabad', 'Muzaffarnagar',
    'Shamli', 'Baraut', 'Saharanpur', 'Modinagar', 'Sakoti', 'Shikohabad', 'Firozabad'
)
DISPLAY    = {'Meerut_City': 'Meerut'}
VALID_DAYS = (1, 3, 7, 14, 30)   # accepted ?days= values

app = Flask(__name__)

# Per-period cache: key = days (None = all-time, 1, 3, 7, 14, 30)
_caches        = {}
_refreshing_set = set()
_refresh_lock  = threading.Lock()


# ──────────────────────────────────────────────
# CACHE HELPERS
# ──────────────────────────────────────────────

def _empty_cache():
    return {'recharge': None, 'funnel': None, 'tickets': None, 'usage': None,
            'generated_at': None, 'error': None}

def get_cache(days):
    return _caches.get(days, _empty_cache())

def is_refreshing(days):
    return days in _refreshing_set

def _parse_days(raw):
    """Convert ?days= query param to int or None."""
    if raw is None or raw == 'all':
        return None
    try:
        d = int(raw)
        return d if d in VALID_DAYS else None
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────
# DATA LAYER
# ──────────────────────────────────────────────

def run_sql(sql: str):
    resp = requests.post(
        f'{METABASE_URL}/api/dataset',
        headers=HEADERS,
        json={'database': DATABASE_ID, 'type': 'native', 'native': {'query': sql}},
        timeout=180
    )
    resp.raise_for_status()
    d = resp.json()
    if 'error' in d:
        raise RuntimeError(d['error'])
    return d['data']['rows']


def _cities_str():
    return ', '.join(f"'{c}'" for c in TARGET_CITIES)

def _pcm_cte(cities_str):
    return f"""partner_city_map AS (
  SELECT DISTINCT PARTNER_ID, PARTNER_CITY AS city
  FROM PROD_DB.PUBLIC.INSTALLED_DATA_RAW_CH
  WHERE PARTNER_CITY IN ({cities_str})
    AND PARTNER_ID IS NOT NULL
)"""


# ── Section 1: Recharge ───────────────────────

def fetch_recharge_data(days=None):
    cs = _cities_str()
    # With a date filter: total = recharged in period; active = still active from that cohort
    pf = '' if days is None else f"AND TRY_TO_TIMESTAMP(h.PLAN_START_TIME::STRING) >= DATEADD(day, -{days}, CURRENT_DATE())"

    rows = run_sql(f"""
WITH {_pcm_cte(cs)},
total_cx AS (
  SELECT pc.city, COUNT(DISTINCT h.MOBILE) AS total
  FROM PROD_DB.DYNAMODB_READ.HOME_ROUTER_PLAN_INFO h
  JOIN partner_city_map pc ON pc.PARTNER_ID = h.LCO_ACCOUNT_ID
  WHERE 1=1 {pf}
  GROUP BY pc.city
),
active_paid AS (
  SELECT h.MOBILE, h.LCO_ACCOUNT_ID AS pid
  FROM PROD_DB.DYNAMODB_READ.HOME_ROUTER_PLAN_INFO h
  WHERE h.CHARGES > 0 AND h.TIME_PLAN > 0
    AND TRY_TO_TIMESTAMP(h.PLAN_START_TIME::STRING) IS NOT NULL
    {pf}
    AND DATEADD(second, h.TIME_PLAN, TRY_TO_TIMESTAMP(h.PLAN_START_TIME::STRING)) > CURRENT_TIMESTAMP()
  QUALIFY ROW_NUMBER() OVER (PARTITION BY h.MOBILE ORDER BY
    DATEADD(second, h.TIME_PLAN, TRY_TO_TIMESTAMP(h.PLAN_START_TIME::STRING)) DESC) = 1
),
city_active AS (
  SELECT pc.city, COUNT(DISTINCT ap.MOBILE) AS active
  FROM active_paid ap JOIN partner_city_map pc ON pc.PARTNER_ID = ap.pid
  GROUP BY pc.city
),
expiring AS (
  SELECT h.MOBILE, h.LCO_ACCOUNT_ID AS pid
  FROM PROD_DB.DYNAMODB_READ.HOME_ROUTER_PLAN_INFO h
  WHERE h.CHARGES > 0 AND h.TIME_PLAN > 0
    AND TRY_TO_TIMESTAMP(h.PLAN_START_TIME::STRING) IS NOT NULL
    {pf}
    AND DATEADD(second, h.TIME_PLAN, TRY_TO_TIMESTAMP(h.PLAN_START_TIME::STRING)) > CURRENT_TIMESTAMP()
    AND DATEADD(second, h.TIME_PLAN, TRY_TO_TIMESTAMP(h.PLAN_START_TIME::STRING)) < DATEADD(day, 7, CURRENT_TIMESTAMP())
  QUALIFY ROW_NUMBER() OVER (PARTITION BY h.MOBILE ORDER BY
    DATEADD(second, h.TIME_PLAN, TRY_TO_TIMESTAMP(h.PLAN_START_TIME::STRING)) DESC) = 1
),
city_expiring AS (
  SELECT pc.city, COUNT(DISTINCT e.MOBILE) AS expiring
  FROM expiring e JOIN partner_city_map pc ON pc.PARTNER_ID = e.pid
  GROUP BY pc.city
)
SELECT tc.city, tc.total,
  COALESCE(ca.active, 0) AS active,
  COALESCE(ce.expiring, 0) AS expiring
FROM total_cx tc
LEFT JOIN city_active ca ON ca.city = tc.city
LEFT JOIN city_expiring ce ON ce.city = tc.city
ORDER BY active DESC
""")
    result = {}
    for r in rows:
        city_raw = str(r[0])
        total    = int(r[1]) if r[1] else 0
        active   = int(r[2]) if r[2] else 0
        expiring = int(r[3]) if r[3] else 0
        result[city_raw] = {
            'total': total, 'active': active,
            'inactive': total - active, 'expiring': expiring,
            'pct': round(active / total * 100, 1) if total > 0 else 0.0,
        }
    return result


# ── Section 2: Install Funnel ─────────────────

def fetch_funnel_data(days=None):
    cs = _cities_str()
    # Filter ASSIGNED events by date; installed/cancelled check remains all-time
    # (a lead assigned on day 1 may install on day 3 — we keep that link)
    af = '' if days is None else f"AND t.ADDED_TIME::DATE >= DATEADD(day, -{days}, CURRENT_DATE())"

    rows = run_sql(f"""
WITH {_pcm_cte(cs)},
assigned AS (
  SELECT DISTINCT t.mobile, t.account_id::STRING AS partner_id
  FROM PROD_DB.PUBLIC.TASK_LOGS t
  WHERE t.event_name = 'ASSIGNED'
  {af}
),
city_leads AS (
  SELECT a.mobile, pc.city
  FROM assigned a JOIN partner_city_map pc ON pc.PARTNER_ID = a.partner_id
),
installed AS (
  SELECT DISTINCT mobile FROM PROD_DB.PUBLIC.TASK_LOGS WHERE event_name = 'OTP_VERIFIED'
),
cancelled AS (
  SELECT DISTINCT mobile FROM PROD_DB.PUBLIC.booking_logs
  WHERE event_name IN ('cancelled','cancelled_fe')
)
SELECT cl.city,
  COUNT(DISTINCT cl.mobile)                                                        AS assigned,
  COUNT(DISTINCT CASE WHEN i.mobile IS NOT NULL THEN cl.mobile END)               AS installed,
  COUNT(DISTINCT CASE WHEN c.mobile IS NOT NULL AND i.mobile IS NULL THEN cl.mobile END) AS cancelled,
  COUNT(DISTINCT CASE WHEN i.mobile IS NULL AND c.mobile IS NULL THEN cl.mobile END)     AS pending
FROM city_leads cl
LEFT JOIN installed i ON i.mobile = cl.mobile
LEFT JOIN cancelled c ON c.mobile = cl.mobile
GROUP BY cl.city ORDER BY assigned DESC
""")
    result = {}
    for r in rows:
        city_raw  = str(r[0])
        assigned  = int(r[1]) if r[1] else 0
        installed = int(r[2]) if r[2] else 0
        cancelled = int(r[3]) if r[3] else 0
        pending   = int(r[4]) if r[4] else 0
        result[city_raw] = {
            'assigned': assigned, 'installed': installed,
            'cancelled': cancelled, 'pending': pending,
            'install_pct': round(installed / assigned * 100, 1) if assigned > 0 else 0.0,
        }
    return result


# ── Section 3: Service Tickets ────────────────

def fetch_tickets_data(days=None):
    cs = _cities_str()
    tf = '' if days is None else f"AND t.CREATED_TIME >= DATEADD(day, -{days}, CURRENT_DATE())"

    rows = run_sql(f"""
WITH {_pcm_cte(cs)},
cx AS (
  SELECT DISTINCT h.MOBILE, pc.city
  FROM PROD_DB.DYNAMODB_READ.HOME_ROUTER_PLAN_INFO h
  JOIN partner_city_map pc ON pc.PARTNER_ID = h.LCO_ACCOUNT_ID
)
SELECT cx.city,
  COUNT(DISTINCT t.TICKET_ID) AS total,
  COUNT(DISTINCT CASE WHEN LOWER(t.STATUS) = 'open' THEN t.TICKET_ID END) AS open_t,
  COUNT(DISTINCT CASE WHEN LOWER(t.STATUS) IN ('resolved','closed') THEN t.TICKET_ID END) AS resolved_t,
  COUNT(DISTINCT CASE WHEN t.RESOLUTION_TIME IS NOT NULL
    AND DATEDIFF(minute,t.CREATED_TIME,t.RESOLUTION_TIME) <= 240 THEN t.TICKET_ID END) AS within_tat,
  COUNT(DISTINCT CASE WHEN t.RESOLUTION_TIME IS NOT NULL
    AND DATEDIFF(minute,t.CREATED_TIME,t.RESOLUTION_TIME) > 240 THEN t.TICKET_ID END) AS out_of_tat
FROM PROD_DB.DYNAMODB_READ.TICKETS t
JOIN cx ON cx.MOBILE = t.MOBILE
WHERE t._FIVETRAN_DELETED = FALSE
  AND t.TITLE NOT ILIKE '%Primary%Existing Customer%'
  {tf}
GROUP BY cx.city ORDER BY total DESC
""")
    result = {}
    for r in rows:
        city_raw = str(r[0])
        total    = int(r[1]) if r[1] else 0
        open_t   = int(r[2]) if r[2] else 0
        resolved = int(r[3]) if r[3] else 0
        in_tat   = int(r[4]) if r[4] else 0
        out_tat  = int(r[5]) if r[5] else 0
        tat_pct  = round(in_tat / (in_tat + out_tat) * 100, 1) if (in_tat + out_tat) > 0 else 0.0
        result[city_raw] = {
            'total': total, 'open': open_t, 'resolved': resolved,
            'in_tat': in_tat, 'out_tat': out_tat, 'tat_pct': tat_pct,
        }
    return result


# ── Section 4: Data Usage ─────────────────────

def fetch_usage_data(days=None):
    cs = _cities_str()
    uf = '' if days is None else f"AND u.ADDED_DATE >= DATEADD(day, -{days}, CURRENT_DATE())"

    rows = run_sql(f"""
WITH {_pcm_cte(cs)},
nas AS (
  SELECT DISTINCT h.MOBILE, h.NAS_ID, pc.city
  FROM PROD_DB.DYNAMODB_READ.HOME_ROUTER_PLAN_INFO h
  JOIN partner_city_map pc ON pc.PARTNER_ID = h.LCO_ACCOUNT_ID
  WHERE h.NAS_ID IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY h.MOBILE ORDER BY h.PLAN_START_TIME DESC) = 1
),
usage_per_cx AS (
  SELECT n.city, n.MOBILE,
    ROUND(SUM(u.DOWNLOAD + u.UPLOAD) / 1073741824.0, 2) AS total_gb
  FROM nas n
  JOIN PROD_DB.PUBLIC.CUSTOMER_DAILY_DATA_USAGE u ON u.NASID = n.NAS_ID
  WHERE 1=1 {uf}
  GROUP BY n.city, n.MOBILE
)
SELECT city,
  COUNT(DISTINCT MOBILE)                                      AS users,
  ROUND(AVG(total_gb), 1)                                    AS avg_gb,
  COUNT(CASE WHEN total_gb >= 30 THEN 1 END)                 AS high,
  COUNT(CASE WHEN total_gb >= 10 AND total_gb < 30 THEN 1 END) AS medium,
  COUNT(CASE WHEN total_gb < 10 THEN 1 END)                  AS low_u
FROM usage_per_cx
GROUP BY city ORDER BY avg_gb DESC
""")
    result = {}
    for r in rows:
        city_raw = str(r[0])
        users  = int(r[1])   if r[1] else 0
        avg_gb = float(r[2]) if r[2] else 0.0
        high   = int(r[3])   if r[3] else 0
        medium = int(r[4])   if r[4] else 0
        low    = int(r[5])   if r[5] else 0
        high_pct = round(high / users * 100, 1) if users > 0 else 0.0
        result[city_raw] = {
            'users': users, 'avg_gb': avg_gb,
            'high': high, 'medium': medium, 'low': low,
            'high_pct': high_pct,
        }
    return result


# ── Refresh orchestrator ──────────────────────

def refresh_data_for(days=None):
    with _refresh_lock:
        if days in _refreshing_set:
            return
        _refreshing_set.add(days)
    try:
        _caches[days] = {
            'recharge': fetch_recharge_data(days),
            'funnel':   fetch_funnel_data(days),
            'tickets':  fetch_tickets_data(days),
            'usage':    fetch_usage_data(days),
            'generated_at': datetime.now().strftime('%d %b %Y %H:%M IST'),
            'error': None,
        }
    except Exception as e:
        if days not in _caches:
            _caches[days] = _empty_cache()
        _caches[days]['error'] = str(e)
        _caches[days]['generated_at'] = (
            _caches[days].get('generated_at') or datetime.now().strftime('%d %b %Y %H:%M IST')
        )
    finally:
        _refreshing_set.discard(days)


# ──────────────────────────────────────────────
# HTML BUILDERS
# ──────────────────────────────────────────────

def pct_color(pct):
    if pct >= 70: return '#008043'
    if pct >= 40: return '#FF8000'
    return '#E01E00'


def _th(label, align='right'):
    cls = 'th-left' if align == 'left' else 'th-right'
    return f'<th class="{cls}">{label}</th>'

def _td(val, color=None, bold=False):
    style = ''
    if color: style += f'color:{color};'
    if bold:  style += 'font-weight:700;'
    s = f' style="{style}"' if style else ''
    return f'<td class="td-num"{s}>{val}</td>'

def _mini_bar(pct, color):
    w = min(pct, 100)
    return (
        f'<div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">'
        f'<div style="width:56px;height:5px;background:rgba(255,255,255,.1);border-radius:3px;overflow:hidden;flex-shrink:0">'
        f'<div style="width:{w}%;height:100%;background:{color};border-radius:3px"></div></div>'
        f'<span style="color:{color};font-weight:700">{pct}%</span></div>'
    )

def _period_label(days):
    """Human-readable period string."""
    if days is None: return 'All Time'
    return f'Last {days} Day{"s" if days > 1 else ""}'

def _section_nav(sections, days=None):
    """Sticky nav: section anchors (left) + period filter buttons (right)."""
    links = ''
    for sid, label, icon in sections:
        links += f'<a href="#{sid}" class="nav-tab">{icon} {label}</a>'

    filter_html = '<div class="filter-sep"></div><span class="filter-label">Period:</span>'
    for d, lbl in [(None, 'All'), (1, '1d'), (3, '3d'), (7, '7d'), (14, '14d'), (30, '30d')]:
        href = '/' if d is None else f'/?days={d}'
        active_cls = ' factive' if d == days else ''
        filter_html += f'<a href="{href}" class="filter-btn{active_cls}">{lbl}</a>'

    return f'<div class="section-nav">{links}{filter_html}</div>'


# ── Section 1: Recharge Overview ─────────────

def build_overview_section(recharge: dict, days=None) -> str:
    pl = _period_label(days)
    cities_data = []
    for c in TARGET_CITIES:
        d = recharge.get(c, {})
        cities_data.append({
            'city_raw': c, 'city': DISPLAY.get(c, c),
            'total': d.get('total', 0), 'active': d.get('active', 0),
            'inactive': d.get('inactive', 0), 'expiring': d.get('expiring', 0),
            'pct': d.get('pct', 0.0),
        })
    cities_data.sort(key=lambda x: x['active'], reverse=True)

    total_cx    = sum(d['total']    for d in cities_data)
    total_act   = sum(d['active']   for d in cities_data)
    total_inact = sum(d['inactive'] for d in cities_data)
    total_exp   = sum(d['expiring'] for d in cities_data)
    overall_pct = round(total_act / total_cx * 100, 1) if total_cx > 0 else 0.0
    oc = pct_color(overall_pct)

    if days:
        kpi1_lbl  = f'Recharged ({pl})'
        kpi1_sub  = 'Plan started in period'
        kpi2_lbl  = 'Still Active'
        kpi2_sub  = 'Plan not yet expired'
        kpi3_lbl  = 'Already Expired'
        kpi3_sub  = 'Recharged in period, plan ended'
        bar_lbl   = f'Active Rate ({pl})'
    else:
        kpi1_lbl  = 'Total Customers'
        kpi1_sub  = 'All UP cities combined'
        kpi2_lbl  = 'Active Paid'
        kpi2_sub  = 'Valid recharge now'
        kpi3_lbl  = 'Inactive / Expired'
        kpi3_sub  = 'No active plan'
        bar_lbl   = 'Overall Recharge Rate'

    kpis = (
        f'<div class="kpi-card" style="--acc:#D9008D"><div class="kpi-label">{kpi1_lbl}</div>'
        f'<div class="kpi-value">{total_cx:,}</div><div class="kpi-sub">{kpi1_sub}</div></div>'
        f'<div class="kpi-card" style="--acc:#008043"><div class="kpi-label">{kpi2_lbl}</div>'
        f'<div class="kpi-value" style="color:#008043">{total_act:,}</div><div class="kpi-sub">{kpi2_sub}</div></div>'
        f'<div class="kpi-card" style="--acc:#E01E00"><div class="kpi-label">{kpi3_lbl}</div>'
        f'<div class="kpi-value" style="color:#E01E00">{total_inact:,}</div><div class="kpi-sub">{kpi3_sub}</div></div>'
        f'<div class="kpi-card" style="--acc:#FF8000"><div class="kpi-label">Expiring in 7 Days</div>'
        f'<div class="kpi-value" style="color:#FF8000">{total_exp:,}</div><div class="kpi-sub">Need renewal outreach</div></div>'
    )

    overall_bar = (
        f'<div class="overall-bar-wrap">'
        f'<span class="obar-label">{bar_lbl}</span>'
        f'<div class="obar-track"><div class="obar-fill" style="width:{min(overall_pct,100)}%;background:{oc}"></div></div>'
        f'<span class="obar-pct" style="color:{oc}">{overall_pct}%</span>'
        f'</div>'
    )

    cards = ''
    for d in cities_data:
        if d['total'] == 0:
            continue
        c = pct_color(d['pct'])
        bw = min(d['pct'], 100)
        exp_badge = f'<span class="exp-badge">&#9888; {d["expiring"]:,} expiring</span>' if d['expiring'] > 0 else ''
        cards += (
            f'<div class="city-card">'
            f'<div class="city-name">&#128205; {d["city"]}</div>'
            f'<div class="city-metrics">'
            f'<div class="metric"><div class="metric-label">{"Recharged" if days else "Total"}</div><div class="metric-value">{d["total"]:,}</div></div>'
            f'<div class="metric"><div class="metric-label">Active</div><div class="metric-value" style="color:{c}">{d["active"]:,}</div></div>'
            f'<div class="metric"><div class="metric-label">{"Expired" if days else "Inactive"}</div><div class="metric-value" style="color:#A7A1B2">{d["inactive"]:,}</div></div>'
            f'</div>'
            f'<div class="city-bottom">'
            f'<div class="recharge-row">'
            f'<div class="prog-track"><div class="prog-fill" style="width:{bw}%;background:{c}"></div></div>'
            f'<span style="color:{c};font-size:12px;font-weight:700;white-space:nowrap">{d["pct"]}% active</span>'
            f'</div>{exp_badge}</div></div>'
        )

    col1_hdr = 'Recharged' if days else 'Total'
    t_rows = ''
    for i, d in enumerate(cities_data, 1):
        c = pct_color(d['pct'])
        exp_cell = (f'<span style="color:#FF8000;font-weight:600">{d["expiring"]:,}</span>'
                    if d['expiring'] > 0 else '<span style="color:#A7A1B2">&#8212;</span>')
        t_rows += (
            f'<tr><td class="td-rank">{i}</td><td class="td-city">{d["city"]}</td>'
            + _td(f'{d["total"]:,}')
            + _td(f'{d["active"]:,}', color=c, bold=True)
            + _td(f'{d["inactive"]:,}', color='#A7A1B2')
            + f'<td class="td-num">{exp_cell}</td>'
            + f'<td class="td-num">{_mini_bar(d["pct"], c)}</td></tr>'
        )

    return (
        f'<div id="recharge" class="section-block">'
        f'<div class="section-header"><span>&#128179;</span> Customer Overview &amp; Recharge'
        f'<span class="period-badge">{pl}</span></div>'
        f'<div class="kpi-grid">{kpis}</div>'
        f'{overall_bar}'
        f'<div class="city-grid">{cards}</div>'
        f'<div class="table-wrap"><table>'
        f'<thead><tr>'
        + _th('#','left') + _th('City','left')
        + _th(col1_hdr) + _th('Active Paid') + _th('Inactive / Expired') + _th('Expiring 7d') + _th('Active %')
        + f'</tr></thead><tbody>{t_rows}</tbody></table></div></div>'
    )


# ── Section 2: Install Funnel ─────────────────

def build_funnel_section(funnel: dict, days=None) -> str:
    pl = _period_label(days)
    cities_data = []
    for c in TARGET_CITIES:
        d = funnel.get(c, {})
        cities_data.append({
            'city': DISPLAY.get(c, c), 'city_raw': c,
            'assigned': d.get('assigned', 0), 'installed': d.get('installed', 0),
            'cancelled': d.get('cancelled', 0), 'pending': d.get('pending', 0),
            'install_pct': d.get('install_pct', 0.0),
        })
    cities_data.sort(key=lambda x: x['assigned'], reverse=True)

    tot_assigned  = sum(d['assigned']  for d in cities_data)
    tot_installed = sum(d['installed'] for d in cities_data)
    tot_cancelled = sum(d['cancelled'] for d in cities_data)
    tot_pending   = sum(d['pending']   for d in cities_data)
    tot_pct = round(tot_installed / tot_assigned * 100, 1) if tot_assigned > 0 else 0.0

    assign_lbl = f'Assigned ({pl})' if days else 'Total Assigned'
    assign_sub = 'Leads in period' if days else 'Leads in pipeline'

    kpis = (
        f'<div class="kpi-card" style="--acc:#D9008D"><div class="kpi-label">{assign_lbl}</div>'
        f'<div class="kpi-value">{tot_assigned:,}</div><div class="kpi-sub">{assign_sub}</div></div>'
        f'<div class="kpi-card" style="--acc:#008043"><div class="kpi-label">Installed</div>'
        f'<div class="kpi-value" style="color:#008043">{tot_installed:,}</div><div class="kpi-sub">OTP Verified &#10003;</div></div>'
        f'<div class="kpi-card" style="--acc:#E01E00"><div class="kpi-label">Cancelled</div>'
        f'<div class="kpi-value" style="color:#E01E00">{tot_cancelled:,}</div><div class="kpi-sub">Dropped off</div></div>'
        f'<div class="kpi-card" style="--acc:#FF8000"><div class="kpi-label">Pending Install</div>'
        f'<div class="kpi-value" style="color:#FF8000">{tot_pending:,}</div><div class="kpi-sub">Assigned, not yet done</div></div>'
    )

    t_rows = ''
    for i, d in enumerate(cities_data, 1):
        c = pct_color(d['install_pct'])
        t_rows += (
            f'<tr><td class="td-rank">{i}</td><td class="td-city">{d["city"]}</td>'
            + _td(f'{d["assigned"]:,}')
            + _td(f'{d["installed"]:,}', color='#008043', bold=True)
            + _td(f'{d["cancelled"]:,}', color='#E01E00')
            + _td(f'{d["pending"]:,}',   color='#FF8000')
            + f'<td class="td-num">{_mini_bar(d["install_pct"], c)}</td></tr>'
        )

    return (
        f'<div id="funnel" class="section-block">'
        f'<div class="section-header"><span>&#128736;</span> Install Funnel'
        f'<span class="period-badge">{pl}</span></div>'
        f'<div class="kpi-grid">{kpis}</div>'
        f'<div class="table-wrap"><table>'
        f'<thead><tr>'
        + _th('#','left') + _th('City','left')
        + _th('Assigned') + _th('Installed') + _th('Cancelled') + _th('Pending') + _th('Install %')
        + f'</tr></thead><tbody>{t_rows}</tbody></table></div></div>'
    )


# ── Section 3: Service Tickets ────────────────

def build_tickets_section(tickets: dict, days=None) -> str:
    pl = _period_label(days)
    cities_data = []
    for c in TARGET_CITIES:
        d = tickets.get(c, {})
        cities_data.append({
            'city': DISPLAY.get(c, c), 'city_raw': c,
            'total': d.get('total', 0), 'open': d.get('open', 0),
            'resolved': d.get('resolved', 0), 'in_tat': d.get('in_tat', 0),
            'out_tat': d.get('out_tat', 0), 'tat_pct': d.get('tat_pct', 0.0),
        })
    cities_data.sort(key=lambda x: x['total'], reverse=True)

    tot_total    = sum(d['total']    for d in cities_data)
    tot_open     = sum(d['open']     for d in cities_data)
    tot_resolved = sum(d['resolved'] for d in cities_data)
    tot_in_tat   = sum(d['in_tat']   for d in cities_data)
    tot_out_tat  = sum(d['out_tat']  for d in cities_data)
    overall_tat  = round(tot_in_tat / (tot_in_tat + tot_out_tat) * 100, 1) if (tot_in_tat + tot_out_tat) > 0 else 0.0
    oc = pct_color(overall_tat)

    total_sub = f'Created in {pl}' if days else 'All time'

    kpis = (
        f'<div class="kpi-card" style="--acc:#D9008D"><div class="kpi-label">Total Tickets</div>'
        f'<div class="kpi-value">{tot_total:,}</div><div class="kpi-sub">{total_sub}</div></div>'
        f'<div class="kpi-card" style="--acc:#FF8000"><div class="kpi-label">Open Tickets</div>'
        f'<div class="kpi-value" style="color:#FF8000">{tot_open:,}</div><div class="kpi-sub">Currently unresolved</div></div>'
        f'<div class="kpi-card" style="--acc:#008043"><div class="kpi-label">Resolved</div>'
        f'<div class="kpi-value" style="color:#008043">{tot_resolved:,}</div><div class="kpi-sub">Closed tickets</div></div>'
        f'<div class="kpi-card" style="--acc:{oc}"><div class="kpi-label">Within TAT (240 min)</div>'
        f'<div class="kpi-value" style="color:{oc}">{overall_tat}%</div>'
        f'<div class="kpi-sub">{tot_in_tat:,} in TAT &nbsp;/&nbsp; {tot_out_tat:,} out</div></div>'
    )

    t_rows = ''
    for i, d in enumerate(cities_data, 1):
        c = pct_color(d['tat_pct'])
        t_rows += (
            f'<tr><td class="td-rank">{i}</td><td class="td-city">{d["city"]}</td>'
            + _td(f'{d["total"]:,}')
            + _td(f'{d["open"]:,}',     color='#FF8000' if d['open'] > 0 else '#A7A1B2')
            + _td(f'{d["resolved"]:,}', color='#008043')
            + _td(f'{d["in_tat"]:,}',   color='#008043')
            + _td(f'{d["out_tat"]:,}',  color='#E01E00' if d['out_tat'] > 0 else '#A7A1B2')
            + f'<td class="td-num">{_mini_bar(d["tat_pct"], c)}</td></tr>'
        )

    return (
        f'<div id="tickets" class="section-block">'
        f'<div class="section-header"><span>&#127915;</span> Service Tickets'
        f'<span class="period-badge">{pl}</span></div>'
        f'<div class="kpi-grid">{kpis}</div>'
        f'<div class="table-wrap"><table>'
        f'<thead><tr>'
        + _th('#','left') + _th('City','left')
        + _th('Total') + _th('Open') + _th('Resolved') + _th('Within TAT') + _th('Out of TAT') + _th('TAT %')
        + f'</tr></thead><tbody>{t_rows}</tbody></table></div></div>'
    )


# ── Section 4: Data Usage ─────────────────────

def build_usage_section(usage: dict, days=None) -> str:
    pl = _period_label(days)
    cities_data = []
    for c in TARGET_CITIES:
        d = usage.get(c, {})
        cities_data.append({
            'city': DISPLAY.get(c, c), 'city_raw': c,
            'users': d.get('users', 0), 'avg_gb': d.get('avg_gb', 0.0),
            'high': d.get('high', 0), 'medium': d.get('medium', 0),
            'low': d.get('low', 0), 'high_pct': d.get('high_pct', 0.0),
        })
    cities_data.sort(key=lambda x: x['avg_gb'], reverse=True)

    tot_users  = sum(d['users']  for d in cities_data)
    tot_high   = sum(d['high']   for d in cities_data)
    tot_medium = sum(d['medium'] for d in cities_data)
    tot_low    = sum(d['low']    for d in cities_data)
    avg_all    = round(sum(d['avg_gb'] for d in cities_data if d['users'] > 0) /
                       max(1, sum(1 for d in cities_data if d['users'] > 0)), 1)

    avg_sub = f'In {pl}' if days else 'Lifetime total'

    kpis = (
        f'<div class="kpi-card" style="--acc:#D9008D"><div class="kpi-label">Users with Usage Data</div>'
        f'<div class="kpi-value">{tot_users:,}</div><div class="kpi-sub">Across all cities</div></div>'
        f'<div class="kpi-card" style="--acc:#6D17CE"><div class="kpi-label">Avg Data / User</div>'
        f'<div class="kpi-value" style="color:#9d5bff">{avg_all:,} GB</div><div class="kpi-sub">{avg_sub}</div></div>'
        f'<div class="kpi-card" style="--acc:#008043"><div class="kpi-label">High Usage (30+ GB)</div>'
        f'<div class="kpi-value" style="color:#008043">{tot_high:,}</div><div class="kpi-sub">Power users</div></div>'
        f'<div class="kpi-card" style="--acc:#FF8000"><div class="kpi-label">Low Usage (&lt;10 GB)</div>'
        f'<div class="kpi-value" style="color:#FF8000">{tot_low:,}</div><div class="kpi-sub">At-risk users</div></div>'
    )

    t_rows = ''
    for i, d in enumerate(cities_data, 1):
        high_pct_c = pct_color(d['high_pct'])
        t_rows += (
            f'<tr><td class="td-rank">{i}</td><td class="td-city">{d["city"]}</td>'
            + _td(f'{d["users"]:,}')
            + _td(f'{d["avg_gb"]:,} GB', color='#9d5bff', bold=True)
            + _td(f'{d["high"]:,}',   color='#008043')
            + _td(f'{d["medium"]:,}', color='#FF8000')
            + _td(f'{d["low"]:,}',    color='#E01E00' if d['low'] > 0 else '#A7A1B2')
            + f'<td class="td-num">{_mini_bar(d["high_pct"], high_pct_c)}</td></tr>'
        )

    return (
        f'<div id="usage" class="section-block">'
        f'<div class="section-header"><span>&#128246;</span> Data Usage'
        f'<span class="period-badge">{pl}</span></div>'
        f'<div class="kpi-grid">{kpis}</div>'
        f'<div class="table-wrap"><table>'
        f'<thead><tr>'
        + _th('#','left') + _th('City','left')
        + _th('Users w/ Data') + _th('Avg GB/User') + _th('High 30+ GB') + _th('Medium 10-30') + _th('Low &lt;10 GB') + _th('High %')
        + f'</tr></thead><tbody>{t_rows}</tbody></table></div></div>'
    )


# ── Main HTML ─────────────────────────────────

def build_html(days=None) -> str:
    cache        = get_cache(days)
    recharge     = cache.get('recharge') or {}
    funnel       = cache.get('funnel')   or {}
    tickets      = cache.get('tickets')  or {}
    usage        = cache.get('usage')    or {}
    generated_at = cache.get('generated_at') or '&#8212;'
    error        = cache.get('error')
    refreshing   = is_refreshing(days)

    pl = _period_label(days)
    days_qs = '' if days is None else f'?days={days}'

    status_html = ''
    if refreshing:
        status_html = (
            f'<div class="banner banner-info">&#128260; Loading <strong>{pl}</strong> data from Snowflake'
            f' &#8212; takes 2&#8211;3 minutes. Page auto-reloads when ready.</div>'
        )
    if error:
        status_html += f'<div class="banner banner-error">&#9888; Error: {error}</div>'

    data_loaded = bool(recharge)

    if not data_loaded:
        main_content = (
            '<div class="loading-state">'
            '<div style="font-size:52px;margin-bottom:16px">&#9203;</div>'
            f'<div style="font-size:18px;font-weight:700;color:#FAF9FC;margin-bottom:8px">Loading {pl} data from Snowflake&#8230;</div>'
            '<div style="font-size:13px;color:#A7A1B2">All 4 sections load together. Takes 2&#8211;3 minutes.</div>'
            '</div>'
            f'<script>setTimeout(()=>location.reload(),8000);</script>'
        )
        nav_html = ''
    else:
        nav_html = _section_nav([
            ('recharge', 'Recharge',       '&#128179;'),
            ('funnel',   'Install Funnel', '&#128736;'),
            ('tickets',  'Tickets',        '&#127915;'),
            ('usage',    'Usage',          '&#128246;'),
        ], days=days)
        main_content = (
            build_overview_section(recharge, days) +
            build_funnel_section(funnel,     days) +
            build_tickets_section(tickets,   days) +
            build_usage_section(usage,       days)
        )

    active_cities = sum(1 for c in TARGET_CITIES if (recharge.get(c) or {}).get('total', 0) > 0)
    refresh_label = f'&#128260; Refresh {pl}' if days else '&#128260; Refresh Data'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UP Cities Dashboard &#8212; Wiom</title>
<style>
:root {{
  --brand:#D9008D; --brand2:#443152;
  --bg:#161021; --surface:#352D42; --surface2:#1E1730;
  --text:#FAF9FC; --muted:#A7A1B2;
  --positive:#008043; --negative:#E01E00; --warning:#FF8000; --info:#6D17CE;
  --border:rgba(250,249,252,0.1);
  --r:12px; --r-sm:8px;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;min-height:100vh;font-size:14px;line-height:1.5}}

/* ── Header ── */
.header{{background:linear-gradient(135deg,var(--brand2) 0%,#2a1f35 100%);border-bottom:3px solid var(--brand);padding:18px 32px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:300;box-shadow:0 4px 24px rgba(0,0,0,.45)}}
.hdr-title{{font-size:19px;font-weight:800;letter-spacing:-.3px}}
.hdr-title em{{font-style:normal;color:var(--brand)}}
.hdr-meta{{font-size:12px;color:var(--muted);margin-top:3px}}
.hdr-right{{display:flex;align-items:center;gap:12px}}
.live-dot{{width:9px;height:9px;background:var(--positive);border-radius:50%;animation:livepulse 2.2s ease-in-out infinite}}
@keyframes livepulse{{0%,100%{{box-shadow:0 0 0 0 rgba(0,128,67,.5)}}50%{{box-shadow:0 0 0 7px rgba(0,128,67,0)}}}}
.btn-refresh{{background:var(--brand);color:#fff;border:none;padding:9px 18px;border-radius:var(--r-sm);cursor:pointer;font-size:13px;font-weight:600;transition:opacity .2s,transform .1s}}
.btn-refresh:hover{{opacity:.85;transform:translateY(-1px)}}
.btn-refresh:disabled{{opacity:.45;cursor:not-allowed;transform:none}}

/* ── Section nav + filter strip ── */
.section-nav{{display:flex;align-items:center;gap:6px;padding:10px 32px;background:var(--surface2);border-bottom:1px solid var(--border);position:sticky;top:63px;z-index:200;overflow-x:auto;flex-wrap:nowrap}}
.nav-tab{{color:var(--muted);text-decoration:none;font-size:13px;font-weight:600;padding:5px 13px;border-radius:20px;white-space:nowrap;transition:background .15s,color .15s}}
.nav-tab:hover{{background:rgba(217,0,141,.15);color:var(--text)}}
.filter-sep{{flex:1;min-width:16px}}
.filter-label{{font-size:11px;color:var(--muted);white-space:nowrap;font-weight:600;letter-spacing:.4px;text-transform:uppercase}}
.filter-btn{{color:var(--muted);text-decoration:none;font-size:12px;font-weight:700;padding:4px 12px;border-radius:20px;white-space:nowrap;border:1px solid transparent;transition:all .15s}}
.filter-btn:hover{{color:var(--text);border-color:rgba(217,0,141,.4)}}
.filter-btn.factive{{background:var(--brand);color:#fff;border-color:var(--brand)}}

/* ── Period badge in section headers ── */
.period-badge{{margin-left:10px;font-size:11px;font-weight:700;color:var(--brand);background:rgba(217,0,141,.12);padding:2px 9px;border-radius:12px;letter-spacing:.3px}}

/* ── Main ── */
.main{{padding:26px 32px;max-width:1440px;margin:0 auto}}
.section-block{{margin-bottom:48px}}
.section-header{{font-size:16px;font-weight:800;color:var(--text);margin-bottom:18px;padding-bottom:10px;border-bottom:2px solid var(--border);display:flex;align-items:center;gap:8px;scroll-margin-top:120px}}

/* ── Banners ── */
.banner{{padding:10px 16px;border-radius:var(--r-sm);font-size:13px;margin-bottom:16px;display:flex;align-items:center;gap:8px}}
.banner-info{{background:rgba(109,23,206,.15);color:#a06fff;border:1px solid rgba(109,23,206,.3)}}
.banner-error{{background:rgba(224,30,0,.15);color:#ff7070;border:1px solid rgba(224,30,0,.3)}}

/* ── KPI Grid ── */
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px}}
.kpi-card{{background:var(--surface);border-radius:var(--r);padding:20px;border:1px solid var(--border);position:relative;overflow:hidden}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--acc,var(--brand))}}
.kpi-label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:7px}}
.kpi-value{{font-size:32px;font-weight:800;line-height:1}}
.kpi-sub{{font-size:12px;color:var(--muted);margin-top:6px}}

/* ── Overall bar ── */
.overall-bar-wrap{{background:var(--surface);border-radius:var(--r);border:1px solid var(--border);padding:14px 20px;display:flex;align-items:center;gap:16px;margin-bottom:24px}}
.obar-label{{font-size:11px;color:var(--muted);white-space:nowrap;min-width:160px;text-transform:uppercase;letter-spacing:.5px}}
.obar-track{{flex:1;background:rgba(255,255,255,.1);border-radius:6px;height:12px;overflow:hidden}}
.obar-fill{{height:100%;border-radius:6px;transition:width .8s ease}}
.obar-pct{{font-size:18px;font-weight:800;min-width:60px;text-align:right}}

/* ── City cards ── */
.city-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(265px,1fr));gap:14px;margin-bottom:28px}}
.city-card{{background:var(--surface);border-radius:var(--r);padding:18px;border:1px solid var(--border);transition:border-color .2s,transform .15s,box-shadow .15s}}
.city-card:hover{{border-color:rgba(217,0,141,.5);transform:translateY(-2px);box-shadow:0 8px 28px rgba(0,0,0,.35)}}
.city-name{{font-size:15px;font-weight:700;margin-bottom:13px}}
.city-metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;text-align:center;margin-bottom:12px}}
.metric-label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.3px;margin-bottom:2px}}
.metric-value{{font-size:20px;font-weight:800}}
.city-bottom{{display:flex;align-items:center;gap:10px}}
.recharge-row{{display:flex;align-items:center;gap:8px;flex:1}}
.prog-track{{flex:1;background:rgba(255,255,255,.1);border-radius:4px;height:5px;overflow:hidden}}
.prog-fill{{height:100%;border-radius:4px;transition:width .6s ease}}
.exp-badge{{font-size:11px;font-weight:700;color:#FF8000;white-space:nowrap}}

/* ── Tables ── */
.table-wrap{{background:var(--surface);border-radius:var(--r);border:1px solid var(--border);overflow:hidden;margin-bottom:0}}
table{{width:100%;border-collapse:collapse}}
thead tr{{background:var(--surface2);border-bottom:1px solid var(--border)}}
.th-left{{padding:12px 16px;text-align:left;font-size:11px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}}
.th-right{{padding:12px 16px;text-align:right;font-size:11px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}}
tbody tr{{border-bottom:1px solid var(--border);transition:background .12s}}
tbody tr:last-child{{border-bottom:none}}
tbody tr:hover{{background:rgba(217,0,141,.04)}}
.td-rank{{padding:12px 16px;text-align:left;color:var(--muted);font-size:12px}}
.td-city{{padding:12px 16px;text-align:left;font-weight:700}}
.td-num{{padding:12px 16px;text-align:right}}

/* ── Loading ── */
.loading-state{{text-align:center;padding:80px 20px;color:var(--muted)}}

/* ── Footer ── */
.footer{{text-align:center;color:var(--muted);font-size:12px;padding:24px 32px;border-top:1px solid var(--border)}}
.brand{{color:var(--brand);font-weight:800}}

@media(max-width:900px){{
  .kpi-grid{{grid-template-columns:repeat(2,1fr)}}
  .main{{padding:16px}}
  .header{{padding:14px 16px;flex-direction:column;align-items:flex-start;gap:10px}}
  .section-nav{{top:0;position:relative;padding:10px 16px}}
  .overall-bar-wrap{{flex-direction:column;align-items:flex-start;gap:8px}}
}}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="hdr-title">&#128506; Wiom &nbsp;<em>UP Cities</em> &#8212; Live Dashboard</div>
    <div class="hdr-meta">{active_cities} active cities &nbsp;&#183;&nbsp; Updated: {generated_at} &nbsp;&#183;&nbsp; Showing: <strong>{pl}</strong></div>
  </div>
  <div class="hdr-right">
    <div class="live-dot"></div>
    <button class="btn-refresh" id="btnRefresh" onclick="triggerRefresh()">{refresh_label}</button>
  </div>
</div>

{nav_html}

<div class="main">
  {status_html}
  {main_content}
</div>

<div class="footer">
  <span class="brand">Wiom</span> UP Cities Dashboard &nbsp;&#183;&nbsp;
  Data: Snowflake via Metabase (DB 113) &nbsp;&#183;&nbsp;
  Modelled on Bagda Dashboard (card 1220) &nbsp;&#183;&nbsp;
  Excludes Delhi NCR &#183; Mumbai &#183; Lucknow &#183; Prayagraj &#183; Gorakhpur
</div>

<script>
  var _polling = false;
  var _daysQs = '{days_qs}';

  function triggerRefresh() {{
    var btn = document.getElementById('btnRefresh');
    btn.disabled = true;
    btn.textContent = 'Loading…';
    fetch('/api/refresh' + _daysQs)
      .then(function() {{ _polling = true; pollStatus(); }})
      .catch(function() {{ btn.disabled = false; btn.textContent = '{refresh_label}'; }});
  }}

  function pollStatus() {{
    if (!_polling) return;
    fetch('/api/status' + _daysQs)
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        if (!d.refreshing) {{ _polling = false; location.reload(); }}
        else {{ setTimeout(pollStatus, 3000); }}
      }});
  }}

  // Auto-reload page every hour
  setTimeout(function() {{ location.reload(); }}, 3600000);
</script>
</body>
</html>"""


# ──────────────────────────────────────────────
# FLASK ROUTES
# ──────────────────────────────────────────────

@app.route('/')
def index():
    days = _parse_days(freq.args.get('days'))
    cache = get_cache(days)
    if not cache.get('recharge') and not is_refreshing(days):
        threading.Thread(target=refresh_data_for, args=(days,), daemon=True).start()
    return build_html(days)

@app.route('/api/refresh')
def api_refresh():
    days = _parse_days(freq.args.get('days'))
    threading.Thread(target=refresh_data_for, args=(days,), daemon=True).start()
    return jsonify({'status': 'refreshing', 'days': days})

@app.route('/api/status')
def api_status():
    days  = _parse_days(freq.args.get('days'))
    cache = get_cache(days)
    return jsonify({
        'refreshing':   is_refreshing(days),
        'generated_at': cache.get('generated_at'),
        'error':        cache.get('error'),
        'days':         days,
    })


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 52)
    print('  Wiom UP Cities Dashboard')
    print(f'  http://localhost:{PORT}')
    print('  Sections: Recharge / Funnel / Tickets / Usage')
    print('  Filters:  All | 1d | 3d | 7d | 14d | 30d')
    print('=' * 52)
    print('Pre-fetching All-Time data from Snowflake (~2-3 min)...')

    threading.Thread(target=refresh_data_for, args=(None,), daemon=True).start()

    if '--no-open' not in sys.argv:
        threading.Timer(3.0, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()

    app.run(host='0.0.0.0', port=PORT, debug=False)
