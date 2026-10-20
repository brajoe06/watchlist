#!/usr/bin/env python3
"""
daily_watchlist.py - ranks a watchlist on live market data and writes a briefing.

NOT financial advice. A research aid: it scores stocks on public data and tells
you what changed. It cannot predict prices, and the political inputs
(congressional trades, policy headlines) are noisy and delayed.

QUICK START (needs Python 3.9+ and an internet connection)
    python daily_watchlist.py
The first run installs pandas, numpy and yfinance automatically.

More options
    python daily_watchlist.py --watch 15            # refresh every 15 minutes
    python daily_watchlist.py --tickers NVDA,BE,XOM # use your own list
    python daily_watchlist.py --congress auto       # best-effort public feeds
    python daily_watchlist.py --congress trades.csv # your own export
    python daily_watchlist.py --html docs/index.html  # also write a web page
    python daily_watchlist.py --demo                # synthetic data, offline

Watchlist: edit watchlist.json (next to this file) instead of the code. If
congress.csv exists it is used with --congress auto.

Prices are Yahoo Finance quotes: roughly real time, but can lag ~15 minutes
and are stale outside US market hours (the report says which).

Congress CSV columns:  ticker,member,date,type,amount_low,amount_high
    BE,Nancy Pelosi,2026-07-24,buy,1000001,5000000
"""
import importlib.util
import subprocess
import sys


def ensure_deps():
    needed = ["pandas", "numpy"] + ([] if "--demo" in sys.argv else ["yfinance"])
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    if missing:
        print(f"Installing {', '.join(missing)} (first run only)...", flush=True)
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", *missing]
        # Some systems (newer Debian/Ubuntu/macOS Homebrew Python) refuse system-wide installs;
        # retry with the override flag, then give clear instructions.
        if subprocess.call(cmd) != 0 and subprocess.call(cmd + ["--user", "--break-system-packages"]) != 0:
            sys.exit(f"Could not install {', '.join(missing)}. Run:  {sys.executable} -m venv .venv && "
                     f".venv/bin/pip install {' '.join(missing)}  then run the script with .venv/bin/python")


ensure_deps()

import argparse  # noqa: E402
import html as htmllib  # noqa: E402
import re  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from datetime import date, datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------
# CONFIG - edit freely
# --------------------------------------------------------------------------
# policy_tilt: your own judgement, -2 (policy headwind) .. +2 (policy tailwind).
# Review it when tariffs, export controls, subsidies or regulation change.
WATCHLIST = {
    "NVDA":  {"sector": "AI chips",           "policy_tilt": 0},
    "INTC":  {"sector": "Semiconductors",     "policy_tilt": 0},
    "BE":    {"sector": "Power / fuel cells", "policy_tilt": 0},
    "AAPL":  {"sector": "Consumer tech",      "policy_tilt": 0},
    "GOOGL": {"sector": "Internet",           "policy_tilt": 0},
    "MSFT":  {"sector": "Software / cloud",   "policy_tilt": 0},
    "AB":    {"sector": "Asset management",   "policy_tilt": 0},
    "UBER":  {"sector": "Mobility",           "policy_tilt": 0},
    "XOM":   {"sector": "Energy",             "policy_tilt": 0},
    "LMT":   {"sector": "Defense",            "policy_tilt": 0},
    "JNJ":   {"sector": "Healthcare",         "policy_tilt": 0},
    "PG":    {"sector": "Staples",            "policy_tilt": 0},
    "MA":    {"sector": "Payments",           "policy_tilt": 0},
    "CSX":   {"sector": "Rail / industrial",  "policy_tilt": 0},
}

WEIGHTS = {  # must sum to 1.0
    "momentum": 0.25, "trend": 0.10, "quality": 0.15, "valuation": 0.10,
    "sentiment": 0.10, "congress": 0.05, "policy": 0.10, "risk": 0.15,
}

BENCH, VIX = "SPY", "^VIX"
OUT_DIR = Path("watchlist_reports")

POS_WORDS = ["beats", "upgrade", "record", "surge", "raises", "contract", "wins",
             "approval", "partnership", "expands", "soars"]
NEG_WORDS = ["miss", "downgrade", "lawsuit", "probe", "investigation", "cuts",
             "plunge", "recall", "delay", "fraud", "halt", "layoffs", "short seller"]
POLICY_WORDS = ["tariff", "sanction", "export control", "regulation", "subsidy",
                "tax credit", "antitrust", "ban ", "congress", "senate",
                "white house", "executive order", "fda", "ftc", "doj"]

FACTOR_LABEL = {
    "momentum": "price momentum", "trend": "trend vs 50/200-day averages",
    "quality": "growth/margins/debt", "valuation": "valuation & analyst upside",
    "sentiment": "news tone", "congress": "congressional trading",
    "policy": "your policy tilt", "risk": "volatility/drawdown",
}

# Best-effort public congressional feeds (community-run; may be stale or offline).
CONGRESS_FEEDS = [
    ("https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
     "representative"),
    ("https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json",
     "senator"),
]


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def _naive(idx):
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx


def fetch_live(tickers):
    import yfinance as yf

    syms = tickers + [BENCH, VIX]
    hist = yf.download(syms, period="15mo", interval="1d", auto_adjust=True,
                       progress=False, threads=True)["Close"]
    hist = hist.dropna(how="all")
    hist.index = _naive(hist.index)
    meta = {"asof": None}

    # Overlay the freshest intraday price on top of the daily history.
    try:
        intra = yf.download(syms, period="1d", interval="1m", auto_adjust=True,
                            progress=False, threads=True)["Close"].dropna(how="all")
        if not intra.empty:
            stamp = pd.Timestamp(intra.index[-1])
            meta["asof"] = stamp if stamp.tzinfo else stamp.tz_localize("UTC")
            day = stamp.tz_localize(None).normalize() if stamp.tzinfo else stamp.normalize()
            latest = intra.ffill().iloc[-1].reindex(hist.columns).dropna()
            hist = hist.reindex(hist.index.union([day]))
            hist.loc[day, latest.index] = latest.values
            hist = hist.sort_index().ffill()
    except Exception as e:
        print(f"note: intraday refresh failed ({e}); using daily closes.", file=sys.stderr)

    def one(t):
        tk = yf.Ticker(t)
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        titles = []
        try:
            for item in tk.news or []:
                title = item.get("title") or (item.get("content") or {}).get("title")
                if title:
                    titles.append(title)
        except Exception:
            pass
        ed = None
        try:
            cal = tk.calendar
            eds = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if eds:
                ed = pd.Timestamp(eds[0]).date()
        except Exception:
            pass
        return t, info, titles[:15], ed

    info, news, earnings = {}, {}, {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for t, i, n, ed in ex.map(one, tickers):
            info[t], news[t] = i, n
            if ed:
                earnings[t] = ed
    return hist, info, news, earnings, meta


def fetch_demo(tickers):
    rng = np.random.default_rng(date.today().toordinal())
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=320)
    data = {}
    for c in tickers + [BENCH, VIX]:
        if c == VIX:
            data[c] = np.clip(18 + np.cumsum(rng.normal(0, 0.6, len(idx))), 11, 45)
            continue
        vol = rng.uniform(0.010, 0.035)
        r = rng.normal(rng.normal(0.0004, 0.0006), vol, len(idx))
        data[c] = 100 * np.exp(np.cumsum(r))
    prices = pd.DataFrame(data, index=idx)
    heads = ["{} beats estimates, raises outlook", "{} faces tariff uncertainty",
             "Analyst downgrade for {}", "{} wins new contract", "Senate probe touches {}"]
    info, news, earnings = {}, {}, {}
    for t in tickers:
        px = float(prices[t].iloc[-1])
        info[t] = {"revenueGrowth": rng.normal(0.1, 0.15), "profitMargins": rng.normal(0.12, 0.1),
                   "debtToEquity": abs(rng.normal(80, 60)), "forwardPE": abs(rng.normal(25, 12)),
                   "targetMeanPrice": px * rng.uniform(0.85, 1.3)}
        news[t] = [h.format(t) for h in rng.choice(heads, size=3, replace=False)]
        if rng.random() < 0.2:
            earnings[t] = (pd.Timestamp.today() + pd.Timedelta(days=int(rng.integers(1, 20)))).date()
    return prices, info, news, earnings, {"asof": None, "demo": True}


EMPTY_CONGRESS = pd.DataFrame(columns=["ticker", "date", "type", "amount_low", "amount_high"])


def _amount(s):
    try:
        parts = str(s).replace("$", "").replace(",", "").split("-")
        lo = float(parts[0].strip())
        hi = float(parts[1].strip()) if len(parts) > 1 else lo
        return lo, hi
    except Exception:
        return np.nan, np.nan


def fetch_congress_auto(tickers):
    rows = []
    for url, who in CONGRESS_FEEDS:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                        timeout=30) as r:
                data = json.load(r)
        except Exception as e:
            print(f"note: congress feed unavailable ({url.split('/')[2]}): {e}", file=sys.stderr)
            continue
        for d in data:
            t = str(d.get("ticker", "")).upper()
            if t not in tickers:
                continue
            kind = str(d.get("type", "")).lower()
            lo, hi = _amount(d.get("amount"))
            rows.append({"ticker": t, "date": d.get("transaction_date"),
                         "type": "buy" if "purchase" in kind else "sell",
                         "amount_low": lo, "amount_high": hi})
    if not rows:
        print("note: no congressional data loaded; that factor will be neutral.", file=sys.stderr)
        return EMPTY_CONGRESS
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "amount_low"])
    if not df.empty and (pd.Timestamp.today() - df["date"].max()).days > 60:
        print(f"warning: newest congress record is {df['date'].max():%Y-%m-%d}; the free feed looks "
              "stale. Use --congress trades.csv for fresher data.", file=sys.stderr)
    return df


def load_congress(arg, tickers):
    if not arg:
        return EMPTY_CONGRESS
    if arg == "auto":
        df = fetch_congress_auto(tickers)
        if Path("congress.csv").exists():
            extra = pd.read_csv("congress.csv")
            extra["date"] = pd.to_datetime(extra["date"])
            extra["ticker"] = extra["ticker"].str.upper()
            df = pd.concat([df, extra], ignore_index=True)
        return df
    df = pd.read_csv(arg)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].str.upper()
    return df


# --------------------------------------------------------------------------
# FEATURES
# --------------------------------------------------------------------------
def rsi(series, n=14):
    d = series.diff().dropna()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) else np.nan


def congress_signal(df, ticker, today):
    d = df[df["ticker"] == ticker]
    if d.empty:
        return 0.0
    age = (pd.Timestamp(today) - d["date"]).dt.days
    keep = age.between(0, 90)
    d, age = d[keep], age[keep]
    if d.empty:
        return 0.0
    amt = ((d["amount_low"] + d["amount_high"]) / 2).clip(lower=1000)
    sign = np.where(d["type"].str.lower().str.startswith("b"), 1.0, -1.0)
    return float(np.sum(sign * np.log10(amt) * np.exp(-age / 45)))


def news_scores(titles):
    sent, pol = 0, []
    for t in titles:
        low = t.lower() + " "
        sent += sum(w in low for w in POS_WORDS) - sum(w in low for w in NEG_WORDS)
        if any(w in low for w in POLICY_WORDS):
            pol.append(t)
    return sent, pol


def build_features(tickers, prices, info, news, earnings, congress, today):
    rows = {}
    for t in tickers:
        p = prices[t].dropna() if t in prices else pd.Series(dtype=float)
        if len(p) < 70:
            print(f"warning: not enough price history for {t}, skipping", file=sys.stderr)
            continue
        i = info.get(t, {})
        last = float(p.iloc[-1])
        sma50 = p.rolling(50).mean().iloc[-1]
        sma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
        ret = p.pct_change().dropna()
        sent, pol = news_scores(news.get(t, []))
        tgt, pe, ed = i.get("targetMeanPrice"), i.get("forwardPE"), earnings.get(t)
        rows[t] = {
            "price": last,
            "chg_1d": float(ret.iloc[-1]),
            "mom_12_1": p.iloc[-21] / p.iloc[-252] - 1 if len(p) >= 252 else np.nan,
            "mom_3m": last / p.iloc[-63] - 1,
            "trend": float(last > sma50) + (float(sma50 > sma200) if not np.isnan(sma200) else 0.5),
            "vol": float(ret.iloc[-63:].std() * np.sqrt(252)),
            "drawdown": last / float(p.iloc[-252:].max()) - 1,
            "rsi": rsi(p),
            "rev_growth": i.get("revenueGrowth", np.nan),
            "margin": i.get("profitMargins", np.nan),
            "debt_eq": i.get("debtToEquity", np.nan),
            "fwd_pe": pe if (pe and pe > 0) else np.nan,
            "upside": (tgt / last - 1) if tgt else np.nan,
            "target": tgt if tgt else np.nan,
            "news": sent,
            "policy_headlines": pol,
            "congress": congress_signal(congress, t, today),
            "policy_tilt": WATCHLIST[t]["policy_tilt"],
            "sector": WATCHLIST[t]["sector"],
            "earnings_in_days": (ed - today).days if ed and 0 <= (ed - today).days <= 14 else None,
        }
    return pd.DataFrame(rows).T


def z(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 3:
        return (s * 0).fillna(0)
    s = s.clip(s.quantile(0.05), s.quantile(0.95))
    sd = s.std()
    return ((s - s.mean()) / sd).fillna(0) if sd and not np.isnan(sd) else (s * 0).fillna(0)


def score(f):
    g = pd.DataFrame(index=f.index)
    g["momentum"] = (z(f["mom_12_1"]) + z(f["mom_3m"])) / 2
    g["trend"] = z(f["trend"])
    g["quality"] = (z(f["rev_growth"]) + z(f["margin"]) + z(-pd.to_numeric(f["debt_eq"], errors="coerce"))) / 3
    g["valuation"] = (z(-pd.to_numeric(f["fwd_pe"], errors="coerce")) + z(f["upside"])) / 2
    g["sentiment"] = z(f["news"])
    g["congress"] = z(f["congress"])
    g["policy"] = pd.to_numeric(f["policy_tilt"], errors="coerce").fillna(0)
    g["risk"] = (z(-pd.to_numeric(f["vol"])) + z(f["drawdown"])) / 2
    contrib = g.mul(pd.Series(WEIGHTS))
    f = f.copy()
    f["score"] = contrib.sum(axis=1)
    return f, contrib


# --------------------------------------------------------------------------
# REGIME, LABELS, SIZING
# --------------------------------------------------------------------------
def market_regime(prices):
    out = {"defensive": False, "text": "Market data unavailable."}
    try:
        spy = prices[BENCH].dropna()
        above = spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]
        vix = float(prices[VIX].dropna().iloc[-1])
        out["defensive"] = (not above) or vix > 28
        out["text"] = (f"S&P 500 proxy ({BENCH}) is {'above' if above else 'BELOW'} its 200-day average; "
                       f"VIX {vix:.1f}. "
                       + ("Defensive regime: favor smaller positions and wait for confirmation."
                          if out["defensive"] else "Risk-on regime: normal sizing rules apply."))
    except Exception:
        pass
    return out


def freshness(meta):
    if meta.get("demo"):
        return "DEMO DATA (synthetic, not real prices)"
    ts = meta.get("asof")
    if ts is None:
        return "Daily closes only (intraday feed unavailable)."
    age = (pd.Timestamp.now(tz="UTC") - ts.tz_convert("UTC")).total_seconds() / 60
    when = ts.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M ET")
    if age < 30:
        return f"Live-ish quotes as of {when} (Yahoo can lag ~15 min)."
    return f"Market closed or feed delayed: last trade {when} ({age / 60:.1f}h ago)."


def label(row, rank, n, defensive):
    flags = []
    if row["rsi"] > 75:
        flags.append(f"extended (RSI {row['rsi']:.0f}), wait for a pullback")
    if row["earnings_in_days"] is not None:
        flags.append(f"earnings in {row['earnings_in_days']}d, event risk")
    if row["trend"] == 0:
        flags.append("below both moving averages")
    if rank <= max(1, n // 4) and not flags:
        lab = "RESEARCH / buy candidate"
    elif rank > n - max(1, n // 4) or row["trend"] == 0:
        lab = "AVOID / reduce"
    elif flags:
        lab = "WATCH (wait)"
    else:
        lab = "HOLD / neutral"
    if defensive and lab.startswith("RESEARCH"):
        lab = "WATCH (defensive market)"
    return lab, flags


def max_weight(vol):
    """Risk budget: lose at most 1% of the portfolio on a 2-sigma monthly drop, cap 10%."""
    monthly = vol / np.sqrt(12)
    return min(0.10, 0.01 / (2 * monthly)) if monthly > 0 else 0.10


# --------------------------------------------------------------------------
# REPORT
# --------------------------------------------------------------------------
def pct(x, d=1):
    return "n/a" if pd.isna(x) else f"{x * 100:+.{d}f}%"


def build_report(f, contrib, regime, prev, today, top_n, meta):
    f = f.sort_values("score", ascending=False)
    n = len(f)
    L = [f"# Watchlist briefing - {today:%A, %d %B %Y}", "",
         "_Research aid, not financial advice. Scores rank stocks against each other; they do not predict returns._",
         "", f"**Data:** {freshness(meta)}", "", "## Market backdrop", regime["text"], ""]

    movers = f["chg_1d"].sort_values()
    L += ["**Biggest moves today:** "
          + ", ".join(f"{t} {pct(movers[t])}" for t in list(movers.index[:2]) + list(movers.index[-2:])), ""]

    L += ["## Ranking", "", "| # | Ticker | Score | vs prev day | Price | Today | Action | Max position |",
          "|---|---|---|---|---|---|---|---|"]
    notes = []
    for rank, (t, r) in enumerate(f.iterrows(), 1):
        lab, flags = label(r, rank, n, regime["defensive"])
        pr = prev.get(t)
        move = "new" if pr is None else (f"{pr - rank:+d}" if pr != rank else "=")
        L.append(f"| {rank} | {t} | {r['score']:+.2f} | {move} | {r['price']:.2f} | {pct(r['chg_1d'])} "
                 f"| {lab} | {max_weight(r['vol']) * 100:.1f}% |")
        c = contrib.loc[t].sort_values()
        up = [f"{FACTOR_LABEL[k]} ({v:+.2f})" for k, v in c[::-1].items() if v > 0.02][:2]
        dn = [f"{FACTOR_LABEL[k]} ({v:+.2f})" for k, v in c.items() if v < -0.02][:2]
        txt = f"**{t}** ({r['sector']}): "
        txt += ("helped by " + ", ".join(up) if up else "no standout strengths")
        txt += ("; held back by " + ", ".join(dn) if dn else "")
        txt += f". 3m {pct(r['mom_3m'], 0)}, from 52w high {pct(r['drawdown'], 0)}, vol {r['vol'] * 100:.0f}% annualised."
        if flags:
            txt += " Flags: " + "; ".join(flags) + "."
        if abs(r["congress"]) > 0.5:
            txt += (" Congress " + ("net buying" if r["congress"] > 0 else "net selling")
                    + " in last 90d (disclosed up to 45 days late).")
        for h in r["policy_headlines"][:2]:
            txt += f" Policy headline: \"{h}\"."
        notes.append((rank, txt))

    L += ["", f"## Insight on the top {top_n} and the bottom 2", ""]
    L += [f"- {t}" for rk, t in notes if rk <= top_n or rk > n - 2]
    L += ["", "## How to read this", "",
          "- **Max position** = size at which a 2-sigma monthly drop costs ~1% of your portfolio (cap 10%).",
          "- **RESEARCH** means 'worth reading the filings and news before acting', not 'buy'.",
          "- Politics enters through congressional trades (lagged, range-based, often spouse accounts), "
          "policy headline flags, and the policy_tilt you set per stock.",
          "- Backtest any rule you plan to rely on; a ranking that looks good today may not have predictive value."]
    return "\n".join(L)



# --------------------------------------------------------------------------
# HTML PAGE
# --------------------------------------------------------------------------
def _inline(s):
    s = htmllib.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", s)


def _cell(c):
    cls = ""
    if c.startswith("RESEARCH"):
        cls = " class='good'"
    elif c.startswith("AVOID"):
        cls = " class='bad'"
    elif c.startswith("WATCH"):
        cls = " class='warn'"
    return f"<td{cls}>{_inline(c)}</td>"


def md_to_body(md):
    lines, out, i, ul = md.split("\n"), [], 0, False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            split = lambda b: [c.strip() for c in b.strip().strip("|").split("|")]  # noqa: E731
            out.append("<div class='scroll'><table><thead><tr>"
                       + "".join(f"<th>{_inline(h)}</th>" for h in split(block[0]))
                       + "</tr></thead><tbody>")
            for b in block[2:]:
                out.append("<tr>" + "".join(_cell(c) for c in split(b)) + "</tr>")
            out.append("</tbody></table></div>")
            continue
        if ul and not ln.startswith("- "):
            out.append("</ul>")
            ul = False
        if ln.startswith("# "):
            out.append(f"<h1>{_inline(ln[2:])}</h1>")
        elif ln.startswith("## "):
            out.append(f"<h2>{_inline(ln[3:])}</h2>")
        elif ln.startswith("- "):
            if not ul:
                out.append("<ul>")
                ul = True
            out.append(f"<li>{_inline(ln[2:])}</li>")
        elif ln.strip():
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    if ul:
        out.append("</ul>")
    return "\n".join(out)



CORE_JS = r"""/* ---- scoring engine: a JavaScript port of the Python scoring ---- */
const num = v => (v === null || v === undefined || Number.isNaN(v)) ? NaN : v;
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
function quantile(sorted, q) {
  const pos = q * (sorted.length - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
function zscore(arr) {
  const valid = arr.filter(Number.isFinite);
  if (valid.length < 3) return arr.map(() => 0);
  const sorted = [...valid].sort((a, b) => a - b);
  const lo = quantile(sorted, 0.05), hi = quantile(sorted, 0.95);
  const c = arr.map(x => Number.isFinite(x) ? Math.min(hi, Math.max(lo, x)) : NaN);
  const cv = c.filter(Number.isFinite), m = mean(cv), sd = std(cv);
  if (!(sd > 0)) return arr.map(() => 0);
  return c.map(x => Number.isFinite(x) ? (x - m) / sd : 0);
}
function rsi14(p) {
  let up = 0, dn = 0, init = false;
  const a = 1 / 14;
  for (let i = 1; i < p.length; i++) {
    const d = p[i] - p[i - 1], u = Math.max(d, 0), v = Math.max(-d, 0);
    if (!init) { up = u; dn = v; init = true; } else { up = (1 - a) * up + a * u; dn = (1 - a) * dn + a * v; }
  }
  return dn === 0 ? NaN : 100 - 100 / (1 + up / dn);
}
function daysBetween(a, b) {
  return Math.round((Date.parse(b + 'T00:00:00Z') - Date.parse(a + 'T00:00:00Z')) / 86400000);
}
function maxWeight(vol) {
  const monthly = vol / Math.sqrt(12);
  return monthly > 0 ? Math.min(0.10, 0.01 / (2 * monthly)) : 0.10;
}
function labelFor(r, rank, n, defensive) {
  const flags = [];
  if (r.rsi > 75) flags.push('extended (RSI ' + r.rsi.toFixed(0) + '), wait for a pullback');
  if (r.earnings_in_days !== null) flags.push('earnings in ' + r.earnings_in_days + 'd, event risk');
  if (r.trend === 0) flags.push('below both moving averages');
  const q = Math.max(1, Math.floor(n / 4));
  let lab;
  if (rank <= q && !flags.length) lab = 'RESEARCH / buy candidate';
  else if (rank > n - q || r.trend === 0) lab = 'AVOID / reduce';
  else if (flags.length) lab = 'WATCH (wait)';
  else lab = 'HOLD / neutral';
  if (defensive && lab.startsWith('RESEARCH')) lab = 'WATCH (defensive market)';
  return { lab, flags };
}
function computeAll(P, S, today) {
  const syms = Object.keys(P.stocks).filter(s => S[s] && S[s].closes.length >= 70);
  const rows = syms.map(s => {
    const st = P.stocks[s], p = S[s].closes, n = p.length, last = p[n - 1];
    const sma50 = mean(p.slice(n - 50));
    const sma200 = n >= 200 ? mean(p.slice(n - 200)) : NaN;
    const rets = []; for (let i = 1; i < n; i++) rets.push(p[i] / p[i - 1] - 1);
    let ed = null;
    if (st.earnings_date) { const d = daysBetween(today, st.earnings_date); if (d >= 0 && d <= 14) ed = d; }
    const pe = num(st.fwd_pe), tgt = num(st.target);
    return {
      sym: s, sector: st.sector, price: last, chg_1d: rets[rets.length - 1],
      mom_12_1: n >= 252 ? p[n - 21] / p[n - 252] - 1 : NaN,
      mom_3m: last / p[n - 63] - 1,
      trend: (last > sma50 ? 1 : 0) + (Number.isFinite(sma200) ? (sma50 > sma200 ? 1 : 0) : 0.5),
      vol: std(rets.slice(-63)) * Math.sqrt(252),
      drawdown: last / Math.max(...p.slice(Math.max(0, n - 252))) - 1,
      rsi: rsi14(p),
      rev_growth: num(st.rev_growth), margin: num(st.margin), debt_eq: num(st.debt_eq),
      fwd_pe: Number.isFinite(pe) ? pe * last / st.px_run : NaN,
      upside: Number.isFinite(tgt) ? tgt / last - 1 : NaN,
      news: st.news, congress: st.congress, policy_tilt: st.policy_tilt,
      policy_headlines: st.policy_headlines || [], earnings_in_days: ed,
    };
  });
  const col = k => rows.map(r => r[k]);
  const neg = a => a.map(x => -x);
  const avg = (...arrs) => arrs[0].map((_, i) => arrs.reduce((s, a) => s + a[i], 0) / arrs.length);
  const g = {
    momentum: avg(zscore(col('mom_12_1')), zscore(col('mom_3m'))),
    trend: zscore(col('trend')),
    quality: avg(zscore(col('rev_growth')), zscore(col('margin')), zscore(neg(col('debt_eq')))),
    valuation: avg(zscore(neg(col('fwd_pe'))), zscore(col('upside'))),
    sentiment: zscore(col('news')),
    congress: zscore(col('congress')),
    policy: col('policy_tilt').map(x => Number.isFinite(x) ? x : 0),
    risk: avg(zscore(neg(col('vol'))), zscore(col('drawdown'))),
  };
  rows.forEach((r, i) => {
    r.contrib = {}; r.score = 0;
    for (const k of Object.keys(P.weights)) { r.contrib[k] = g[k][i] * P.weights[k]; r.score += r.contrib[k]; }
  });
  rows.sort((a, b) => b.score - a.score);

  let regime = { defensive: false, text: 'Market data unavailable.' };
  try {
    const spy = S.SPY.closes;
    const above = spy.length >= 200 && spy[spy.length - 1] > mean(spy.slice(spy.length - 200));
    const vix = P.vix;
    if (vix === null || vix === undefined) throw new Error('no vix');
    const defensive = !above || vix > 28;
    regime = { defensive, text: 'S&P 500 proxy (SPY) is ' + (above ? 'above' : 'BELOW') + ' its 200-day average; VIX ' + vix.toFixed(1) + '. ' +
      (defensive ? 'Defensive regime: favor smaller positions and wait for confirmation.' : 'Risk-on regime: normal sizing rules apply.') };
  } catch (e) { /* keep default */ }

  rows.forEach((r, i) => {
    r.rank = i + 1;
    const l = labelFor(r, r.rank, rows.length, regime.defensive);
    r.label = l.lab; r.flags = l.flags; r.maxw = maxWeight(r.vol);
  });
  return { rows, regime };
}
"""

APP_JS = r"""/* ---- page: render + optional live prices (Finnhub) ---- */
const P = JSON.parse(document.getElementById('payload').textContent);
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const FL = P.factor_label;
const pct = (x, d = 1) => Number.isFinite(x) ? (x >= 0 ? '+' : '') + (x * 100).toFixed(d) + '%' : 'n/a';
const sgn = (v, d = 2) => (v >= 0 ? '+' : '') + v.toFixed(d);
const etDate = ms => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York' }).format(new Date(ms));
const etTime = ms => new Date(ms).toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour12: false });

const S = {};
for (const [t, st] of Object.entries(P.stocks)) S[t] = { dates: st.dates.slice(), closes: st.closes.slice() };
S.SPY = { dates: P.spy.dates.slice(), closes: P.spy.closes.slice() };

let live = { mode: 'snapshot', lastTick: 0, ws: null, timer: null, msg: '' };
const shown = {};

function render() {
  const { rows, regime } = computeAll(P, S, etDate(Date.now()));
  const n = rows.length;
  $('regime').textContent = regime.text;
  const byMove = rows.slice().sort((a, b) => a.chg_1d - b.chg_1d);
  const mv = byMove.slice(0, 2).concat(byMove.slice(-2));
  $('movers').innerHTML = '<strong>Biggest moves today:</strong> ' + mv.map(r => esc(r.sym) + ' ' + pct(r.chg_1d)).join(', ');
  $('tbody').innerHTML = rows.map(r => {
    const pr = P.prev_ranks[r.sym];
    const move = pr === undefined ? 'new' : (pr === r.rank ? '=' : (pr - r.rank > 0 ? '+' : '') + (pr - r.rank));
    const cls = r.label.startsWith('RESEARCH') ? 'good' : r.label.startsWith('AVOID') ? 'bad' : r.label.startsWith('WATCH') ? 'warn' : '';
    const was = shown[r.sym]; shown[r.sym] = r.price;
    const flash = was !== undefined && was !== r.price ? (r.price > was ? ' up' : ' down') : '';
    return '<tr><td>' + r.rank + '</td><td>' + esc(r.sym) + '</td><td>' + sgn(r.score) + '</td><td>' + move +
      '</td><td class="px' + flash + '">' + r.price.toFixed(2) + '</td><td>' + pct(r.chg_1d) + '</td><td class="' + cls + '">' +
      esc(r.label) + '</td><td>' + (r.maxw * 100).toFixed(1) + '%</td></tr>';
  }).join('');
  $('ins-title').textContent = 'Insight on the top ' + P.top_n + ' and the bottom 2';
  $('insights').innerHTML = rows.filter(r => r.rank <= P.top_n || r.rank > n - 2).map(r => {
    const c = Object.entries(r.contrib).sort((a, b) => a[1] - b[1]);
    const up = c.slice().reverse().filter(([, v]) => v > 0.02).slice(0, 2).map(([k, v]) => FL[k] + ' (' + sgn(v) + ')');
    const dn = c.filter(([, v]) => v < -0.02).slice(0, 2).map(([k, v]) => FL[k] + ' (' + sgn(v) + ')');
    let t = '<strong>' + esc(r.sym) + '</strong> (' + esc(r.sector) + '): ' + (up.length ? 'helped by ' + esc(up.join(', ')) : 'no standout strengths') +
      (dn.length ? '; held back by ' + esc(dn.join(', ')) : '') + '. 3m ' + pct(r.mom_3m, 0) + ', from 52w high ' + pct(r.drawdown, 0) +
      ', vol ' + (r.vol * 100).toFixed(0) + '% annualised.';
    if (r.flags.length) t += ' Flags: ' + esc(r.flags.join('; ')) + '.';
    if (Math.abs(r.congress) > 0.5) t += ' Congress ' + (r.congress > 0 ? 'net buying' : 'net selling') + ' in last 90d (disclosed up to 45 days late).';
    r.policy_headlines.slice(0, 2).forEach(h => { t += ' Policy headline: "' + esc(h) + '".'; });
    return '<li>' + t + '</li>';
  }).join('');
  renderStatus();
}

function renderStatus() {
  const pill = $('pill'), snap = new Date(P.generated).toLocaleString();
  let cls = 'snap', txt = 'Snapshot only';
  if (live.mode === 'connecting') { cls = 'wait'; txt = 'Connecting...'; }
  else if (live.mode === 'live') {
    const age = (Date.now() - live.lastTick) / 60000;
    if (live.lastTick && age < 20) { cls = 'on'; txt = 'LIVE - last tick ' + etTime(live.lastTick) + ' ET'; }
    else { cls = 'wait'; txt = 'Connected - market closed or quiet' + (live.lastTick ? ', last trade ' + etDate(live.lastTick) + ' ' + etTime(live.lastTick) + ' ET' : ''); }
  } else if (live.mode === 'error') { cls = 'off'; txt = 'Live feed error'; }
  pill.className = 'pill ' + cls; pill.textContent = txt;
  $('livemsg').textContent = live.msg;
  $('data').innerHTML = '<strong>Data:</strong> ' + esc(P.freshness) + ' Snapshot built ' + esc(snap) + '. ' +
    (live.mode === 'live' ? 'Prices and rankings below update live; fundamentals, news and congressional data are from the snapshot.' : '');
  $('keybox').hidden = live.mode === 'live' || live.mode === 'connecting';
  $('stopbtn').hidden = !(live.mode === 'live' || live.mode === 'connecting');
}

let pending = false;
function scheduleRender() {
  if (pending) return;
  pending = true;
  setTimeout(() => { pending = false; render(); }, 1000);
}

function applyTick(sym, price, tsMs, prevClose) {
  const s = S[sym]; if (!s || !(price > 0)) return;
  const day = etDate(tsMs), last = s.dates[s.dates.length - 1];
  if (day === last) s.closes[s.closes.length - 1] = price;
  else if (day > last) {
    if (prevClose > 0 && daysBetween(last, day) <= 4) s.closes[s.closes.length - 1] = prevClose;
    s.dates.push(day); s.closes.push(price);
  } else return;
  if (tsMs > live.lastTick && sym === 'SPY') live.lastTick = tsMs;
}

const fh = s => s.replace('-', '.');
const back = {};
Object.keys(S).forEach(s => { back[fh(s)] = s; });

async function fetchQuotes(key) {
  for (const sym of Object.keys(S)) {
    const r = await fetch('https://finnhub.io/api/v1/quote?symbol=' + encodeURIComponent(fh(sym)) + '&token=' + encodeURIComponent(key));
    if (r.status === 401 || r.status === 403) throw new Error('That API key was rejected.');
    if (r.status === 429) throw new Error('Rate limited by Finnhub; try again in a minute.');
    if (!r.ok) continue;
    const q = await r.json();
    if (q && q.c > 0 && q.t > 0) applyTick(sym, q.c, q.t * 1000, q.pc);
    await new Promise(res => setTimeout(res, 120));
  }
}

function openSocket(key, attempt) {
  const ws = new WebSocket('wss://ws.finnhub.io?token=' + encodeURIComponent(key));
  live.ws = ws;
  ws.onopen = () => Object.keys(S).forEach(s => ws.send(JSON.stringify({ type: 'subscribe', symbol: fh(s) })));
  ws.onmessage = e => {
    let m; try { m = JSON.parse(e.data); } catch (_) { return; }
    if (m.type !== 'trade') return;
    for (const t of m.data) applyTick(back[t.s], t.p, t.t);
    scheduleRender();
  };
  ws.onclose = () => {
    if (live.ws !== ws || live.mode !== 'live') return;
    if (attempt >= 3) { startPolling(key); return; }
    setTimeout(() => { if (live.mode === 'live') openSocket(key, attempt + 1); }, 2000 * (attempt + 1));
  };
}

function startPolling(key) {
  live.msg = 'Streaming unavailable; refreshing every 30 seconds instead.';
  clearInterval(live.timer);
  live.timer = setInterval(async () => { try { await fetchQuotes(key); render(); } catch (e) { live.msg = e.message; renderStatus(); } }, 30000);
}

async function connect(key) {
  stop(true);
  live.mode = 'connecting'; live.msg = ''; renderStatus();
  try {
    await fetchQuotes(key);
    live.mode = 'live';
    try { localStorage.setItem('fh_key', key); } catch (_) { /* private mode */ }
    openSocket(key, 0);
    render();
  } catch (e) {
    live.mode = 'error'; live.msg = e.message || 'Could not connect.'; renderStatus();
  }
}

function stop(keep) {
  if (live.ws) { const w = live.ws; live.ws = null; live.mode = 'snapshot'; try { w.close(); } catch (_) { } }
  clearInterval(live.timer);
  if (!keep) {
    live.mode = 'snapshot'; live.msg = '';
    try { localStorage.removeItem('fh_key'); } catch (_) { }
    Object.keys(S).forEach(t => { const src = t === 'SPY' ? P.spy : P.stocks[t]; S[t].dates = src.dates.slice(); S[t].closes = src.closes.slice(); });
    render();
  }
}

$('connect').onclick = () => { const k = $('key').value.trim(); if (k) connect(k); };
$('stopbtn').onclick = () => stop(false);
if (P.demo) { $('live').hidden = true; }
render();
setInterval(renderStatus, 5000);
try { const saved = localStorage.getItem('fh_key'); if (saved && !P.demo) connect(saved); } catch (_) { }
"""

PAGE_CSS = """
:root{color-scheme:light dark;--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff;
--good:#1a7f45;--bad:#b3261e;--warn:#9a6700}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ececea;--mut:#9a9a94;--line:#2c2c29;--card:#1c1c1a;
--good:#4cc38a;--bad:#ff7b72;--warn:#e3b341}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.5rem;margin:.2em 0 .6em}h2{font-size:1.15rem;margin:1.6em 0 .5em}
.scroll{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:600}tr:last-child td{border-bottom:0}
td.good{color:var(--good);font-weight:600}td.bad{color:var(--bad);font-weight:600}td.warn{color:var(--warn);font-weight:600}
ul{padding-left:1.2em}li{margin:.5em 0}em,.mut{color:var(--mut)}
.live{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.85rem;font-weight:600;border:1px solid var(--line)}
.pill.on{color:var(--good);border-color:var(--good)}.pill.wait{color:var(--warn);border-color:var(--warn)}
.pill.off{color:var(--bad);border-color:var(--bad)}.pill.snap{color:var(--mut)}
.live input{padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);min-width:240px}
.live button{padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);cursor:pointer}
.live p{margin:.5em 0}
td.up{animation:fu 1.2s}td.down{animation:fd 1.2s}
@keyframes fu{from{background:color-mix(in srgb,var(--good) 35%,transparent)}to{background:transparent}}
@keyframes fd{from{background:color-mix(in srgb,var(--bad) 35%,transparent)}to{background:transparent}}
"""

PAGE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Watchlist briefing</title>
<style>__CSS__</style></head><body><main>
<h1>Watchlist briefing</h1>
<p><em>Research aid, not financial advice. Scores rank stocks against each other; they do not predict returns.</em></p>
<div class="live" id="live">
  <p><span class="pill snap" id="pill">Snapshot only</span> <button id="stopbtn" hidden>Disconnect live prices</button></p>
  <div id="keybox"><p class="mut">For real-time prices, paste a free <a href="https://finnhub.io/register" target="_blank" rel="noopener">Finnhub</a> API key.
  It is stored only in this browser and sent only to Finnhub.</p>
  <p><input id="key" type="password" placeholder="Finnhub API key" autocomplete="off"> <button id="connect">Connect</button></p></div>
  <p class="mut" id="livemsg"></p>
</div>
<p id="data"></p>
<h2>Market backdrop</h2><p id="regime"></p><p id="movers"></p>
<h2>Ranking</h2>
<div class="scroll"><table><thead><tr><th>#</th><th>Ticker</th><th>Score</th><th>vs prev day</th><th>Price</th><th>Today</th><th>Action</th><th>Max position</th></tr></thead><tbody id="tbody"></tbody></table></div>
<h2 id="ins-title"></h2><ul id="insights"></ul>
<h2>How to read this</h2>
<ul>
<li><strong>Max position</strong> = size at which a 2-sigma monthly drop costs ~1% of your portfolio (cap 10%).</li>
<li><strong>RESEARCH</strong> means "worth reading the filings and news before acting", not "buy".</li>
<li>Politics enters through congressional trades (lagged, range-based, often spouse accounts), policy headline flags, and the policy_tilt you set per stock.</li>
<li>With live prices connected, price-based factors and rankings update as trades arrive; fundamentals, news and congressional data stay as of the snapshot.</li>
<li>Backtest any rule you plan to rely on; a ranking that looks good today may not have predictive value.</li>
</ul>
<noscript><hr>__NOSCRIPT__</noscript>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
__CORE__
__APP__
</script></body></html>"""


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isinf(x)) else x


def _series(s, keep=300):
    s = s.dropna().iloc[-keep:]
    return {"dates": [d.strftime("%Y-%m-%d") for d in s.index], "closes": [round(float(v), 4) for v in s.values]}


def build_payload(scored, prices, earnings, prev, meta, today, top_n):
    stocks = {}
    for t, r in scored.iterrows():
        d = _series(prices[t])
        d.update({
            "sector": r["sector"], "px_run": float(r["price"]),
            "rev_growth": _num(r["rev_growth"]), "margin": _num(r["margin"]), "debt_eq": _num(r["debt_eq"]),
            "fwd_pe": _num(r["fwd_pe"]), "target": _num(r["target"]),
            "news": float(r["news"]), "congress": float(r["congress"]), "policy_tilt": float(r["policy_tilt"]),
            "earnings_date": str(earnings[t]) if t in earnings else None,
            "policy_headlines": list(r["policy_headlines"]), "snap_score": float(r["score"]),
        })
        stocks[t] = d
    vix = None
    try:
        vix = _num(prices[VIX].dropna().iloc[-1])
    except Exception:
        pass
    return {
        "generated": datetime.now(timezone.utc).isoformat(), "today": str(today), "demo": bool(meta.get("demo")),
        "freshness": freshness(meta), "weights": WEIGHTS, "factor_label": FACTOR_LABEL, "top_n": top_n,
        "vix": vix, "spy": _series(prices[BENCH]), "prev_ranks": prev, "stocks": stocks,
    }


def build_live_html(payload, report_md):
    data = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return (PAGE_HTML.replace("__CSS__", PAGE_CSS).replace("__NOSCRIPT__", md_to_body(report_md))
            .replace("__PAYLOAD__", data).replace("__CORE__", CORE_JS).replace("__APP__", APP_JS))


# --------------------------------------------------------------------------
def run_once(a, tickers, congress):
    today = date.today()
    prices, info, news, earnings, meta = (fetch_demo if a.demo else fetch_live)(tickers)
    feats = build_features(tickers, prices, info, news, earnings, congress, today)
    if feats.empty:
        sys.exit("No usable data. Check your internet connection and tickers.")
    scored, contrib = score(feats)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / "state.json"
    state = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {}
    # "previous day" ranks: yesterday's if the saved state is from an earlier date
    prev = state.get("prev", {}) if state.get("date") == str(today) else state.get("ranks", {})

    report = build_report(scored, contrib, market_regime(prices), prev, today, a.top, meta)
    print(report)
    (out / f"watchlist_{today:%Y-%m-%d}.md").write_text(report, encoding="utf-8")
    if a.html:
        hp = Path(a.html)
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(build_live_html(build_payload(scored, prices, earnings, prev, meta, today, a.top), report),
                      encoding="utf-8")

    ranks = {t: i for i, t in enumerate(scored.sort_values("score", ascending=False).index, 1)}
    state_file.write_text(json.dumps({"date": str(today), "ranks": ranks, "prev": prev}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline)")
    ap.add_argument("--congress", help="'auto' or path to a CSV of congressional trades")
    ap.add_argument("--tickers", help="comma-separated tickers to use instead of the built-in list")
    ap.add_argument("--watch", type=float, metavar="MIN", help="refresh every MIN minutes")
    ap.add_argument("--html", help="also write a web page to this path, e.g. docs/index.html")
    ap.add_argument("--config", default="watchlist.json", help="watchlist file (JSON)")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()

    cfg = Path(a.config)
    if cfg.exists():
        try:
            loaded = json.loads(cfg.read_text(encoding="utf-8"))["tickers"]
            WATCHLIST.clear()
            WATCHLIST.update({k.upper(): {"sector": v.get("sector", "Custom"),
                                          "policy_tilt": v.get("policy_tilt", 0)} for k, v in loaded.items()})
        except Exception as e:
            sys.exit(f"Could not read {cfg}: {e}")

    if a.tickers:
        for t in [x.strip().upper() for x in a.tickers.split(",") if x.strip()]:
            WATCHLIST.setdefault(t, {"sector": "Custom", "policy_tilt": 0})
        tickers = [x.strip().upper() for x in a.tickers.split(",") if x.strip()]
    else:
        tickers = list(WATCHLIST)

    congress = load_congress(a.congress, tickers)
    try:
        while True:
            run_once(a, tickers, congress)
            if not a.watch:
                break
            print(f"\n--- next refresh in {a.watch:g} min (Ctrl+C to stop) ---\n", flush=True)
            time.sleep(a.watch * 60)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
