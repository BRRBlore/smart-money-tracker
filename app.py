# ============================================================
# app.py — Cloud-safe Smart Money Tracker
# ============================================================

import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Smart Money Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cloud detection ───────────────────────────────────────────────────────────
IS_CLOUD = not os.path.exists("data/smart_money.db")

# ── Safe imports — only pure Python / no local-machine dependencies ───────────
from scoring import calculate_score
from config  import APP_TITLE, DATA_CSV, SCORE_WEIGHTS

# ── Plotly charts ─────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px

# ── Column rename map: SQLite lowercase → app CamelCase ──────────────────────
COL_MAP = {
    "ticker":"Ticker","name":"Name","sector":"Sector",
    "sector_pe":"Sector_PE","last_updated":"Last_Updated",
    "price":"Price","price_1m_ret":"Price_1M_Ret",
    "price_3m_ret":"Price_3M_Ret","price_6m_ret":"Price_6M_Ret",
    "low_52w":"52W_Low","high_52w":"52W_High",
    "pct_above_52w_low":"Pct_Above_52W_Low",
    "price_trend":"Price_Trend","volume_trend":"Volume_Trend",
    "pe":"PE","pb":"PB","roe":"RoE","roce":"RoCE",
    "de":"DE","revenue_growth":"Revenue_Growth",
    "market_cap_cr":"Market_Cap_Cr","screener_url":"Screener_URL",
    "fii_pct":"FII_Pct","dii_pct":"DII_Pct","promoter_pct":"Promoter_Pct",
    "fii_q1":"FII_Q1","fii_q2":"FII_Q2","fii_q3":"FII_Q3","fii_q4":"FII_Q4",
    "dii_q1":"DII_Q1","dii_q2":"DII_Q2","dii_q3":"DII_Q3","dii_q4":"DII_Q4",
    "fii_selling_4q":"FII_Selling_4Q","dii_buying_4q":"DII_Buying_4Q",
    "fii_trend_pct":"FII_Trend_Pct","dii_trend_pct":"DII_Trend_Pct",
    "fii_label":"FII_Label","dii_label":"DII_Label",
    "fair_value":"Fair_Value","buy_zone_low":"Buy_Zone_Low",
    "buy_zone_high":"Buy_Zone_High","strong_buy_below":"Strong_Buy_Below",
    "value_signal":"Value_Signal","valuation_methods":"Valuation_Methods",
    "smart_money_score":"Smart_Money_Score","grade":"Grade",
    "score_fii":"Score_FII","score_dii":"Score_DII",
    "score_pe":"Score_PE","score_pb":"Score_PB",
    "score_roe":"Score_RoE","score_roce":"Score_RoCE",
    "score_debt":"Score_Debt","score_revgrowth":"Score_RevGrowth",
    "score_promoter":"Score_Promoter","score_52w":"Score_52W",
    "scrape_status":"Scrape_Status",
}

def _rename(df):
    df = df.copy()
    df.columns = [COL_MAP.get(c, c) for c in df.columns]
    return df

# ── Local-only imports ────────────────────────────────────────────────────────
if not IS_CLOUD:
    try:
        from data_pipeline import build_dataset
        from database import (init_db, get_all_stocks, get_scrape_progress,
                               get_alerts, acknowledge_alerts)
        from batch_scraper import start_batch_scrape, stop_scrape
        from alert_engine  import run_all_alerts
        from ai_engine     import (generate_stock_insight,
                                   generate_portfolio_summary,
                                   natural_language_query,
                                   check_ollama_status)
        init_db()
        LOCAL_OK = True
    except Exception as e:
        LOCAL_OK = False
        st.sidebar.warning(f"Local modules error: {e}")
else:
    LOCAL_OK = False


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    # 1. SQLite (local only)
    if LOCAL_OK:
        try:
            df = get_all_stocks()
            if not df.empty:
                return _rename(df)
        except Exception:
            pass

    # 2. cloud_data.csv — exported from local SQLite
    for csv_path in ["data/cloud_data.csv", "data/smart_money_data.csv"]:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    df = _rename(df)
                    if "Smart_Money_Score" not in df.columns:
                        df = calculate_score(df)
                    return df.sort_values("Smart_Money_Score",
                                         ascending=False).reset_index(drop=True)
            except Exception:
                pass
    return pd.DataFrame()


# ── Charts ────────────────────────────────────────────────────────────────────
def chart_leaderboard(df):
    grade_colors = {"A+":"#1B5E20","A ":"#388E3C","B ":"#F9A825",
                    "C ":"#E65100","D ":"#B71C1C"}
    colors = []
    for g in df.get("Grade", pd.Series()):
        c = "#888"
        for k, v in grade_colors.items():
            if k.strip() in str(g): c = v; break
        colors.append(c)
    fig = go.Figure(go.Bar(
        x=df["Smart_Money_Score"], y=df["Name"], orientation="h",
        marker_color=colors,
        text=df.get("Grade",""), textposition="outside",
    ))
    fig.add_vline(x=60, line_dash="dash", line_color="#1565C0",
                  annotation_text="Buy threshold")
    fig.update_layout(
        title="Smart Money Score Leaderboard",
        xaxis=dict(title="Score (0–100)", range=[0, 115]),
        yaxis=dict(title=""), height=max(350, len(df)*32),
        margin=dict(l=160, r=80, t=50, b=40),
    )
    return fig


def chart_value_zone(df):
    df = df.sort_values("Smart_Money_Score", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["Name"], x=df.get("Buy_Zone_High",0) - df.get("Buy_Zone_Low",0),
        base=df.get("Buy_Zone_Low",0), name="Value Buy Zone",
        orientation="h", marker_color="rgba(56,142,60,0.5)",
    ))
    fig.add_trace(go.Scatter(
        y=df["Name"], x=df.get("Fair_Value",0),
        mode="markers", name="Fair Value",
        marker=dict(symbol="line-ns", size=12, color="#1565C0",
                    line=dict(width=2, color="#1565C0")),
    ))
    colors = []
    for sig in df.get("Value_Signal", pd.Series()):
        s = str(sig).upper()
        if "STRONG" in s: colors.append("#1B5E20")
        elif "BUY" in s:  colors.append("#388E3C")
        elif "WATCH" in s: colors.append("#F9A825")
        else:              colors.append("#C62828")
    fig.add_trace(go.Scatter(
        y=df["Name"], x=df.get("Price",0),
        mode="markers", name="Current Price",
        marker=dict(size=9, color=colors),
    ))
    fig.update_layout(
        barmode="overlay", title="Price vs Value Buy Zones",
        xaxis_title="Price (₹)", yaxis_title="",
        height=max(350, len(df)*38),
        margin=dict(l=160, r=80, t=50, b=40),
    )
    return fig


def chart_fii_dii(df):
    fii_x  = df["FII_Trend_Pct"]  if "FII_Trend_Pct"     in df.columns else pd.Series([0]*len(df))
    dii_y  = df["DII_Trend_Pct"]  if "DII_Trend_Pct"     in df.columns else pd.Series([0]*len(df))
    names  = df["Name"]            if "Name"              in df.columns else pd.Series([""]*len(df))
    scores = df["Smart_Money_Score"] if "Smart_Money_Score" in df.columns else pd.Series([50]*len(df))
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-20, x1=0, y0=0, y1=20,
                  fillcolor="rgba(56,142,60,0.08)", line_width=0)
    fig.add_trace(go.Scatter(
        x=fii_x, y=dii_y, mode="markers+text",
        text=names, textposition="top center", textfont=dict(size=9),
        marker=dict(size=scores/5+8, color=scores, colorscale="RdYlGn",
                    colorbar=dict(title="Score"),
                    line=dict(width=1, color="white"), showscale=True),
    ))
    fig.add_hline(y=0, line_color="grey", line_width=1)
    fig.add_vline(x=0, line_color="grey", line_width=1)
    fig.add_annotation(x=-10, y=8, text="🟢 Smart Money Signal",
                       showarrow=False, font=dict(color="#388E3C", size=11))
    fig.update_layout(
        title="FII vs DII 4-Quarter Trend",
        xaxis=dict(title="FII 4Q Change % (Negative = Selling)"),
        yaxis=dict(title="DII 4Q Change % (Positive = Buying)"),
        height=480, margin=dict(t=60, b=60, l=60, r=40),
    )
    return fig


def chart_radar(row):
    factors = ["FII Signal","DII Signal","PE Discount","PB Ratio",
               "RoE","RoCE","Debt Safety","Rev Growth","Promoter","52W Safety"]
    score_cols = ["Score_FII","Score_DII","Score_PE","Score_PB",
                  "Score_RoE","Score_RoCE","Score_Debt",
                  "Score_RevGrowth","Score_Promoter","Score_52W"]
    scores = [float(row.get(c, 0)) for c in score_cols]
    fig = go.Figure(go.Scatterpolar(
        r=scores+[scores[0]], theta=factors+[factors[0]],
        fill="toself", fillcolor="rgba(33,150,243,0.2)",
        line=dict(color="#1565C0", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,10])),
        showlegend=False,
        title=f"Factor Breakdown — {row.get('Name','')}",
        height=350, margin=dict(t=60, b=20, l=40, r=40),
    )
    return fig


def signal_badge(sig):
    s = str(sig).upper()
    if "STRONG BUY" in s: return "✅ " + sig
    if "BUY"        in s: return "🟢 " + sig
    if "WATCH"      in s: return "🟡 " + sig
    if "OVERVALUED" in s: return "🔴 " + sig
    return sig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")

    if IS_CLOUD:
        st.info("☁️ Cloud mode\nAI & batch scraping run locally only.")
    st.divider()

    if LOCAL_OK:
        st.subheader("📡 Data Refresh")
        try:
            prog = get_scrape_progress()
            if prog.get("running"):
                total = int(prog.get("total",0))
                done  = int(prog.get("completed",0))
                if total > 0:
                    st.progress(done/total,
                                text=f"Scraping {done}/{total}")
        except Exception:
            pass

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Refresh New", use_container_width=True):
                st.toast(start_batch_scrape(force=False))
                st.cache_data.clear()
        with c2:
            if st.button("🔁 Force All", use_container_width=True):
                st.toast(start_batch_scrape(force=True))
                st.cache_data.clear()
        if st.button("⚡ Quick Refresh (watchlist)", use_container_width=True):
            st.cache_data.clear()
            build_dataset(force_refresh=True)
            st.rerun()
    else:
        if st.button("🔄 Reload Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.subheader("🔍 Filters")
    min_score   = st.slider("Min Score", 0, 100, 40, step=5)
    fii_filter  = st.checkbox("FII Selling 4Q+")
    dii_filter  = st.checkbox("DII Buying 4Q+")
    sig_filter  = st.multiselect("Value Signal",
                    ["STRONG BUY","BUY","WATCH","FAIR VALUE","OVERVALUED"])
    st.divider()
    with st.expander("📐 Score Weights"):
        for f, w in SCORE_WEIGHTS.items():
            st.caption(f"{f.replace('_',' ').title()}: **{w}%**")


# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()

st.title(APP_TITLE)

if df.empty:
    if IS_CLOUD:
        st.warning(
            "No data found. **To fix:** Run `python export_to_csv.py` "
            "locally → commit `data/cloud_data.csv` → push to GitHub."
        )
    else:
        st.warning("No data. Click **Quick Refresh** in sidebar.")
    st.stop()

# Sector filter
with st.sidebar:
    secs = sorted(df["Sector"].unique().tolist()) if "Sector" in df.columns else []
    sectors = st.multiselect("Sectors", options=secs, default=secs,
                              key="sec_filter")

# Apply all filters
flt = df.copy()
if "Smart_Money_Score" in flt.columns:
    flt = flt[flt["Smart_Money_Score"] >= min_score]
if sectors and "Sector" in flt.columns:
    flt = flt[flt["Sector"].isin(sectors)]
if sig_filter and "Value_Signal" in flt.columns:
    flt = flt[flt["Value_Signal"].str.contains(
        "|".join(sig_filter), na=False)]
if fii_filter and "FII_Selling_4Q" in flt.columns:
    flt = flt[flt["FII_Selling_4Q"].astype(bool)]
if dii_filter and "DII_Buying_4Q" in flt.columns:
    flt = flt[flt["DII_Buying_4Q"].astype(bool)]


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_names = ["🏆 Leaderboard","💰 Value Zones",
             "📡 Smart Money Map","🔬 Deep Dive"]
if LOCAL_OK:
    tab_names += ["🤖 AI Assistant","🔔 Alerts"]

tabs = st.tabs(tab_names)
t1,t2,t3,t4 = tabs[0],tabs[1],tabs[2],tabs[3]
t5 = tabs[4] if LOCAL_OK and len(tabs)>4 else None
t6 = tabs[5] if LOCAL_OK and len(tabs)>5 else None


# ══════ TAB 1 — LEADERBOARD ══════════════════════════════════════════════════
with t1:
    buy_n = (flt["Value_Signal"].str.contains("BUY", na=False).sum()
             if "Value_Signal" in flt.columns else 0)
    avg_s = flt["Smart_Money_Score"].mean() if "Smart_Money_Score" in flt.columns else 0
    fii_n = flt["FII_Selling_4Q"].astype(bool).sum() if "FII_Selling_4Q" in flt.columns else 0
    dii_n = flt["DII_Buying_4Q"].astype(bool).sum()  if "DII_Buying_4Q"  in flt.columns else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Stocks",         str(len(flt)))
    c2.metric("Avg Score",      f"{avg_s:.1f}/100")
    c3.metric("In Value Zone",  str(buy_n))
    c4.metric("FII Sell / DII Buy", f"{int(fii_n)} / {int(dii_n)}")

    if not flt.empty:
        st.plotly_chart(chart_leaderboard(flt), use_container_width=True)

    # Table
    show = [c for c in ["Name","Sector","Price","Smart_Money_Score","Grade",
             "PE","PB","RoE","DE","FII_Label","DII_Label",
             "Value_Signal","Buy_Zone_Low","Buy_Zone_High"] if c in flt.columns]

    def _sig_color(val):
        s = str(val)
        if "STRONG BUY" in s: return "background-color:#E8F5E9;color:#1B5E20;font-weight:bold"
        if "BUY"        in s: return "background-color:#F1F8E9;color:#2E7D32"
        if "WATCH"      in s: return "background-color:#FFF9C4;color:#7F4F00"
        if "OVERVALUED" in s: return "background-color:#FFEBEE;color:#B71C1C"
        return ""

    num_fmt = {"Price":"₹{:.0f}","Buy_Zone_Low":"₹{:.0f}",
               "Buy_Zone_High":"₹{:.0f}","PE":"{:.1f}x","PB":"{:.1f}x",
               "RoE":"{:.1f}%","DE":"{:.2f}x","Smart_Money_Score":"{:.0f}"}
    fmt = {k:v for k,v in num_fmt.items() if k in show}

    ren = {"Smart_Money_Score":"Score","Buy_Zone_Low":"Buy Low ₹",
           "Buy_Zone_High":"Buy High ₹","Value_Signal":"Signal",
           "FII_Label":"FII Trend","DII_Label":"DII Trend"}

    try:
        styled = (flt[show].rename(columns=ren)
            .style
            .map(_sig_color, subset=["Signal"] if "Signal" in
                 list(ren.values()) else [])
            .format({ren.get(k,k):v for k,v in fmt.items()})
            .background_gradient(subset=["Score"],
                                 cmap="RdYlGn", vmin=0, vmax=100))
        st.dataframe(styled, use_container_width=True, height=420)
    except Exception:
        st.dataframe(flt[show], use_container_width=True, height=420)


# ══════ TAB 2 — VALUE ZONES ══════════════════════════════════════════════════
with t2:
    st.subheader("Price vs Value Buy Zones")
    if not flt.empty:
        st.plotly_chart(chart_value_zone(flt), use_container_width=True)

    if "Value_Signal" in flt.columns:
        vdf = flt[flt["Value_Signal"].str.contains("BUY|STRONG",na=False,regex=True)]
        if vdf.empty:
            st.info("No stocks in value zone with current filters.")
        else:
            for _, row in vdf.iterrows():
                fv = float(row.get("Fair_Value",0) or 0)
                pr = float(row.get("Price",0) or 0)
                disc = (fv-pr)/fv*100 if fv > 0 else 0
                with st.expander(
                    f"{signal_badge(row.get('Value_Signal',''))}  "
                    f"**{row.get('Name','')}** — ₹{pr:.0f} "
                    f"({disc:+.1f}% vs fair value ₹{fv:.0f})"
                ):
                    a,b,c,d = st.columns(4)
                    a.metric("Price",      f"₹{pr:.0f}")
                    b.metric("Fair Value", f"₹{fv:.0f}")
                    c.metric("Buy Zone",
                             f"₹{float(row.get('Buy_Zone_Low',0) or 0):.0f}"
                             f"–₹{float(row.get('Buy_Zone_High',0) or 0):.0f}")
                    d.metric("Strong Buy",
                             f"₹{float(row.get('Strong_Buy_Below',0) or 0):.0f}")
                    st.caption(str(row.get("Valuation_Methods","")))


# ══════ TAB 3 — SMART MONEY MAP ══════════════════════════════════════════════
with t3:
    st.subheader("FII vs DII Quadrant")
    st.caption("🟢 Top-left = FII selling + DII buying = contrarian signal")
    if not flt.empty:
        st.plotly_chart(chart_fii_dii(flt), use_container_width=True)

    sh = [c for c in ["Name","Promoter_Pct","FII_Pct","DII_Pct",
          "FII_Trend_Pct","DII_Trend_Pct","FII_Selling_4Q","DII_Buying_4Q"]
          if c in flt.columns]
    if sh:
        tmp = flt[sh].copy()
        for col in ["FII_Selling_4Q","DII_Buying_4Q"]:
            if col in tmp.columns:
                tmp[col] = tmp[col].map({True:"✅",False:"—",1:"✅",0:"—"})
        st.dataframe(tmp, use_container_width=True)


# ══════ TAB 4 — DEEP DIVE ════════════════════════════════════════════════════
with t4:
    st.subheader("Stock Deep Dive")
    names = flt["Name"].tolist() if "Name" in flt.columns else []
    if not names:
        st.info("No stocks to display.")
    else:
        sel = st.selectbox("Select stock", options=names)
        row = flt[flt["Name"]==sel].iloc[0]

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Price",      f"₹{float(row.get('Price',0) or 0):.0f}",
                  delta=f"{float(row.get('Price_3M_Ret',0) or 0):+.1f}% (3M)")
        c2.metric("Score",      f"{float(row.get('Smart_Money_Score',0) or 0):.0f}/100")
        c3.metric("Signal",     signal_badge(row.get("Value_Signal","")))
        c4.metric("Fair Value", f"₹{float(row.get('Fair_Value',0) or 0):.0f}")
        c5.metric("Grade",      str(row.get("Grade","—")))

        left, right = st.columns(2)
        with left:
            if any(c.startswith("Score_") for c in row.index):
                st.plotly_chart(chart_radar(row), use_container_width=True)
        with right:
            st.markdown("#### Fundamentals")
            for k,v in {
                "P/E":  f"{float(row.get('PE',0) or 0):.1f}x  (Sector: {float(row.get('Sector_PE',0) or 0):.1f}x)",
                "P/B":  f"{float(row.get('PB',0) or 0):.2f}x",
                "RoE":  f"{float(row.get('RoE',0) or 0):.1f}%",
                "RoCE": f"{float(row.get('RoCE',0) or 0):.1f}%",
                "D/E":  f"{float(row.get('DE',0) or 0):.2f}x",
                "Rev Growth": f"{float(row.get('Revenue_Growth',0) or 0):.1f}%",
                "Mkt Cap": f"₹{float(row.get('Market_Cap_Cr',0) or 0):,.0f} Cr",
            }.items():
                a,b = st.columns([2,3]); a.caption(k); b.write(v)

            st.markdown("#### Smart Money")
            for k,v in {
                "FII": f"{float(row.get('FII_Pct',0) or 0):.1f}% → {row.get('FII_Label','')}",
                "DII": f"{float(row.get('DII_Pct',0) or 0):.1f}% → {row.get('DII_Label','')}",
                "Promoter": f"{float(row.get('Promoter_Pct',0) or 0):.1f}%",
                "Returns": (f"1M: {float(row.get('Price_1M_Ret',0) or 0):+.1f}% | "
                            f"3M: {float(row.get('Price_3M_Ret',0) or 0):+.1f}% | "
                            f"6M: {float(row.get('Price_6M_Ret',0) or 0):+.1f}%"),
                "52W Range": (f"₹{float(row.get('52W_Low',0) or 0):.0f} – "
                              f"₹{float(row.get('52W_High',0) or 0):.0f}"),
            }.items():
                a,b = st.columns([2,3]); a.caption(k); b.write(v)

        st.markdown("#### Value Buy Zone")
        a,b,c,d = st.columns(4)
        a.metric("Price",      f"₹{float(row.get('Price',0) or 0):.0f}")
        b.metric("Fair Value", f"₹{float(row.get('Fair_Value',0) or 0):.0f}")
        c.metric("Buy Zone",   (f"₹{float(row.get('Buy_Zone_Low',0) or 0):.0f}"
                                f"–₹{float(row.get('Buy_Zone_High',0) or 0):.0f}"))
        d.metric("Strong Buy", f"₹{float(row.get('Strong_Buy_Below',0) or 0):.0f}")
        st.caption(str(row.get("Valuation_Methods","")))
        url = str(row.get("Screener_URL",""))
        if url and url.startswith("http"):
            st.link_button("🔗 Open on Screener.in", url)


# ══════ TAB 5 — AI (local only) ══════════════════════════════════════════════
if t5 and LOCAL_OK:
    with t5:
        st.subheader("🤖 AI Assistant (Ollama — local only)")
        try:
            ollama_ok, ollama_msg = check_ollama_status()
            st.caption(ollama_msg)
        except Exception:
            ollama_ok = False
        sel_ai = st.selectbox("Pick stock",
                              options=flt["Name"].tolist() if "Name" in flt.columns else [],
                              key="ai_sel")
        if st.button("Generate Insight", disabled=not ollama_ok):
            ai_row = flt[flt["Name"]==sel_ai].iloc[0]
            with st.spinner(f"Analysing {sel_ai}..."):
                try:
                    st.markdown(f"```\n{generate_stock_insight(ai_row)}\n```")
                except Exception as e:
                    st.error(str(e))
        st.divider()
        st.markdown("### Natural Language Screener")
        q = st.text_area("Ask:", height=80,
                          placeholder="e.g. show high RoE stocks where FII is selling")
        if st.button("Run", disabled=not ollama_ok):
            with st.spinner("Running..."):
                try:
                    st.markdown(natural_language_query(flt, q))
                except Exception as e:
                    st.error(str(e))


# ══════ TAB 6 — ALERTS (local only) ══════════════════════════════════════════
if t6 and LOCAL_OK:
    with t6:
        st.subheader("🔔 Alert Centre")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Check Alerts Now"):
                n = run_all_alerts(flt)
                st.success(f"{n} alert(s) fired." if n else "No new alerts.")
        with c2:
            if st.button("Mark All Read"):
                acknowledge_alerts()
                st.rerun()
        try:
            hist = get_alerts(100)
            if not hist.empty:
                st.dataframe(
                    hist[["fired_at","name","alert_type","message","price","score"]],
                    use_container_width=True, height=400)
            else:
                st.info("No alerts yet.")
        except Exception:
            st.info("Alert history not available.")
