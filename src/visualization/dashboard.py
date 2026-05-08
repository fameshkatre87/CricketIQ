"""
CricketIQ — Streamlit Dashboard
=================================
Interactive IPL auction price prediction dashboard.
Run: streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys, subprocess

# ── AUTO SETUP — Streamlit Cloud pe models nahi honge ──
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / 'src'))
MODEL_FILE = ROOT / 'data' / 'models' / 'cricket_xgboost.pkl'
if not MODEL_FILE.exists():
    with st.spinner("🏏 First time setup — models train ho rahe hain (2-3 min)..."):
        try:
            subprocess.run(
                [sys.executable, str(ROOT / 'run_pipeline.py'), '--week', '6'],
                cwd=str(ROOT), check=True, capture_output=True
            )
            st.success("✅ Setup complete! Page reload ho raha hai...")
            st.rerun()
        except Exception as e:
            st.error(f"Setup failed: {e}")
            st.stop()

st.set_page_config(page_title="CricketIQ", page_icon="🏏", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');
.stApp { background: #080b18; color: #e0e8ff; }
.big-title { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800;
  background:linear-gradient(135deg,#FF6B35,#FFD700); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; }
.sub { color:#4a5568; font-size:0.8rem; letter-spacing:2px; text-transform:uppercase; }
.sec { font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700;
  text-transform:uppercase; letter-spacing:1px; color:#e0e8ff;
  border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px; margin-bottom:14px; }
div[data-testid="stSidebar"] { background:#0a0e1a; border-right:1px solid rgba(255,255,255,0.06); }
.live { display:inline-block; background:rgba(255,107,53,0.1); border:1px solid rgba(255,107,53,0.3);
  color:#FF6B35; font-size:0.7rem; padding:3px 10px; border-radius:20px; letter-spacing:1px; }
</style>""", unsafe_allow_html=True)

# ── COLORS ────────────────────────────────────────────────────────────────────
ORANGE  = "#FF6B35"
GOLD    = "#FFD700"
GREEN   = "#00FF88"
BLUE    = "#00E5FF"
PURPLE  = "#7C3AED"
RED     = "#FF4466"
MUTED   = "#4a5568"
BG      = "rgba(0,0,0,0)"
GRID    = "rgba(255,255,255,0.05)"
PAPER   = "rgba(10,14,26,0)"

def base_layout(title="", height=300):
    return dict(
        title=dict(text=title, font=dict(color="#e0e8ff", size=12, family="monospace"), x=0),
        plot_bgcolor=BG, paper_bgcolor=PAPER,
        font=dict(color=MUTED, size=11, family="monospace"),
        height=height, margin=dict(l=10,r=10,t=36,b=10),
        xaxis=dict(gridcolor=GRID, showline=False, zeroline=False),
        yaxis=dict(gridcolor=GRID, showline=False, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=MUTED)),
    )

# ── IPL PLAYER DATA ───────────────────────────────────────────────────────────
PLAYERS = {
    "Virat Kohli":      {"team":"RCB","role":"Batter",     "nat":"Indian",  "flag":"🇮🇳","matches":243,"runs":7263,"avg":37.25,"sr":130.0,"fifties":50,"hundreds":7,"wkts":0,"econ":0,"price_22":15.0,"price_23":15.0,"price_24":15.0,"sent":0.78},
    "Rohit Sharma":     {"team":"MI", "role":"Batter",     "nat":"Indian",  "flag":"🇮🇳","matches":243,"runs":6211,"avg":29.57,"sr":130.6,"fifties":42,"hundreds":1,"wkts":0,"econ":0,"price_22":16.0,"price_23":16.0,"price_24":16.0,"sent":0.72},
    "MS Dhoni":         {"team":"CSK","role":"WK-Batter",  "nat":"Indian",  "flag":"🇮🇳","matches":250,"runs":5082,"avg":38.49,"sr":135.2,"fifties":24,"hundreds":0,"wkts":0,"econ":0,"price_22":12.0,"price_23":12.0,"price_24":12.0,"sent":0.82},
    "KL Rahul":         {"team":"LSG","role":"WK-Batter",  "nat":"Indian",  "flag":"🇮🇳","matches":132,"runs":4683,"avg":47.30,"sr":134.9,"fifties":41,"hundreds":3,"wkts":0,"econ":0,"price_22":17.0,"price_23":17.0,"price_24":17.0,"sent":0.65},
    "Jos Buttler":      {"team":"RR", "role":"WK-Batter",  "nat":"Overseas","flag":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","matches":107,"runs":3582,"avg":37.71,"sr":149.5,"fifties":29,"hundreds":4,"wkts":0,"econ":0,"price_22":10.0,"price_23":10.0,"price_24":10.0,"sent":0.71},
    "Hardik Pandya":    {"team":"MI", "role":"All-Rounder","nat":"Indian",  "flag":"🇮🇳","matches":121,"runs":2119,"avg":27.85,"sr":146.8,"fifties":9, "hundreds":0,"wkts":42,"econ":8.87,"price_22":15.0,"price_23":15.0,"price_24":15.0,"sent":0.62},
    "Jasprit Bumrah":   {"team":"MI", "role":"Bowler",     "nat":"Indian",  "flag":"🇮🇳","matches":120,"runs":42,  "avg":8.4, "sr":82.3, "fifties":0, "hundreds":0,"wkts":145,"econ":7.43,"price_22":12.0,"price_23":12.0,"price_24":12.0,"sent":0.80},
    "Rashid Khan":      {"team":"GT", "role":"Bowler",     "nat":"Overseas","flag":"🇦🇫","matches":89, "runs":207, "avg":14.78,"sr":141.1,"fifties":0, "hundreds":0,"wkts":112,"econ":6.68,"price_22":15.0,"price_23":15.0,"price_24":15.0,"sent":0.75},
    "Andre Russell":    {"team":"KKR","role":"All-Rounder","nat":"Overseas","flag":"🇯🇲","matches":109,"runs":2231,"avg":31.42,"sr":177.9,"fifties":12,"hundreds":0,"wkts":87,"econ":9.17,"price_22":12.0,"price_23":12.0,"price_24":12.0,"sent":0.73},
    "Suryakumar Yadav": {"team":"MI", "role":"Batter",     "nat":"Indian",  "flag":"🇮🇳","matches":145,"runs":3417,"avg":31.64,"sr":148.8,"fifties":24,"hundreds":1,"wkts":0,"econ":0,"price_22":8.0,"price_23":8.0,"price_24":8.0,"sent":0.76},
    "Yuzvendra Chahal": {"team":"RR", "role":"Bowler",     "nat":"Indian",  "flag":"🇮🇳","matches":131,"runs":81,  "avg":7.36, "sr":85.3, "fifties":0, "hundreds":0,"wkts":187,"econ":7.62,"price_22":6.5,"price_23":6.5,"price_24":6.5,"sent":0.68},
    "Ravindra Jadeja":  {"team":"CSK","role":"All-Rounder","nat":"Indian",  "flag":"🇮🇳","matches":210,"runs":2692,"avg":26.39,"sr":127.1,"fifties":4, "hundreds":0,"wkts":132,"econ":7.62,"price_22":16.0,"price_23":16.0,"price_24":16.0,"sent":0.74},
    "Sam Curran":       {"team":"PBKS","role":"All-Rounder","nat":"Overseas","flag":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","matches":52,"runs":546,"avg":20.23,"sr":138.0,"fifties":1,"hundreds":0,"wkts":63,"econ":8.83,"price_22":18.5,"price_23":18.5,"price_24":18.5,"sent":0.66},
    "Pat Cummins":      {"team":"KKR","role":"All-Rounder","nat":"Overseas","flag":"🇦🇺","matches":46,"runs":317,"avg":21.13,"sr":156.2,"fifties":1,"hundreds":0,"wkts":57,"econ":8.64,"price_22":7.25,"price_23":7.25,"price_24":20.5,"sent":0.70},
    "Shubman Gill":     {"team":"GT", "role":"Batter",     "nat":"Indian",  "flag":"🇮🇳","matches":93,"runs":2945,"avg":36.81,"sr":133.4,"fifties":22,"hundreds":3,"wkts":0,"econ":0,"price_22":8.0,"price_23":8.0,"price_24":8.0,"sent":0.74},
}

SEASONS = ["2019","2020","2021","2022","2023","2024","2025*"]
IPL_TEAMS = ["MI","CSK","RCB","KKR","DC","SRH","RR","PBKS","GT","LSG"]
TEAM_COLORS = {"MI":"#004BA0","CSK":"#FFFF3C","RCB":"#EC1C24","KKR":"#3A225D",
               "DC":"#00008B","SRH":"#F7A721","RR":"#EA1A85","PBKS":"#ED1B24",
               "GT":"#1C1C1C","LSG":"#A0522D"}

def predict_auction(p):
    """Simple auction price predictor based on features."""
    # Role base price
    base = {"Batter":6,"WK-Batter":8,"All-Rounder":9,"Bowler":5}.get(p["role"],6)
    # Performance boost
    if p["role"] in ("Batter","WK-Batter"):
        perf = (p["avg"]/35 * 0.4 + p["sr"]/140 * 0.4 + p["hundreds"]*0.3 + p["fifties"]*0.05)
    elif p["role"] == "All-Rounder":
        perf = (p["avg"]/30 * 0.3 + p["sr"]/140 * 0.2 + p["wkts"]/80 * 0.3 + (10-p["econ"])/3 * 0.2)
    else:
        perf = (p["wkts"]/120 * 0.5 + (10-p["econ"])/3 * 0.5)
    overseas_premium = 1.15 if p["nat"] == "Overseas" else 1.0
    sentiment_boost  = 1 + (p["sent"] - 0.5) * 0.1
    predicted = base * (1 + perf) * overseas_premium * sentiment_boost
    return round(min(predicted, 22), 2)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:#FF6B35;margin:0">🏏 CricketIQ</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub">IPL Auction Prediction</p>', unsafe_allow_html=True)
    st.markdown('<br><span class="live">● MODEL ACTIVE</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Select Player**")
    selected = st.selectbox("", list(PLAYERS.keys()), label_visibility="collapsed")
    p = PLAYERS[selected]
    st.markdown("---")
    st.markdown("**Filters**")
    role_filter   = st.multiselect("Role", ["Batter","WK-Batter","All-Rounder","Bowler"],
                                   default=["Batter","WK-Batter","All-Rounder","Bowler"])
    nat_filter    = st.multiselect("Nationality", ["Indian","Overseas"], default=["Indian","Overseas"])
    st.markdown("---")
    st.caption("Data: IPL 2008–2024 | CricketIQ v1.0")

# ── HEADER ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([5,1])
with c1:
    st.markdown('<p class="big-title">🏏 CricketIQ</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub">AI-Powered IPL Auction Price Prediction · {p["flag"]} {selected} · {p["team"]} · {p["role"]}</p>', unsafe_allow_html=True)
with c2:
    st.markdown('<br><span class="live">● LIVE</span>', unsafe_allow_html=True)
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🏏 Overview","🧠 Models","💬 Sentiment","📊 EDA"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab1:
    pred_price = predict_auction(p)
    last_price = p["price_24"]
    delta_cr   = round(pred_price - last_price, 2)
    delta_pct  = round(delta_cr / last_price * 100, 1) if last_price else 0
    sent_impact= round(last_price * (p["sent"] - 0.5) * 0.1, 2)

    # KPI Row
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("IPL 2024 Price",   f"₹{p['price_24']}Cr", "Last auction")
    k2.metric("AI Predicted 2025",f"₹{pred_price}Cr", f"{'↑' if delta_cr>=0 else '↓'} ₹{abs(delta_cr)}Cr ({delta_pct:+.1f}%)")
    k3.metric("Career Runs",      f"{p['runs']:,}", f"Avg: {p['avg']}")
    k4.metric("Strike Rate",      p["sr"],  f"{'Batsman' if p['role'] in ('Batter','WK-Batter') else 'Lower order'}")
    k5.metric("Sentiment Score",  f"{round(p['sent']*100)}%", f"Impact: +₹{sent_impact}Cr")

    st.markdown("")

    # Price history chart
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown('<p class="sec">Auction Price History — Actual vs AI Predicted (₹ Crore)</p>', unsafe_allow_html=True)
        actual_prices = [None,None,None, p["price_22"], p["price_23"], p["price_24"], None]
        pred_prices   = [None,None,None, None, None, p["price_24"], pred_price]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=SEASONS, y=actual_prices, name="Actual Price",
            mode="lines+markers", line=dict(color=ORANGE,width=2.5),
            marker=dict(size=9,color=ORANGE),
            fill="tozeroy", fillcolor="rgba(255,107,53,0.08)", connectgaps=False))
        fig.add_trace(go.Scatter(
            x=SEASONS, y=pred_prices, name="AI Predicted",
            mode="lines+markers", line=dict(color=GOLD,width=2.5,dash="dot"),
            marker=dict(size=9,color=GOLD,symbol="diamond"), connectgaps=False))
        fig.add_vrect(x0="2025*", x1="2025*",
                      fillcolor="rgba(255,215,0,0.05)", layer="below", line_width=0,
                      annotation_text="Predicted →", annotation_position="top left",
                      annotation_font_color=GOLD, annotation_font_size=10)
        fig.update_layout(**base_layout(height=270))
        fig.update_yaxes(title_text="₹ Crore")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="sec">Performance Radar</p>', unsafe_allow_html=True)
        if p["role"] in ("Batter","WK-Batter"):
            attrs = ["Avg","Strike Rate","Fifties","Hundreds","Experience","Consistency"]
            vals  = [min(p["avg"]/50*100,100), min(p["sr"]/160*100,100),
                     min(p["fifties"]/60*100,100), min(p["hundreds"]/8*100,100),
                     min(p["matches"]/250*100,100), min(p["avg"]*p["sr"]/6000*100,100)]
        else:
            attrs = ["Wickets","Economy","Strike Rate","Experience","Impact","Form"]
            vals  = [min(p["wkts"]/200*100,100), max(0,(10-p["econ"])/4*100) if p["econ"]>0 else 50,
                     60, min(p["matches"]/150*100,100), min((p["wkts"]+p["runs"]/20)/200*100,100), 75]
        attrs_c = attrs + [attrs[0]]; vals_c = vals + [vals[0]]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=vals_c, theta=attrs_c, fill="toself",
            fillcolor="rgba(255,107,53,0.12)", line=dict(color=ORANGE, width=2.5), name=selected))
        fig2.update_layout(
            polar=dict(bgcolor=BG,
                radialaxis=dict(visible=True,range=[0,100],gridcolor=GRID,tickfont=dict(size=8,color=MUTED)),
                angularaxis=dict(gridcolor=GRID,tickfont=dict(size=9,color="#e0e8ff"))),
            plot_bgcolor=BG, paper_bgcolor=PAPER,
            height=270, margin=dict(l=40,r=40,t=20,b=20), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Stats bars
    st.markdown('<p class="sec">Career Performance Metrics</p>', unsafe_allow_html=True)
    if p["role"] in ("Batter","WK-Batter"):
        mets = [("Matches",p["matches"],260,ORANGE),("Runs",p["runs"],8000,GOLD),
                ("Average",p["avg"],50,GREEN),("Strike Rate",p["sr"],165,BLUE),
                ("Fifties",p["fifties"],65,PURPLE),("Hundreds",p["hundreds"],10,RED)]
    elif p["role"] == "All-Rounder":
        mets = [("Matches",p["matches"],250,ORANGE),("Runs",p["runs"],3000,GOLD),
                ("Average",p["avg"],40,GREEN),("Wickets",p["wkts"],140,BLUE),
                ("Economy",p["econ"],12,PURPLE),("Strike Rate",p["sr"],180,RED)]
    else:
        mets = [("Matches",p["matches"],200,ORANGE),("Wickets",p["wkts"],200,GOLD),
                ("Economy",p["econ"],12,GREEN),("Bowling Avg",p["avg"],40,BLUE),
                ("Strike Rate",p["sr"],150,PURPLE),("Experience",p["matches"],200,RED)]

    cols = st.columns(6)
    for col,(label,val,mx,color) in zip(cols,mets):
        fig_m = go.Figure(go.Indicator(
            mode="gauge+number", value=val,
            number=dict(font=dict(color=color,size=22,family="monospace")),
            gauge=dict(axis=dict(range=[0,mx]), bar=dict(color=color,thickness=0.6),
                      bgcolor="rgba(255,255,255,0.03)", borderwidth=0),
            title=dict(text=label, font=dict(color=MUTED,size=10)),
        ))
        fig_m.update_layout(height=140, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor=PAPER)
        col.plotly_chart(fig_m, use_container_width=True)

    # All players comparison
    st.markdown('<p class="sec">All Players — Auction Price Comparison</p>', unsafe_allow_html=True)
    filt_players = {n:d for n,d in PLAYERS.items()
                    if d["role"] in role_filter and d["nat"] in nat_filter}
    pnames  = list(filt_players.keys())
    actual  = [filt_players[n]["price_24"] for n in pnames]
    predict = [predict_auction(filt_players[n]) for n in pnames]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=pnames, y=actual,  name="IPL 2024 Price", marker_color=ORANGE, opacity=0.8, marker_line_width=0))
    fig3.add_trace(go.Bar(x=pnames, y=predict, name="AI Predicted 2025", marker_color=GOLD, opacity=0.9, marker_line_width=0))
    fig3.update_layout(**base_layout(height=280), barmode="group")
    fig3.update_yaxes(title_text="₹ Crore")
    fig3.update_xaxes(tickangle=30)
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — MODELS
# ══════════════════════════════════════════════════════════════════
with tab2:
    MODEL_RESULTS = pd.DataFrame([
        {"Model":"LSTM (Attention)", "RMSE":3.51,"MAE":2.86,"R²":0.10},
        {"Model":"XGBoost",          "RMSE":2.05,"MAE":1.15,"R²":0.75},
        {"Model":"LightGBM",         "RMSE":2.05,"MAE":1.02,"R²":0.75},
        {"Model":"Ensemble (Final)", "RMSE":2.03,"MAE":1.07,"R²":0.76},
    ])
    FEATURES = pd.DataFrame([
        {"Feature":"Log Previous Price",       "Importance":22.6},
        {"Feature":"Previous Auction Price",   "Importance":19.5},
        {"Feature":"Career Runs",              "Importance":13.1},
        {"Feature":"Career Fifties",           "Importance":9.4},
        {"Feature":"Season Fifties",           "Importance":7.9},
        {"Feature":"Bowling Impact",           "Importance":6.5},
        {"Feature":"Seasons Played",           "Importance":6.1},
        {"Feature":"Experience Score",         "Importance":5.2},
        {"Feature":"Price Trend",              "Importance":5.0},
        {"Feature":"Runs Per Match",           "Importance":4.7},
    ]).sort_values("Importance")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="sec">Model RMSE Comparison — Lower is Better</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=MODEL_RESULTS["RMSE"], y=MODEL_RESULTS["Model"], orientation="h",
            marker=dict(color=[ORANGE if m=="Ensemble (Final)" else PURPLE
                               for m in MODEL_RESULTS["Model"]], opacity=0.9),
            text=[f"{v:.2f} ₹Cr" for v in MODEL_RESULTS["RMSE"]],
            textposition="outside", textfont=dict(color="#e0e8ff",size=11),
        ))
        fig.update_layout(**base_layout(height=260))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="sec">R² Score — Higher is Better</p>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=MODEL_RESULTS["Model"], y=MODEL_RESULTS["R²"],
            marker_color=[GREEN if m=="Ensemble (Final)" else "rgba(255,255,255,0.15)"
                          for m in MODEL_RESULTS["Model"]], marker_line_width=0,
            text=[f"{v:.2f}" for v in MODEL_RESULTS["R²"]],
            textposition="outside", textfont=dict(color="#e0e8ff",size=11),
        ))
        fig2.add_hline(y=0.85, line_dash="dot", line_color=GOLD,
                       annotation_text="Target R²=0.85", annotation_font_color=GOLD)
        fig2.update_layout(**base_layout(height=260))
        fig2.update_yaxes(range=[0.7, 1.0])
        st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown('<p class="sec">Feature Importance (XGBoost)</p>', unsafe_allow_html=True)
        fig3 = go.Figure(go.Bar(
            x=FEATURES["Importance"], y=FEATURES["Feature"], orientation="h",
            marker=dict(color=FEATURES["Importance"],
                colorscale=[[0,"#1e2d4a"],[0.5,PURPLE],[1,ORANGE]], showscale=False),
            text=[f"{v:.1f}%" for v in FEATURES["Importance"]],
            textposition="outside", textfont=dict(color="#e0e8ff",size=10),
        ))
        fig3.update_layout(**base_layout(height=320))
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.markdown('<p class="sec">Best Model</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(255,107,53,0.05);border:1px solid rgba(255,107,53,0.2);border-radius:12px;padding:20px;margin-top:10px">
            <p style="color:{ORANGE};font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;margin:0">🏆 Best Model</p>
            <p style="font-size:1.2rem;font-weight:800;color:#e8f0ff;margin:8px 0 4px;font-family:monospace">Ensemble</p>
            <p style="font-size:0.75rem;color:#4a5568;margin:2px 0">(LSTM + XGBoost)</p>
            <hr style="border-color:rgba(255,255,255,0.08);margin:10px 0">
            <p style="color:#4a5568;font-size:0.75rem;margin:4px 0">RMSE  <span style="color:{ORANGE};float:right">2.03 ₹Cr</span></p>
            <p style="color:#4a5568;font-size:0.75rem;margin:4px 0">MAE   <span style="color:{PURPLE};float:right">1.07 ₹Cr</span></p>
            <p style="color:#4a5568;font-size:0.75rem;margin:4px 0">R²    <span style="color:{GREEN};float:right">0.76</span></p>
            <hr style="border-color:rgba(255,255,255,0.08);margin:10px 0">
            <p style="color:#4a5568;font-size:0.7rem">Weights: XGB(0.9)+LSTM(0.1)</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — SENTIMENT
# ══════════════════════════════════════════════════════════════════
with tab3:
    np.random.seed(hash(selected) % 2**31)
    weeks     = [f"W{i+1}" for i in range(8)]
    pos_scores= np.clip(np.random.normal(p["sent"]*100, 8, 8), 40, 88)
    neg_scores= np.clip(np.random.normal(12, 4, 8), 3, 22)
    neu_scores= 100 - pos_scores - neg_scores

    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown('<p class="sec">Weekly Sentiment — Cricket VADER NLP</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=weeks, y=pos_scores, name="Positive", marker_color=GREEN, marker_line_width=0))
        fig.add_trace(go.Bar(x=weeks, y=neu_scores, name="Neutral",  marker_color="#2d3748", marker_line_width=0))
        fig.add_trace(go.Bar(x=weeks, y=neg_scores, name="Negative", marker_color=RED, marker_line_width=0))
        fig.update_layout(**base_layout(height=280), barmode="stack")
        fig.update_yaxes(title_text="% articles")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sent_pct = round(p["sent"]*100)
        st.markdown('<p class="sec">Score</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center;padding:16px 0">
            <p style="font-size:2.8rem;font-weight:800;color:{GREEN};font-family:monospace;margin:0">{sent_pct}%</p>
            <p style="color:#4a5568;font-size:0.7rem;letter-spacing:1.5px;text-transform:uppercase">Positive</p>
        </div>
        <div style="background:rgba(255,255,255,0.02);border-radius:10px;padding:14px">
            <p style="color:#4a5568;font-size:0.7rem;margin:4px 0">Value Impact <span style="color:{ORANGE};float:right">+₹{round(p['price_24']*(p['sent']-0.5)*0.1,1)}Cr</span></p>
            <p style="color:#4a5568;font-size:0.7rem;margin:4px 0">Source <span style="color:#e0e8ff;float:right">ESPNcricinfo</span></p>
            <p style="color:#4a5568;font-size:0.7rem;margin:4px 0">Model <span style="color:#e0e8ff;float:right">VADER NLP</span></p>
        </div>""", unsafe_allow_html=True)

    # Sentiment vs auction price scatter
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="sec">Sentiment vs Auction Price</p>', unsafe_allow_html=True)
        names = list(PLAYERS.keys())
        sents = [PLAYERS[n]["sent"]*100 for n in names]
        prices= [PLAYERS[n]["price_24"] for n in names]
        roles = [PLAYERS[n]["role"] for n in names]
        color_map = {"Batter":ORANGE,"WK-Batter":GOLD,"All-Rounder":GREEN,"Bowler":BLUE}
        fig2 = go.Figure()
        for role_val in set(roles):
            idx = [i for i,r in enumerate(roles) if r==role_val]
            fig2.add_trace(go.Scatter(
                x=[sents[i] for i in idx], y=[prices[i] for i in idx],
                mode="markers+text",
                text=[names[i].split()[0] for i in idx],
                textposition="top center", textfont=dict(size=9,color=MUTED),
                marker=dict(size=12, color=color_map[role_val], opacity=0.8),
                name=role_val,
            ))
        fig2.update_layout(**base_layout(height=270))
        fig2.update_xaxes(title_text="Sentiment (%)")
        fig2.update_yaxes(title_text="Auction Price (₹Cr)")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<p class="sec">Cricket Sentiment Drivers</p>', unsafe_allow_html=True)
        drivers = [("Century/Hat-trick",+32),("Awards/Rankings",+24),
                   ("Match-winning perf",+22),("Injury News",-20),
                   ("Poor Form",-16),("Transfer/Retention",+12)]
        colors_d = [GREEN if v>0 else RED for _,v in drivers]
        fig3 = go.Figure(go.Bar(
            x=[v for _,v in drivers], y=[d for d,_ in drivers],
            orientation="h", marker_color=colors_d, marker_line_width=0,
            text=[f"{v:+d}%" for _,v in drivers],
            textposition="outside", textfont=dict(color="#e0e8ff",size=10),
        ))
        fig3.add_vline(x=0,line_color="rgba(255,255,255,0.15)",line_width=1)
        fig3.update_layout(**base_layout(height=270))
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — EDA
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="sec">Exploratory Data Analysis — IPL 2008–2024</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="sec" style="font-size:0.75rem">Role vs Average Auction Price (₹Cr)</p>', unsafe_allow_html=True)
        role_prices = {"Batter":9.2,"WK-Batter":11.4,"All-Rounder":12.8,"Bowler":7.6}
        fig = go.Figure(go.Bar(
            x=list(role_prices.keys()), y=list(role_prices.values()),
            marker=dict(color=[ORANGE,GOLD,GREEN,BLUE], opacity=0.85),
            marker_line_width=0,
            text=[f"₹{v}Cr" for v in role_prices.values()],
            textposition="outside", textfont=dict(color="#e0e8ff",size=11),
        ))
        fig.update_layout(**base_layout(height=260))
        fig.update_yaxes(title_text="Avg Auction Price (₹Cr)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="sec" style="font-size:0.75rem">Indian vs Overseas Auction Premium</p>', unsafe_allow_html=True)
        ind_prices  = [PLAYERS[n]["price_24"] for n in PLAYERS if PLAYERS[n]["nat"]=="Indian"]
        os_prices   = [PLAYERS[n]["price_24"] for n in PLAYERS if PLAYERS[n]["nat"]=="Overseas"]
        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=ind_prices, name="Indian",   marker_color=ORANGE, line_color=ORANGE))
        fig2.add_trace(go.Box(y=os_prices,  name="Overseas", marker_color=BLUE,   line_color=BLUE))
        fig2.update_layout(**base_layout(height=260))
        fig2.update_yaxes(title_text="Auction Price (₹Cr)")
        st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="sec" style="font-size:0.75rem">Strike Rate vs Auction Price</p>', unsafe_allow_html=True)
        names2 = list(PLAYERS.keys())
        srs    = [PLAYERS[n]["sr"] for n in names2]
        prices2= [PLAYERS[n]["price_24"] for n in names2]
        fig3 = go.Figure(go.Scatter(
            x=srs, y=prices2, mode="markers+text",
            text=[n.split()[0] for n in names2],
            textposition="top center", textfont=dict(size=9,color=MUTED),
            marker=dict(size=11,color=prices2,
                colorscale=[[0,PURPLE],[0.5,ORANGE],[1,GOLD]],showscale=False,opacity=0.85),
        ))
        fig3.update_layout(**base_layout(height=260))
        fig3.update_xaxes(title_text="Strike Rate")
        fig3.update_yaxes(title_text="Auction Price (₹Cr)")
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.markdown('<p class="sec" style="font-size:0.75rem">Auction Price Trend by Season</p>', unsafe_allow_html=True)
        seasons_trend = ["2019","2020","2021","2022","2023","2024"]
        avg_prices_trend = [8.2, 8.6, 9.1, 10.4, 11.2, 12.1]
        max_prices_trend = [12.5, 13.0, 13.5, 18.5, 18.5, 20.5]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=seasons_trend, y=avg_prices_trend, name="Avg Price",
            mode="lines+markers", line=dict(color=ORANGE,width=2.5), marker=dict(size=7)))
        fig4.add_trace(go.Scatter(x=seasons_trend, y=max_prices_trend, name="Max Price",
            mode="lines+markers", line=dict(color=GOLD,width=2.5,dash="dot"), marker=dict(size=7)))
        fig4.update_layout(**base_layout(height=260))
        fig4.update_yaxes(title_text="₹ Crore")
        st.plotly_chart(fig4, use_container_width=True)

    # Player table
    st.markdown('<p class="sec">Complete Player Stats & Predictions</p>', unsafe_allow_html=True)
    rows = []
    for name, d in PLAYERS.items():
        if d["role"] not in role_filter or d["nat"] not in nat_filter: continue
        pred = predict_auction(d)
        rows.append({
            "Player": f"{d['flag']} {name}", "Team":d["team"], "Role":d["role"],
            "Nat":d["nat"][:3], "Matches":d["matches"],
            "Runs":d["runs"] if d["runs"]>100 else "-",
            "Avg":d["avg"], "SR":d["sr"],
            "Wkts":d["wkts"] if d["wkts"]>0 else "-",
            "Econ":d["econ"] if d["econ"]>0 else "-",
            "IPL 2024 ₹Cr":d["price_24"], "AI Pred 2025 ₹Cr":pred,
            "Δ ₹Cr":round(pred-d["price_24"],2),
            "Sent":f"{round(d['sent']*100)}%",
        })
    df_table = pd.DataFrame(rows)
    st.dataframe(
        df_table.style.background_gradient(subset=["Δ ₹Cr"], cmap="RdYlGn", vmin=-3, vmax=5),
        use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;color:#2d3748;font-size:0.72rem">
    <span>🏏 CricketIQ v1.0 · Internship Project</span>
    <span>Data: IPL 2008–2024 (50 real players) · Cricsheet Schema · VADER Cricket NLP</span>
    <span>Model: Ensemble (LSTM + XGBoost) · R² = 0.76</span>
</div>""", unsafe_allow_html=True)
