# 🏏 CricketIQ — AI IPL Auction Price Prediction

> **Infosys Springboard 6.0 Internship Project**
> 
> Presented by: Famesh Katre
> 
> Mentor: Pranaya Ma'am
> 
> Duration: 8 Weeks

> **Internship Project** — Dynamic IPL Player Auction Value Prediction using AI and Multi-source Data
> 
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-orange)](https://cricketiq-fvk.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-fameshkatre87-black)](https://github.com/fameshkatre87/CricketIQ)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-orange)](https://streamlit.io)

---

## 📌 Project Overview

CricketIQ is an AI-driven system that predicts IPL player auction prices by integrating **multi-source data** — batting/bowling performance statistics, news sentiment, historical auction prices, and player profile features.

The system uses a stacked ensemble of **LSTM time-series models** and **XGBoost/LightGBM** to produce season-by-season auction price forecasts.

---

## 🏗️ Architecture

```
Data Sources                  Pipeline                     Output
──────────────────────────────────────────────────────────────────
Cricsheet Ball-by-Ball ──┐
IPL Auction History    ──┤── Preprocessing ──► LSTM ──┐
News Sentiment (VADER) ──┤   Feature Eng.              ├── Ensemble ──► ₹Cr Prediction
Player Profiles        ──┘                   XGBoost ──┘
```

---

## 📁 Project Structure

```
CricketIQ/
│
├── data/
│   ├── raw/
│   │   ├── ipl_batting.csv          # Season batting stats (50 players)
│   │   ├── ipl_bowling.csv          # Season bowling stats
│   │   └── ipl_auction.csv          # Historical auction prices 2019–2024
│   ├── processed/
│   │   ├── cricket_feature_matrix.csv
│   │   ├── scaler.pkl
│   │   └── evaluation_report.csv
│   ├── models/
│   │   ├── cricket_lstm.pt
│   │   ├── cricket_xgboost.pkl
│   │   └── cricket_ensemble.pkl
│   └── sentiment/
│       └── ipl_sentiment.csv
│
├── src/
│   ├── data_collection/
│   │   ├── cricsheet_loader.py       # IPL ball-by-ball + auction data
│   │   └── cricket_sentiment.py     # VADER NLP + cricket lexicon
│   │
│   ├── preprocessing/
│   │   └── cricket_feature_engineer.py
│   │
│   ├── models/
│   │   └── cricket_models.py         # LSTM + XGBoost + Ensemble
│   │
│   └── visualization/
│       └── dashboard.py             # Streamlit dashboard
│
├── api/
│   └── main.py                      # FastAPI REST API
│
├── run_pipeline.py                  # Master pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/fameshkatre87/CricketIQ.git
cd CricketIQ

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install packages
pip install -r requirements.txt

# 4. Run full pipeline
python run_pipeline.py

# 5. Launch dashboard
streamlit run src/visualization/dashboard.py

# 6. Start API (optional)
uvicorn api.main:app --reload --port 8000
```

---

## 📊 Data Sources

| Source | Data | Records |
|--------|------|---------|
| Cricsheet (IPL) | Ball-by-ball → batting/bowling stats | 50 players × 6 seasons |
| IPL Auction DB | Historical prices 2019–2024 | 284 auction records |
| News VADER NLP | Cricket sentiment scores | 139 weekly records |
| Player Profiles | Role, nationality, experience | 50 real IPL players |

### Real Players Included (50 total)

**Batters:** Virat Kohli, Rohit Sharma, David Warner, KL Rahul, Shubman Gill, Faf du Plessis, Jos Buttler, Suryakumar Yadav, Ruturaj Gaikwad, Yashasvi Jaiswal, MS Dhoni, AB de Villiers, Rishabh Pant, Sanju Samson, Ishan Kishan, Tilak Varma...

**All-Rounders:** Hardik Pandya, Ravindra Jadeja, Andre Russell, Glenn Maxwell, Ben Stokes, Sam Curran, Axar Patel, Pat Cummins...

**Bowlers:** Jasprit Bumrah, Yuzvendra Chahal, Rashid Khan, Bhuvneshwar Kumar, Mohammed Shami, Trent Boult, Kagiso Rabada, Kuldeep Yadav...

---

## 🔧 Feature Engineering (62 total features)

**Batting:** runs_per_match, impact_score, milestone_score (50s + 100s), consistency_score (avg × SR), boundary_pct, six_pct, dot_ball_risk

**Bowling:** bowling_impact, death_specialist flag, powerplay_bowler flag, wicket_taker_score, economy_score

**All-Rounder:** is_allrounder, allrounder_score, dual_threat_bonus

**Experience:** seasons_played, veteran_flag (≥8 seasons), young_prospect (≤3 seasons), peak_experience flag

**Market:** prev_auction_price, price_trend, log_prev_price, overseas_premium

**Sentiment:** compound_score, positive_pct, negative_pct, value_impact_cr

---

## 📈 Model Results

| Model | RMSE (₹Cr) | MAE (₹Cr) | R² |
|-------|-----------|----------|-----|
| LSTM (Attention) | 5.74 | 5.39 | 0.81 |
| XGBoost | 4.63 | 3.98 | 0.87 |
| LightGBM | 4.70 | 4.05 | 0.86 |
| **Ensemble (Final)** | **4.25** | **3.62** | **0.91** |

---

## 🔌 API Endpoints

```
GET  /                        Health check
GET  /players                 All 50 IPL players
GET  /player/{name}           Player stats + prediction
POST /predict                 Predict auction price
GET  /predict/top             Top value players for 2025
GET  /models/comparison       Model metrics
GET  /features/importance     XGBoost feature importance
GET  /sentiment/{name}        Player sentiment score
```

---

## 🏏 Cricket-Specific VADER Lexicon (28 terms)

Positive additions: `century (+3.5)`, `hat-trick (+3.5)`, `six (+2.0)`, `masterclass (+3.2)`, `match-winning (+3.2)`, `orange cap (+2.8)`

Negative additions: `duck (-2.8)`, `golden duck (-3.2)`, `injured (-2.8)`, `ruled out (-3.0)`, `poor form (-2.5)`, `expensive (-2.0)`

---

## 🗓️ Weekly Milestones

| Week | Task | Status |
|------|------|--------|
| 1 | IPL batting/bowling/auction data collection | ✅ |
| 2 | Feature engineering (62 features) | ✅ |
| 3–4 | Advanced features + sentiment integration | ✅ |
| 5 | LSTM with attention mechanism | ✅ |
| 6 | XGBoost + LightGBM + Ensemble | ✅ |
| 7 | Evaluation + comparison report | ✅ |
| 8 | Streamlit dashboard + API + docs | ✅ |



---

*Built with ❤️ as an Internship Project — CricketIQ v1.0 🏏*
