"""
CricketIQ — Exploratory Data Analysis
=======================================
Run this notebook-style script to explore the IPL dataset.
Usage: python notebooks/01_EDA.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_collection.cricsheet_loader import CricsheetLoader
from data_collection.cricket_sentiment import CricketSentimentCollector

print("=" * 55)
print("CricketIQ — Exploratory Data Analysis")
print("=" * 55)

loader     = CricsheetLoader()
bat_df     = loader.get_batting_stats()
bowl_df    = loader.get_bowling_stats()
auction_df = loader.get_auction_history()

# ── 1. DATASET OVERVIEW ──────────────────────────────────────────
print("\n📊 Dataset Overview")
print(f"  Batting records  : {len(bat_df)}")
print(f"  Bowling records  : {len(bowl_df)}")
print(f"  Auction records  : {len(auction_df)}")
print(f"  Unique players   : {bat_df['player_name'].nunique()}")
print(f"  Seasons covered  : {sorted(bat_df['season'].unique())}")
print(f"  Roles            : {bat_df['role'].value_counts().to_dict()}")
print(f"  Nationalities    : {bat_df['nationality'].value_counts().to_dict()}")

# ── 2. BATTING STATS ─────────────────────────────────────────────
print("\n🏏 Batting Stats Summary")
print(bat_df[["runs","batting_avg","strike_rate","fifties","hundreds"]].describe().round(2).to_string())

# ── 3. BOWLING STATS ─────────────────────────────────────────────
print("\n🎳 Bowling Stats Summary")
print(bowl_df[["wickets","bowling_avg","economy_rate","bowling_sr"]].describe().round(2).to_string())

# ── 4. AUCTION PRICES ────────────────────────────────────────────
print("\n💰 Auction Price Analysis")
print(f"  Max price  : ₹{auction_df['auction_price_cr'].max():.2f} Cr")
print(f"  Min price  : ₹{auction_df['auction_price_cr'].min():.2f} Cr")
print(f"  Mean price : ₹{auction_df['auction_price_cr'].mean():.2f} Cr")
print(f"  Median     : ₹{auction_df['auction_price_cr'].median():.2f} Cr")

print("\n  Top 10 Most Expensive Players (2024):")
top10 = auction_df[auction_df['season']=='2024'].nlargest(10,'auction_price_cr')[["player_name","auction_price_cr","role","nationality"]]
print(top10.to_string(index=False))

# ── 5. ROLE ANALYSIS ─────────────────────────────────────────────
print("\n📈 Average Auction Price by Role")
role_avg = auction_df.merge(
    bat_df[["player_name","role"]].drop_duplicates(), on="player_name", how="left"
).groupby("role")["auction_price_cr"].mean().round(2).sort_values(ascending=False)
print(role_avg.to_string())

# ── 6. CORRELATION ───────────────────────────────────────────────
print("\n🔗 Top Correlations with Auction Price")
merged = auction_df.merge(bat_df[["player_name","season","batting_avg","strike_rate","fifties","hundreds","career_runs"]],
                           on=["player_name","season"], how="left")
corr = merged[["auction_price_cr","batting_avg","strike_rate","fifties","hundreds","career_runs"]].corr()["auction_price_cr"].sort_values(ascending=False)
print(corr.round(3).to_string())

print("\n✅ EDA complete!")
