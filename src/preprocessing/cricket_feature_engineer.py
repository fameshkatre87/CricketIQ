"""
Cricket Feature Engineer — CricketIQ
=======================================
Merges batting + bowling + auction + sentiment data.
Creates ML-ready feature matrix for LSTM and XGBoost models.
Target variable: auction_price_cr (IPL auction price in ₹ Crore)

Usage:
    fe = CricketFeatureEngineer()
    df = fe.build_feature_matrix(bat_df, bowl_df, auction_df, sentiment_df)
    X_train, X_val, X_test, y_train, y_val, y_test = fe.train_val_test_split(df)
"""

import pandas as pd
import numpy as np
import logging, pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class CricketFeatureEngineer:

    def __init__(self):
        self.scaler   = StandardScaler()
        self.encoders = {}
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "processed").mkdir(exist_ok=True)

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def build_feature_matrix(self, bat_df, bowl_df, auction_df,
                              sentiment_df=None) -> pd.DataFrame:
        log.info("Building cricket feature matrix...")
        df = self._merge_all(bat_df, bowl_df, auction_df, sentiment_df)
        df = self._handle_missing(df)
        df = self._encode_categoricals(df)
        df = self._batting_features(df)
        df = self._bowling_features(df)
        df = self._allrounder_features(df)
        df = self._age_experience_features(df)
        df = self._auction_trend_features(df, auction_df)
        df = self._create_target(df)
        log.info(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude  = ["player_id", "target", "auction_price_cr"]
        num_cols = [c for c in num_cols if c not in exclude]
        if fit:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
        return df

    def train_val_test_split(self, df, val_frac=0.15, test_frac=0.15,
                             random_state=42):
        """Random stratified split — better for small datasets."""
        from sklearn.model_selection import train_test_split as _tts
        import numpy as _np
        num_cols = df.select_dtypes(include=[_np.number]).columns.tolist()
        excl = ["target", "auction_price_cr"]
        feat_cols = [c for c in num_cols if c not in excl]
        X = _np.nan_to_num(df[feat_cols].values.astype("float32"), nan=0.0)
        y = df["target"].values.astype("float32")
        X_tr, X_tmp, y_tr, y_tmp = _tts(
            X, y, test_size=val_frac + test_frac, random_state=random_state)
        X_val, X_te, y_val, y_te = _tts(
            X_tmp, y_tmp, test_size=0.5, random_state=random_state)
        return X_tr, X_val, X_te, y_tr, y_val, y_te

    def build_lstm_sequences(self, auction_df, bat_df=None,
                              bowl_df=None, seq_len=3) -> tuple:
        """Build time-series sequences per player for LSTM."""
        log.info(f"Building LSTM sequences (seq_len={seq_len})...")
        seasons_order = ["2019","2020","2021","2022","2023","2024"]
        sequences_X, sequences_y = [], []

        # Merge batting into auction
        if bat_df is not None:
            bat_agg = bat_df.groupby(["player_name","season"]).agg(
                runs=("runs","sum"), batting_avg=("batting_avg","mean"),
                strike_rate=("strike_rate","mean"), fifties=("fifties","sum"),
            ).reset_index()
            merged = auction_df.merge(bat_agg, on=["player_name","season"], how="left")
        else:
            merged = auction_df.copy()
            merged["runs"] = 0; merged["batting_avg"] = 0; merged["strike_rate"] = 0

        merged["season_order"] = merged["season"].apply(
            lambda s: seasons_order.index(s) if s in seasons_order else -1)
        merged = merged[merged["season_order"] >= 0]

        for player in merged["player_name"].unique():
            pdata = merged[merged["player_name"] == player].sort_values("season_order").reset_index(drop=True)
            if len(pdata) < seq_len + 1:
                continue
            for i in range(len(pdata) - seq_len):
                window = pdata.iloc[i:i+seq_len]
                target = pdata.iloc[i+seq_len]["auction_price_cr"]
                feats  = []
                for _, row in window.iterrows():
                    feats.append([
                        float(row.get("auction_price_cr", 5)),
                        float(row.get("runs", 0)),
                        float(row.get("batting_avg", 20)),
                        float(row.get("strike_rate", 120)),
                        float(row.get("is_overseas", 0)),
                    ])
                arr = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0)
                sequences_X.append(arr)
                if not np.isnan(target): sequences_y.append(float(target))
                else: sequences_y.append(5.0)

        X = np.array(sequences_X)
        y = np.array(sequences_y, dtype=np.float32)
        log.info(f"LSTM sequences: X={X.shape}, y={y.shape}")
        return X, y

    def save(self, df, filename):
        path = DATA_DIR / filename
        df.to_csv(path, index=False)
        log.info(f"Saved → {path}")

    def save_scaler(self):
        path = DATA_DIR / "processed" / "scaler.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        log.info(f"Scaler saved → {path}")

    # ── MERGING ───────────────────────────────────────────────────────────────

    def _merge_all(self, bat_df, bowl_df, auction_df, sentiment_df):
        # Aggregate batting per player per season
        bat_agg = bat_df.groupby(["player_name","season","role","nationality","team"]).agg(
            runs        = ("runs",        "sum"),
            batting_avg = ("batting_avg", "mean"),
            strike_rate = ("strike_rate", "mean"),
            fifties     = ("fifties",     "sum"),
            hundreds    = ("hundreds",    "sum"),
            dot_ball_pct= ("dot_ball_pct","mean"),
            boundary_pct= ("boundary_pct","mean"),
            six_pct     = ("six_pct",     "mean"),
            career_avg  = ("career_avg",  "first"),
            career_sr   = ("career_sr",   "first"),
            career_runs = ("career_runs", "first"),
            career_fifties = ("career_fifties","first"),
            career_hundreds= ("career_hundreds","first"),
            seasons_played = ("seasons_played","first"),
        ).reset_index()

        # Aggregate bowling per player per season
        bowl_agg = bowl_df.groupby(["player_name","season"]).agg(
            wickets        = ("wickets",        "sum"),
            bowling_avg    = ("bowling_avg",    "mean"),
            economy_rate   = ("economy_rate",   "mean"),
            bowling_sr     = ("bowling_sr",     "mean"),
            bowl_dot_pct   = ("dot_ball_pct",   "mean"),
            death_economy  = ("death_economy",  "mean"),
            pp_wickets     = ("powerplay_wickets","sum"),
            career_wickets = ("career_wickets", "first"),
            career_economy = ("career_economy", "first"),
            career_bowl_avg= ("career_bowling_avg","first"),
        ).reset_index()

        # Merge
        df = auction_df.merge(bat_agg, on=["player_name","season"], how="left")
        df = df.merge(bowl_agg, on=["player_name","season"], how="left")

        # Sentiment
        if sentiment_df is not None and not sentiment_df.empty:
            sent_agg = sentiment_df.groupby("player_name").agg(
                sentiment_compound = ("compound_score","mean"),
                sentiment_positive = ("positive_pct", "mean"),
                sentiment_negative = ("negative_pct", "mean"),
                sentiment_articles = ("article_count","sum"),
            ).reset_index()
            df = df.merge(sent_agg, on="player_name", how="left")
        else:
            df["sentiment_compound"] = 0.0
            df["sentiment_positive"] = 55.0
            df["sentiment_negative"] = 12.0
            df["sentiment_articles"] = 0

        return df

    # ── FEATURE ENGINEERING ──────────────────────────────────────────────────

    def _handle_missing(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        imp = SimpleImputer(strategy="median")
        df[num_cols] = imp.fit_transform(df[num_cols])
        df = df.fillna("Unknown")
        return df

    def _encode_categoricals(self, df):
        if "role" in df.columns:
            role_dummies = pd.get_dummies(df["role"], prefix="role")
            df = pd.concat([df, role_dummies], axis=1)
        for col in ["team","nationality","season"]:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        return df

    def _batting_features(self, df):
        # Run-scoring ability
        df["runs_per_match"]     = df.get("runs", pd.Series(0, index=df.index)) / df.get("seasons_played", pd.Series(1, index=df.index)).replace(0, 1)
        df["impact_score"]       = (df.get("batting_avg", 0) * 0.4 + df.get("strike_rate", 0) * 0.3 + df.get("career_sr", 0) * 0.3) / 100
        df["milestone_score"]    = df.get("fifties", 0) * 1.0 + df.get("hundreds", 0) * 2.5 + df.get("career_hundreds", 0) * 1.5
        df["consistency_score"]  = df.get("career_avg", 0) * df.get("career_sr", 0) / 100
        df["powerplay_value"]    = df.get("six_pct", 0) * 0.5 + df.get("boundary_pct", 0) * 0.3
        df["dot_ball_risk"]      = df.get("dot_ball_pct", 30) / 100
        return df

    def _bowling_features(self, df):
        df["bowling_impact"]     = df.get("career_wickets", 0) * 0.4 + (1 / df.get("career_economy", 8).replace(0, 8)) * 30
        df["death_specialist"]   = (df.get("death_economy", 9) < 8.5).astype(int)
        df["powerplay_bowler"]   = (df.get("pp_wickets", 0) > 3).astype(int)
        df["wicket_taker_score"] = df.get("wickets", 0) * 1.5 + df.get("career_wickets", 0) * 0.3
        df["economy_score"]      = (10 - df.get("economy_rate", 8)).clip(0, 5)
        return df

    def _allrounder_features(self, df):
        role_col = df.get("role", pd.Series("Batter", index=df.index))
        is_ar    = role_col.astype(str).str.contains("All-Rounder").astype(int)
        df["is_allrounder"]      = is_ar
        df["allrounder_score"]   = is_ar * (df.get("impact_score", 0) + df.get("bowling_impact", 0))
        df["dual_threat_bonus"]  = is_ar * (df.get("career_runs", 0) / 1000 + df.get("career_wickets", 0) / 20)
        return df

    def _age_experience_features(self, df):
        df["experience_score"]   = df.get("seasons_played", 1) / 16  # max 16 IPL seasons
        df["veteran_flag"]       = (df.get("seasons_played", 0) >= 8).astype(int)
        df["young_prospect"]     = (df.get("seasons_played", 0) <= 3).astype(int)
        df["peak_experience"]    = ((df.get("seasons_played", 0) >= 4) & (df.get("seasons_played", 0) <= 10)).astype(int)
        df["overseas_premium"]   = df.get("is_overseas", 0)  # overseas slots scarce → premium
        return df

    def _auction_trend_features(self, df, auction_df):
        # Previous season price as a feature
        auction_sorted = auction_df.sort_values(["player_name","season"])
        auction_sorted["prev_price"] = auction_sorted.groupby("player_name")["auction_price_cr"].shift(1)
        auction_sorted["price_trend"] = auction_sorted["auction_price_cr"] - auction_sorted["prev_price"].fillna(0)
        trend_df = auction_sorted[["player_name","season","prev_price","price_trend"]]
        df = df.merge(trend_df, on=["player_name","season"], how="left")
        df["prev_price"]  = df["prev_price"].fillna(df.get("auction_price_cr", 5))
        df["price_trend"] = df["price_trend"].fillna(0)
        df["log_prev_price"] = np.log1p(df["prev_price"])
        return df

    def _create_target(self, df):
        df["target"] = df["auction_price_cr"].astype(float)  # raw ₹Cr - consistent with LSTM
        return df


# ── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from data_collection.cricsheet_loader    import CricsheetLoader
    from data_collection.cricket_sentiment   import CricketSentimentCollector

    loader  = CricsheetLoader()
    bat_df  = loader.get_batting_stats()
    bowl_df = loader.get_bowling_stats()
    auction_df = loader.get_auction_history()

    collector  = CricketSentimentCollector()
    sent_df    = collector.get_player_sentiment(
        bat_df["player_name"].unique().tolist()[:10], days_back=60)

    fe = CricketFeatureEngineer()
    feat_df = fe.build_feature_matrix(bat_df, bowl_df, auction_df, sent_df)
    feat_df = fe.scale_features(feat_df, fit=True)
    fe.save(feat_df, "processed/cricket_feature_matrix.csv")
    fe.save_scaler()

    print("\n=== Feature Matrix ===")
    show = ["player_name","season","role","runs","batting_avg","strike_rate",
            "wickets","economy_rate","auction_price_cr","impact_score","target"]
    print(feat_df[[c for c in show if c in feat_df.columns]].head(10).to_string(index=False))

    X_tr, X_val, X_te, y_tr, y_val, y_te = fe.train_val_test_split(feat_df)
    print(f"\nTrain: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

    X_seq, y_seq = fe.build_lstm_sequences(auction_df, bat_df, bowl_df, seq_len=3)
    print(f"LSTM sequences: X={X_seq.shape}, y={y_seq.shape}")
