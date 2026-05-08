"""
CricketIQ — Master Pipeline
=============================
End-to-end IPL auction price prediction pipeline.
  Week 1: Data collection (Cricsheet + Auction + Sentiment)
  Week 2: Feature engineering
  Week 5: LSTM training
  Week 6: XGBoost + LightGBM + Ensemble
  Week 7: Evaluation

Usage:
    python run_pipeline.py              # full pipeline
    python run_pipeline.py --week 1     # data only
"""

import sys, argparse, logging, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("cricketiq_pipeline.log")]
)
log = logging.getLogger("CricketIQ")
DIV = "=" * 60


def week1_collect():
    log.info(f"\n{DIV}\nWEEK 1 — DATA COLLECTION (IPL)\n{DIV}")
    from data_collection.cricsheet_loader  import CricsheetLoader
    from data_collection.cricket_sentiment import CricketSentimentCollector

    loader = CricsheetLoader()
    bat_df    = loader.get_batting_stats()
    bowl_df   = loader.get_bowling_stats()
    auction_df= loader.get_auction_history()
    loader.save(bat_df,     "raw/ipl_batting.csv")
    loader.save(bowl_df,    "raw/ipl_bowling.csv")
    loader.save(auction_df, "raw/ipl_auction.csv")
    log.info(f"  ✓ Batting:  {bat_df.shape[0]} records, {bat_df['player_name'].nunique()} players")
    log.info(f"  ✓ Bowling:  {bowl_df.shape[0]} records, {bowl_df['player_name'].nunique()} players")
    log.info(f"  ✓ Auction:  {auction_df.shape[0]} records")

    top15 = bat_df["player_name"].unique().tolist()[:15]
    collector  = CricketSentimentCollector()
    sent_df    = collector.get_player_sentiment(top15, days_back=60)
    collector.save(sent_df, "sentiment/ipl_sentiment.csv")
    log.info(f"  ✓ Sentiment: {len(sent_df)} weekly records for {len(top15)} players")
    log.info("\n✅ Week 1 complete.")
    return bat_df, bowl_df, auction_df, sent_df


def week2_features(bat_df, bowl_df, auction_df, sent_df):
    log.info(f"\n{DIV}\nWEEK 2 — FEATURE ENGINEERING\n{DIV}")
    from preprocessing.cricket_feature_engineer import CricketFeatureEngineer

    fe      = CricketFeatureEngineer()
    feat_df = fe.build_feature_matrix(bat_df, bowl_df, auction_df, sent_df)
    feat_df = fe.scale_features(feat_df, fit=True)
    fe.save(feat_df, "processed/cricket_feature_matrix.csv")
    fe.save_scaler()
    log.info(f"  ✓ Feature matrix: {feat_df.shape}")

    X_seq, y_seq = fe.build_lstm_sequences(auction_df, bat_df, bowl_df, seq_len=3)
    log.info(f"  ✓ LSTM sequences: X={X_seq.shape}, y={y_seq.shape}")
    log.info("\n✅ Week 2 complete.")
    return feat_df, X_seq, y_seq, fe


def week5_lstm(X_seq, y_seq):
    log.info(f"\n{DIV}\nWEEK 5 — LSTM TRAINING\n{DIV}")
    from models.cricket_models import CricketLSTMTrainer

    from sklearn.model_selection import train_test_split as _tts
    X_tr, X_tmp, y_tr, y_tmp = _tts(X_seq, y_seq, test_size=0.30, random_state=42)
    X_val, X_te, y_val, y_te = _tts(X_tmp, y_tmp, test_size=0.50, random_state=42)
    log.info(f"  Sequences → Train:{len(X_tr)}, Val:{len(X_val)}, Test:{len(X_te)}")

    lstm = CricketLSTMTrainer(input_size=X_seq.shape[2], hidden_size=64, n_layers=2)
    lstm.train(X_tr, y_tr, X_val, y_val, epochs=50)
    m = lstm.evaluate(X_te, y_te)
    lstm.save("cricket_lstm.pt")
    log.info(f"  ✓ LSTM → RMSE:{m['rmse']:.4f} | R²:{m['r2']:.4f}")
    log.info("\n✅ Week 5 complete.")
    return lstm, X_val, y_val, X_te, y_te


def week6_ensemble(feat_df, fe, lstm, X_sq_val, y_sq_val, X_sq_te, y_sq_te):
    log.info(f"\n{DIV}\nWEEK 6 — ENSEMBLE TRAINING\n{DIV}")
    from models.cricket_models import CricketXGBoost, CricketLightGBM, CricketEnsemble

    X_tr, X_val, X_te, y_tr, y_val, y_te = fe.train_val_test_split(feat_df)
    feat_cols = [c for c in feat_df.select_dtypes(include=[np.number]).columns
                 if c not in ["target","auction_price_cr"]]

    xgb_model = CricketXGBoost()
    xgb_model.train(X_tr, y_tr, X_val, y_val, feature_names=feat_cols)
    xgb_model.save()
    log.info("  ✓ XGBoost trained")

    log.info("\n  --- Top 10 Features (XGBoost) ---")
    for _, row in xgb_model.top_features(10).iterrows():
        bar = "█" * int(row["importance_pct"] / 2)
        log.info(f"  {row['feature']:35s} {bar} {row['importance_pct']:.1f}%")

    lgbm = CricketLightGBM()
    lgbm.train(X_tr, y_tr, X_val, y_val, feature_names=feat_cols)
    log.info("  ✓ LightGBM trained")

    mn = min(len(X_sq_val), len(X_val))
    mt = min(len(X_sq_te),  len(X_te))
    ensemble = CricketEnsemble(lstm, xgb_model, lgbm)
    ensemble.fit_weights(X_sq_val[:mn], X_val[:mn], y_sq_val[:mn])
    ensemble.save()
    log.info("  ✓ Ensemble weights optimised")
    log.info("\n✅ Week 6 complete.")
    return ensemble, X_te, y_te, mt   # X_te/y_te are feat-based test sets


def week7_evaluation(ensemble, X_sq_te, X_feat_te, y_sq_te, y_feat_te, mt):
    log.info(f"\n{DIV}\nWEEK 7 — EVALUATION\n{DIV}")
    comp = ensemble.full_comparison(X_sq_te, X_feat_te, y_sq_te, y_feat_te)

    log.info("\n╔══════════════════════════════════════════════════════╗")
    log.info("║        CRICKETIQ — FINAL MODEL COMPARISON           ║")
    log.info("╠══════════════════════════════════════════════════════╣")
    log.info(f"║ {'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}       ║")
    log.info("╠══════════════════════════════════════════════════════╣")
    for _, row in comp.iterrows():
        tag = " ◄ BEST" if row["Model"] == "Ensemble (Final)" else ""
        log.info(f"║ {row['Model']:<22} {row['rmse']:>8.4f} {row['mae']:>8.4f} {row['r2']:>8.4f}{tag:<7}║")
    log.info("╚══════════════════════════════════════════════════════╝")

    comp.to_csv("data/processed/evaluation_report.csv", index=False)
    log.info("\n✅ Week 7 complete.")
    return comp


def main():
    parser = argparse.ArgumentParser(description="CricketIQ Pipeline")
    parser.add_argument("--week", type=int, default=0)
    args = parser.parse_args()

    log.info(f"\n{'='*60}")
    log.info("🏏 CricketIQ — IPL Auction Price Prediction Pipeline")
    log.info(f"{'='*60}")

    bat_df, bowl_df, auction_df, sent_df = week1_collect()
    if args.week == 1: return

    feat_df, X_seq, y_seq, fe = week2_features(bat_df, bowl_df, auction_df, sent_df)
    if args.week == 2: return

    lstm, X_sq_val, y_sq_val, X_sq_te, y_sq_te = week5_lstm(X_seq, y_seq)
    if args.week == 5: return

    ensemble, X_feat_te, y_feat_te, mt = week6_ensemble(
        feat_df, fe, lstm, X_sq_val, y_sq_val, X_sq_te, y_sq_te)
    if args.week == 6: return

    comp = week7_evaluation(ensemble, X_sq_te, X_feat_te, y_sq_te, y_feat_te, mt)

    best = comp[comp["r2"] == comp["r2"].max()].iloc[0]
    log.info(f"\n🏆 Best Model : {best['Model']}")
    log.info(f"   RMSE       : {best['rmse']:.4f} (log ₹Cr)")
    log.info(f"   R²         : {best['r2']:.4f}")
    log.info(f"\n   Dashboard  : streamlit run src/visualization/dashboard.py")
    log.info(f"   API        : uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
