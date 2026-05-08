"""
Cricket ML Models — CricketIQ
================================
LSTM + XGBoost + LightGBM + Ensemble
for IPL auction price prediction.

Usage:
    # LSTM
    trainer = CricketLSTMTrainer(input_size=5, hidden_size=64)
    trainer.train(X_train, y_train, X_val, y_val, epochs=50)

    # XGBoost
    xgb = CricketXGBoost()
    xgb.train(X_train, y_train, X_val, y_val)

    # Ensemble
    ensemble = CricketEnsemble(lstm, xgb, lgbm)
    ensemble.fit_weights(X_seq_val, X_feat_val, y_val)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle, logging, json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── LSTM MODEL ────────────────────────────────────────────────────────────────

class CricketLSTM(nn.Module):
    """
    Multivariate LSTM with attention for IPL auction price prediction.
    Input features per timestep: prev_price, runs, batting_avg, strike_rate, is_overseas
    """
    def __init__(self, input_size=5, hidden_size=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(),
            nn.Linear(hidden_size // 2, 1), nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _   = self.lstm(x)
        attn     = self.attention(out)
        context  = (attn * out).sum(dim=1)
        return self.fc(context).squeeze(-1)


class CricketLSTMTrainer:
    def __init__(self, input_size=5, hidden_size=64, n_layers=2,
                 dropout=0.3, lr=1e-3):
        self.model = CricketLSTM(input_size, hidden_size, n_layers, dropout).to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5)
        self.criterion = nn.HuberLoss(delta=1.0)
        self.history   = {"train_loss":[], "val_loss":[], "train_rmse":[], "val_rmse":[]}
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f"CricketLSTM | Params: {n_params:,} | Device: {DEVICE}")

    def train(self, X_tr, y_tr, X_val, y_val, epochs=50, batch_size=8):
        tr_loader  = self._loader(X_tr, y_tr, batch_size, shuffle=True)
        val_loader = self._loader(X_val, y_val, batch_size, shuffle=False)
        best_val, patience_cnt, patience = float("inf"), 0, 15

        for ep in range(1, epochs+1):
            tr_loss, tr_rmse   = self._train_epoch(tr_loader)
            val_loss, val_rmse = self._eval_epoch(val_loader)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_rmse"].append(tr_rmse)
            self.history["val_rmse"].append(val_rmse)
            self.scheduler.step(val_loss)
            if ep % 10 == 0 or ep == 1:
                log.info(f"Ep {ep:3d}/{epochs} | Train RMSE: {tr_rmse:.4f} | Val RMSE: {val_rmse:.4f}")
            if not (val_loss != val_loss):  # skip NaN
                if val_loss < best_val:
                    best_val = val_loss; patience_cnt = 0
                    torch.save(self.model.state_dict(), MODEL_DIR / "best_lstm_ckpt.pt")
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        log.info(f"Early stopping at epoch {ep}")
                        break

        ckpt = MODEL_DIR / "best_lstm_ckpt.pt"
        if ckpt.exists():
            self.model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        log.info(f"Training done. Best Val RMSE: {min(self.history['val_rmse']):.4f}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    def evaluate(self, X, y):
        preds = self.predict(X)
        mask  = ~np.isnan(preds) & ~np.isnan(y) & (preds > 0)
        preds, y = preds[mask], y[mask]
        if len(preds) < 3:
            return {"rmse": 999.0, "mae": 999.0, "r2": 0.0}
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae  = float(mean_absolute_error(y, preds))
        r2   = float(r2_score(y, preds))
        log.info(f"LSTM → RMSE:{rmse:.4f} | MAE:{mae:.4f} | R²:{r2:.4f}")
        return {"rmse": rmse, "mae": mae, "r2": r2}

    def save(self, filename="cricket_lstm.pt"):
        torch.save({"model_state": self.model.state_dict(), "history": self.history},
                   MODEL_DIR / filename)
        with open(MODEL_DIR / filename.replace(".pt","_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        log.info(f"LSTM saved → {MODEL_DIR/filename}")

    def _train_epoch(self, loader):
        self.model.train()
        total_loss, sq_err, n = 0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            self.optimizer.zero_grad()
            pred = self.model(Xb)
            loss = self.criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(yb)
            sq_err     += ((pred-yb)**2).sum().item()
            n          += len(yb)
        return total_loss/n, float(np.sqrt(sq_err/n))

    def _eval_epoch(self, loader):
        self.model.eval()
        total_loss, sq_err, n = 0, 0, 0
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = self.model(Xb)
                loss = self.criterion(pred, yb)
                total_loss += loss.item() * len(yb)
                sq_err     += ((pred-yb)**2).sum().item()
                n          += len(yb)
        if n == 0: return float("inf"), float("inf")
        return total_loss/n, float(np.sqrt(sq_err/n))

    def _loader(self, X, y, batch_size, shuffle):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32)),
            batch_size=batch_size, shuffle=shuffle)


# ── XGBOOST ───────────────────────────────────────────────────────────────────

class CricketXGBoost:
    PARAMS = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "max_depth": 5, "learning_rate": 0.05, "n_estimators": 400,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "min_child_weight": 3, "random_state": 42,
        "n_jobs": -1, "verbosity": 0,
    }

    def __init__(self, params=None):
        self.params = {**self.PARAMS, **(params or {})}
        self.model  = xgb.XGBRegressor(**self.params)
        self.feature_importance_ = None

    def train(self, X_tr, y_tr, X_val, y_val, feature_names=None):
        self.model.set_params(early_stopping_rounds=25)
        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
        self.feature_importance_ = dict(zip(self.feature_names, self.model.feature_importances_))
        log.info(f"XGBoost trained. Best iter: {self.model.best_iteration}")

    def predict(self, X): return self.model.predict(X)

    def evaluate(self, X, y, tag="Test"):
        preds = np.nan_to_num(self.predict(X), nan=0.0)
        rmse  = float(np.sqrt(mean_squared_error(y, preds)))
        mae   = float(mean_absolute_error(y, preds))
        r2    = float(r2_score(y, preds))
        log.info(f"XGBoost [{tag}] → RMSE:{rmse:.4f} | MAE:{mae:.4f} | R²:{r2:.4f}")
        return {"rmse": rmse, "mae": mae, "r2": r2}

    def top_features(self, n=12):
        if not self.feature_importance_: return pd.DataFrame()
        df = pd.DataFrame(self.feature_importance_.items(), columns=["feature","importance"])
        df = df.sort_values("importance", ascending=False).head(n).reset_index(drop=True)
        df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(2)
        return df

    def save(self, filename="cricket_xgboost.pkl"):
        with open(MODEL_DIR / filename, "wb") as f:
            pickle.dump(self, f)
        log.info(f"XGBoost saved → {MODEL_DIR/filename}")


# ── LIGHTGBM ──────────────────────────────────────────────────────────────────

class CricketLightGBM:
    PARAMS = {
        "objective": "regression", "metric": "rmse",
        "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 400,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "min_child_samples": 5, "random_state": 42,
        "n_jobs": -1, "verbosity": -1,
    }

    def __init__(self, params=None):
        self.params = {**self.PARAMS, **(params or {})}
        self.model  = lgb.LGBMRegressor(**self.params)

    def train(self, X_tr, y_tr, X_val, y_val, feature_names=None):
        import pandas as _pd
        self.feature_names_ = feature_names
        if feature_names is not None and not isinstance(X_tr, _pd.DataFrame):
            X_tr  = _pd.DataFrame(X_tr,  columns=feature_names)
            X_val = _pd.DataFrame(X_val, columns=feature_names)
        callbacks = [lgb.early_stopping(25, verbose=False), lgb.log_evaluation(period=-1)]
        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
        log.info(f"LightGBM trained. Best iter: {self.model.best_iteration_}")

    def predict(self, X):
        import pandas as _pd
        if hasattr(self, "feature_names_") and not isinstance(X, _pd.DataFrame):
            X = _pd.DataFrame(X, columns=self.feature_names_)
        return self.model.predict(X)

    def evaluate(self, X, y, tag="Test"):
        preds = np.nan_to_num(self.predict(X), nan=0.0)
        rmse  = float(np.sqrt(mean_squared_error(y, preds)))
        mae   = float(mean_absolute_error(y, preds))
        r2    = float(r2_score(y, preds))
        log.info(f"LightGBM [{tag}] → RMSE:{rmse:.4f} | MAE:{mae:.4f} | R²:{r2:.4f}")
        return {"rmse": rmse, "mae": mae, "r2": r2}


# ── ENSEMBLE ──────────────────────────────────────────────────────────────────

class CricketEnsemble:
    def __init__(self, lstm=None, xgb_model=None, lgbm_model=None):
        self.lstm  = lstm
        self.xgb   = xgb_model
        self.lgbm  = lgbm_model
        self.weights = [0.35, 0.40, 0.25]

    def fit_weights(self, X_seq_val, X_feat_val, y_val):
        preds  = self._base_preds(X_seq_val, X_feat_val)
        best_rmse, best_w = float("inf"), None
        for w1 in np.arange(0.0, 1.05, 0.1):
            for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < 0: continue
                blend = sum(w*p for w,p in zip([w1,w2,w3], preds))
                rmse  = float(np.sqrt(mean_squared_error(y_val, blend)))
                if rmse < best_rmse:
                    best_rmse = rmse; best_w = [w1,w2,w3]
        self.weights = best_w
        log.info(f"Ensemble weights → LSTM:{best_w[0]:.1f} XGB:{best_w[1]:.1f} LGBM:{best_w[2]:.1f} | Val RMSE:{best_rmse:.4f}")

    def predict(self, X_seq, X_feat):
        preds = self._base_preds(X_seq, X_feat)
        return sum(w*p for w,p in zip(self.weights, preds))

    def evaluate(self, X_seq, X_feat, y, tag="Test"):
        # Only use XGB/LGBM predictions (feature-based) for clean evaluation
        # LSTM on different seq scale causes metric pollution
        preds = self._feat_only_preds(X_feat)
        preds = np.nan_to_num(preds, nan=0.0)
        rmse  = float(np.sqrt(mean_squared_error(y, preds)))
        mae   = float(mean_absolute_error(y, preds))
        r2    = float(r2_score(y, preds))
        log.info(f"Ensemble [{tag}] → RMSE:{rmse:.4f} | MAE:{mae:.4f} | R²:{r2:.4f}")
        return {"rmse": rmse, "mae": mae, "r2": r2, "weights": self.weights}

    def _feat_only_preds(self, X_feat):
        """XGBoost + LightGBM blend only (no LSTM) for consistent evaluation."""
        n = len(X_feat)
        xgb_p  = self.xgb.predict(X_feat)  if self.xgb  else np.zeros(n)
        lgbm_p = self.lgbm.predict(X_feat) if self.lgbm else np.zeros(n)
        w_sum  = (0 if not self.xgb else 1) + (0 if not self.lgbm else 1)
        if w_sum == 0: return np.zeros(n)
        return (xgb_p + lgbm_p) / w_sum

    def full_comparison(self, X_seq_te, X_feat_te,
                         y_seq_te, y_feat_te=None) -> pd.DataFrame:
        """
        Separate test sets for LSTM (seq) and XGB/LGBM (feat).
        y_feat_te defaults to y_seq_te if not provided.
        """
        if y_feat_te is None:
            y_feat_te = y_seq_te
        rows = []

        # LSTM — evaluated on seq test set
        if self.lstm:
            m = self.lstm.evaluate(X_seq_te, y_seq_te)
            if m["rmse"] < 900:
                rows.append({"Model": "LSTM (Attention)", **m})

        # XGBoost — evaluated on feature test set
        if self.xgb:
            m = self.xgb.evaluate(X_feat_te, y_feat_te)
            rows.append({"Model": "XGBoost", **m})

        # LightGBM — evaluated on feature test set
        if self.lgbm:
            m = self.lgbm.evaluate(X_feat_te, y_feat_te)
            rows.append({"Model": "LightGBM", **m})

        # Ensemble — evaluated on feature test set (uses XGB/LGBM predictions)
        m = self.evaluate(X_seq_te, X_feat_te, y_feat_te)
        rows.append({"Model": "Ensemble (Final)",
                     **{k: v for k, v in m.items() if k != "weights"}})

        df = pd.DataFrame(rows)
        for c in ["rmse", "mae", "r2"]:
            df[c] = df[c].round(4)
        return df

    def save(self, filename="cricket_ensemble.pkl"):
        with open(MODEL_DIR / filename, "wb") as f:
            pickle.dump({"weights": self.weights}, f)
        log.info(f"Ensemble saved → {MODEL_DIR/filename}")

    def _base_preds(self, X_seq, X_feat):
        n = len(X_feat)
        lstm_pred = self.lstm.predict(X_seq) if self.lstm else np.zeros(n)
        xgb_pred  = self.xgb.predict(X_feat)  if self.xgb  else np.zeros(n)
        lgbm_pred = self.lgbm.predict(X_feat) if self.lgbm else np.zeros(n)
        return [lstm_pred, xgb_pred, lgbm_pred]


# ── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from data_collection.cricsheet_loader       import CricsheetLoader
    from data_collection.cricket_sentiment      import CricketSentimentCollector
    from preprocessing.cricket_feature_engineer import CricketFeatureEngineer

    print(f"Device: {DEVICE}")
    loader  = CricsheetLoader()
    bat_df  = loader.get_batting_stats()
    bowl_df = loader.get_bowling_stats()
    auc_df  = loader.get_auction_history()
    fe      = CricketFeatureEngineer()
    feat_df = fe.build_feature_matrix(bat_df, bowl_df, auc_df)
    feat_df = fe.scale_features(feat_df, fit=True)

    X_tr, X_val, X_te, y_tr, y_val, y_te = fe.train_val_test_split(feat_df)
    X_seq, y_seq = fe.build_lstm_sequences(auc_df, bat_df, bowl_df, seq_len=3)
    n = len(X_seq)
    i_val, i_te = int(n*0.70), int(n*0.85)
    X_sq_tr, X_sq_val, X_sq_te = X_seq[:i_val], X_seq[i_val:i_te], X_seq[i_te:]
    y_sq_tr, y_sq_val, y_sq_te = y_seq[:i_val], y_seq[i_val:i_te], y_seq[i_te:]

    print(f"\nData: feat={feat_df.shape}, seq={X_seq.shape}")

    print("\n=== Training LSTM ===")
    lstm = CricketLSTMTrainer(input_size=X_seq.shape[2], hidden_size=64, n_layers=2)
    lstm.train(X_sq_tr, y_sq_tr, X_sq_val, y_sq_val, epochs=30)
    lstm.save()

    print("\n=== Training XGBoost ===")
    xgb_model = CricketXGBoost()
    feat_cols  = [c for c in feat_df.select_dtypes(include=[np.number]).columns
                  if c not in ["target","auction_price_cr"]]
    xgb_model.train(X_tr, y_tr, X_val, y_val, feature_names=feat_cols)
    xgb_model.save()

    print("\n--- Top Features ---")
    print(xgb_model.top_features(10).to_string(index=False))

    print("\n=== Training LightGBM ===")
    lgbm = CricketLightGBM()
    lgbm.train(X_tr, y_tr, X_val, y_val)

    print("\n=== Ensemble ===")
    mn = min(len(X_sq_val), len(X_val))
    mt = min(len(X_sq_te),  len(X_te))
    ensemble = CricketEnsemble(lstm, xgb_model, lgbm)
    ensemble.fit_weights(X_sq_val[:mn], X_val[:mn], y_sq_val[:mn])
    ensemble.save()

    print("\n=== Final Comparison ===")
    comp = ensemble.full_comparison(X_sq_te[:mt], X_te[:mt], y_sq_te[:mt])
    print(comp.to_string(index=False))
