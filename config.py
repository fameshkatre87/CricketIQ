"""
CricketIQ — Config
===================
Central config. Edit values here or set as env variables.
"""
import os
from pathlib import Path

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = DATA_DIR / "models"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
SENT_DIR   = DATA_DIR / "sentiment"

NEWS_API_KEY        = os.getenv("NEWS_API_KEY", "")
TRANSFERMARKT_DELAY = int(os.getenv("TRANSFERMARKT_DELAY", "5"))
SEQ_LEN             = int(os.getenv("SEQ_LEN", "3"))
EPOCHS              = int(os.getenv("EPOCHS", "50"))
BATCH_SIZE          = int(os.getenv("BATCH_SIZE", "8"))
LSTM_HIDDEN         = int(os.getenv("LSTM_HIDDEN", "64"))
LSTM_LAYERS         = int(os.getenv("LSTM_LAYERS", "2"))
XGB_N_ESTIMATORS    = int(os.getenv("XGB_N_ESTIMATORS", "400"))
N_PLAYERS           = int(os.getenv("N_PLAYERS", "50"))
