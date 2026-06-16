"""Konfigurácia parametrov pre tréning modelu rozpoznávania kľúčových slov."""

import torch
from kws.c_model.m_w2v import Wav2VecKWS
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# IDENTIFIKÁTORY EXPERIMENTU
# ────────────────────────────────────────────────────────────────
DATASET_SLUG = "d2"                     # identifikátor použitého datasetu
MODEL_SLUG = "m_w2v"                    # identifikátor modelu
CONFIG_SLUG = "c"                       # identifikátor konfigurácie
MODEL_CLASSES = {"m_w2v": Wav2VecKWS,} # mapovanie slugov na triedy modelov

# ────────────────────────────────────────────────────────────────
# ZÁKLADNÉ PARAMETRE TRÉNINGU
# ────────────────────────────────────────────────────────────────
EPOCHS = 6                              # počet tréningových epoch
LEARNING_RATE = 0.00005                 # základná rýchlosť učenia
BATCH_SIZE = 16                         # veľkosť batchu
WEIGHT_DECAY = 0.0005                   # regularizácia L2
SAMPLE_RATE = 16000                     # vzorkovacia frekvencia audia (Hz)
TARGET_SAMPLES = 16000                  # cieľový počet vzoriek (1 sekunda)
NUM_WORKERS = 8                         # počet procesov pre DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # výpočtové zariadenie
AUGMENT_TRAINING_DATA = True           # či sa má používať augmentácia dát

# ────────────────────────────────────────────────────────────────
# CESTY K DÁTAM A VÝSLEDKOM
# ────────────────────────────────────────────────────────────────

project_root = Path(__file__).resolve().parents[2]
DATA_ROOT = project_root / "kws" / "b_dataset" / DATASET_SLUG
RESULTS_BASE_DIR = project_root / "kws" / "h_result"
