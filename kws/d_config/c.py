"""
Konfigurácia parametrov pre tréning modelu rozpoznávania kľúčových slov (Keyword Spotting).

Tento súbor obsahuje všetky nastavenia potrebné pre tréningový pipeline:
- Identifikátory experimentu (dataset, model, konfigurácia)
- Hyperparametre tréningu (learning rate, batch size, epochs, scheduler...)
- Nastavenia modelu (freezing vrstiev, augmentácia)
- Cesty k dátam a výstupným priečinkom

Tento config sa načíta dynamicky hlavným tréningovým skriptom.
"""

import torch
from kws.c_model.m_w2v import Wav2VecKWS
from pathlib import Path


# ────────────────────────────────────────────────────────────────
# IDENTIFIKÁTORY EXPERIMENTU
# ────────────────────────────────────────────────────────────────
DATASET_SLUG = "d2"                     # identifikátor použitého datasetu
MODEL_SLUG = "m_w2v"                    # identifikátor modelu
CONFIG_SLUG = "c"                       # identifikátor konfigurácie
MODEL_CLASSES = {"m_w2v": Wav2VecKWS,}  # mapovanie slugov na triedy modelov


# ────────────────────────────────────────────────────────────────
# ZÁKLADNÉ NASTAVENIA TRÉNINGU
# ────────────────────────────────────────────────────────────────
EPOCHS = 12                             # počet tréningových epoch
LEARNING_RATE = 0.00005                 # základná rýchlosť učenia
BATCH_SIZE = 16                         # veľkosť batchu
WEIGHT_DECAY = 0.0005                   # regularizácia L2

CLASS_WEIGHTS = None                    # počítajú sa automaticky (alebo ručne napr. [1.0, 15.0])
AUGMENT_TRAINING_DATA = True            # či sa má používať augmentácia dát

FREEZE_FEATURE_EXTRACTOR = True
FREEZE_ENCODER = True
UNFREEZE_LAST_N_LAYERS = 3

SCHEDULER_PATIENCE = 2        # počet epoch bez zlepšenia pred znížením LR
SCHEDULER_FACTOR = 0.5        # množstvo zníženia LR (0.5 = polovica)
SCHEDULER_COOLDOWN = 1        # čakanie v epochách po znížení LR
SCHEDULER_MIN_LR = 1e-7       # minimálny learning rate

SAMPLE_RATE = 16000           # vzorkovacia frekvencia audia (Hz)
TARGET_SAMPLES = 16000        # cieľový počet vzoriek (1 sekunda)
NUM_WORKERS = 8               # počet procesov pre DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # výpočtové zariadenie


# ────────────────────────────────────────────────────────────────
# CESTY K DÁTAM A VÝSLEDKOM
# ────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
DATA_ROOT = project_root / "kws" / "b_dataset" / DATASET_SLUG
RESULTS_BASE_DIR = project_root / "kws" / "g_result"