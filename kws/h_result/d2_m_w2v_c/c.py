import torch
from kws.c_model.m_w2v import Wav2VecKWS

# ────────────────────────────────────────────────────────────────
# IDENTIFIKÁTORY EXPERIMENTU
# ────────────────────────────────────────────────────────────────
DATASET_SLUG = "d2"
MODEL_SLUG   = "m_w2v"
CONFIG_SLUG  = "c"
MODEL_CLASSES = {"m_w2v": Wav2VecKWS,}


# ────────────────────────────────────────────────────────────────
# ZÁKLADNÉ PARAMETRE TRÉNINGU
# ────────────────────────────────────────────────────────────────
EPOCHS        = 8
LEARNING_RATE = 0.0001
BATCH_SIZE    = 4
WEIGHT_DECAY  = 0.0005
SAMPLE_RATE   = 16000
TARGET_SAMPLES = 16000  # 1 sekunda pri 16000 Hz
NUM_WORKERS   = 4       # pre DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUGMENT_TRAINING_DATA = True


# ────────────────────────────────────────────────────────────────
# CESTY K DÁTAM A VÝSLEDKOM
# ────────────────────────────────────────────────────────────────
DATA_ROOT = fr"C:\Users\peter\Moje\Diplomka\DiplomovaPraca\kws\b_dataset\{DATASET_SLUG}"
RESULTS_BASE_DIR = r"C:\Users\peter\Moje\Diplomka\DiplomovaPraca\kws\h_result"


