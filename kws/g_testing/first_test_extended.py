from pathlib import Path
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from kws.c_model.m_w2v import Wav2VecKWS

# Import from the new Wav2Vec2 utils
from kws.g_testing.first_test_utils import (
    list_audio_files,
    sentence_probability,
    get_sentence_info_from_txt_by_file_id,
)

# ────────────────────────────────────────────────────────────────
# PARAMETERS
# ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
WINDOW_LENGTH_SEC = 1.0
WINDOW_HOP_SEC = 0.1
SENTENCE_THRESHOLD = 0.5
AGGREGATION_MODE = "max"

AUDIO_EXTENSIONS = {".wav", ".wave", ".flac", ".mp3", ".ogg", ".m4a"}

# ────────────────────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
testing_set_dir = project_root / "kws" / "g_testing" / "dataset_first_test" / "testing_set_cze"

DATASET = "d2"
model_path = project_root / "kws" / "h_result" / f"{DATASET}_m_w2v_c" / f"{DATASET}_m_w2v_c.pth"
txt_path = project_root / "kws" / "g_testing" / "dataset_first_test" / "test_file_cze.txt"


model_dir = model_path.parent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
detailed_dir = model_dir / "detailed_sliding_curves"
detailed_dir.mkdir(parents=True, exist_ok=True)

print(f"Detailné sliding window grafy budú uložené do:\n{detailed_dir}\n")

# ────────────────────────────────────────────────────────────────
# LOAD MODEL
# ────────────────────────────────────────────────────────────────
model = Wav2VecKWS(num_classes=2)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ────────────────────────────────────────────────────────────────
# LOAD AUDIO FILES
# ────────────────────────────────────────────────────────────────
audio_files = list_audio_files(testing_set_dir, AUDIO_EXTENSIONS)
print(f"Nájdených {len(audio_files)} testovacích súborov\n")

# ────────────────────────────────────────────────────────────────
# GENERATE GRAPHS FOR EACH SAMPLE
# ────────────────────────────────────────────────────────────────
for i, audio_path in enumerate(audio_files, 1):
    file_id = audio_path.stem

    # Get sentence text and type
    try:
        sentence_text, typ = get_sentence_info_from_txt_by_file_id(txt_path, file_id)
    except Exception:
        sentence_text = "[Text vety sa nepodarilo načítať]"
        typ = "unknown"
        print(f"Varovanie: Nepodarilo sa načítať info pre {file_id}")

    # NEW CALL – without MFCC parameters
    sent_prob, all_window_probs = sentence_probability(
        model=model,
        audio_path=audio_path,
        sample_rate=SAMPLE_RATE,
        window_length_sec=WINDOW_LENGTH_SEC,
        window_hop_sec=WINDOW_HOP_SEC,
        aggregation_mode=AGGREGATION_MODE,
        return_all_probs=True
    )

    # Time axis (center of each window)
    times = [
        (step * WINDOW_HOP_SEC) + (WINDOW_LENGTH_SEC / 2.0)
        for step in range(len(all_window_probs))
    ]

    # ───── PLOT ─────
    plt.figure(figsize=(13, 7))
    plt.plot(times, all_window_probs, 'b-', linewidth=2.5, label='Pravdepodobnosť v okne')
    plt.axhline(y=SENTENCE_THRESHOLD, color='red', linestyle='--', linewidth=2.5,
                label=f'Threshold = {SENTENCE_THRESHOLD:.2f}')
    plt.axhline(y=sent_prob, color='green', linestyle='-', linewidth=2.5,
                label=f'Maximum = {sent_prob:.4f}')

    plt.xlabel('Čas [sekundy]', fontsize=13)
    plt.ylabel('Pravdepodobnosť', fontsize=13)
    plt.title(f'Sliding Window Analýza – {file_id}\n'
              f'Typ: {typ.upper()} | Max prob: {sent_prob:.4f}\n'
              f'Veta: "{sentence_text}"', fontsize=13, pad=20, loc='left')

    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.35, linestyle=':')
    plt.legend(fontsize=11)
    plt.tight_layout()

    # Save graph
    graph_name = f"{file_id}_sliding_{sent_prob:.4f}.png"
    graph_path = detailed_dir / graph_name
    plt.savefig(graph_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"{i:3d}/{len(audio_files)} → {graph_name} | {typ} | {sent_prob:.4f}")

print("\n" + "=" * 80)
print(f"HOTOVO!")
print(f"Všetky detailné grafy boli uložené do:")
print(detailed_dir)
print("=" * 80)