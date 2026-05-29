from pathlib import Path
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from kws.c_model.m_w2v import Wav2VecKWS
from first_test_utils import (
    list_audio_files,
    sentence_probability,
    get_status_from_txt_by_file_id,
)

# ────────────────────────────────────────────────────────────────
# PARAMETERS
# ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
WINDOW_LENGTH_SEC = 1.0
WINDOW_HOP_SEC = 0.2
SENTENCE_THRESHOLD = 0.5
AGGREGATION_MODE = "max"          # "max" or "mean"
AUDIO_EXTENSIONS = {".wav"}

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
log_path = model_dir / f"test_log_{timestamp}.txt"

# ────────────────────────────────────────────────────────────────
# LOAD MODEL
# ────────────────────────────────────────────────────────────────
print("Načítavam Wav2Vec2 model...")
model = Wav2VecKWS(num_classes=2)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
print(f"Model úspešne načítaný: {model_path.name}\n")

# ────────────────────────────────────────────────────────────────
# LOAD AUDIO FILES
# ────────────────────────────────────────────────────────────────
audio_files = list_audio_files(testing_set_dir, AUDIO_EXTENSIONS)
print(f"Nájdených {len(audio_files)} testovacích súborov\n")

# Colors for terminal
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# ────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ────────────────────────────────────────────────────────────────
log_lines = []
log_lines.append(f"Test run: {timestamp}")
log_lines.append(f"Model: {model_path.name}")
log_lines.append(f"Dataset folder: {testing_set_dir}")
log_lines.append(f"Threshold: {SENTENCE_THRESHOLD:.3f} | Aggregation: {AGGREGATION_MODE}")
log_lines.append("-" * 80)

# ────────────────────────────────────────────────────────────────
# MAIN TESTING LOOP
# ────────────────────────────────────────────────────────────────
probs_by_type = defaultdict(list)
all_probs = []          # list of tuples: (probability, type, original_index)

tp_detected = fn_detected = tp_partial = fn_partial = 0
fp_none = tn_none = fp_confuse = tn_confuse = 0

for idx, audio_path in enumerate(audio_files, 1):   # idx starts from 1
    sent_prob = sentence_probability(
        model=model,
        audio_path=audio_path,
        sample_rate=SAMPLE_RATE,
        window_length_sec=WINDOW_LENGTH_SEC,
        window_hop_sec=WINDOW_HOP_SEC,
        aggregation_mode=AGGREGATION_MODE,
    )

    detected = sent_prob > SENTENCE_THRESHOLD
    file_id = audio_path.stem
    # Extract numeric prefix (001, 002, etc.)
    numeric_id = file_id.split('_')[0] if '_' in file_id else file_id
    status = "True" if detected else "False"
    color = GREEN if detected else RED

    try:
        typ = get_status_from_txt_by_file_id(txt_path, numeric_id)
    except KeyError:
        print(f"⚠ Skipping '{file_id}' - '{numeric_id}' not found in lookup file")
        log_lines.append(f"⚠ Skipped: '{file_id}'")
        continue

    line = f"{numeric_id} - Prob: {sent_prob:.4f} - {color}{status}{RESET} - {typ}"
    print(line)
    log_lines.append(line)

    probs_by_type[typ].append(sent_prob)
    all_probs.append((sent_prob, typ, idx))   # store original sample number

    # Statistics
    if typ == "detected":
        if detected: tp_detected += 1
        else: fn_detected += 1
    elif typ == "partial":
        if detected: tp_partial += 1
        else: fn_partial += 1
    elif typ == "none":
        if detected: fp_none += 1
        else: tn_none += 1
    elif typ == "confuse":
        if detected: fp_confuse += 1
        else: tn_confuse += 1

# ────────────────────────────────────────────────────────────────
# STATISTICS
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ŠTATISTIKY – PRIEMERNÉ PRAVDEPODOBNOSTI PODĽA TYPU")
print("-" * 80)
for typ in sorted(probs_by_type):
    hodnoty = probs_by_type[typ]
    if not hodnoty: continue
    arr = np.array(hodnoty)
    stat_line = (
        f"{typ:18} | počet: {len(arr):4} | "
        f"priemer: {arr.mean():.4f} ± {arr.std():.4f} | "
        f"medián: {np.median(arr):.4f} | "
        f">= {SENTENCE_THRESHOLD:.2f}: {(arr >= SENTENCE_THRESHOLD).sum():3} ks"
    )
    print(stat_line)
    log_lines.append(stat_line)

print("=" * 80)
print("\nVýsledky detekcie:")
print(f"TP detected : {tp_detected}")
print(f"FN detected : {fn_detected}")
print(f"TP partial : {tp_partial}")
print(f"FN partial : {fn_partial}")
print(f"FP none : {fp_none}")
print(f"TN none : {tn_none}")
print(f"FP confuse : {fp_confuse}")
print(f"TN confuse : {tn_confuse}")

log_lines.append("\nVýsledky detekcie:")
log_lines.extend([
    f"TP detected : {tp_detected}", f"FN detected : {fn_detected}",
    f"TP partial : {tp_partial}", f"FN partial : {fn_partial}",
    f"FP none : {fp_none}", f"TN none : {tn_none}",
    f"FP confuse : {fp_confuse}", f"TN confuse : {tn_confuse}"
])

# ────────────────────────────────────────────────────────────────
# PLOT - GROUPED + ORIGINAL SAMPLE NUMBERS ABOVE BARS
# ────────────────────────────────────────────────────────────────
print("\nGenerujem graf...")

color_map = {
    'detected': '#006400',
    'partial': '#90EE90',
    'none': '#808080',
    'confuse': '#FFA500',
}
default_color = '#4682B4'

group_order = ['detected', 'partial', 'confuse', 'none']

# Sort by group, then by probability ascending inside each group
sorted_samples = []
for group in group_order:
    group_samples = [item for item in all_probs if item[1] == group]
    group_samples.sort(key=lambda x: x[0])        # lowest prob first
    sorted_samples.extend(group_samples)

# Prepare data
probs = [p for p, _, _ in sorted_samples]
bar_colors = [color_map.get(t, default_color) for _, t, _ in sorted_samples]
original_numbers = [idx for _, _, idx in sorted_samples]   # original sample numbers

plt.figure(figsize=(18, 8))

plt.bar(range(1, len(probs) + 1), probs, color=bar_colors,
        width=0.75, edgecolor='black', linewidth=0.4)

plt.axhline(y=SENTENCE_THRESHOLD, color='red', linestyle='--', linewidth=2.5,
            label=f'Threshold = {SENTENCE_THRESHOLD:.2f}')

# Add ORIGINAL sample numbers above each bar
for i, (prob, orig_num) in enumerate(zip(probs, original_numbers), 1):
    plt.text(i, prob + 0.012, str(orig_num), ha='center', va='bottom',
             fontsize=8.5, rotation=90, fontweight='bold')

# Group separators
cumulative = 0
for group in group_order[:-1]:
    size = len([s for s in sorted_samples if s[1] == group])
    cumulative += size
    plt.axvline(x=cumulative + 0.5, color='black', linestyle=':', alpha=0.5, linewidth=1.2)

# Clean x-axis as requested
plt.xticks([])                    # no numbers on horizontal axis
# no xlabel

plt.ylabel('Pravdepodobnosť detekcie', fontsize=12)
plt.title(f'Pravdepodobnosti detekcie – všetky testovacie vzorky (n = {len(audio_files)})\n'
          f'Zoskupené: detected → partial → confuse → none | v rámci skupiny podľa rastúcej pravdepodobnosti',
          fontsize=13, pad=25)

plt.ylim(0, 1.10)
plt.grid(axis='y', alpha=0.3, linestyle=':')

plt.tight_layout()
chart_path = model_dir / f"prob_bars_grouped_{timestamp[:12]}.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"Graf uložený: {chart_path.name}")
log_lines.append(f"\nGraf uložený: {chart_path.name}")

# ────────────────────────────────────────────────────────────────
# SAVE LOG
# ────────────────────────────────────────────────────────────────
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print(f"\nVýsledky uložené do: {log_path}")
print("Hotovo.")