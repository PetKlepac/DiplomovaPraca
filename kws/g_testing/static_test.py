"""
Skript na testovanie modelu Wav2VecKWS na statickom testovacom datasete.
Vyhodnocuje pravdepodobnosti detekcie, generuje logy, štatistiky a boxploty.
"""

from pathlib import Path
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re
from datetime import datetime
from kws.c_model.m_w2v import Wav2VecKWS

# ────────────────────────────────────────────────────────────────
# PARAMETERS
# ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000  # cieľová vzorkovacia frekvencia
SENTENCE_THRESHOLD = 0.5  # hranica pre detekciu pozitívnej triedy
AUDIO_EXTENSIONS = {".wav"}  # podporované prípony súborov

# ────────────────────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
testing_set_dir = project_root / "kws" / "g_testing" / "dataset_static_test" / "segments_static_test_augmented"
DATASET = "d2"
model_path = project_root / "kws" / "h_result" / f"{DATASET}_m_w2v_c" / f"{DATASET}_m_w2v_c.pth"
model_dir = model_path.parent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = model_dir / f"test_log_{timestamp}.txt"
metadata_path = model_dir / f"metadata_{timestamp}.txt"

# ────────────────────────────────────────────────────────────────
# LOAD MODEL
# ────────────────────────────────────────────────────────────────
print("Načítavam Wav2Vec2 model...")
model = Wav2VecKWS(num_classes=2)  # inicializácia modelu s 2 triedami
state_dict = torch.load(model_path, map_location="cpu")  # načítanie váh na CPU
model.load_state_dict(state_dict)
model.eval()  # prepnutie do eval módu (vypnutie dropout atď.)
print(f"Model úspešne načítaný: {model_path.name}\n")


# ────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────
def list_audio_files(directory: Path, extensions: set) -> list[Path]:
    """Vráti zoradený zoznam audio súborov v priečinku."""
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files)


def infer_positive_prob(model: torch.nn.Module, waveform: torch.Tensor) -> float:
    """Vráti pravdepodobnosť pozitívnej triedy (index 1)."""
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)  # prevod stereo na mono (priemer kanálov)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # pridanie batch dimenzie
    waveform = waveform.float()

    with torch.no_grad():
        logits = model(waveform)
        if isinstance(logits, tuple):
            logits = logits[0]  # niektoré modely vracajú tuple (logits, ...)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return float(probs[1].item())  # pravdepodobnosť triedy 1 (pozitívna)


def get_base_id(file_stem: str) -> str:
    """Odstráni variant (_clean/_noisy) a vráti základné ID."""
    stem = re.sub(r'_(clean|noisy)$', '', file_stem)
    match = re.match(r'(\d{2}_[a-z_]+?)(?:_|$)', stem)
    return match.group(1) if match else stem


def get_numeric_id(file_stem: str) -> str:
    """Extrahuje číselnú časť ID (prvé dve číslice)."""
    match = re.match(r'(\d{2})', file_stem)
    return match.group(1) if match else file_stem


def get_variant(file_stem: str) -> str:
    """Určí variant súboru podľa prípony mena."""
    if file_stem.endswith('_clean'):
        return 'clean'
    elif file_stem.endswith('_noisy'):
        return 'noisy'
    return 'unknown'


def get_category(file_stem: str) -> str:
    """Kategorizuje súbor podľa kľúčových častí mena."""
    stem = file_stem.lower()
    if re.match(r'0[1-2]_pos_final', stem):
        return 'pos_final'
    elif 'pos_partial' in stem:
        return 'pos_partial'
    elif 'pos_prefix' in stem:
        return 'pos_prefix'
    elif 'pos_sufix' in stem:
        return 'pos_sufix'
    elif 'neg_soft' in stem:
        return 'neg_soft'
    elif 'neg_harder' in stem:
        return 'neg_harder'
    elif 'neg_hard' in stem:
        return 'neg_hard'
    elif 'noise' in stem:
        return 'noise'
    return 'other'


# ────────────────────────────────────────────────────────────────
# MAIN TESTING LOOP
# ────────────────────────────────────────────────────────────────
# Hlavný testovací cyklus - tu prebieha kompletné vyhodnotenie modelu
# Postup:
# 1. Načíta všetky audio súbory
# 2. Pre každý súbor samostatne:
#    • Načíta waveform
#    • Resampluje na 16kHz (ak treba)
#    • Spustí inferenciu modelu
#    • Určí detekciu podľa thresholdu
#    • Extrahuje metadata z názvu súboru
#    • Uloží výsledky do logov a štatistických štruktúr

audio_files = list_audio_files(testing_set_dir, AUDIO_EXTENSIONS)
print(f"Nájdených {len(audio_files)} súborov\n")

log_lines = []  # riadky pre finálny textový log
metadata_lines = []  # riadky pre podrobný metadata súbor

# Štruktúry na zber dát pre neskoršie štatistiky a grafy
base_probs_clean = defaultdict(list)  # prob. hodnoty pre clean varianty (podľa base_id)
base_probs_noisy = defaultdict(list)  # prob. hodnoty pre noisy varianty (podľa base_id)
base_to_typ = {}  # base_id → kategória (pos_final, neg_hard, ...)
base_to_numeric = {}  # base_id → číselné ID vzorky (pre os x v grafe)

for audio_path in audio_files:
    # === Načítanie audio súboru ===
    waveform_np, orig_sr = sf.read(str(audio_path), dtype="float32")
    waveform = torch.from_numpy(waveform_np).float()

    # === Resamplovanie na požadovanú frekvenciu (ak je potrebné) ===
    if orig_sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=SAMPLE_RATE
        )
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)  # resamplovanie na 16kHz

    # === Inferencia modelu ===
    prob = infer_positive_prob(model, waveform)  # pravdepodobnosť pozitívnej triedy (0..1)
    detected = prob > SENTENCE_THRESHOLD  # rozhodnutie: True = vetu detegoval

    # === Extrakcia informácií z názvu súboru ===
    file_id = audio_path.stem
    variant = get_variant(file_id)  # clean / noisy / unknown
    typ = get_category(file_id)  # pos_final, neg_hard, noise, ...
    base_id = get_base_id(file_id)  # základné ID (napr. 01_pozdrav)
    numeric_id = get_numeric_id(file_id)  # číselná časť pre graf (napr. 01)

    # === Výpis a logovanie výsledku ===
    line = f"{file_id:45} Prob: {prob:.4f} {'True' if detected else 'False'} {typ:12} base: {base_id} ({variant})"
    print(line)

    log_lines.append(line)
    metadata_lines.append(
        f"{file_id} | base={base_id} | variant={variant} | "
        f"category={typ} | prob={prob:.4f} | detected={detected}"
    )

    # === Uloženie výsledkov pre neskoršie analýzy ===
    if base_id not in base_to_typ:
        base_to_typ[base_id] = typ
        base_to_numeric[base_id] = numeric_id

    # Rozdelenie podľa variantu pre samostatné boxploty
    if variant == 'clean':
        base_probs_clean[base_id].append(prob)
    else:
        base_probs_noisy[base_id].append(prob)


# ────────────────────────────────────────────────────────────────
# FUNKCIA NA VYTVORENIE GRAFU
# ────────────────────────────────────────────────────────────────
def create_boxplot(base_probs_dict, title_suffix, filename_suffix):
    """Vytvorí boxplot distribúcie pravdepodobností pre danú skupinu."""
    if not base_probs_dict:
        return

    sorted_bases = sorted(base_probs_dict.keys(), key=lambda x: int(x[:2]))
    data = [base_probs_dict[b] for b in sorted_bases]

    fig_width = max(19, len(sorted_bases) * 0.52)
    plt.figure(figsize=(fig_width, 13.5))  # zväčšená výška grafu

    bp = plt.boxplot(data,
                     patch_artist=True,
                     showmeans=True,
                     meanline=True,
                     widths=0.95,  # takmer dotýkajúce sa boxy
                     boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
                     medianprops=dict(color='black', linewidth=4.0),
                     meanprops=dict(color='#1f77b4', linestyle=':', linewidth=2.5),
                     flierprops=dict(marker='o', color='black', alpha=0.6))

    plt.axhline(y=SENTENCE_THRESHOLD, color='red', linestyle='--', linewidth=3)  # hranica detekcie
    labels = [base_to_numeric[b] for b in sorted_bases]

    plt.xticks(range(1, len(labels) + 1), labels, rotation=90, ha='center', fontsize=16)
    plt.ylabel('Pravdepodobnosť detekcie', fontsize=22)
    plt.xlabel('ID vzorky', fontsize=22)
    plt.title(f'Distribúcia pravdepodobností – {title_suffix}', fontsize=24, pad=35)
    plt.ylim(0, 1.1)
    plt.yticks(np.arange(0, 1.01, 0.2))
    plt.grid(axis='y', alpha=0.3, linestyle=':')
    plt.tight_layout()

    chart_path = model_dir / f"prob_boxplots_{filename_suffix}_{timestamp[:12]}.png"
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Graf uložený: {chart_path.name}")


# ────────────────────────────────────────────────────────────────
# VYTVORENIE DVOCH GRAFOV
# ────────────────────────────────────────────────────────────────
print("\nGenerujem dva boxploty...")
create_boxplot(base_probs_clean, "CLEAN variants", "clean")
create_boxplot(base_probs_noisy, "NOISY variants", "noisy")


# ────────────────────────────────────────────────────────────────
# ŠTATISTIKY
# ────────────────────────────────────────────────────────────────
def print_stats(title, base_probs_dict):
    """Vypíše štatistiky pre danú skupinu variantov."""
    print("\n" + "=" * 100)
    print(f"ŠTATISTIKY – {title}")
    print("=" * 100)

    local_probs = defaultdict(list)
    for b in base_probs_dict:
        typ = base_to_typ[b]
        local_probs[typ].extend(base_probs_dict[b])

    for typ in sorted(local_probs):
        arr = np.array(local_probs[typ])
        print(f"{typ:18} | počet: {len(arr):4} | priemer: {arr.mean():.4f} ± {arr.std():.4f} | "
              f"medián: {np.median(arr):.4f} | >= {SENTENCE_THRESHOLD:.2f}: {(arr >= SENTENCE_THRESHOLD).sum():3}")


print_stats("CLEAN variants", base_probs_clean)
print_stats("NOISY variants", base_probs_noisy)

# ────────────────────────────────────────────────────────────────
# SAVE LOG + METADATA
# ────────────────────────────────────────────────────────────────
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

with open(metadata_path, "w", encoding="utf-8") as f:
    f.write("file_id | base | variant | category | prob | detected\n")
    f.write("-" * 80 + "\n")
    f.write("\n".join(metadata_lines))

print(f"\nLog uložený: {log_path}")
print(f"Metadata uložené: {metadata_path}")
print("Hotovo.")