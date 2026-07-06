"""
KWS Vyhodnotenie – Metriky nezávislé od prahu

Tento skript vykonáva komplexné vyhodnotenie Keyword Spotting modelu,
ktoré nie je závislé od konkrétnej hodnoty prahu.

Vypočítava a vypisuje:
- ROC AUC
- Average Precision (AP)
- Equal Error Rate (EER) + prah
- Best F1 + prah
- Najmenšiu maximálnu pravdepodobnosť v pozitívnych vzorkách

Generuje grafy:
- Distribúciu pravdepodobností pozitívnych a negatívnych vzoriek
- PR krivku
- ROC krivku

Výstup:
    - Výpis metrík do konzoly (vrátane najmenšej pravdepodobnosti)
    - TXT súbor s metrikami (eval_..._metrics.txt)
    - Tri PNG grafy
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score
)

# ────────────────────────────────────────────────────────────────
# KONFIGURÁCIA
# ────────────────────────────────────────────────────────────────
MODEL_NAME = "d3_m_w2v_c"
EPOCH_TO_USE = 11

# ────────────────────────────────────────────────────────────────
# CESTY
# ────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
output_base = project_root / "kws" / "g_result" / MODEL_NAME / "testing"

if MODEL_NAME == "whisper":
    file_suffix = f"{MODEL_NAME}"
else:
    file_suffix = f"{MODEL_NAME}_e{EPOCH_TO_USE:02d}"

output_dir = output_base / file_suffix
csv_path = output_dir / f"table_{file_suffix}.csv"

print("=" * 75)
print("KWS VYHODNOTENIE – METRIKY NEZÁVISLÉ OD PRAHU")
print("=" * 75)
print(f"Model : {MODEL_NAME}")
print(f"Epócha: {EPOCH_TO_USE}")
print(f"CSV : {csv_path}")
print("=" * 75)

# ────────────────────────────────────────────────────────────────
# NAČÍTANIE DÁT
# ────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
df['y_true'] = (df['folder'] == 'positive').astype(int)
y_true = df['y_true'].values
y_score = df['max_prob'].values

n_pos = int((y_true == 1).sum())
n_neg = int((y_true == 0).sum())

print(f"\nPočet pozitívnych vzoriek : {n_pos}")
print(f"Počet negatívnych vzoriek : {n_neg}")

pos_scores = y_score[y_true == 1]
neg_scores = y_score[y_true == 0]

print(f"Pozitívne – priemer/medián : {np.mean(pos_scores):.4f} / {np.median(pos_scores):.4f}")
print(f"Negatívne  – priemer/medián : {np.mean(neg_scores):.4f} / {np.median(neg_scores):.4f}")

# === NOVÉ: Jasný výpis najmenšej pravdepodobnosti ===
min_pos_prob = float(np.min(pos_scores))
print(f"\n>>> Najmenšia maximálna pravdepodobnosť v pozitívnych vzorkách: {min_pos_prob:.4f} <<<")

# ────────────────────────────────────────────────────────────────
# VÝPOČTY METRÍK
# ────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

ap = average_precision_score(y_true, y_score)

# EER
fpr_eer, tpr_eer, thresholds_eer = roc_curve(y_true, y_score)
eer_idx = np.nanargmin(np.abs(fpr_eer - (1 - tpr_eer)))
eer = fpr_eer[eer_idx]
eer_threshold = thresholds_eer[eer_idx]

# Best F1
thresholds_f1 = np.linspace(0.05, 0.95, 181)
f1_scores = [f1_score(y_true, (y_score >= th).astype(int), zero_division=0) for th in thresholds_f1]
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds_f1[best_f1_idx]
best_f1_value = f1_scores[best_f1_idx]

# Výpis metrík
print(f"\nROC AUC               : {roc_auc:.4f}")
print(f"Average Precision (AP): {ap:.4f}")
print(f"EER                   : {eer*100:.2f}% (prah = {eer_threshold:.4f})")
print(f"Best F1               : {best_f1_value:.4f} (prah = {best_f1_threshold:.4f})")

# ────────────────────────────────────────────────────────────────
# ULOŽENIE METRÍK DO TXT SÚBORU
# ────────────────────────────────────────────────────────────────
txt_path = output_dir / f"eval_{file_suffix}_metrics.txt"

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("KWS VYHODNOTENIE – METRIKY NEZÁVISLÉ OD PRAHU\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Model : {MODEL_NAME}\n")
    f.write(f"Epócha: {EPOCH_TO_USE}\n\n")

    f.write(f"Počet pozitívnych vzoriek : {n_pos}\n")
    f.write(f"Počet negatívnych vzoriek : {n_neg}\n\n")

    f.write("Štatistiky pozitívnych vzoriek:\n")
    f.write(f"  Priemer  : {np.mean(pos_scores):.4f}\n")
    f.write(f"  Medián   : {np.median(pos_scores):.4f}\n")
    f.write(f"  Minimum  : {min_pos_prob:.4f}\n\n")

    f.write("Štatistiky negatívnych vzoriek:\n")
    f.write(f"  Priemer  : {np.mean(neg_scores):.4f}\n")
    f.write(f"  Medián   : {np.median(neg_scores):.4f}\n\n")

    f.write("-" * 50 + "\n")
    f.write("METRIKY NEZÁVISLÉ OD PRAHU\n")
    f.write("-" * 50 + "\n")
    f.write(f"ROC AUC               : {roc_auc:.4f}\n")
    f.write(f"Average Precision (AP): {ap:.4f}\n")
    f.write(f"EER                   : {eer*100:.2f}%   (prah = {eer_threshold:.4f})\n")
    f.write(f"Best F1               : {best_f1_value:.4f}   (prah = {best_f1_threshold:.4f})\n")
    f.write("-" * 50 + "\n")

print(f"\n✓ Uložené metriky do TXT: {txt_path.name}")

# ────────────────────────────────────────────────────────────────
# GRAF 1: ROZLOŽENIE PRAVDEPODOBNOSTÍ
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bins = np.linspace(0, 1, 51)

ax.hist(neg_scores, bins=bins, alpha=0.55, label=f'Negatívne (n={n_neg})',
        color='crimson', density=True, edgecolor='black', linewidth=0.3)
ax.hist(pos_scores, bins=bins, alpha=0.65, label=f'Pozitívne (n={n_pos})',
        color='limegreen', density=True, edgecolor='black', linewidth=0.3)

ax.set_xlabel('Maximálna pravdepodobnosť (kľúčového slova)', fontsize=12)
ax.set_ylabel('Hustota', fontsize=12)
ax.set_title(f'Rozloženie pravdepodobností – {file_suffix}', fontsize=14, pad=12)
ax.legend(loc='upper center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(0, 1.02)

plt.tight_layout()
plt.savefig(output_dir / f"eval_{file_suffix}_rozlozenie_pravdepodobnosti.png", dpi=220, bbox_inches='tight')
plt.close()
print("✓ Uložené: rozloženie pravdepodobností")

# ────────────────────────────────────────────────────────────────
# GRAF 2: PR KRIVKA
# ────────────────────────────────────────────────────────────────
precision, recall, _ = precision_recall_curve(y_true, y_score)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(recall, precision, color='darkgreen', lw=2.5)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title(f'PR krivka – {file_suffix}', fontsize=14, pad=12)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(output_dir / f"eval_{file_suffix}_krivka_pr.png", dpi=220, bbox_inches='tight')
plt.close()
print("✓ Uložené: PR krivka")

# ────────────────────────────────────────────────────────────────
# GRAF 3: ROC KRIVKA
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr, tpr, color='darkorange', lw=2.5)
ax.plot([0, 1], [0, 1], color='navy', lw=1.2, linestyle='--')
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title(f'ROC krivka – {file_suffix}', fontsize=14, pad=12)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig(output_dir / f"eval_{file_suffix}_krivka_roc.png", dpi=220, bbox_inches='tight')
plt.close()
print("✓ Uložené: ROC krivka")

print("\n" + "=" * 75)
print("Hotovo – vyhodnotenie dokončené.")
print("=" * 75)