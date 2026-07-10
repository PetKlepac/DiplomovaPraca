"""
KWS Vyhodnotenie – Metriky závislé od hodnoty prahu

Tento skript načíta výsledky testovania z CSV tabuľky a pre zadané prahy
vypočíta kľúčové metriky Keyword Spotting systému.

Pre každý prah vypočíta:
- Confusion Matrix (TP, FP, TN, FN)
- Recall, Precision, Specificity
- Počet segmentov, ktoré je potrebné manuálne skontrolovať
- Workload Reduction (percentuálne zníženie pracovnej záťaže)

Vstup:
    CSV súbor s výsledkami testovania (stĺpce: folder, max_prob)

Výstup:
    - Prehľadná tabuľka v konzole
    - CSV súbor s porovnaním metrík pre všetky prahy
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix


# ────────────────────────────────────────────────────────────────
# KONFIGURÁCIA
# ────────────────────────────────────────────────────────────────
MODEL_NAME = "d2_m_w2v_c"
EPOCH_TO_USE = 10

# === TU ZADÁŠ SVOJE TRI PRAHY ===
THRESHOLDS = [0.95, 0.80, 0.50]


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

print("=" * 80)
print("KWS VYHODNOTENIE – METRIKY ZÁVISLÉ OD PRAHU")
print("=" * 80)
print(f"Model : {MODEL_NAME}")
print(f"Epócha : {EPOCH_TO_USE}")
print(f"Prahy : {THRESHOLDS}")
print("=" * 80)


# ────────────────────────────────────────────────────────────────
# NAČÍTANIE DÁT
# ────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
df['y_true'] = (df['folder'] == 'positive').astype(int)

y_true = df['y_true'].values
y_score = df['max_prob'].values
total_samples = len(y_true)
n_pos = int((y_true == 1).sum())

results = []

for threshold in THRESHOLDS:
    y_pred = (y_score >= threshold).astype(int)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metriky
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    total_to_review = tp + fp
    workload_reduction = (1 - total_to_review / total_samples) * 100

    results.append({
        "Prah": threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Recall": round(recall, 4),
        "Precision": round(precision, 4),
        "Specificity": round(specificity, 4),
        "Segmenty na kontrolu": total_to_review,
        "Workload Reduction %": round(workload_reduction, 2)
    })


# ────────────────────────────────────────────────────────────────
# VÝSTUPNÁ TABUĽKA
# ────────────────────────────────────────────────────────────────
result_df = pd.DataFrame(results)

print("\nPorovnanie metrík pre jednotlivé prahy:\n")
print(result_df.to_string(index=False))

# Uloženie do CSV
output_csv = output_dir / f"threshold_comparison_{file_suffix}.csv"
result_df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"\n✓ Tabuľka uložená ako CSV: {output_csv.name}")
print("\n" + "=" * 80)
print("Hotovo.")
print("=" * 80)