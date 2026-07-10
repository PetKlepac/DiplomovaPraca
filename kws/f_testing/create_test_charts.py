"""
Skript na generovanie grafov z existujúcej tabuľky výsledkov testovania Keyword Spotting.

Tento skript načíta CSV tabuľku s výsledkami testovania (stĺpce: global_id, folder, max_prob)
a vygeneruje prehľadné stĺpcové grafy pre:
- Pozitívne vzorky
- Negatívne vzorky (rozdelené do dvoch grafov pre lepšiu čitateľnosť)

Hlavné funkcie:
- Používa lokálne číslovanie vzoriek v grafoch
- Negatívne vzorky sú rozdelené na dve časti s pokračujúcim číslovaním
- Všetky grafy sú uložené do priečinka konkrétnej epochy
- Zachováva rovnakú štruktúru výstupných súborov ako testovací skript

Vstup:
    CSV súbor s výsledkami testovania (table_*.csv)

Výstup:
    Viacero PNG grafov uložených do output_dir
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────
# KONFIGURÁCIA
# ────────────────────────────────────────────────────────────────
MODEL_NAME = "d2_m_w2v_c"
EPOCH_TO_USE = 10
THRESHOLD = 0.50


# ────────────────────────────────────────────────────────────────
# CESTY
# ────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
output_base = project_root / "kws" / "g_result" / MODEL_NAME / "testing"

if MODEL_NAME == "whisper":
    file_suffix = f"{MODEL_NAME}"
else:
    file_suffix = f"{MODEL_NAME}_e{EPOCH_TO_USE:02d}"

# Vytvoríme / použijeme rovnaký priečinok ako test.py
output_dir = output_base / file_suffix
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / f"table_{file_suffix}.csv"

print("=" * 75)
print("GENERÁTOR GRAFOV PRE KWS (ukladanie do podpriečinka epochy)")
print("=" * 75)
print(f"Model: {MODEL_NAME}")
print(f"Epoch: {EPOCH_TO_USE}")
print(f"Threshold: {THRESHOLD}")
print(f"Tabuľka: {csv_path}")
print(f"Výstupný priečinok: {output_dir}")
print("=" * 75)


# ────────────────────────────────────────────────────────────────
# NAČÍTANIE TABUĽKY
# ────────────────────────────────────────────────────────────────
if not csv_path.exists():
    raise FileNotFoundError(f"Tabuľka nenájdená: {csv_path}")

df = pd.read_csv(csv_path)

if 'detected' in df.columns:
    df = df.drop(columns=['detected'])

print(f"Načítaných {len(df)} záznamov.\n")

required = {'global_id', 'folder', 'max_prob'}
if not required.issubset(df.columns):
    raise ValueError(f"Chýbajú stĺpce: {required - set(df.columns)}")


# ────────────────────────────────────────────────────────────────
# FUNKCIA NA TVORBU GRAFU S LOKÁLNYM ČÍSLOVANÍM
# ────────────────────────────────────────────────────────────────
def create_bar_chart(subset, title, color, filename, threshold, start_id=1):
    """Vytvorí stĺpcový graf s lokálnym číslovaním vzoriek."""
    if subset.empty:
        print(f" Preskakujem: {title}")
        return

    subset = subset.sort_values('global_id').reset_index(drop=True)
    n = len(subset)
    local_ids = list(range(start_id, start_id + n))

    fig, ax = plt.subplots(figsize=(18, 7))

    ax.bar(
        local_ids,
        subset['max_prob'],
        color=color,
        width=0.85,
        alpha=0.92,
        edgecolor='black',
        linewidth=0.4
    )

    ax.axhline(
        y=threshold,
        color='black',
        linestyle='--',
        linewidth=2.3,
        label=f'Prah = {threshold}'
    )

    step = max(1, n // 28)
    tick_positions = local_ids[::step]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, rotation=90, ha='center', fontsize=9)
    ax.set_title(title, fontsize=15, pad=18, fontweight='bold')
    ax.set_xlabel('Vzorky', fontsize=11)
    ax.set_ylabel('Maximálna pravdepodobnosť', fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=220, bbox_inches='tight')
    plt.close()

    print(f" ✓ Uložený: {filename}")


# ────────────────────────────────────────────────────────────────
# 1. POSITIVE
# ────────────────────────────────────────────────────────────────
positive_df = df[df['folder'] == 'positive'].copy()

create_bar_chart(
    subset=positive_df,
    title=f"Pozitívne vzorky",
    color='limegreen',
    filename=f"charts_{file_suffix}_positives.png",
    threshold=THRESHOLD,
    start_id=1
)


# ────────────────────────────────────────────────────────────────
# 2. NEGATIVE — rozdelené s pokračujúcim číslovaním
# ────────────────────────────────────────────────────────────────
negative_df = df[df['folder'] == 'negative'].copy().sort_values('global_id').reset_index(drop=True)

half = len(negative_df) // 2
neg_part1 = negative_df.iloc[:half]
neg_part2 = negative_df.iloc[half:]

create_bar_chart(
    subset=neg_part1,
    title=f"Negatívne vzorky 1–{len(neg_part1)}",
    color='crimson',
    filename=f"charts_{file_suffix}_negatives_part1.png",
    threshold=THRESHOLD,
    start_id=1
)

create_bar_chart(
    subset=neg_part2,
    title=f"Negatívne vzorky {len(neg_part1)+1}–{len(negative_df)}",
    color='crimson',
    filename=f"charts_{file_suffix}_negatives_part2.png",
    threshold=THRESHOLD,
    start_id=len(neg_part1) + 1
)


# ────────────────────────────────────────────────────────────────
# ZHRNUTIE
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("GRAFOVANIE DOKONČENÉ")
print("=" * 75)
print(f"Positive samples : {len(positive_df):>3} (číslovanie od 1)")
print(f"Negative Part 1 : {len(neg_part1):>3} (číslovanie od 1)")
print(f"Negative Part 2 : {len(neg_part2):>3} (pokračovanie číslovania)")
print(f"Threshold : {THRESHOLD}")
print(f"Výstupný priečinok: {output_dir}")
print("=" * 75)