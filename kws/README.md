# Dokumentácia

Tento repozitár obsahuje kompletný kód diplomovej práce na tému **Detekce klíčových slov v letecké komunikaci pomocí metod strojového učení**.

Autor: **Bc. Peter Klepáč**

Vedúci práce: **Ing. Matouš Cejnek, Ph.D.**

(ČVUT FS, Ústav přístrojové a řídicí techniky, 2026)  

Cieľom je automaticky detegovať kľúčové slovo "finále" v audio nahrávkach leteckej komunikácie (LKJH) a tým 
znížiť pracovnú záťaž operátorov letiska.

## Použitá metóda
- **Model**: Wav2Vec2ForSequenceClassification (Hugging Face) fine-tuned na binary klasifikáciu (positive = kľúčové slovo, negative = ostatné).
- **Stratégia zamrazovania**: feature extractor + väčšina encoderu zamrazená, posledné 3 vrstvy + classification head sa trénujú.
- **Augmentácia**: clean verzie + 11 noisy verzií s náhodným SNR (5–15 dB).
- **Štnadardizácia dát**: všetky audio súbory sú mono, 16 kHz, presne 1 s, –22 dBFS.
- **Testovanie**: inferencia na dlhých nahrávkach pomocou **sliding window** (1 s okná, krok 0.05 s) → max. pravdepodobnosť.

## Štruktúra repozitára

```
kws/
├── a_data_preparation/          # Príprava a anotácia audio dát
│   ├── prepare_noises.py
│   ├── extract_long_segments_from_long_records.py
│   ├── extract_short_negative_segments_from_short_records.py
│   └── annotate_long_segments.py          
├── b_dataset/                   # Tvorba a ukladanie finálnych datasetov,
│   └── create_dataset.py
├── c_model/                     # Definícia modelu
│   └── m_w2v.py                 # Wav2VecKWS trieda
├── d_config/                    # Konfigurácia experimentu
│   └── c.py
├── f_training/                  # Tréningový proces
│   ├── train.py                 # Vstupný bod tréningu
│   ├── trainer.py
│   └── dataset.py
├── g_testing/                   # Testovanie a vyhodnotenie modelu
│   ├── test.py                  # Testovanie (sliding-window inferencia)
│   ├── evaluate_independent.py  # Nezávislé metriky
│   ├── evaluate_dependent.py    # Závislé metriky
│   └── create_test_charts.py    # Priame grafy z výsledkov testovania
└── h_result/                    # Výsledky tréningov a testovaní
    └── d3_m_w2v_c/              # (príklad experimentu)
        ├── models/              # .pth checkpointy
        ├── training/            # Súbory tréningu
        └── testing/             # Súbory testovania a vyhodnotenia
```

---

## Inštalácia závislostí

V koreňovom adresári repozitára je potrebné vytvoriť súbor requirements.txt s obsahom:

```txt
torch==2.12.0.dev20260309+cu128
torchaudio==2.11.0.dev20260407+cu128
transformers==5.7.0
pydub==0.25.1
pygame==2.6.1
numpy==2.4.4
pandas==3.0.2
matplotlib==3.10.9
scikit-learn==1.9.0
soundfile==0.13.1
tqdm==4.67.3
joblib==1.5.3
```

Závislosti je možné nainštalovať príkazom:

```bash
pip install -r requirements.txt
```

Odporúča sa vytvoriť virtuálne prostredie (napr. python -m venv venv && source venv/bin/activate).

---

## Obecný postup

**Dôležité upozornenie!**  
Verejne dostupný repozitár **neobsahuje surové audio dáta**.

---

### 1. Príprava dát

Vstupné audio dáta sa delia na dve základné kategórie:

- **Long records** – dlhé nahrávky obsahujúce tiché pauzy medzi vetami, ktoré vznikajú nahrávaním celého dňa.
- **Short records** – krátke nahrávky bez páuz, ktoré vznikajú nahrávaním celého dňa s použitím VoiceActivation.

Podľa typu dostupných nahrávok sa volí postup získavania vzoriek. Nasledujúci opis uvádza odporúčaný postup.

#### Príprava šumových súborov

Skript `prepare_noises.py` normalizuje a štandardizuje súbory so šumom (premenovanie, orezanie na maximálnu dĺžku).  
Výstupné súbory slúžia neskôr ako zdroj šumu pri augmentácii tréningových dát.

#### Získanie viet z dlhých nahrávok (long records)

Ak sú k dispozícii dlhé nahrávky s pauzami, použije sa skript `extract_long_segments_from_long_records.py`.  
Na základe detekcie ticha rozdelí nahrávku na samostatné rečové segmenty (vety).  
Výstupom sú súbory použiteľné pre testovanie (inferenciu).

#### Anotácia získaných segmentov

Segmenty vytvorené v predchádzajúcom kroku sa anotujú pomocou skriptu `annotate_long_segments.py`.  
Používateľ interaktívne priraďuje jednotlivým segmentom kategórie (positive, negative, engpositive, bin) pomocou klávesnice.  
Anotované súbory sa automaticky triedia do príslušných podpriečinkov.

#### Získanie negatívnych vzoriek z krátkych nahrávok

Pre nahrávky bez pauz sa používa skript `extract_short_negative_segments_from_short_records.py`.  
Generuje rovnomerne rozložené krátke segmenty a zároveň umožňuje ich interaktívne schvaľovanie.  
Tento prístup sa používa najmä preto, že krátke nahrávky môžu obsahovať aj pozitívne vzorky, a preto je potrebné ich overiť hneď pri extrakcii.

### 2. Vytvorenie datasetu

Po získaní a anotovaní pozitívnych aj negatívnych vzoriek sa spustí skript `b_dataset/create_dataset.py`.  
Skript zhromažďuje anotované súbory podľa definovanej konfigurácie priečinkov a vytvára štruktúrovaný dataset rozdelený na tréningovú a validačnú časť.

### 3. Model

Priečinok `c_model` obsahuje definíciu modelu. 
Súbor `m_w2v.py` definuje triedu `Wav2VecKWS`, ktorá obaľuje predtrénovaný model `Wav2Vec2ForSequenceClassification`. 
Model podporuje selektívne zamrazovanie vrstiev feature extractora a enkodéra a umožňuje odomknutie posledných vrstiev enkodéra.

### 4. Konfigurácia

Priečinok `d_config` obsahuje konfiguračné súbory jednotlivých experimentov. 
Každý súbor, napríklad `c.py`, definuje identifikátory experimentu v podobe `DATASET_SLUG`, `MODEL_SLUG` a `CONFIG_SLUG`. 
Obsahuje všetky hyperparametre potrebné pre tréning, ako sú počet epoch, learning rate, veľkosť batchu, weight decay, nastavenia scheduleru, stratégia zamrazovania vrstiev a použitie augmentácie. 
Určuje tiež cesty k tréningovým dátam a k výstupným priečinkom výsledkov.

### 5. Tréning

Priečinok `e_training` obsahuje všetko potrebné na tréning modelu. 
Súbor `dataset.py` definuje triedu `RawWaveformDataset`, ktorá načítava surové waveformy zo súborov .wav z priečinkov positive a negative a nastavujú sa v nej aj augmentácie počas tréningu. 
Súbor `train.py` predstavuje hlavný tréningový skript, ktorý načíta konfiguráciu, pripraví DataLoadery, vypočíta váhy tried, vytvorí model a spustí tréning. 
Súbor `trainer.py` implementuje samotnú tréningovú slučku vrátane validácie, použitia ReduceLROnPlateau scheduleru, ukladania modelov po každej epoche a generovania grafov a CSV súborov s metrikami.

### 6. Testovanie a vyhodnocovanie

Priečinok `f_testing` obsahuje skripty na testovanie a vyhodnocovanie natrénovaného modelu. 
Súbor `test.py` vykonáva sliding window inferenciu na dlhších audio súboroch a vytvára CSV tabuľku s maximálnou pravdepodobnosťou pre každý testovací súbor. 
Súbor `evaluate_independent.py` počíta metriky nezávislé od prahu, ako sú ROC AUC, Average Precision, EER a Best F1, a generuje príslušné grafy vrátane distribúcie pravdepodobností, ROC krivky a PR krivky. 
Súbor `evaluate_dependent.py` počíta metriky závislé od prahu, ako sú Recall, Precision, Specificity a Workload Reduction, pre zadané prahové hodnoty. 
Súbor `analyze_model.py` analyzuje natrénovaný model z hľadiska počtu parametrov, času inferencie a spotreby GPU pamäte. 
Súbor `create_test_charts.py` generuje prehľadné stĺpcové grafy z výsledkov testovania uložených v CSV tabuľke.

### 7. Výsledky

Priečinok `g_result` je hlavný výstupný priečinok projektu. Má štruktúru `<DATASET>_<MODEL>_<CONFIG>`, napríklad `d2_m_w2v_c`. 
Vo vnútri každého experimentu sa nachádza priečinok `models`, ktorý obsahuje uložené checkpointy modelu po jednotlivých epochách. 
Priečinok `training` obsahuje tréningové krivky, CSV súbory s metrikami a kópiu konfiguračného súboru. 
Priečinok `testing` obsahuje výsledky testovania vrátane CSV tabuliek, grafov a vyhodnotení. 
Táto štruktúra umožňuje prehľadné ukladanie a porovnávanie viacerých experimentov.

---

**Zhrnutie pipeline**

Na prípravu dát slúžia skripty `prepare_noises.py`, `extract_long_segments_from_long_records.py`, `annotate_long_segments.py` a `extract_short_negative_segments_from_short_records.py`.  
Po ich použití sa spustí `b_dataset/create_dataset.py`.  
Pred spustením tréningu sa upraví konfiguračný súbor `d_config/c.py` a je možné upraviť augmentácie v súbore `e_training/dataset.py`.  
Následne sa spustí tréning `e_training/train.py`.  
Na testovanie poslúžia skripty v `f_testing` ako `test.py`, `evaluate_independent.py`, `evaluate_dependent.py`, `analyze_model.py` a `create_test_charts.py`.
Výsledky tréningu a testovania je možné prehliadať v `g_result`.