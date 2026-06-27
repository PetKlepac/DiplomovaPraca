"""
Hlavný skript na spustenie tréningu Keyword Spotting modelu.

Tento skript vykonáva kompletný tréningový pipeline:
1. Načíta konfiguráciu z externého súboru (zložka d_config)
2. Vytvorí priečinok pre experiment s unikátnym názvom
3. Pripraví train a val DataLoader-y s RawWaveformDataset
4. Vypočíta váhy tried (class weights) pre vyvážený tréning
5. Vytvorí model podľa konfigurácie (s možnosťou freezovania vrstiev)
6. Spustí tréning pomocou funkcie train_model()
7. Uloží výsledky, checkpointy a históriu do experiment_dir

Výstup:
    - Priečinok experimentu obsahujúci model, logy a výsledky
    - Vypísané informácie o datasete a priebehu tréningu
"""

import os
import warnings
from pathlib import Path
import importlib.util
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
from kws.e_training.dataset import RawWaveformDataset
from kws.e_training.trainer import train_model
from transformers import logging as hf_logging


def main() -> None:
    # ────────────────────────────────────────────────────────────────
    # POTLAČENIE SPRÁV
    # ────────────────────────────────────────────────────────────────
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    warnings.filterwarnings("ignore", module="huggingface_hub")
    warnings.filterwarnings("ignore", module="transformers")
    hf_logging.set_verbosity_error()

    # ────────────────────────────────────────────────────────────────
    # NAČÍTANIE CONFIGU
    # ────────────────────────────────────────────────────────────────
    CONFIG_NAME = "c"
    project_root = Path(__file__).resolve().parents[2]
    CONFIG_PATH = project_root / "kws" / "d_config" / f"{CONFIG_NAME}.py"

    spec = importlib.util.spec_from_file_location(CONFIG_NAME, CONFIG_PATH)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)  # načítanie config súboru ako modulu

    # ────────────────────────────────────────────────────────────────
    # VYTVORENIE PRIEČINKOV EXPERIMENTU
    # ────────────────────────────────────────────────────────────────
    experiment_name = f"{config.DATASET_SLUG}_{config.MODEL_SLUG}_{config.CONFIG_SLUG}"
    experiment_dir = Path(config.RESULTS_BASE_DIR) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)  # vytvorenie priečinka experimentu

    print(f"Experiment: {experiment_name}")
    print(f"Priečinok: {experiment_dir}\n")

    # ────────────────────────────────────────────────────────────────
    # DATASET A DATALOADERS
    # ────────────────────────────────────────────────────────────────
    print("Načítavam dataset...")

    train_dataset = RawWaveformDataset(
        subdir="train",
        root_dir=config.DATA_ROOT,                    # cesta k datasetu
        target_samples=config.TARGET_SAMPLES,         # fixná dĺžka waveformu
        augment_training=config.AUGMENT_TRAINING_DATA,  # zapnutie augmentácií
    )

    val_dataset = RawWaveformDataset(
        subdir="val",
        root_dir=config.DATA_ROOT,
        target_samples=config.TARGET_SAMPLES,
        augment_training=False,  # augmentácie vypnuté pri validácii
    )

    pin_memory = str(config.DEVICE).startswith("cuda") and torch.cuda.is_available()  # optimalizácia pre GPU

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,           # náhodné miešanie tréningových dát
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,  # zamknutá pamäť pre rýchly prenos na GPU
        drop_last=True,         # zahodí posledný neúplný batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # ────────────────────────────────────────────────────────────────
    # VÁHY PRE CrossEntropyLoss
    # ────────────────────────────────────────────────────────────────
    print("\nPočítam class weights...")

    temp_dataset = RawWaveformDataset(
        subdir="train",
        root_dir=config.DATA_ROOT,
        target_samples=config.TARGET_SAMPLES,
        augment_training=False,
    )

    pos_count = sum(1 for _, label in temp_dataset if label == 1)
    total_count = len(temp_dataset)
    neg_count = total_count - pos_count

    print(f"Trénovací dataset: {total_count:,} vzoriek")
    print(f"Pozitívne: {pos_count:,} | Negatívne: {neg_count:,}")

    if config.CLASS_WEIGHTS is None:
        w_pos = neg_count / pos_count if pos_count > 0 else 1.0
        class_weights = torch.tensor([1.0, w_pos])
        print(f"CLASS_WEIGHTS = None")
        print(f"Váhy: neg: 1.0000 | pos: {w_pos:.4f}")
    else:
        class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32)
        print(f"CLASS_WEIGHTS = {config.CLASS_WEIGHTS} - použité manuálne nastavenie")

    print("")

    # ────────────────────────────────────────────────────────────────
    # MODEL A FREEZING
    # ────────────────────────────────────────────────────────────────
    print("Načítavam model...")

    model = config.MODEL_CLASSES[config.MODEL_SLUG](
        num_classes=2,
        freeze_feature_extractor=config.FREEZE_FEATURE_EXTRACTOR,
        freeze_encoder=config.FREEZE_ENCODER,
        unfreeze_last_n_layers=config.UNFREEZE_LAST_N_LAYERS,
        verbose=True,
    )

    # ────────────────────────────────────────────────────────────────
    # TRÉNING
    # ────────────────────────────────────────────────────────────────
    print("Trénujem...")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        class_weights=class_weights,
        experiment_dir=experiment_dir,
        model_name=experiment_name,
    )

    print(f"Hotovo! Výsledky → {experiment_dir}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # bezpečné pre Windows multiprocessing
    main()