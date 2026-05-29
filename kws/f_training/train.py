"""Hlavný skript na spustenie tréningu Keyword Spotting modelu."""

import os
import sys
import warnings
from pathlib import Path
import shutil
import importlib.util
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader

# ====================== WINDOWS + HF SETUP ======================
def setup_environment():
    """Všetko nastavenie prostredia na jednom mieste"""
    # Windows UTF-8
    if sys.platform == "win32":
        try:
            sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
            sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
        except:
            pass

    # Suppress HuggingFace warnings
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    warnings.filterwarnings("ignore", module="huggingface_hub")
    warnings.filterwarnings("ignore", module="transformers")

    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()


# ====================== IMPORTY PROJEKTU ======================
# Imported lazily inside main() to keep Windows spawn workers import-safe.

def main() -> None:
    setup_environment()

    from kws.f_training.dataset import RawWaveformDataset
    from kws.f_training.trainer import train_model

    # ────────────────────────────────────────────────────────────────
    # NAČÍTANIE CONFIGU
    # ────────────────────────────────────────────────────────────────
    CONFIG_NAME = "c"
    CONFIG_PATH = Path(r"C:\Users\peter\Moje\Diplomka\DiplomovaPraca\kws\d_config") / f"{CONFIG_NAME}.py"

    spec = importlib.util.spec_from_file_location(CONFIG_NAME, CONFIG_PATH)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)                 # načítanie config súboru ako modulu

    # ────────────────────────────────────────────────────────────────
    # VYTVORENIE EXPERIMENTU
    # ────────────────────────────────────────────────────────────────
    experiment_name = f"{config.DATASET_SLUG}_{config.MODEL_SLUG}_{config.CONFIG_SLUG}"
    experiment_dir = Path(config.RESULTS_BASE_DIR) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)   # vytvorenie priečinka experimentu

    print(f"Experiment: {experiment_name}")
    print(f"Priečinok: {experiment_dir}\n")

    shutil.copy(CONFIG_PATH, experiment_dir / f"{config.CONFIG_SLUG}.py")  # záloha configu

    # ────────────────────────────────────────────────────────────────
    # DATASET + DATALOADERS
    # ────────────────────────────────────────────────────────────────
    print("Načítavam dataset...")

    train_dataset = RawWaveformDataset(
        subdir="train",
        root_dir=config.DATA_ROOT,                  # cesta k datasetu
        target_samples=config.TARGET_SAMPLES,       # fixná dĺžka waveformu
        augment_training=config.AUGMENT_TRAINING_DATA,  # zapnutie augmentácií
    )

    val_dataset = RawWaveformDataset(
        subdir="val",
        root_dir=config.DATA_ROOT,
        target_samples=config.TARGET_SAMPLES,
        augment_training=False,                     # augmentácie vypnuté pri validácii
    )

    pin_memory = str(config.DEVICE).startswith("cuda") and torch.cuda.is_available()  # optimalizácia pre GPU

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,                               # náhodné miešanie tréningových dát
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,                      # zamknutá pamäť pre rýchly prenos na GPU
        drop_last=True,                             # zahodí posledný neúplný batch
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
    print("\n Počítam class weights...")

    # dočasný dataset BEZ augmentácií pre spočítanie
    temp_dataset = RawWaveformDataset(
        subdir="train",
        root_dir=config.DATA_ROOT,
        target_samples=config.TARGET_SAMPLES,
        augment_training=False,                     # <<< dôležité!
    )

    pos_count = sum(1 for _, label in temp_dataset if label == 1)
    total_count = len(temp_dataset)
    neg_count = total_count - pos_count

    print(f"Train dataset: {total_count:,} vzoriek")
    print(f"Pozitívne: {pos_count:,} | Negatívne: {neg_count:,}")

    w_pos = neg_count / pos_count if pos_count > 0 else 1.0
    class_weights = torch.tensor([1.0, w_pos])      # váhy pre vyváženie tried

    print(f"Váhy → neg: 1.0000 | pos: {w_pos:.4f}\n")

    # ────────────────────────────────────────────────────────────────
    # MODEL
    # ────────────────────────────────────────────────────────────────
    print("Načítavam model...")

    model = config.MODEL_CLASSES[config.MODEL_SLUG](
        num_classes=2,
        freeze_feature_extractor=True,              # zamrazenie CNN častí
        freeze_encoder=True,                        # zamrazenie Transformer encoderu
    )

    # ────────────────────────────────────────────────────────────────
    # TRÉNING
    # ────────────────────────────────────────────────────────────────
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        device=config.DEVICE,
        experiment_dir=experiment_dir,
        model_name=experiment_name,
        class_weights=class_weights,
    )

    # ────────────────────────────────────────────────────────────────
    # UKLADANIE
    # ────────────────────────────────────────────────────────────────
    model_path = experiment_dir / f"{experiment_name}.pth"
    torch.save(model.state_dict(), model_path)      # uloženie váh modelu

    print(f"\nModel uložený: {model_path}")
    print(f"Hotovo! Výsledky → {experiment_dir}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)        # bezpečné pre Windows multiprocessing
    main()