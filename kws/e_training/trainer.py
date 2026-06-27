"""
Trénovacia funkcia pre PyTorch modely v Keyword Spotting.

Táto funkcia vykonáva kompletný tréningový cyklus s validáciou:
- Tréning s možnosťou váženej CrossEntropyLoss
- Validácia po každej epoche (voliteľná)
- Použitie ReduceLROnPlateau scheduleru
- Ukladanie modelu po každej epoche
- Zaznamenávanie histórie (loss + accuracy)
- Automatické uloženie metrík do CSV a grafov (loss + accuracy)
- Podpora experimentov s ukladaním konfiguračného súboru

Vstupy:
    model, train_loader, val_loader, config, class_weights, experiment_dir

Výstup:
    history (dict s train_loss, train_acc, val_loss, val_acc)
    + uložené modely, CSV a grafy do experiment_dir/training/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def accuracy(output, target):
    """Vypočíta presnosť v percentách pre jeden batch."""
    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return 100.0 * correct / target.size(0)


def train_model(
    model,                    # model na trénovanie
    train_loader,             # DataLoader pre tréningové dáta
    val_loader=None,          # voliteľný DataLoader pre validáciu
    config=None,              # konfigurácia (DEVICE, BATCH_SIZE, EPOCHS...)
    class_weights=None,       # váhy tried pre CrossEntropyLoss
    experiment_dir=None,      # priečinok na uloženie výsledkov
    model_name="experiment",  # názov experimentu
):
    if config is None:
        raise ValueError("Config must be provided")

    device = config.DEVICE
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # ────────────────────────────────────────────────────────────────
    # LOSS FUNCTION AND OPTIMIZER
    # ────────────────────────────────────────────────────────────────
    if class_weights is not None:
        # Bezpečné spracovanie tensoru (odstráni warning)
        class_weights_tensor = class_weights.clone().detach().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print(f"CrossEntropyLoss s váhami | neg={class_weights_tensor[0]:.4f}, "
              f"pos={class_weights_tensor[1]:.4f}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("CrossEntropyLoss bez váh")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )  # Adam s L2 regularizáciou

    # ────────────────────────────────────────────────────────────────
    # SCHEDULER
    # ────────────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        cooldown=config.SCHEDULER_COOLDOWN,
        min_lr=config.SCHEDULER_MIN_LR,
    )
    print("Scheduler: ReduceLROnPlateau")

    # ────────────────────────────────────────────────────────────────
    # PRIEČINKY
    # ────────────────────────────────────────────────────────────────
    experiment_dir = Path(experiment_dir)
    training_dir = experiment_dir / "training"
    models_dir = experiment_dir / "models"

    training_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Uloženie konfiguračného súboru do priečinka experimentu
    shutil.copy(
        Path(__file__).parent.parent / "d_config" / f"{config.CONFIG_SLUG}.py",
        training_dir / f"{config.CONFIG_SLUG}.py"
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print(f"Tréning modelu: {model_name}")
    print(f"Device: {device} | Epochy: {config.EPOCHS} | Batch size: {train_loader.batch_size}")
    print(f"Learning rate: {config.LEARNING_RATE:.6f} | Weight decay: {config.WEIGHT_DECAY:.6f}")
    print(f"Models: {models_dir}")
    print(f"Training artifacts: {training_dir}")
    print("-" * 80)

    for epoch in range(1, config.EPOCHS + 1):
        # ────────────────────────────────────────────────────────────────
        # TRÉNING
        # ────────────────────────────────────────────────────────────────
        model.train()  # prepnutie modelu do tréningového režimu
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]")

        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            train_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            train_total += batch_size

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ────────────────────────────────────────────────────────────────
        # VALIDÁCIA
        # ────────────────────────────────────────────────────────────────
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_acc = 0.0

        if val_loader is not None:
            model.eval()  # prepnutie modelu do eval režimu
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Val]", leave=False)

                for inputs, targets in val_bar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    batch_size = inputs.size(0)
                    val_loss += loss.item() * batch_size
                    val_correct += outputs.argmax(dim=1).eq(targets).sum().item()
                    val_total += batch_size

                if val_total > 0:
                    val_loss /= val_total
                    val_acc = 100.0 * val_correct / val_total

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

        # Scheduler step
        scheduler.step(val_loss if val_loader is not None else train_loss)

        # ────────────────────────────────────────────────────────────────
        # VÝPIS A PRIEBEŽNÉ ULOŽENIE MODELOV
        # ────────────────────────────────────────────────────────────────
        epoch_path = models_dir / f"{model_name}_e{epoch:02d}.pth"
        torch.save(model.state_dict(), epoch_path)

        msg = f"Epoch {epoch:3d}/{config.EPOCHS} | Train loss: {train_loss:.4f} | Train acc: {train_acc:5.2f}%"
        if val_loader is not None:
            msg += f" | Val loss: {val_loss:.4f} | Val acc: {val_acc:5.2f}%"
        print(msg)

    # ────────────────────────────────────────────────────────────────
    # KONIEC TRÉNINGU + ULOŽENIE VÝSLEDKOV
    # ────────────────────────────────────────────────────────────────
    print("-" * 80)
    print(f"Tréning dokončený.")

    if experiment_dir is not None:
        experiment_dir = Path(experiment_dir)
        training_dir = experiment_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)

        base_name = model_name

        # Uloženie metrík do CSV
        df = pd.DataFrame({
            'epoch': list(range(1, config.EPOCHS + 1)),
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
        })
        if 'val_loss' in history:
            df['val_loss'] = history['val_loss']
            df['val_acc'] = history['val_acc']

        df.to_csv(training_dir / f"{base_name}_metrics.csv", index=False, float_format='%.4f')
        print(f"Metriky uložené: {training_dir / f'{base_name}_metrics.csv'}")

        # Uloženie grafov
        epochs_range = list(range(1, config.EPOCHS + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
        if 'val_loss' in history:
            ax1.plot(epochs_range, history["val_loss"], label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, history["train_acc"], label="Train Acc")
        if 'val_acc' in history:
            ax2.plot(epochs_range, history["val_acc"], label="Val Acc")
        ax2.set_title("Accuracy (%)")
        ax2.set_ylim(0, 105)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = training_dir / f"{base_name}_curves.png"
        plt.savefig(plot_path, dpi=180, bbox_inches="tight")
        print(f"Grafy uložené: {plot_path}")
        plt.close(fig)

    print(f"\nHotovo!")
    print(f"→ Modely: {models_dir}")
    print(f"→ Training folder: {training_dir}")

    return history