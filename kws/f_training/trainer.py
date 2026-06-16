"""Trénovacia funkcia pre PyTorch modely.
Podporuje tréning s validáciou, váženú stratu, scheduler, ukladanie histórie a grafov.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path


def accuracy(output, target):
    """Presnosť v percentách pre jeden batch"""
    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return 100.0 * correct / target.size(0)


def train_model(
    model,                    # model na trénovanie
    train_loader,             # DataLoader pre tréningové dáta
    val_loader=None,          # voliteľný DataLoader pre validáciu
    epochs=None,              # počet epoch
    lr=None,                  # learning rate
    weight_decay=None,        # L2 regularizácia
    device=None,              # zariadenie (cuda/cpu)
    experiment_dir=None,      # priečinok na uloženie výsledkov
    model_name="experiment",  # názov experimentu
    class_weights=None        # očakávame tensor [w_neg, w_pos]

):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ────────────────────────────────────────────────────────────────
    # LOSS FUNCTION
    # ────────────────────────────────────────────────────────────────
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # vážená strata podľa tried
        print(f"CrossEntropyLoss s váhami | neg={class_weights[0]:.4f}, pos={class_weights[1]:.4f}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("CrossEntropyLoss bez váh")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Adam s L2 regularizáciou

    # ────────────────────────────────────────────────────────────────
    # SCHEDULER
    # ────────────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        cooldown=1,
        min_lr=1e-7,
    )
    print("Scheduler: ReduceLROnPlateau")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = -1.0
    best_epoch = 0

    print(f"Tréning modelu: {model_name}")
    print(f"Device: {device} | Epochy: {epochs} | Batch size: {train_loader.batch_size}")
    print(f"Learning rate: {lr:.6f} | Weight decay: {weight_decay:.6f}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        # ────────────────────────────────────────────────────────────────
        # TRAIN
        # ────────────────────────────────────────────────────────────────
        model.train()      # prepnutie modelu do tréningového režimu (zapne dropout, batchnorm atď.)
        train_loss = 0.0   # akumulátor straty pre celú epochu
        train_correct = 0  # počet správne klasifikovaných vzoriek
        train_total = 0    # celkový počet spracovaných vzoriek (na výpočet priemerov)

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")  # progress bar pre tréning
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)  # presun dát na zariadenie

            optimizer.zero_grad()                 # vynulovanie gradientov pred novým krokom
            outputs = model(inputs)               # forward pass
            loss = criterion(outputs, targets)
            loss.backward()                       # spätná propagácia (výpočet gradientov)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()                      # aktualizácia váh modelu

            batch_size = inputs.size(0)           # aktuálna veľkosť batchu

            train_loss += loss.item() * batch_size  # akumulácia váženého loss
            train_correct += outputs.argmax(dim=1).eq(targets).sum().item()  # počet správne klasifikovaných vzoriek
            train_total += batch_size  # celkový počet spracovaných vzoriek

            train_bar.set_postfix(loss=f"{loss.item():.4f}")  # zobrazenie aktuálnej loss v progress bare

        train_loss /= train_total  # priemerná strata za epochu
        train_acc = 100.0 * train_correct / train_total  # výpočet presnosti v %

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ────────────────────────────────────────────────────────────────
        # VALIDATION
        # ────────────────────────────────────────────────────────────────
        val_loss = 0.0  # akumulátor straty pre celú validáciu
        val_correct = 0  # počet správne klasifikovaných vzoriek
        val_total = 0  # celkový počet spracovaných vzoriek
        val_acc = 0.0  # validačná presnosť (bude nastavená nižšie)

        if val_loader is not None:
            model.eval()  # prepnutie modelu do eval režimu (vypne dropout, batchnorm atď.)
            with torch.no_grad():  # vypnutie sledovania gradientov – šetrí pamäť a zrýchľuje výpočet
                val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False)

                for inputs, targets in val_bar:
                    inputs, targets = inputs.to(device), targets.to(device)  # presun dát na zariadenie

                    outputs = model(inputs)  # forward pass
                    loss = criterion(outputs, targets)  # výpočet straty pre aktuálny batch

                    batch_size = inputs.size(0)
                    val_loss += loss.item() * batch_size  # akumulácia váženej straty (pre správny priemer)
                    val_correct += outputs.argmax(dim=1).eq(targets).sum().item()  # počet správnych predikcií
                    val_total += batch_size  # akumulácia počtu vzoriek

                if val_total > 0:
                    val_loss /= val_total  # výpočet priemernej validačnej straty
                    val_acc = 100.0 * val_correct / val_total  # výpočet validačnej presnosti v %

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

        # Scheduler step
        scheduler.step(val_loss if val_loader is not None else train_loss)  # zníženie LR podľa validácie

        # ────────────────────────────────────────────────────────────────
        # VÝPIS
        # ────────────────────────────────────────────────────────────────
        msg = f"Epoch {epoch:3d}/{epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:5.2f}%"
        if val_loader is not None:
            msg += f" | Val loss: {val_loss:.4f} | Val acc: {val_acc:5.2f}%"
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            msg += " ← BEST"
        print(msg)

    # ────────────────────────────────────────────────────────────────
    # KONIEC TRÉNINGU + UKLADANIE VÝSLEDKOV
    # ────────────────────────────────────────────────────────────────
    print("-" * 80)
    print(f"Tréning dokončený. Najlepšia val acc: {best_val_acc:.2f}% (epoch {best_epoch})")

    if experiment_dir is not None:
        experiment_dir = Path(experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        base_name = model_name

        # JSON
        with open(experiment_dir / f"{base_name}_history.json", 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        print(f"História uložená: {experiment_dir / f'{base_name}_history.json'}")

        # CSV
        df = pd.DataFrame({
            'epoch': list(range(1, epochs + 1)),
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
        })
        if 'val_loss' in history:
            df['val_loss'] = history['val_loss']
            df['val_acc'] = history['val_acc']
        df.to_csv(experiment_dir / f"{base_name}_metrics.csv", index=False, float_format='%.4f')
        print(f"Metriky uložené: {experiment_dir / f'{base_name}_metrics.csv'}")

        # Grafy
        epochs_range = list(range(1, epochs + 1))
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
        plot_path = experiment_dir / f"{base_name}_curves.png"
        plt.savefig(plot_path, dpi=180, bbox_inches="tight")
        print(f"Grafy uložené: {plot_path}")
        plt.close(fig)

    print("\nHotovo – výsledky sú v priečinku experimentu.")
    return history