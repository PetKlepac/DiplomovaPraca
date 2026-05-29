"""
Skript na zobrazenie stromovej štruktúry priečinkov negatívnych dát
s počtom súborov v každom priečinku (rekurzívne).
Používa sa na rýchlu kontrolu štruktúry a veľkosti datasetu negatívnych vzoriek.
"""

from pathlib import Path


ROOT_PATH = Path(r"C:\Users\peter\Moje\Diplomka\DiplomovaPraca\kws\a_data\positive\1sec_hovorene_final_a_finale")


def count_files_in_folder(folder: Path) -> int:
    """Vráti celkový počet súborov v priečinku vrátane všetkých podpriečinkov."""
    return sum(1 for _ in folder.rglob("*") if _.is_file())


def print_tree_with_counts(root_path: Path) -> None:
    """Vypíše stromovú štruktúru priečinkov s počtom súborov v každom priečinku."""
    print(f"Štruktúra datasetu: {root_path.name}\n")

    for current in sorted(root_path.rglob("*")):
        if current.is_dir():
            # výpočet odsadenia podľa hĺbky priečinka
            depth = len(current.relative_to(root_path).parts)
            indent = " " * depth

            # počet súborov v tomto priečinku a všetkých jeho podpriečinkoch
            file_count = count_files_in_folder(current)
            folder_name = current.name

            print(f"{indent}{folder_name} - {file_count} súborov")


def main() -> None:
    # kontrola existencie root priečinka
    if not ROOT_PATH.exists() or not ROOT_PATH.is_dir():
        print(f"Chyba: Priečinok neexistuje alebo nie je priečinok: {ROOT_PATH}")
        return

    try:
        print_tree_with_counts(ROOT_PATH)
    except Exception as e:
        print(f"Chyba pri prechádzaní priečinka: {e}")


if __name__ == "__main__":
    main()