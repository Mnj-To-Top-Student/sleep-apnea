import argparse
import os
from collections import Counter

import wfdb


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect MIT-BIH PSG annotation symbols.")
    parser.add_argument(
        "--data-dir",
        default="mitbih_psg_data",
        help="Directory containing slpdb files.",
    )
    parser.add_argument(
        "--ann-ext",
        default="st",
        help="Annotation extension to inspect (default: st).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    records = sorted(
        name[:-4]
        for name in os.listdir(args.data_dir)
        if name.endswith(".hea")
    )

    if not records:
        raise RuntimeError(f"No records found in {args.data_dir}")

    global_symbol_counter = Counter()
    global_aux_counter = Counter()
    apnea_like_counter = Counter()

    print(f"Found {len(records)} records")
    for rec in records:
        ann = wfdb.rdann(os.path.join(args.data_dir, rec), args.ann_ext)
        symbols = ann.symbol
        aux_notes = [note.strip() for note in ann.aux_note]

        symbol_counter = Counter(symbols)
        aux_counter = Counter(aux_notes)

        global_symbol_counter.update(symbol_counter)
        global_aux_counter.update(aux_counter)

        apnea_like = Counter(
            note for note in aux_notes if ("apn" in note.lower() or "apnea" in note.lower())
        )
        apnea_like_counter.update(apnea_like)

        print(
            f"{rec}: total_ann={len(symbols)} "
            f"unique_symbols={sorted(symbol_counter.keys())} "
            f"unique_aux={len(aux_counter)}"
        )

    print("\nGlobal symbol counts:")
    for symbol, count in sorted(global_symbol_counter.items(), key=lambda x: x[0]):
        print(f"  {symbol}: {count}")

    print("\nTop aux_note values:")
    for note, count in global_aux_counter.most_common(30):
        printable = note if note else "<empty>"
        print(f"  {printable}: {count}")

    print("\nApnea-like aux_note values:")
    if apnea_like_counter:
        for note, count in apnea_like_counter.most_common():
            printable = note if note else "<empty>"
            print(f"  {printable}: {count}")
    else:
        print("  <none detected by keyword search>")


if __name__ == "__main__":
    main()
