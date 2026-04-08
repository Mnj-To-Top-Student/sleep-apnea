import argparse

import wfdb


def parse_args():
    parser = argparse.ArgumentParser(description="Download MIT-BIH Polysomnographic Database via WFDB.")
    parser.add_argument(
        "--db",
        default="slpdb",
        help="PhysioNet database name. For MIT-BIH PSG use 'slpdb'.",
    )
    parser.add_argument(
        "--out",
        default="mitbih_psg_data",
        help="Output directory for downloaded files.",
    )
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Optional list of record names to download (default: all records).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Downloading database '{args.db}' to '{args.out}'...")
    if args.records:
        wfdb.dl_database(args.db, dl_dir=args.out, records=args.records)
    else:
        wfdb.dl_database(args.db, dl_dir=args.out)
    print("Download complete.")


if __name__ == "__main__":
    main()
