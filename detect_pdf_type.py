#!/usr/bin/env python3
"""Classify PDFs in a folder as text, scanned, or unreadable."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


TEXT_PDF = "text_pdf"
SCANNED_PDF = "scanned_pdf"
UNREADABLE_PDF = "unreadable_pdf"


def iter_pdf_files(folder: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    # Sort for stable output across runs.
    yield from sorted(folder.glob(pattern), key=lambda p: str(p).lower())


def classify_pdf(pdf_path: Path, threshold: int) -> dict[str, object]:
    try:
        reader = PdfReader(str(pdf_path))
        pages = len(reader.pages)

        extracted_chars = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            extracted_chars += len(text.strip())

        classification = TEXT_PDF if extracted_chars >= threshold else SCANNED_PDF

        return {
            "path": str(pdf_path.resolve()),
            "filename": pdf_path.name,
            "pages": pages,
            "extracted_chars": extracted_chars,
            "classification": classification,
        }
    except Exception:
        return {
            "path": str(pdf_path.resolve()),
            "filename": pdf_path.name,
            "pages": "",
            "extracted_chars": "",
            "classification": UNREADABLE_PDF,
        }


def write_csv(rows: Iterable[dict[str, object]], output_csv: Path) -> None:
    fieldnames = ["path", "filename", "pages", "extracted_chars", "classification"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a folder of PDFs and classify each file by text extractability."
    )
    parser.add_argument("folder", type=Path, help="Folder containing PDF files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("pdf_classification.csv"),
        help="Output CSV path (default: pdf_classification.csv)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=100,
        help="Minimum extracted character count to label as text_pdf (default: 100)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subfolders recursively",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.folder

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {folder}")

    pdf_files = list(iter_pdf_files(folder, recursive=args.recursive))
    results = [classify_pdf(pdf, threshold=args.threshold) for pdf in pdf_files]
    write_csv(results, args.output)

    print(f"Scanned {len(pdf_files)} PDF file(s).")
    print(f"Saved CSV report to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
