#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Simran S. Sangha
# Copyright (c) 2026, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
ESDIS Statistics Runner
Wrapper script to execute country-level user analysis.

Usage:
    python run_esdis_stats.py \
        --input_dir my_data --start_year 2023 --end_year 2025
"""

import argparse
import sys
from pathlib import Path
from esdis_analytics import CountryUserAnalyzer

# Resolve the absolute path of the directory containing this script.
# This ensures it always finds the repo's internal folders, 
# even if run from a different working directory via PATH.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input_spreadsheets"

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run ESDIS Country User Statistics Analysis"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing source Excel files (default: {DEFAULT_INPUT_DIR})"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="stats_output",
        help="Directory to save results (default: stats_output)"
    )

    parser.add_argument(
        "--start_year",
        type=int,
        default=2023,
        help="Analysis start year (default: 2023)"
    )

    parser.add_argument(
        "--end_year",
        type=int,
        default=2025,
        help="Analysis end year (default: 2025)"
    )

    args = parser.parse_args()

    # Validate Input Directory
    in_path = Path(args.input_dir)
    if not in_path.exists():
        print(f"Error: Input directory '{in_path}' does not exist.")
        sys.exit(1)

    print("--- Starting Analysis ---")
    print(f"Input: {in_path}")
    print(f"Output: {args.output_dir}")
    print(f"Range: {args.start_year} to {args.end_year}")

    analyzer = CountryUserAnalyzer(args.input_dir, args.output_dir)

    # 1. Process Data
    df_results = analyzer.process_data(args.start_year, args.end_year)

    if df_results.empty:
        print("No data found or processed. Exiting.")
        sys.exit(0)

    # 2. Save Main Spreadsheet
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    excel_path = out_path / "OPERA_User_Distribution_Summary_Updated.xlsx"
    df_results.to_excel(excel_path)
    print(f"Spreadsheet saved: {excel_path}")

    # 3. Generate Visuals & PDF
    analyzer.generate_outputs(df_results, args.start_year, args.end_year)

    print("--- Analysis Complete ---")


if __name__ == "__main__":
    main()

