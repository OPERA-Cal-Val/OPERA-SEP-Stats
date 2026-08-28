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
        --input_dir my_data --start_year 2023 --end_year 2026
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from esdis_analytics import (
    CountryUserAnalyzer, 
    SummaryMetricsAnalyzer, 
    DisciplineAnalyzer,
    ProductAnalyzer
)

# Dynamically resolve the absolute path to the repository directory
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input_spreadsheets"


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run ESDIS User Statistics Analysis"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=(
            f"Directory containing source Excel files "
            f"(default: {DEFAULT_INPUT_DIR})"
        )
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
        default=2026,
        help="Analysis end year (default: 2026)"
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

    # Create the output directory before running any analytics
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate Overall Summary Metrics
    summary_analyzer = SummaryMetricsAnalyzer(args.input_dir, args.output_dir)
    fig_summary = summary_analyzer.generate_summary_plots(
        args.start_year, args.end_year
    )

    # 2. Process Discipline (PSA) Data & Capture Figure
    discipline_analyzer = DisciplineAnalyzer(args.input_dir, args.output_dir)
    df_discipline = discipline_analyzer.process_data(args.start_year, args.end_year)
    
    fig_discipline = None
    if not df_discipline.empty:
        fig_discipline = discipline_analyzer.generate_outputs(
            df_discipline, args.start_year, args.end_year
        )

    # Process Product Data
    product_analyzer = ProductAnalyzer(args.input_dir, args.output_dir)
    df_product = product_analyzer.process_data(args.start_year, args.end_year)
    
    fig_product = None
    if not df_product.empty:
        fig_product = product_analyzer.generate_outputs(
            df_product, args.start_year, args.end_year
        )

    # 3. Process Country Data
    country_analyzer = CountryUserAnalyzer(args.input_dir, args.output_dir)
    df_results = country_analyzer.process_data(args.start_year, args.end_year)

    if df_results.empty:
        print("No data found or processed. Exiting.")
        sys.exit(0)

    # 4. Save Main Spreadsheet (Multi-Tab)
    excel_path = out_path / "OPERA_User_Distribution_Summary_Updated.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Country_Distribution')
        
        if not df_discipline.empty:
            df_disc_out = df_discipline.reset_index()
            df_disc_out.columns = ['Discipline', 'Total Users']
            df_disc_out.to_excel(writer, sheet_name='Discipline_Distribution', index=False)
            
        # NEW: Save Product Distribution
        if not df_product.empty:
            df_product.to_excel(writer, sheet_name='Product_Distribution', index=False)
            
    print(f"Spreadsheet saved with tracking tabs: {excel_path}")

    # 5. Generate Visuals & Compile PDF
    country_analyzer.generate_outputs(
        df_results, args.start_year, args.end_year, 
        summary_fig=fig_summary, 
        discipline_fig=fig_discipline, 
        product_fig=fig_product # NEW
    )

    print("--- Analysis Complete ---")


if __name__ == "__main__":
    main()

