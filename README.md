# OPERA SEP Statistics Analytics

This repository contains tools for aggregating, analyzing, and visualizing user distribution statistics for OPERA (Observational Products for End-Users from Remote Sensing Analysis) data products, specifically for the Stakeholder Engagement Program (SEP).

The workflow ingests ESDIS (Earth Science Data and Information System) metrics spreadsheets to generate comprehensive reports, including global distribution maps, pie charts of top user countries, and aggregate summary spreadsheets.

> [!CAUTION]
> THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.

## Features

- **Automated Aggregation**: Merges historical cumulative data with new monthly reports.
- **Geospatial Visualization**: Generates global maps showing active user distribution, correcting for political name variants (e.g., merging "Puerto Rico" into "United States").
- **Statistical Reporting**: Produces breakdown pie charts (Top 10, 15, 20 countries) and calculates global adoption ratios (e.g., "166 out of 195 countries").
- **Smart Formatting**: Handles map resolution limitations (e.g., listing microstates like Singapore or Tuvalu in footers even if not visible on low-res maps).
- **PDF Generation**: Compiles all visual assets into a single, executive-ready PDF report.

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/OPERA-Cal-Val/OPERA-SEP-Stats.git](https://github.com/OPERA-Cal-Val/OPERA-SEP-Stats.git)
   cd OPERA-SEP-Stats
   ```

2. **Create the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate OPERA-SEP-Stats
   ```

## Usage
The primary entry point is the `run_esdis_stats.py` wrapper script.

### Basic Run
By default, the script looks for Excel files in `input_spreadsheets/` and outputs results to `stats_output/` for the years 2023–2025.
   ```bash
   python run_esdis_stats.py
   ```

### Custom Run
You can specify custom input/output directories and analysis year ranges:
   ```bash
   python run_esdis_stats.py \
    --input_dir /path/to/my_data \
    --output_dir /path/to/results \
    --start_year 2024 \
    --end_year 2026
   ```

## Input Data Guidelines

To update the statistics, simply add new ESDIS Excel reports to the `input_spreadsheets` directory.

### IMPORTANT: Data Convention Change (September 2025)
The tool handles a specific shift in how ESDIS data was reported:

1.  **Historical Data (Up to September 2025):**
    * Data is expected in a single cumulative file (e.g., `OPERA_dist_September_2025.xlsx`).
    * This file contains aggregate counts going back to the start of the mission (November 2022).

2.  **Monthly Data (October 2025 onwards):**
    * Data is reported in individual monthly files (e.g., `OPERA_dist_October_2025.xlsx`).
    * The tool dynamically identifies these files, parses the "Summaries" tab, and merges them with the historical baseline.

## Supported Statistics

**Current Version:**
* **Aggregation Key:** Country of User.
* **Metrics:** Unique users, distribution by country.
* **Visuals:** Global maps (cumulative & yearly), Pie charts (Top N vs Others).

**Future Roadmap:**
We plan to expand the tool to support, feel free to issue tickets for requests:
* **Filtering:** Date ranges and specific product suites (e.g., DSWx, DISP).
* **Additional Keys:** Aggregation by Affiliation (e.g., Academia, Government, Commercial), User Group, and Product Type.
* **Interactive Outputs:** HTML-based reports.

## Repo Structure

```text
.
├── esdis_analytics.py      # Core analysis logic and plotting classes
├── run_esdis_stats.py      # Command-line wrapper
├── environment.yml         # Conda environment definition
├── input_spreadsheets/     # Place source Excel files here
│   ├── OPERA_dist_September_2025.xlsx  # Historical baseline
│   └── OPERA_dist_October_2025.xlsx    # Monthly updates...
└── stats_output/           # Generated results (created automatically)
    ├── OPERA_User_Distribution_Summary_Updated.xlsx
    ├── User_Analysis_Report_2023_2025.pdf
    └── users_by_country/   # Individual PNGs
```

## License
Copyright (c) 2026, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
