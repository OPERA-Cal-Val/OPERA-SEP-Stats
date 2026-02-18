# OPERA SEP Statistics Analytics

This repository contains tools for aggregating, analyzing, and visualizing user distribution statistics for OPERA (Observational Products for End-Users from Remote Sensing Analysis) data products, specifically for the Stakeholder Engagement Program (SEP).

The workflow ingests ESDIS (Earth Science Data and Information System) metrics spreadsheets to generate comprehensive reports, including global distribution maps, pie charts of top user countries, and aggregate summary spreadsheets.

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
