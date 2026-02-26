# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Simran S. Sangha
# Copyright (c) 2026, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
ESDIS Analytics Library
Contains classes and functions for processing ESDIS statistics.

EXTENSIBILITY GUIDE:
--------------------
This module is designed to be extended for different aggregation types
(e.g., Affiliation, Data Product, User Group).

To add a new analysis variant:
1. Define a new class (e.g., `AffiliationAnalyzer`).
2. Implement a `process_data(self, start_year, end_year)` method that returns
   a cleaned DataFrame appropriate for that data type.
3. Implement a `generate_outputs(self, df, start_year, end_year)` method
   that creates the relevant visualizations (e.g., Bar Charts instead of
   Maps) and saves them to the appropriate subdirectory.
4. Instantiate your new class in the execution wrapper (`run_esdis_stats.py`).
"""

import logging
import re
import textwrap
import warnings
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import country_converter as coco
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    DrawingArea,
    HPacker,
    OffsetImage,
    TextArea,
    VPacker,
)
from matplotlib.patches import Rectangle

# ==========================================
# CONSTANTS & CONFIG
# ==========================================

PLOT_TITLE_SIZE = 22
PLOT_LABEL_SIZE = 14
COLOR_OTHERS = '#D3D3D3'  # Gray color strictly for "Others"
# Standardize figure size for consistent PDF pages (Width, Height)
FIG_SIZE_STD = (16, 9)

# Mapping for flag images
_TWEMOJI_VERSION = "14.0.2"
_TWEMOJI_BASE_URL = (
    f"https://cdn.jsdelivr.net/gh/twitter/twemoji@{_TWEMOJI_VERSION}"
    "/assets/72x72"
)

# Hardcoded list of microstates often missing from low-res maps
# We force-add these to the "World" set to ensure they are reported
# as missing if they have no data.
MICROSTATES_TO_INCLUDE = [
    'Vatican City', 'Tuvalu', 'Nauru', 'Palau', 'San Marino',
    'Liechtenstein', 'Monaco', 'Marshall Islands', 'Saint Kitts and Nevis',
    'Maldives', 'Malta', 'Singapore', 'Bahrain', 'Tonga', 'Samoa'
]

# Country name normalization
COUNTRY_NAME_OVERRIDES = {
    # --- China Merges ---
    'Hong Kong': 'China',
    'Hong Kong SAR': 'China',
    'Hong Kong SAR China': 'China',
    'Macao': 'China',
    'Macau': 'China',

    # --- USA Merges ---
    'Puerto Rico': 'United States',
    'Virgin Islands, U.S.': 'United States',
    'U.S. Virgin Islands': 'United States',
    'Guam': 'United States',
    'American Samoa': 'United States',
    'Northern Mariana Islands': 'United States',

    # --- France Merges ---
    'Reunion': 'France',
    'Réunion': 'France',
    'Martinique': 'France',
    'Guadeloupe': 'France',
    'French Guiana': 'France',
    'New Caledonia': 'France',
    'Mayotte': 'France',
    'Saint Martin': 'France',
    'Saint Barthelemy': 'France',
    'French Southern Territories': 'France',
    'Saint Pierre and Miquelon': 'France',
    'Wallis and Futuna': 'France',
    'French Polynesia': 'France',

    # --- UK Merges ---
    'Virgin Islands, British': 'United Kingdom',
    'British Virgin Islands': 'United Kingdom',
    'Cayman Islands': 'United Kingdom',
    'Bermuda': 'United Kingdom',
    'Turks and Caicos Islands': 'United Kingdom',
    'Gibraltar': 'United Kingdom',
    'Falkland Islands': 'United Kingdom',
    'South Georgia and the South Sandwich Islands': 'United Kingdom',
    'Pitcairn': 'United Kingdom',
    'Anguilla': 'United Kingdom',
    'Montserrat': 'United Kingdom',
    'Saint Helena': 'United Kingdom',

    # --- Denmark Merges ---
    'Greenland': 'Denmark',
    'Faroe Islands': 'Denmark',

    # --- Netherlands Merges ---
    'Aruba': 'Netherlands',
    'Curacao': 'Netherlands',
    'Sint Maarten': 'Netherlands',
    'Bonaire, Sint Eustatius and Saba': 'Netherlands',
    'The Netherlands': 'Netherlands',

    # --- African Name Variants ---
    'DR Congo': 'Democratic Republic of Congo',
    'Democratic Republic of the Congo': 'Democratic Republic of Congo',
    'Congo, Dem. Rep.': 'Democratic Republic of Congo',
    
    'Congo Republic': 'Republic of Congo',
    'Republic of the Congo': 'Republic of Congo',
    'Congo, Rep.': 'Republic of Congo',
    'Congo': 'Republic of Congo',
    
    'Eswatini': 'Eswatini',
    'Swaziland': 'Eswatini',
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Cote d’Ivoire": "Cote d'Ivoire",
    "Côte d’Ivoire": "Cote d'Ivoire",
    "Cote d Ivoire": "Cote d'Ivoire",

    # --- Asian/Middle East Variants ---
    'Brunei Darussalam': 'Brunei',
    'Palestinian Territory, Occupied': 'Palestine',
    'Viet Nam': 'Vietnam',
    'Timor Leste': 'Timor-Leste',
    'Türkiye': 'Turkey',
    'Turkiye': 'Turkey',
    'T¸rkiye': 'Turkey',
    'T?rkiye': 'Turkey',
    'Republic of Korea': 'South Korea',
    'Korea, Republic of': 'South Korea',
    'Korea (South)': 'South Korea',
    'Korea, South': 'South Korea',
    'Korea South': 'South Korea',
    'Burma': 'Myanmar',

    # --- European Variants ---
    'Russian Federation': 'Russia',
    'Russia Federation': 'Russia',
    'Czech Republic': 'Czechia',
    'Macedonia': 'North Macedonia',
    'Holy See (Vatican City State)': 'Vatican City',
    'Vatican': 'Vatican City',
    'Moldova, Republic of': 'Moldova',

    # --- Special Cases ---
    'Antarctica': None,  # Explicitly ignore
    'Unresolvable': 'Unknown',
}


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_safe_font_family():
    """Return a safe font family list."""
    try:
        font_paths = font_manager.findSystemFonts(
            fontpaths=None, fontext='ttf'
        )
        if any('NotoColorEmoji' in f for f in font_paths):
            return ['Noto Color Emoji', 'sans-serif']
    except Exception:  # pylint: disable=broad-except
        pass
    return ['sans-serif']


def clean_country_name(name):
    """Normalize country names."""
    if not name:
        return None
    name = str(name).strip()
    if name in COUNTRY_NAME_OVERRIDES:
        return COUNTRY_NAME_OVERRIDES[name]

    # Simple fix for encoding issues
    if "T¸rkiye" in name or "T?rkiye" in name:
        return "Turkey"
    if name == "African Republic":
        return "Central African Republic"
    if name in ["Country", "nan", "Grand Total", "Total"]:
        return None
    return name


def _iso2_to_twemoji_code(iso2: str) -> str:
    """Convert ISO2 country code to Twemoji filename code."""
    iso2 = iso2.upper()
    codepoints = []
    for ch in iso2:
        if not "A" <= ch <= "Z":
            return ""
        codepoints.append(f"{0x1F1E6 + (ord(ch) - ord('A')):x}")
    return "-".join(codepoints)


def fetch_twemoji_flag_png(iso2: str, cache_dir: Path, timeout: int = 10):
    """Fetch and cache a Twemoji flag PNG."""
    if not iso2:
        return None

    code = _iso2_to_twemoji_code(iso2)
    if not code:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{code}.png"

    if not out_path.exists():
        url = f"{_TWEMOJI_BASE_URL}/{code}.png"
        try:
            with urlopen(url, timeout=timeout) as resp:
                out_path.write_bytes(resp.read())
        except (URLError, OSError, ValueError):
            return None

    try:
        return mpimg.imread(str(out_path))
    except (OSError, ValueError):
        return None


def _sanitize_filename(text):
    """Clean text for terminal-friendly filenames."""
    # Replace spaces with underscores, remove parens/quotes
    cleaned = text.replace(' ', '_').replace('(', '').replace(')', '')
    cleaned = cleaned.replace("'", "").replace('"', '')
    return cleaned


def _add_color_flag_legend(ax, entries, prop, loc_anchor):
    """Add a custom legend with colors, text, and optional flags."""
    if not entries:
        return

    label_size = prop.get_size_in_points()
    mono_prop = FontProperties(family="DejaVu Sans Mono", size=label_size)

    countries = [str(e.get("country", "")) for e in entries]
    max_country_len = max(len(c) for c in countries) if countries else 0

    values = [str(e.get("value", 0)) for e in entries]
    max_value_len = max(len(v) for v in values) if values else 0

    rows = []
    for entry in entries:
        country = str(entry.get("country", ""))
        value = str(entry.get("value", 0))
        rgba = entry.get("color", (0.2, 0.2, 0.2, 1.0))
        flag_img = entry.get("flag")

        # Swatch
        swatch = DrawingArea(14, 12, 0, 0)
        rect = Rectangle((0, 0), 14, 12, fc=rgba, ec="none")
        swatch.add_artist(rect)

        # Text
        country_pad = country.ljust(max_country_len)
        country_area = TextArea(
            country_pad, textprops={"fontproperties": mono_prop}
        )

        # Flag
        if flag_img is not None:
            flag_area = OffsetImage(flag_img, zoom=0.25)
        else:
            flag_area = TextArea(
                "", textprops={"fontproperties": mono_prop}
            )

        # Value
        value_pad = value.rjust(max_value_len)
        value_area = TextArea(
            f" ({value_pad})", textprops={"fontproperties": mono_prop}
        )

        row = HPacker(
            children=[swatch, country_area, flag_area, value_area],
            align="center", pad=0, sep=8
        )
        rows.append(row)

    # Tighten vertical separation (sep=0) to prevent overflow on tall lists
    legend_box = VPacker(children=rows, align="left", pad=0, sep=0)

    # Anchor to the provided location (allows side-by-side placement)
    anchored = AnchoredOffsetbox(
        loc="center left", child=legend_box, pad=0.2, frameon=True,
        bbox_to_anchor=loc_anchor, bbox_transform=ax.transAxes,
        borderpad=0.6
    )
    ax.add_artist(anchored)


# ==========================================
# ANALYZER CLASS (COUNTRY)
# ==========================================

class CountryUserAnalyzer:
    """
    Analyzes user distribution by Country.

    Includes specific logic for:
    - GeoPandas World Map generation.
    - Flag retrieval for legends.
    - Parsing 'Summary - Country-Users' sheets.
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.users_by_country_dir = self.output_dir / "users_by_country"
        self.flag_cache_dir = self.output_dir / "_twemoji_flag_cache"
        self.font_family = get_safe_font_family()
        self.world_map_data = None

        # Mute logging
        logging.getLogger('country_converter').setLevel(logging.ERROR)

    def load_world_map(self):
        """Loads and caches the world map data."""
        if self.world_map_data is not None:
            return self.world_map_data

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.world_map_data = gpd.read_file(
                        gpd.datasets.get_path('naturalearth_lowres')
                    )
                except (AttributeError, ValueError):
                    url = (
                        "https://naciscdn.org/naturalearth/110m/cultural/"
                        "ne_110m_admin_0_countries.zip"
                    )
                    self.world_map_data = gpd.read_file(url)
        except (OSError, ValueError) as e:
            print(f"Failed to load map data: {e}")
            return None
        return self.world_map_data

    def process_data(self, start_year, end_year):
        """Orchestrates data loading and merging."""
        print("Parsing historical and monthly data...")

        # Identify files
        hist_file = list(self.input_dir.glob("*September_2025.xlsx"))
        if not hist_file:
            print(f"Warning: Historical file not found in {self.input_dir}")
            return pd.DataFrame()
        hist_file = hist_file[0]

        monthly_files = list(self.input_dir.glob("OPERA_dist_*.xlsx"))
        monthly_files = [f for f in monthly_files if f.name != hist_file.name]

        # Parse
        df_hist = self._parse_historical(hist_file, start_year, end_year)
        df_combined = df_hist

        for f in monthly_files:
            month_col, df_month = self._parse_monthly(f)
            if month_col and df_month is not None:
                # Rename series to the date so columns don't collide
                df_month.name = month_col
                df_combined = df_combined.join(df_month, how='outer')

        df_combined = df_combined.fillna(0)

        # Apply grouping again to ensure overrides merged correctly
        if df_combined.index.name != "Country":
            df_combined.index.name = "Country"
        df_combined = df_combined.groupby("Country").sum()

        # Calculate Totals
        cols = df_combined.columns
        years = range(start_year, end_year + 1)
        cumulative_cols = []

        for y in years:
            y_str = str(y)
            y_cols = [c for c in cols if c.startswith(y_str)]
            if y_cols:
                col_name = f'Total_{y}'
                df_combined[col_name] = df_combined[y_cols].sum(axis=1)
                cumulative_cols.append(col_name)

        if cumulative_cols:
            df_combined['Cumulative_Total'] = (
                df_combined[cumulative_cols].sum(axis=1)
            )

        return df_combined.sort_values('Cumulative_Total', ascending=False)

    def _parse_historical(self, path, start_year, end_year):
        """Parses the historical summary file."""
        try:
            df_raw = pd.read_excel(
                path,
                sheet_name='Summary - Country-Users',
                header=None,
                engine='openpyxl'
            )
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"Err reading hist file: {e}")
            return pd.DataFrame()

        # Row 6 starts data. Col 1 is Country.
        # Years/Months structure is fixed per original requirements
        years_seq = [2022] * 2 + [2023] * 12 + [2024] * 12 + [2025] * 9
        months_seq = (
            [11, 12] +
            list(range(1, 13)) +
            list(range(1, 13)) +
            list(range(1, 10))
        )
        date_cols = [f"{y}-{m:02d}" for y, m in zip(years_seq, months_seq)]

        data_indices = [1] + list(range(2, 37))
        df_hist = df_raw.iloc[6:, data_indices].copy()
        df_hist.columns = ["Country"] + date_cols

        df_hist['Country'] = df_hist['Country'].apply(clean_country_name)
        df_hist = df_hist.dropna(subset=['Country'])

        for col in date_cols:
            df_hist[col] = pd.to_numeric(
                df_hist[col], errors='coerce'
            ).fillna(0)

        # Filter by requested years
        keep_cols = [
            c for c in date_cols
            if start_year <= int(c.split('-')[0]) <= end_year
        ]
        df_hist = df_hist.set_index("Country")[keep_cols]

        return df_hist.groupby(df_hist.index).sum()

    def _parse_monthly(self, path):
        """Parses a monthly distribution file."""
        try:
            df_raw = pd.read_excel(
                path, sheet_name='Summaries', header=None, engine='openpyxl'
            )
        except (FileNotFoundError, ValueError, OSError):
            return None, None

        # Determine date from file content or filename
        date_label = "Unknown_Month"
        try:
            cell_val = str(df_raw.iloc[2, 1])
            match = re.search(r'(For\s+)?([A-Za-z]+)\s+(\d{4})', cell_val)
            if match:
                dt = datetime.strptime(
                    f"{match.group(2)} {match.group(3)}", "%B %Y"
                )
                date_label = dt.strftime("%Y-%m")
            else:
                match_file = re.search(r'dist_([A-Za-z]+)_(\d{4})', str(path))
                if match_file:
                    dt = datetime.strptime(
                        f"{match_file.group(1)} {match_file.group(2)}",
                        "%B %Y"
                    )
                    date_label = dt.strftime("%Y-%m")
        except (ValueError, IndexError):
            pass

        # Parse Data Columns
        try:
            header_row = 5
            headers = df_raw.iloc[header_row].astype(str).tolist()
            c_idx = [i for i, h in enumerate(headers) if "Country" in h][-1]
            sub_h = headers[c_idx:]
            u_rel_idx = next(
                i for i, h in enumerate(sub_h) if "# of Users" in h
            )
            u_idx = c_idx + u_rel_idx

            df_m = df_raw.iloc[header_row + 1:, [c_idx, u_idx]].copy()
            df_m.columns = ["Country", "Users"]
            df_m['Country'] = df_m['Country'].apply(clean_country_name)
            df_m = df_m.dropna(subset=['Country'])
            df_m['Users'] = pd.to_numeric(
                df_m['Users'], errors='coerce'
            ).fillna(0)

            return date_label, df_m.groupby("Country")['Users'].sum()
        except (ValueError, IndexError):
            return None, None

    def generate_outputs(self, df, start_year, end_year):
        """Generates folders, plots, and unified PDF."""
        # Setup categories with updated titles for cumulative
        categories = [(
            'Cumulative_Total',
            f'Cumulative ({start_year}-{end_year})'
        )]

        for y in range(start_year, end_year + 1):
            col_name = f'Total_{y}'
            if col_name in df.columns:
                categories.append((col_name, str(y)))

        # Global color map based on all data
        color_map = self._generate_color_map(df, [c[0] for c in categories])

        # Prepare PDF
        fname_base = f"User_Analysis_Report_{start_year}_{end_year}"
        pdf_name = f"{_sanitize_filename(fname_base)}.pdf"
        pdf_path = self.output_dir / pdf_name

        with PdfPages(pdf_path) as pdf:

            # Title Page - Standardized Size
            first_page = plt.figure(figsize=FIG_SIZE_STD)
            first_page.clf()
            first_page.text(
                0.5, 0.5,
                f"Analysis of Users by Country\n{start_year} - {end_year}",
                ha='center', va='center', fontsize=24, fontweight='bold'
            )
            pdf.savefig(first_page)
            plt.close()

            # Iterate Categories
            for col, label in categories:
                if df[col].sum() == 0:
                    continue

                # Create Subdirectory (Cleaned)
                safe_label = _sanitize_filename(label).lower()
                sub_dir = self.users_by_country_dir / safe_label
                sub_dir.mkdir(parents=True, exist_ok=True)

                print(f"Generating visuals for: {label}...")

                # 1. Global Map
                fig_map = self._create_global_map(df, col, label, sub_dir)
                if fig_map:
                    pdf.savefig(fig_map, dpi=200)
                    plt.close(fig_map)

                # 2. Side-by-Side Pie Charts (10, 15, 20)
                for n in [10, 15, 20]:
                    fig_dual = self._create_side_by_side_pie_charts(
                        df, col, label, n, color_map, sub_dir
                    )
                    if fig_dual:
                        pdf.savefig(fig_dual, dpi=200)
                        plt.close(fig_dual)

        print(f"PDF Report saved to: {pdf_path}")

    def _generate_color_map(self, df, columns):
        unique_c = set()
        for col in columns:
            if col in df.columns:
                top = df.nlargest(25, col).index.tolist()
                unique_c.update(top)

        sorted_c = sorted(list(unique_c))
        cmap = plt.cm.nipy_spectral
        colors = [
            mcolors.to_hex(cmap(i / len(sorted_c)))
            for i in range(len(sorted_c))
        ]

        clean_colors = []
        for c in colors:
            # Avoid the reserved "Others" gray
            if c.upper() == COLOR_OTHERS.upper():
                clean_colors.append('#FF0000')
            # Avoid faint grays
            elif self._is_gray(c):
                clean_colors.append('#008080')
            else:
                clean_colors.append(c)

        return dict(zip(sorted_c, clean_colors))

    def _is_gray(self, hex_color):
        rgb = mcolors.to_rgb(hex_color)
        return (
            abs(rgb[0] - rgb[1]) < 0.05 and
            abs(rgb[1] - rgb[2]) < 0.05 and
            abs(rgb[0] - rgb[2]) < 0.05
        )

    def _create_side_by_side_pie_charts(
        self, df, column, label, n, color_map, out_dir
    ):
        """Creates a figure with two pie charts side-by-side."""
        df_sorted = df[df[column] > 0].sort_values(column, ascending=False)
        actual_n = min(n, len(df_sorted))
        top_n_df = df_sorted.head(actual_n)

        # Base Data
        data_base = top_n_df[column].tolist()
        labels_base = top_n_df.index.tolist()
        colors_base = [color_map.get(c, '#1f77b4') for c in labels_base]

        # "Others" Data
        data_others = list(data_base)
        labels_others = list(labels_base)
        colors_others = list(colors_base)

        total_sum = df_sorted[column].sum()
        others_val = total_sum - sum(data_base)
        others_count = len(df_sorted) - actual_n

        if others_val > 0:
            data_others.append(others_val)
            labels_others.append(f"Others ({others_count} countries)")
            colors_others.append(COLOR_OTHERS)

        # Plot Setup (1 Row, 2 Columns)
        # Use Standardized Size for consistent PDF
        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_STD)

        # --- LEFT PLOT: Standard Top N ---
        axes[0].pie(data_base, startangle=140, colors=colors_base)
        axes[0].set_title(
            f"Top {actual_n} Countries", fontsize=18, fontweight='normal'
        )

        # --- RIGHT PLOT: Top N + Others ---
        wedges2, _ = axes[1].pie(
            data_others, startangle=140, colors=colors_others
        )
        axes[1].set_title(
            f"Top {actual_n} + Others", fontsize=18, fontweight='normal'
        )

        # --- SHARED LEGEND ---
        entries = []
        for wedge, txt, val in zip(wedges2, labels_others, data_others):
            iso2 = None
            clean_txt = txt.split(' (')[0]
            if "Others" not in clean_txt:
                iso2 = coco.convert(
                    names=clean_txt, to='ISO2', not_found=None
                )

            flag_img = (
                fetch_twemoji_flag_png(str(iso2), self.flag_cache_dir)
                if iso2 else None
            )
            entries.append({
                "color": tuple(wedge.get_facecolor()),
                "country": txt,
                "value": int(val),
                "flag": flag_img
            })

        # Reduce legend font size slightly more to fit taller lists
        prop = FontProperties(family=self.font_family, size=11)

        # Add legend anchored closer to the rightmost plot
        # (1.05, 0.5) puts it comfortably outside the axes but on the page
        _add_color_flag_legend(
            axes[1], entries, prop, loc_anchor=(1.05, 0.5)
        )

        # Overall Title (Bold)
        plt.suptitle(
            label, fontsize=PLOT_TITLE_SIZE, fontweight='bold'
        )

        # Clean Filename
        fname_str = f"PieChart_SideBySide_{label}_Top{n}"
        filename = f"{_sanitize_filename(fname_str)}.png"

        # Adjust layout: right=0.72 reserves 28% of width for the legend
        plt.subplots_adjust(
            left=0.05, right=0.72, top=0.85, bottom=0.05, wspace=0.1
        )

        fig.savefig(out_dir / filename, bbox_inches='tight', dpi=200)
        return fig

    def _create_global_map(self, df, column, label, out_dir):
        print(f"  Mapping {label}...")
        world = self.load_world_map()
        if world is None:
            return None

        # Prepare Data
        df_iso = df.copy()
        df_iso['iso3'] = coco.convert(
            names=df.index.tolist(), to='ISO3', not_found=None
        )
        df_map = df_iso.groupby('iso3')[column].sum().reset_index()

        iso_col = next(
            (c for c in ['ISO_A3', 'ADM0_A3'] if c in world.columns), None
        )
        if not iso_col:
            return None

        merged = world.merge(
            df_map, left_on=iso_col, right_on='iso3', how='left'
        )
        merged[column] = merged[column].fillna(0)

        # Filter
        name_col = next(
            (c for c in ['name', 'NAME', 'ADMIN'] if c in merged.columns),
            None
        )
        if name_col:
            merged = merged[merged[name_col] != "Antarctica"]

        # Pins
        active = merged[merged[column] > 0].copy()
        # Reproject for centroid calculation
        active_proj = active.to_crs("+proj=moll")
        active['centroid'] = active_proj.geometry.centroid.to_crs(active.crs)

        # Plot with Standardized Size
        fig, ax = plt.subplots(figsize=FIG_SIZE_STD)
        merged.plot(ax=ax, color='#F0F0F0', edgecolor='#D0D0D0')
        active['centroid'].plot(
            ax=ax, marker='v', color='red', markersize=50, alpha=0.8
        )

        # --- STATS CALCULATION (ROBUST TO TERRITORIES & ISLANDS) ---
        
        # 1. Numerator: Active, unique countries from DATA
        # Uses 'df_iso' which has already been cleaned/grouped.
        # This counts "USA" once, even if input had "USA" and "Puerto Rico".
        active_isos = df_iso[df_iso[column] > 0]['iso3'].unique()
        active_isos = {i for i in active_isos if i is not None}
        total_active_in_df = len(active_isos)

        # 2. Denominator: "Total World" Definition
        # Start with all ISOs from the map (handles 95% of countries)
        world_isos = set(world[iso_col].unique())
        world_isos = {i for i in world_isos if str(i) != '-99'}
        
        # Convert map ISOs to Clean Names to resolve territories (GRL -> Denmark)
        # We assume if the name cleans to something else, it's a territory.
        logging.getLogger('country_converter').setLevel(logging.ERROR)
        
        world_names_set = set()
        
        # 2a. Add Map Countries (Cleaned)
        map_names = coco.convert(
            names=list(world_isos), to='name_short', not_found=None
        )
        if isinstance(map_names, list):
            for name in map_names:
                clean = clean_country_name(name)
                if clean and clean != "Antarctica": 
                    world_names_set.add(clean)
        
        # 2b. Add Active Data Countries (Cleaned) - ensures we don't miss any
        # valid countries that are in our data but somehow not in the map
        data_names = df_iso.index.unique().tolist()
        for name in data_names:
            clean = clean_country_name(name)
            if clean and clean != "Antarctica":
                world_names_set.add(clean)
                
        # 2c. Force-Add Microstates (Cleaned) - ensures Tuvalu etc. are counted
        for ms in MICROSTATES_TO_INCLUDE:
            clean = clean_country_name(ms)
            if clean and clean != "Antarctica":
                world_names_set.add(clean)
        
        total_world_countries = len(world_names_set)
        
        # 3. Missing Calculation
        active_names_set = set()
        active_raw = df_iso[df_iso[column] > 0].index.tolist()
        for name in active_raw:
            clean = clean_country_name(name)
            if clean and clean != "Antarctica":
                active_names_set.add(clean)
                
        missing_names = sorted(list(world_names_set - active_names_set))
        missing_count = len(missing_names)

        # --- HYBRID FOOTER LOGIC ---
        if missing_count > 25:
            display_list = missing_names[:25]
            remaining = missing_count - 25
            list_str = ", ".join(display_list) + f", ...and {remaining} others."
        else:
            list_str = ", ".join(missing_names)

        # 1. Header
        header_text = (
            f"International adoption of OPERA data reaches "
            f"{total_active_in_df} out of {total_world_countries} countries."
        )
        
        # 2. Intro (Explicit newline separation)
        footer_intro = (
            f"The following {missing_count} countries "
            f"have no recorded users:"
        )
        
        # 3. Wrapped List (Wrap only the list content)
        wrapped_list = textwrap.fill(list_str, width=110)
        
        # 4. Combine with explicit newlines
        full_footer = f"{header_text}\n{footer_intro} {wrapped_list}"

        plt.title(
            f"Global Distribution - {label}",
            fontsize=20, fontweight='bold'
        )
        
        plt.figtext(
            0.5, 0.02, full_footer, ha="center", va="bottom", fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5}
        )
        
        plt.subplots_adjust(bottom=0.15)
        ax.set_axis_off()

        fname_str = f"Global_Map_Pins_{label}"
        filename = f"{_sanitize_filename(fname_str)}.png"
        fig.savefig(out_dir / filename, bbox_inches='tight', dpi=300)
        return fig
