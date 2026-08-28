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
import numpy as np
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
COLOR_OTHERS = '#D3D3D3'
FIG_SIZE_STD = (16, 9)

DISCIPLINE_MAPPING = {
    'Land use Land change': [
        'Land Processes', 'Human Dimensions of Global Change'
    ],
    'Solid Earth': ['Geophysics'],
    'Hydrology': ['Hydrologic Cycle'],
    'Atmo Sci': [
        'Atmospheric Aerosols', 'Atmospheric Composition', 'Clouds',
        'Radiation Budget', 'Tropospheric Chemistry',
        'Upper Atmosphere Composition', 'Upper Atmosphere Dynamics'
    ],
    'Environ Sci': ['Biogeochemical Dynamics', 'Global Biosphere'],
    'Cryosphere': ['Cryospheric Studies', 'Sea Ice', 'Polar Processes'],
    'Oceanography': [
        'Air Sea Interaction', 'Physical Oceanography', 
        'Biological Oceanography'
    ]
}

_TWEMOJI_VERSION = "14.0.2"
_TWEMOJI_BASE_URL = (
    f"https://cdn.jsdelivr.net/gh/twitter/twemoji@"
    f"{_TWEMOJI_VERSION}/assets/72x72"
)

MICROSTATES_TO_INCLUDE = [
    'Vatican City', 'Tuvalu', 'Nauru', 'Palau', 'San Marino',
    'Liechtenstein', 'Monaco', 'Marshall Islands', 'Saint Kitts and Nevis',
    'Maldives', 'Malta', 'Singapore', 'Bahrain', 'Tonga', 'Samoa'
]

COUNTRY_NAME_OVERRIDES = {
    'Hong Kong': 'China', 'Hong Kong SAR': 'China', 
    'Hong Kong SAR China': 'China', 'Macao': 'China', 'Macau': 'China', 
    'Puerto Rico': 'United States', 'Virgin Islands, U.S.': 'United States', 
    'U.S. Virgin Islands': 'United States', 'Guam': 'United States', 
    'American Samoa': 'United States', 
    'Northern Mariana Islands': 'United States', 'Reunion': 'France',
    'Réunion': 'France', 'Martinique': 'France', 'Guadeloupe': 'France',
    'French Guiana': 'France', 'New Caledonia': 'France', 'Mayotte': 'France',
    'Saint Martin': 'France', 'Saint Barthelemy': 'France',
    'French Southern Territories': 'France', 
    'Saint Pierre and Miquelon': 'France', 'Wallis and Futuna': 'France', 
    'French Polynesia': 'France', 'Virgin Islands, British': 'United Kingdom', 
    'British Virgin Islands': 'United Kingdom',
    'Cayman Islands': 'United Kingdom', 'Bermuda': 'United Kingdom',
    'Turks and Caicos Islands': 'United Kingdom', 
    'Gibraltar': 'United Kingdom', 'Falkland Islands': 'United Kingdom',
    'South Georgia and the South Sandwich Islands': 'United Kingdom',
    'Pitcairn': 'United Kingdom', 'Anguilla': 'United Kingdom',
    'Montserrat': 'United Kingdom', 'Saint Helena': 'United Kingdom',
    'Greenland': 'Denmark', 'Faroe Islands': 'Denmark', 'Aruba': 'Netherlands',
    'Curacao': 'Netherlands', 'Sint Maarten': 'Netherlands',
    'Bonaire, Sint Eustatius and Saba': 'Netherlands', 
    'The Netherlands': 'Netherlands',
    'DR Congo': 'Democratic Republic of Congo',
    'Democratic Republic of the Congo': 'Democratic Republic of Congo',
    'Congo, Dem. Rep.': 'Democratic Republic of Congo', 
    'Congo Republic': 'Republic of Congo',
    'Republic of the Congo': 'Republic of Congo', 
    'Congo, Rep.': 'Republic of Congo',
    'Congo': 'Republic of Congo', 'Eswatini': 'Eswatini', 
    'Swaziland': 'Eswatini', "Côte d'Ivoire": "Cote d'Ivoire", 
    "Cote d’Ivoire": "Cote d'Ivoire", "Côte d’Ivoire": "Cote d'Ivoire", 
    "Cote d Ivoire": "Cote d'Ivoire", 'Brunei Darussalam': 'Brunei', 
    'Palestinian Territory, Occupied': 'Palestine',
    'Viet Nam': 'Vietnam', 'Timor Leste': 'Timor-Leste', 'Türkiye': 'Turkey',
    'Turkiye': 'Turkey', 'T¸rkiye': 'Turkey', 'T?rkiye': 'Turkey',
    'Republic of Korea': 'South Korea', 'Korea, Republic of': 'South Korea',
    'Korea (South)': 'South Korea', 'Korea, South': 'South Korea',
    'Korea South': 'South Korea', 'Burma': 'Myanmar', 
    'Russian Federation': 'Russia', 'Russia Federation': 'Russia', 
    'Czech Republic': 'Czechia', 'Macedonia': 'North Macedonia', 
    'Holy See (Vatican City State)': 'Vatican City',
    'Vatican': 'Vatican City', 'Moldova, Republic of': 'Moldova',
    'Antarctica': None, 'Unresolvable': 'Unknown',
}


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_safe_font_family():
    try:
        f_paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        if any('NotoColorEmoji' in f for f in f_paths):
            return ['Noto Color Emoji', 'sans-serif']
    except Exception:
        pass
    return ['sans-serif']


def clean_country_name(name):
    if not name:
        return None
    name = str(name).strip()
    if name in COUNTRY_NAME_OVERRIDES:
        return COUNTRY_NAME_OVERRIDES[name]
    if "T¸rkiye" in name or "T?rkiye" in name:
        return "Turkey"
    if name == "African Republic":
        return "Central African Republic"
    if name in ["Country", "nan", "Grand Total", "Total"]:
        return None
    return name


def _iso2_to_twemoji_code(iso2: str) -> str:
    iso2 = iso2.upper()
    codepoints = []
    for ch in iso2:
        if not "A" <= ch <= "Z":
            return ""
        codepoints.append(f"{0x1F1E6 + (ord(ch) - ord('A')):x}")
    return "-".join(codepoints)


def fetch_twemoji_flag_png(iso2: str, cache_dir: Path, timeout: int = 10):
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
    cleaned = text.replace(' ', '_').replace('(', '').replace(')', '')
    return cleaned.replace("'", "").replace('"', '')


def _add_color_flag_legend(ax, entries, prop, loc_anchor):
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

        swatch = DrawingArea(14, 12, 0, 0)
        swatch.add_artist(Rectangle((0, 0), 14, 12, fc=rgba, ec="none"))
        country_area = TextArea(
            country.ljust(max_country_len), 
            textprops={"fontproperties": mono_prop}
        )
        if flag_img is not None:
            flag_area = OffsetImage(flag_img, zoom=0.25)
        else:
            flag_area = TextArea("", textprops={"fontproperties": mono_prop})
        value_area = TextArea(
            f" ({value.rjust(max_value_len)})", 
            textprops={"fontproperties": mono_prop}
        )

        rows.append(HPacker(
            children=[swatch, country_area, flag_area, value_area], 
            align="center", pad=0, sep=8
        ))

    legend_box = VPacker(children=rows, align="left", pad=0, sep=0)
    anchored = AnchoredOffsetbox(
        loc="center left", child=legend_box, pad=0.2, frameon=True,
        bbox_to_anchor=loc_anchor, bbox_transform=ax.transAxes, borderpad=0.6
    )
    ax.add_artist(anchored)


# ==========================================
# UNIVERSAL MONTHLY EXTRACTION 
# ==========================================

def extract_monthly_date(file_path):
    """Universal date extractor for monthly spreadsheets."""
    match = re.search(r'dist_([A-Za-z]+)_(\d{4})', file_path.name)
    if match:
        dt = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%B %Y")
        return dt.strftime("%Y-%m")
    return None


def extract_monthly_table(file_path, sheet_name_hints, col_header_hints):
    """
    Universal table extractor. Dynamically hunts for the target sheet, 
    locates the header row, and safely extracts the [Category, Users] table.
    """
    try:
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        
        target_sheet = next(
            (s for s in xls.sheet_names 
             if any(h.upper() in str(s).upper() for h in sheet_name_hints)), 
            None
        )
        if not target_sheet:
            return None
            
        df_raw = pd.read_excel(xls, sheet_name=target_sheet, header=None)
        
        # Scan up to 20 rows to find the headers
        for r_idx in range(min(20, len(df_raw))):
            row_vals = df_raw.iloc[r_idx].tolist()
            
            # Protected str() cast to bypass float NaN crashes completely
            c_idx = next(
                (i for i, h in enumerate(row_vals) 
                 if any(
                     c.upper() in str(h).upper() for c in col_header_hints
                 )),
                None
            )
            
            if c_idx is not None:
                sub_h = row_vals[c_idx:]
                u_rel_idx = next(
                    (i for i, h in enumerate(sub_h) 
                     if "USERS" in str(h).upper()), 
                    None
                )
                
                if u_rel_idx is not None:
                    u_idx = c_idx + u_rel_idx
                    df_m = df_raw.iloc[r_idx + 1:, [c_idx, u_idx]].copy()
                    df_m.columns = ["Category", "Users"]
                    
                    gt_mask = df_m['Category'].astype(str).str.contains(
                        "Grand Total", case=False, na=False
                    )
                    gt_idx = df_m[gt_mask].index
                    
                    if len(gt_idx) > 0:
                        df_m = df_m.loc[:gt_idx[0] - 1]
                        
                    df_m = df_m.dropna(subset=['Category'])
                    df_m['Users'] = pd.to_numeric(
                        df_m['Users'], errors='coerce'
                    ).fillna(0)
                    return df_m
                    
        return None
    except Exception as e:
        print(f"Extraction failed for {file_path.name}: {e}")
        return None


# ==========================================
# ANALYZER CLASS (SUMMARY METRICS)
# ==========================================

class SummaryMetricsAnalyzer:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def generate_summary_plots(self, start_year, end_year):
        print("Generating summary quarterly plots...")
        files = list(self.input_dir.glob("OPERA_dist_*.xlsx"))
        latest_file = None
        latest_date = datetime.min

        for f in files:
            if "September_2025" in f.name:
                continue
            match = re.search(r'OPERA_dist_([A-Za-z]+)_(\d{4})\.xlsx', f.name)
            if match:
                dt = datetime.strptime(
                    f"{match.group(1)} {match.group(2)}", "%B %Y"
                )
                if dt > latest_date:
                    latest_date = dt
                    latest_file = f

        if not latest_file:
            print("No valid monthly files found for summary plots.")
            return None

        try:
            df_raw = pd.read_excel(
                latest_file, sheet_name='Summary-Total', header=None, 
                engine='openpyxl'
            )
            header_idx, date_col_idx = None, None
            for i, row in df_raw.iterrows():
                row_strs = [str(val).strip().lower() for val in row.values]
                if "date" in row_strs:
                    header_idx = i
                    date_col_idx = row_strs.index("date")
                    break

            if header_idx is None or date_col_idx is None:
                return None

            col_indices = [
                date_col_idx, date_col_idx + 1, 
                date_col_idx + 2, date_col_idx + 3
            ]
            df_summary = df_raw.iloc[header_idx + 1:, col_indices].copy()
            df_summary.columns = ['Date', 'Vol Gb', '#Files', '#Users']

            df_summary['Date'] = pd.to_datetime(
                df_summary['Date'], errors='coerce'
            )
            df_summary = df_summary.dropna(subset=['Date'])
            
            # Compute Federal Fiscal Quarter FIRST
            df_summary['Quarter'] = df_summary['Date'].dt.to_period('Q-SEP')
            
            # Filter using the Fiscal Year, NOT the Calendar Year
            df_summary = df_summary[
                (df_summary['Quarter'].dt.qyear >= start_year) & 
                (df_summary['Quarter'].dt.qyear <= end_year)
            ]

            if df_summary.empty:
                return None

            for col in ['Vol Gb', '#Files', '#Users']:
                df_summary[col] = pd.to_numeric(
                    df_summary[col], errors='coerce'
                ).fillna(0)

            df_q = df_summary.groupby('Quarter')[
                ['Vol Gb', '#Files', '#Users']
            ].sum()

            fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE_STD)

            def _plot_bar(ax, data, title, ylabel, color):
                if not data.empty:
                    data.index = data.index.astype(str)
                    data.plot(
                        kind='bar', ax=ax, color=color, edgecolor='black'
                    )
                ax.set_title(title, fontsize=16)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_xlabel('Quarter', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', linestyle='--', alpha=0.7)

            if 'Vol Gb' in df_q.columns:
                _plot_bar(
                    axes[0], df_q['Vol Gb'], 'Volume (GB)', 'Vol Gb', '#4C72B0'
                )
            if '#Files' in df_q.columns:
                _plot_bar(
                    axes[1], df_q['#Files'], '# Files', 'Count', '#55A868'
                )
            if '#Users' in df_q.columns:
                _plot_bar(
                    axes[2], df_q['#Users'], '# Users', 'Count', '#C44E52'
                )

            plt.suptitle(
                f"OPERA Quarterly Summary ({start_year}-{end_year})", 
                fontsize=PLOT_TITLE_SIZE, fontweight='bold'
            )
            plt.subplots_adjust(
                left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.25
            )
            
            f_name = f"Summary_Quarterly_{start_year}_{end_year}.png"
            out_path = self.output_dir / _sanitize_filename(f_name)
            fig.savefig(out_path, bbox_inches='tight', dpi=300)
            return fig
            
        except Exception as e:
            print(f"SummaryMetrics: Failed to generate plots: {e}")
            return None


# ==========================================
# ANALYZER CLASS (COUNTRY)
# ==========================================

class CountryUserAnalyzer:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.users_by_country_dir = self.output_dir / "users_by_country"
        self.flag_cache_dir = self.output_dir / "_twemoji_flag_cache"
        self.font_family = get_safe_font_family()
        self.world_map_data = None
        logging.getLogger('country_converter').setLevel(logging.ERROR)

    def load_world_map(self):
        if self.world_map_data is not None:
            return self.world_map_data
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    p = gpd.datasets.get_path('naturalearth_lowres')
                    self.world_map_data = gpd.read_file(p)
                except (AttributeError, ValueError):
                    url = (
                        "https://naciscdn.org/naturalearth/110m/cultural/"
                        "ne_110m_admin_0_countries.zip"
                    )
                    self.world_map_data = gpd.read_file(url)
        except Exception:
            return None
        return self.world_map_data

    def process_data(self, start_year, end_year):
        print("Parsing historical and monthly data...")
        hist_file = list(self.input_dir.glob("*September_2025.xlsx"))
        if not hist_file:
            return pd.DataFrame()
        
        hist_file = hist_file[0]
        m_files = [
            f for f in self.input_dir.glob("OPERA_dist_*.xlsx") 
            if f.name != hist_file.name
        ]

        df_combined = self._parse_historical(hist_file, start_year, end_year)

        for f in m_files:
            date_label = extract_monthly_date(f)
            if not date_label: 
                continue
            
            df_m = extract_monthly_table(f, ['Summaries'], ['Country'])
            if df_m is not None:
                df_m['Category'] = df_m['Category'].apply(clean_country_name)
                df_m = df_m.dropna(subset=['Category'])
                df_grouped = df_m.groupby("Category")['Users'].sum()
                df_combined = df_combined.join(
                    df_grouped.rename(date_label), how='outer'
                )

        df_combined = df_combined.fillna(0)
        if df_combined.index.name != "Country":
            df_combined.index.name = "Country"
        df_combined = df_combined.groupby("Country").sum()

        cols = df_combined.columns
        cumulative_cols = []
        for y in range(start_year, end_year + 1):
            y_cols = [c for c in cols if c.startswith(str(y))]
            if y_cols:
                col_name = f'Total_{y}'
                df_combined[col_name] = df_combined[y_cols].sum(axis=1)
                cumulative_cols.append(col_name)

        if cumulative_cols:
            c_tot = df_combined[cumulative_cols].sum(axis=1)
            df_combined['Cumulative_Total'] = c_tot

        return df_combined.sort_values('Cumulative_Total', ascending=False)

    def _parse_historical(self, path, start_year, end_year):
        try:
            df_raw = pd.read_excel(
                path, sheet_name='Summary - Country-Users', 
                header=None, engine='openpyxl'
            )
            y_seq = [2022] * 2 + [2023] * 12 + [2024] * 12 + [2025] * 9
            m_seq = [11, 12] + list(range(1, 13)) * 2 + list(range(1, 10))
            date_cols = [f"{y}-{m:02d}" for y, m in zip(y_seq, m_seq)]
            
            df_hist = df_raw.iloc[6:, [1] + list(range(2, 37))].copy()
            df_hist.columns = ["Country"] + date_cols

            df_hist['Country'] = df_hist['Country'].apply(clean_country_name)
            df_hist = df_hist.dropna(subset=['Country'])

            for col in date_cols:
                df_hist[col] = pd.to_numeric(
                    df_hist[col], errors='coerce'
                ).fillna(0)

            keep_cols = [
                c for c in date_cols 
                if start_year <= int(c.split('-')[0]) <= end_year
            ]
            res_df = df_hist.set_index("Country")[keep_cols]
            return res_df.groupby(level=0).sum()
        except Exception:
            return pd.DataFrame()

    def generate_outputs(
        self, df, start_year, end_year, summary_fig=None, 
        discipline_fig=None, product_fig=None, product_indiv_figs=None, 
        affiliation_fig=None
    ):
        categories = [
            ('Cumulative_Total', f'Cumulative ({start_year}-{end_year})')
        ]
        for y in range(start_year, end_year + 1):
            if f'Total_{y}' in df.columns:
                categories.append((f'Total_{y}', str(y)))

        color_map = self._generate_color_map(df, [c[0] for c in categories])
        f_name = f"User_Analysis_Report_{start_year}_{end_year}.pdf"
        pdf_path = self.output_dir / f_name

        with PdfPages(pdf_path) as pdf:
            first_page = plt.figure(figsize=FIG_SIZE_STD)
            first_page.clf()
            first_page.text(
                0.5, 0.5, 
                f"Analysis of Users by Country\n{start_year} - {end_year}", 
                ha='center', va='center', fontsize=24, fontweight='bold'
            )
            pdf.savefig(first_page)
            plt.close()

            # Pass bbox_inches='tight' to gracefully contain dynamic text
            for fig in [summary_fig, product_fig]:
                if fig: 
                    pdf.savefig(fig, dpi=200, bbox_inches='tight')
                    plt.close(fig)

            if product_indiv_figs:
                for fig in product_indiv_figs:
                    pdf.savefig(fig, dpi=200, bbox_inches='tight')
                    plt.close(fig)

            for fig in [affiliation_fig, discipline_fig]:
                if fig: 
                    pdf.savefig(fig, dpi=200, bbox_inches='tight')
                    plt.close(fig)

            for col, label in categories:
                if df[col].sum() == 0: 
                    continue
                c_label = _sanitize_filename(label).lower()
                sub_dir = self.users_by_country_dir / c_label
                sub_dir.mkdir(parents=True, exist_ok=True)

                print(f"Generating visuals for: {label}...")
                fig_map = self._create_global_map(df, col, label, sub_dir)
                if fig_map: 
                    pdf.savefig(fig_map, dpi=200)
                    plt.close(fig_map)

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
                unique_c.update(df.nlargest(25, col).index.tolist())
        sorted_c = sorted(list(unique_c))
        cmap = plt.cm.nipy_spectral
        colors = [
            mcolors.to_hex(cmap(i / len(sorted_c))) 
            for i in range(len(sorted_c))
        ]
        res = {}
        for c, color in zip(sorted_c, colors):
            if c.upper() == COLOR_OTHERS.upper():
                res[c] = '#FF0000'
            elif self._is_gray(color):
                res[c] = '#008080'
            else:
                res[c] = color
        return res

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
        df_sorted = df[df[column] > 0].sort_values(column, ascending=False)
        top_n_df = df_sorted.head(min(n, len(df_sorted)))
        data_b = top_n_df[column].tolist()
        labels_b = top_n_df.index.tolist()
        
        data_o, labels_o = list(data_b), list(labels_b)
        others_val = df_sorted[column].sum() - sum(data_b)
        if others_val > 0:
            data_o.append(others_val)
            l_str = f"Others ({len(df_sorted) - len(top_n_df)} ctrs)"
            labels_o.append(l_str)

        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_STD)
        c_base = [color_map.get(c, '#1f77b4') for c in labels_b]
        
        # Fixed typo: changed data_base to data_b
        axes[0].pie(
            data_b, startangle=140, 
            colors=c_base
        )
        axes[0].set_title(
            f"Top {len(top_n_df)} Countries", fontsize=18, fontweight='normal'
        )

        c_o = c_base + [COLOR_OTHERS] if others_val > 0 else c_base
        wedges2, _ = axes[1].pie(
            data_o, startangle=140, colors=c_o
        )
        axes[1].set_title(
            f"Top {len(top_n_df)} + Others", fontsize=18, fontweight='normal'
        )

        entries = []
        for w, t, v in zip(wedges2, labels_o, data_o):
            if "Others" not in t:
                iso2 = coco.convert(
                    names=t.split(' (')[0], to='ISO2', not_found=None
                )
                flag = fetch_twemoji_flag_png(str(iso2), self.flag_cache_dir)
            else:
                flag = None
            entries.append({
                "color": w.get_facecolor(), "country": t, 
                "value": int(v), "flag": flag
            })
            
        prop = FontProperties(family=self.font_family, size=11)
        _add_color_flag_legend(axes[1], entries, prop, loc_anchor=(1.05, 0.5))

        plt.suptitle(label, fontsize=PLOT_TITLE_SIZE, fontweight='bold')
        plt.subplots_adjust(
            left=0.05, right=0.72, top=0.85, bottom=0.05, wspace=0.1
        )
        
        f_name = f"PieChart_SideBySide_{_sanitize_filename(label)}_Top{n}.png"
        fig.savefig(out_dir / f_name, bbox_inches='tight', dpi=200)
        return fig

    def _create_global_map(self, df, column, label, out_dir):
        world = self.load_world_map()
        if world is None:
            return None
        df_iso = df.copy()
        df_iso['iso3'] = coco.convert(
            names=df.index.tolist(), to='ISO3', not_found=None
        )
        
        iso_col = next(
            (c for c in ['ISO_A3', 'ADM0_A3'] if c in world.columns), None
        )
        df_grp = df_iso.groupby('iso3')[column].sum().reset_index()
        merged = world.merge(
            df_grp, left_on=iso_col, right_on='iso3', how='left'
        ).fillna(0)
        
        n_col = next(
            (c for c in ['name', 'NAME', 'ADMIN'] if c in merged.columns), None
        )
        if n_col: 
            merged = merged[merged[n_col] != "Antarctica"]

        active = merged[merged[column] > 0].copy()
        c_proj = active.to_crs("+proj=moll").geometry.centroid
        active['centroid'] = c_proj.to_crs(active.crs)

        fig, ax = plt.subplots(figsize=FIG_SIZE_STD)
        merged.plot(ax=ax, color='#F0F0F0', edgecolor='#D0D0D0')
        active['centroid'].plot(
            ax=ax, marker='v', color='red', markersize=50, alpha=0.8
        )

        w_names = set()
        if iso_col:
            cn_res = coco.convert(
                names=list(set(world[iso_col].unique())), 
                to='name_short', not_found=None
            )
            if isinstance(cn_res, list):
                w_names.update({clean_country_name(c) for c in cn_res})
                
        w_names.update({
            clean_country_name(c) 
            for c in df_iso.index.unique().tolist() + MICROSTATES_TO_INCLUDE
        })
        world_names = {n for n in w_names if n and n != "Antarctica"}
        
        a_names = df_iso[df_iso[column] > 0].index.tolist()
        active_names = {clean_country_name(n) for n in a_names}
        active_names = {n for n in active_names if n and n != "Antarctica"}
        
        missing = sorted(list(world_names - active_names))
        m_txt = ", ".join(missing[:25])
        if len(missing) > 25:
            m_txt += f", ...and {len(missing)-25} others."
        missing_text = textwrap.fill(m_txt, width=110)

        plt.title(
            f"Global Distribution - {label}", fontsize=20, fontweight='bold'
        )
        a_isos = len({
            i for i in df_iso[df_iso[column] > 0]['iso3'].unique() if i
        })
        footer_str = (
            f"International adoption reaches {a_isos} of "
            f"{len(world_names)} countries.\nNo users in: {missing_text}"
        )
        plt.figtext(
            0.5, 0.02, footer_str, ha="center", va="bottom", fontsize=12, 
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5}
        )
        plt.subplots_adjust(bottom=0.15)
        ax.set_axis_off()
        f_name = f"Global_Map_Pins_{_sanitize_filename(label)}.png"
        fig.savefig(out_dir / f_name, bbox_inches='tight', dpi=300)
        return fig


# ==========================================
# ANALYZER CLASS (DISCIPLINE / PSA)
# ==========================================

class DisciplineAnalyzer:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def process_data(self, start_year, end_year):
        print("Parsing Discipline (PSA) data...")
        mapped_totals = {key: 0 for key in DISCIPLINE_MAPPING}
        hist_files = list(self.input_dir.glob("*September_2025.xlsx"))
        h_name = hist_files[0].name if hist_files else None
        m_files = [f for f in self.input_dir.glob("OPERA_dist_*.xlsx")]
        monthly_files = [f for f in m_files if f.name != h_name]

        if hist_files:
            try:
                df_raw = pd.read_excel(
                    hist_files[0], sheet_name='Summary - PSA', 
                    header=None, engine='openpyxl'
                )
                start_mask = df_raw.apply(
                    lambda row: row.astype(str).str.contains(
                        'Primary Study Area by Number of Users', na=False
                    ).any(), axis=1
                )
                if start_mask.any():
                    s_idx = start_mask.idxmax()
                    df_users = df_raw.iloc[s_idx + 3:].copy()
                    df_users.columns = df_raw.iloc[
                        s_idx+2
                    ].astype(str).tolist()
                    
                    df_users = df_users.rename(columns={'Month-Year': 'Date'})
                    df_users['Date'] = pd.to_datetime(
                        df_users['Date'], errors='coerce'
                    )
                    df_users = df_users.dropna(subset=['Date'])
                    df_users = df_users[
                        (df_users['Date'].dt.year >= start_year) & 
                        (df_users['Date'].dt.year <= end_year)
                    ]
                    
                    df_clean = df_users.drop(
                        columns=['Date', 'Grand Total', 'nan'], errors='ignore'
                    ).apply(pd.to_numeric, errors='coerce').fillna(0).sum()
                    
                    for raw_cat, val in df_clean.items():
                        for broad_cat, sub_cats in DISCIPLINE_MAPPING.items():
                            if str(raw_cat).strip() in sub_cats: 
                                mapped_totals[broad_cat] += val
                                break
            except Exception as e:
                print(f"Failed to process historical PSA data: {e}")

        for f in monthly_files:
            df_m = extract_monthly_table(
                f, ['Summaries'], ['Primary Study Area', 'PSA']
            )
            if df_m is not None:
                grouped = df_m.groupby("Category")['Users'].sum()
                for raw_cat, val in grouped.items():
                    for b_cat, sub_cats in DISCIPLINE_MAPPING.items():
                        if str(raw_cat).strip() in sub_cats: 
                            mapped_totals[b_cat] += val
                            break

        df_mapped = pd.Series(mapped_totals)
        return df_mapped[df_mapped > 0].sort_values(ascending=False)

    def generate_outputs(self, df_series, start_year, end_year):
        if df_series.empty or df_series.sum() == 0: 
            return None
        print("Generating Discipline pie chart...")
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = [
            '#1a648b', '#eb7c31', '#216c21', '#17a2c8', 
            '#a03198', '#549c3b', '#0f3c53'
        ]
        wedges, _ = ax.pie(
            df_series, colors=colors[:len(df_series)], startangle=90, 
            counterclock=False, 
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        kw = dict(
            arrowprops=dict(arrowstyle="-", color="black"), 
            zorder=0, va="center"
        )

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y, x = np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang))
            pct = df_series.iloc[i] / df_series.sum() * 100
            l_idx = str(df_series.index[i])
            label_name = "Land use\nLand change" if l_idx == "Land use Land change" else l_idx
            
            if pct >= 10:
                ax.text(
                    x * 0.6, y * 0.6, f"{label_name}\n({int(round(pct))}%)", 
                    ha='center', va='center', fontsize=14, 
                    fontweight='bold', color='white'
                )
            else:
                kw["arrowprops"].update(
                    {"connectionstyle": f"angle,angleA=0,angleB={ang}"}
                )
                h_align = {-1: "right", 1: "left"}[int(np.sign(x))]
                ax.annotate(
                    f"{label_name} ({int(round(pct))}%)", xy=(x, y), 
                    xytext=(1.15 * np.sign(x), 1.2 * y), 
                    horizontalalignment=h_align, **kw, 
                    fontsize=14, fontweight='bold', color='black'
                )

        ax.set_title("Aggregate of all users", fontsize=24, pad=20)
        f_path = self.output_dir / "Discipline_Aggregate_PieChart.png"
        fig.savefig(f_path, bbox_inches='tight', dpi=300)
        return fig


# ==========================================
# ANALYZER CLASS (PRODUCT)
# ==========================================

class ProductAnalyzer:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.product_mapping = {
            'CSLC': 'CSLC', 'RTC': 'RTC', 'DISP': 'DISP', 
            'DIST': 'DIST', 'DSWx': 'DSWX'
        }

    def process_data(self, start_year, end_year):
        print("Parsing Product data...")
        monthly_totals = {}
        hist_files = list(self.input_dir.glob("*September_2025.xlsx"))
        h_name = hist_files[0].name if hist_files else None
        m_files = [f for f in self.input_dir.glob("OPERA_dist_*.xlsx")]
        monthly_files = [f for f in m_files if f.name != h_name]

        if hist_files:
            try:
                df_raw = pd.read_excel(
                    hist_files[0], sheet_name='Sumamry-DAAC_Product', 
                    header=None, engine='openpyxl'
                )
                start_mask = df_raw.apply(
                    lambda row: row.astype(str).str.contains(
                        'Provider by Number of Users', case=False, na=False
                    ).any(), axis=1
                )
                if start_mask.any():
                    start_idx = start_mask.idxmax()
                    
                    # Dynamically find the Grand Total column boundary
                    gt_idx = len(df_raw.columns)
                    for r_offset in [2, 3]:
                        if start_idx + r_offset < len(df_raw):
                            for c_i, val in enumerate(df_raw.iloc[start_idx + r_offset].values):
                                if str(val).strip().lower() == 'grand total':
                                    if c_i < gt_idx:
                                        gt_idx = c_i
                                        break
                    
                    # Truncate headers to avoid ffilling into the Grand Total col
                    h_vals = df_raw.iloc[start_idx + 3].values[:gt_idx]
                    headers_row = pd.Series(h_vals).ffill().values
                    
                    mapped_cols = {}
                    for i, col_name in enumerate(headers_row):
                        if str(col_name).lower() in ['nan', 'nat', 'none']: 
                            continue
                        for key, search_val in self.product_mapping.items():
                            s_val = str(search_val).upper()
                            c_val = str(col_name).upper()
                            # Strict regex to prevent partial match overlaps
                            pat = rf"(?:^|[-_]){s_val}(?:[-_\s\d]|$)"
                            if re.search(pat, c_val): 
                                mapped_cols.setdefault(key, []).append(i)

                    df_data = df_raw.iloc[start_idx + 6:].copy()
                    
                    def _is_valid_dt(c):
                        return pd.to_datetime(
                            df_data[c], errors='coerce'
                        ).notna().any()
                        
                    date_col = next(
                        (col for col in df_data.columns if _is_valid_dt(col)), 
                        None
                    )
                    
                    if date_col is not None:
                        df_data['Date'] = pd.to_datetime(
                            df_data[date_col], errors='coerce'
                        )
                        df_data = df_data.dropna(subset=['Date'])
                        for p_key, cols in mapped_cols.items():
                            df_data[p_key] = df_data[cols].apply(
                                pd.to_numeric, errors='coerce'
                            ).fillna(0).sum(axis=1)
                            
                        for _, row in df_data.iterrows():
                            dt_str = row['Date'].strftime("%Y-%m")
                            if dt_str not in monthly_totals: 
                                monthly_totals[dt_str] = {
                                    k: 0 for k in self.product_mapping.keys()
                                }
                            for k in self.product_mapping.keys():
                                if k in df_data.columns: 
                                    monthly_totals[dt_str][k] += row[k]
            except Exception as e:
                print(f"Failed to process historical product data: {e}")

        for f in monthly_files:
            date_label = extract_monthly_date(f)
            if not date_label: 
                continue
            
            df_m = extract_monthly_table(f, ['DAAC_Product'], ['Product'])
            if df_m is not None:
                if date_label not in monthly_totals:
                    monthly_totals[date_label] = {
                        k: 0 for k in self.product_mapping.keys()
                    }
                for _, row in df_m.iterrows():
                    prod_str = str(row['Category']).upper()
                    for key, search in self.product_mapping.items():
                        s_val = str(search).upper()
                        # Strict regex to prevent partial match overlaps
                        pat = rf"(?:^|[-_]){s_val}(?:[-_\s\d]|$)"
                        if re.search(pat, prod_str):
                            monthly_totals[date_label][key] += row['Users']

        if not monthly_totals: 
            return pd.DataFrame()

        df_time = pd.DataFrame.from_dict(
            monthly_totals, orient='index'
        ).reset_index().rename(columns={'index': 'Date'})
        
        df_time['Date'] = pd.to_datetime(df_time['Date'])
        
        # Compute Federal Fiscal Quarter FIRST
        df_time['Quarter'] = df_time['Date'].dt.to_period('Q-SEP')
        
        # Filter using the Fiscal Year, NOT the Calendar Year
        df_time = df_time[
            (df_time['Quarter'].dt.qyear >= start_year) & 
            (df_time['Quarter'].dt.qyear <= end_year)
        ]
        
        df_q = df_time.groupby('Quarter')[
            list(self.product_mapping.keys())
        ].sum().reset_index()
        
        for key in self.product_mapping.keys():
            df_q[f'{key}_Cumulative'] = df_q[key].cumsum()

        # Dynamically extract true deduplicated mission-wide users
        latest_file = None
        latest_date = datetime.min
        for f in self.input_dir.glob("*.xlsx"):
            match = re.search(r'dist_([A-Za-z]+)_(\d{4})', f.name)
            if match:
                dt = datetime.strptime(
                    f"{match.group(1)} {match.group(2)}", "%B %Y"
                )
                if dt > latest_date:
                    latest_date = dt
                    latest_file = f
                    
        if latest_file:
            try:
                df_raw_tot = pd.read_excel(
                    latest_file, sheet_name='Summary-Total', 
                    header=None, engine='openpyxl'
                )
                h_idx, d_idx = None, None
                for i, row in df_raw_tot.iterrows():
                    row_strs = [str(val).strip().lower() for val in row.values]
                    if "date" in row_strs:
                        h_idx, d_idx = i, row_strs.index("date")
                        break
                
                if h_idx is not None and d_idx is not None:
                    df_true = df_raw_tot.iloc[
                        h_idx + 1:, [d_idx, d_idx + 3]
                    ].copy()
                    df_true.columns = ['Date', 'True_Users']
                    df_true['Date'] = pd.to_datetime(
                        df_true['Date'], errors='coerce'
                    )
                    df_true = df_true.dropna(subset=['Date'])
                    df_true['Quarter'] = df_true['Date'].dt.to_period('Q-SEP')
                    df_true = df_true[
                        (df_true['Quarter'].dt.qyear >= start_year) & 
                        (df_true['Quarter'].dt.qyear <= end_year)
                    ]
                    df_true_q = df_true.groupby('Quarter')[
                        'True_Users'
                    ].sum().reset_index()
                    
                    df_q = df_q.merge(df_true_q, on='Quarter', how='left')
                    df_q['True_Users'] = df_q['True_Users'].fillna(0)
                    df_q['Total_Cumulative'] = df_q['True_Users'].cumsum()
                else:
                    k_list = list(self.product_mapping.keys())
                    df_q['Total_Cumulative'] = df_q[k_list].sum(axis=1).cumsum()
            except Exception as e:
                print(f"Failed to read true total product users: {e}")
                k_list = list(self.product_mapping.keys())
                df_q['Total_Cumulative'] = df_q[k_list].sum(axis=1).cumsum()
        else:
            k_list = list(self.product_mapping.keys())
            df_q['Total_Cumulative'] = df_q[k_list].sum(axis=1).cumsum()
        
        return df_q

    def generate_outputs(self, df_q, start_year, end_year):
        if df_q.empty: 
            return None, []
        print("Generating Cumulative Product Users bar charts...")
        
        def _plot_product_bar(ax, x, y_vals, x_labels, title):
            bars = ax.bar(x, y_vals, width=0.45, color='#1b617d', zorder=3)
            ax.plot(
                x, y_vals, linestyle=':', color='#aaaaaa', 
                linewidth=2, zorder=2
            )
            for bar, val in zip(bars, y_vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() + (max(y_vals) * 0.02), 
                    f"{int(val):,}", ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', color='black'
                )
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=14, fontweight='bold')
            ax.tick_params(axis='y', labelsize=14)
            y_fmt = plt.FuncFormatter(lambda val, loc: "{:,}".format(int(val)))
            ax.get_yaxis().set_major_formatter(y_fmt)
            ax.grid(
                axis='y', color='#e0e0e0', linestyle='-', linewidth=1, zorder=0
            )
            ax.tick_params(axis='both', length=0)
            for spine in ['top', 'right', 'left', 'bottom']: 
                ax.spines[spine].set_visible(False)
                
            # Render title reliably inside PDF bounding box
            ax.text(
                0.5, -0.18, title, transform=ax.transAxes, 
                ha="center", va="top", fontsize=24, 
                fontweight='bold', color='black'
            )
            plt.subplots_adjust(bottom=0.2)

        q_list = df_q['Quarter'].astype(str).tolist()
        clean_labels = []
        for i, lbl in enumerate(q_list):
            y_p, q_p = lbl[:4], lbl[4:]
            if i == 0 or y_p != q_list[i-1][:4]:
                clean_labels.append(f"{y_p}\n{q_p}")
            else:
                clean_labels.append(q_p)
        
        fig_agg, ax = plt.subplots(figsize=(14, 8))
        title_agg = "Cumulative number of users across OPERA products"
        _plot_product_bar(
            ax, np.arange(len(clean_labels)), 
            df_q['Total_Cumulative'].values, clean_labels, title_agg
        )
        
        f_agg = self.output_dir / "Product_Cumulative_Users_BarChart.png"
        fig_agg.savefig(f_agg, bbox_inches='tight', dpi=300)
        
        fig_individuals = []
        for prod in ['DSWx', 'DIST', 'CSLC', 'RTC', 'DISP']:
            if f'{prod}_Cumulative' not in df_q.columns: 
                continue
            df_plot = df_q[df_q[f'{prod}_Cumulative'] > 0].copy()
            if df_plot.empty: 
                continue
                
            s_labs = df_plot['Quarter'].astype(str).tolist()
            sub_labels = []
            for i, lbl in enumerate(s_labs):
                y_p, q_p = lbl[:4], lbl[4:]
                if i == 0 or y_p != s_labs[i-1][:4]:
                    sub_labels.append(f"{y_p}\n{q_p}")
                else:
                    sub_labels.append(q_p)
                    
            fig_indiv, ax_sub = plt.subplots(figsize=(14, 8))
            _plot_product_bar(
                ax_sub, np.arange(len(sub_labels)), 
                df_plot[f'{prod}_Cumulative'].values, sub_labels, 
                f"{prod} - Cumulative number of users"
            )
            
            f_n = f"Product_Cumulative_Users_{prod}_BarChart.png"
            fig_indiv.savefig(
                self.output_dir / f_n, bbox_inches='tight', dpi=300
            )
            fig_individuals.append(fig_indiv)

        return fig_agg, fig_individuals


# ==========================================
# ANALYZER CLASS (AFFILIATION / STAKEHOLDER)
# ==========================================

class AffiliationAnalyzer:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.affiliation_mapping = {
            'GOV': 'Federal', 'COM': 'Private sector', 
            'EDU': 'Academia', 'ORG': 'Non-profit'
        }
        self.colors = {
            'Federal': '#1a9fd2', 'Private sector': '#206622', 
            'Academia': '#e06b26', 'Non-profit': '#1a6080'
        }

    def process_data(self, start_year, end_year):
        print("Parsing Affiliation data...")
        monthly_totals = {}
        hist_files = list(self.input_dir.glob("*September_2025.xlsx"))
        h_name = hist_files[0].name if hist_files else None
        m_files = [f for f in self.input_dir.glob("OPERA_dist_*.xlsx")]
        monthly_files = [f for f in m_files if f.name != h_name]

        if hist_files:
            try:
                xls = pd.ExcelFile(hist_files[0], engine='openpyxl')
                target_sheet = next(
                    (s for s in xls.sheet_names 
                     if 'UA' in s.upper() or 'AFFIL' in s.upper()), None
                )
                if target_sheet:
                    df_raw = pd.read_excel(
                        xls, sheet_name=target_sheet, header=None
                    )
                    start_mask = df_raw.apply(
                        lambda row: row.astype(str).str.contains(
                            'User Affiliation by Number of Users', 
                            case=False, na=False
                        ).any(), axis=1
                    )
                    if start_mask.any():
                        r_idx = start_mask.idxmax()
                        s_col = df_raw.iloc[r_idx].astype(str).str.contains(
                            'User Affiliation by Number of Users', 
                            case=False, na=False
                        ).idxmax()
                        df_sub = df_raw.iloc[r_idx + 2:, s_col:s_col+12].copy()
                        df_sub.columns = df_sub.iloc[0].astype(str).values
                        
                        dt_col = next(
                            (c for c in df_sub.columns 
                             if 'DATE' in c.upper() or 'MONTH' in c.upper()), 
                            'DATE_COL'
                        )
                        df_sub = df_sub.iloc[1:].dropna(subset=[dt_col])
                        df_sub['DATE_COL'] = pd.to_datetime(
                            df_sub[dt_col], errors='coerce'
                        )
                        df_sub = df_sub.dropna(subset=['DATE_COL'])
                        
                        for _, row in df_sub.iterrows():
                            dt_str = row['DATE_COL'].strftime("%Y-%m")
                            if dt_str not in monthly_totals: 
                                monthly_totals[dt_str] = {
                                    k: 0 for k in self.colors.keys()
                                }
                            for r_key, clean in self.affiliation_mapping.items():
                                c_idx = next(
                                    (i for i, c in enumerate(df_sub.columns) 
                                     if r_key in str(c).upper()), None
                                )
                                if c_idx is not None:
                                    val = pd.to_numeric(
                                        row.iloc[c_idx], errors='coerce'
                                    )
                                    if not pd.isna(val):
                                        monthly_totals[dt_str][clean] += val
            except Exception as e:
                print(f"Failed to process historical affiliation data: {e}")

        for f in monthly_files:
            date_label = extract_monthly_date(f)
            if not date_label: 
                continue
            
            df_m = extract_monthly_table(
                f, ['UA', 'AFFIL', 'Summaries'], ['AFFIL', 'STAKEHOLDER']
            )
            if df_m is not None:
                if date_label not in monthly_totals:
                    monthly_totals[date_label] = {
                        k: 0 for k in self.colors.keys()
                    }
                for _, row in df_m.iterrows():
                    aff_str = str(row['Category']).upper()
                    for r_key, c_key in self.affiliation_mapping.items():
                        c_upper = c_key.upper()
                        is_gov = r_key == 'GOV' and 'GOVERNMENT' in aff_str
                        is_com = r_key == 'COM' and 'COMMERCIAL' in aff_str
                        is_edu = r_key == 'EDU' and 'EDUCATION' in aff_str
                        is_org = r_key == 'ORG' and (
                            'ORGANIZATION' in aff_str or 
                            'NON-PROFIT' in aff_str
                        )
                        
                        if (r_key in aff_str or c_upper in aff_str or 
                                is_gov or is_com or is_edu or is_org):
                            monthly_totals[date_label][c_key] += row['Users']

        if not monthly_totals: 
            return pd.DataFrame()
            
        df_time = pd.DataFrame.from_dict(
            monthly_totals, orient='index'
        ).reset_index().rename(columns={'index': 'Date'})
        df_time['Date'] = pd.to_datetime(df_time['Date'])
        
        # Enforce Calendar Year logic to match manual historical baseline
        df_time['Year'] = df_time['Date'].dt.year
        df_time = df_time[
            (df_time['Year'] >= start_year) & (df_time['Year'] <= end_year)
        ]
        
        df_yearly = df_time.groupby('Year')[
            list(self.colors.keys())
        ].sum().reset_index()
        for col in self.colors.keys(): 
            df_yearly[col] = df_yearly[col].cumsum()
        return df_yearly

    def generate_outputs(self, df_yearly, start_year, end_year):
        if df_yearly.empty: 
            return None
        print("Generating Cumulative User Affiliation bar chart...")
        
        categories = ['Federal', 'Private sector', 'Academia', 'Non-profit']
        fig, ax = plt.subplots(figsize=(10, 7))
        years = df_yearly['Year'].astype(str).tolist()
        x_indices = np.arange(len(years))
        bottom, bar_width, bars_dict = np.zeros(len(years)), 0.45, {}
        
        for cat in categories:
            values = df_yearly[cat].values
            bars_dict[cat] = ax.bar(
                x_indices, values, bottom=bottom, width=bar_width, 
                color=self.colors[cat], zorder=3
            )
            bottom += values
            
        for spine in ['top', 'right', 'left', 'bottom']: 
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', length=0, labelsize=14)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(years, fontweight='bold', fontsize=16)
        
        y_fmt = plt.FuncFormatter(lambda val, loc: "{:,}".format(int(val)))
        ax.get_yaxis().set_major_formatter(y_fmt)
        
        last_idx, last_x = len(years) - 1, x_indices[-1]
        fig.canvas.draw()
        
        y_ticks = [y for y in ax.get_yticks() if 0 < y <= bottom.max() * 1.1]
        for y in y_ticks: 
            ax.plot(
                [-0.5, last_x + (bar_width / 2)], [y, y], 
                color='#e0e0e0', linestyle='-', linewidth=1, zorder=0
            )
        
        for cat in categories:
            bar = bars_dict[cat][last_idx]
            y_center = bar.get_y() + bar.get_height() / 2
            ax.annotate(
                "", xy=(last_x + 0.325, y_center), 
                xytext=(last_x + 0.825, y_center), 
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2)
            )
            ax.plot(
                last_x + 0.975, y_center, marker='s', markersize=26, 
                color=self.colors[cat], clip_on=False
            )
            # Pushed x-coordinate from 1.125 to 1.25 to add padding
            ax.text(
                last_x + 1.25, y_center, cat, 
                va='center', ha='left', fontsize=20, 
                fontweight='bold', color=self.colors[cat]
            )

        # Expanded chart bounds from 2.0 to 2.2 to accommodate the wider text
        ax.set_xlim(-0.5, len(years) + 2.2)
        
        # Render title inside the axis coordinates to prevent PDF cutoff
        ax.text(
            0.5, -0.18, 
            "Cumulative growth in stakeholder groups across OPERA products", 
            transform=ax.transAxes, ha="center", va="top", 
            fontsize=24, fontweight='bold', color='black'
        )
        plt.subplots_adjust(bottom=0.2)
        
        out_name = "Affiliation_Cumulative_Users_BarChart.png"
        fig.savefig(self.output_dir / out_name, bbox_inches='tight', dpi=300)
        return fig

