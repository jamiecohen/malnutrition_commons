"""
Pull / generate LSFF (Large-Scale Food Fortification) coverage data.

Source: Food Fortification Initiative (FFI) 2023 Country Status Report
        https://www.ffinetwork.org/global-progress
        Cross-checked with GAIN GFDx (https://gfdx.gain.org)

No machine-readable public API is available for this dataset — FFI and
GFDx publish data as PDFs and interactive maps. This script encodes the
2023 country-level status as a curated static dataset and saves it to
data/raw/lsff/ffi_country_status.csv.

Coverage proxy methodology (for lsff_coverage_proxy_pct):
  - Mandatory legislation = 75%  (accounts for artisanal mill non-compliance
    and enforcement gaps in LMICs; conservative vs. FFI's ~86% estimate)
  - Voluntary program     = 20%  (limited uptake typical for voluntary schemes)
  - No program            = 0%

For wheat flour specifically; maize and rice tracked separately where relevant.
Update annually from: https://www.ffinetwork.org/global-progress
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parents[2] / "data" / "raw" / "lsff"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Country LSFF status ───────────────────────────────────────────────────────
# Columns: iso3, wheat_flour, maize_flour
# Values:  "mandatory" | "voluntary" | "no_program"
# Source:  FFI Global Progress, November 2023

LSFF_STATUS = [
    # ── Sub-Saharan Africa ────────────────────────────────────────────────────
    ("NGA", "mandatory",  "mandatory"),   # Nigeria
    ("ETH", "mandatory",  "mandatory"),   # Ethiopia
    ("COD", "voluntary",  "no_program"),  # Congo, Dem. Rep. (DRC)
    ("TZA", "mandatory",  "no_program"),  # Tanzania
    ("KEN", "mandatory",  "no_program"),  # Kenya
    ("UGA", "mandatory",  "no_program"),  # Uganda
    ("MOZ", "mandatory",  "no_program"),  # Mozambique
    ("GHA", "mandatory",  "mandatory"),   # Ghana
    ("MDG", "mandatory",  "no_program"),  # Madagascar
    ("CMR", "mandatory",  "no_program"),  # Cameroon
    ("CIV", "mandatory",  "mandatory"),   # Cote d'Ivoire
    ("MLI", "mandatory",  "mandatory"),   # Mali
    ("BFA", "mandatory",  "mandatory"),   # Burkina Faso
    ("NER", "mandatory",  "mandatory"),   # Niger
    ("MWI", "mandatory",  "no_program"),  # Malawi
    ("ZMB", "mandatory",  "no_program"),  # Zambia
    ("SEN", "mandatory",  "mandatory"),   # Senegal
    ("ZWE", "mandatory",  "no_program"),  # Zimbabwe
    ("TCD", "mandatory",  "no_program"),  # Chad
    ("GIN", "mandatory",  "mandatory"),   # Guinea
    ("SSD", "no_program", "no_program"),  # South Sudan
    ("RWA", "mandatory",  "no_program"),  # Rwanda
    ("BEN", "mandatory",  "mandatory"),   # Benin
    ("BDI", "no_program", "no_program"),  # Burundi
    ("SOM", "no_program", "no_program"),  # Somalia
    ("SLE", "mandatory",  "mandatory"),   # Sierra Leone
    ("AGO", "voluntary",  "no_program"),  # Angola
    ("COG", "mandatory",  "no_program"),  # Congo, Rep.
    ("CAF", "no_program", "no_program"),  # Central African Republic
    ("ERI", "mandatory",  "no_program"),  # Eritrea
    ("TGO", "mandatory",  "mandatory"),   # Togo
    ("LBR", "mandatory",  "mandatory"),   # Liberia
    ("MRT", "mandatory",  "mandatory"),   # Mauritania
    ("NAM", "mandatory",  "no_program"),  # Namibia
    ("BWA", "mandatory",  "no_program"),  # Botswana
    ("LSO", "mandatory",  "no_program"),  # Lesotho
    ("SWZ", "mandatory",  "no_program"),  # Eswatini
    ("GMB", "mandatory",  "mandatory"),   # Gambia
    ("GNB", "mandatory",  "mandatory"),   # Guinea-Bissau
    ("GAB", "mandatory",  "no_program"),  # Gabon
    ("GNQ", "no_program", "no_program"),  # Equatorial Guinea
    ("CPV", "mandatory",  "no_program"),  # Cabo Verde
    ("COM", "no_program", "no_program"),  # Comoros
    ("STP", "no_program", "no_program"),  # Sao Tome and Principe
    ("MUS", "mandatory",  "no_program"),  # Mauritius
    ("SYC", "no_program", "no_program"),  # Seychelles
    ("DJI", "mandatory",  "no_program"),  # Djibouti
    ("SDN", "mandatory",  "no_program"),  # Sudan
    ("ZAF", "mandatory",  "mandatory"),   # South Africa
    # ── South & Southeast Asia ────────────────────────────────────────────────
    ("PAK", "mandatory",  "no_program"),  # Pakistan
    ("BGD", "mandatory",  "no_program"),  # Bangladesh
    ("IND", "voluntary",  "voluntary"),   # India (state-level voluntary)
    ("NPL", "voluntary",  "no_program"),  # Nepal
    ("LKA", "voluntary",  "no_program"),  # Sri Lanka
    ("PHL", "mandatory",  "no_program"),  # Philippines (wheat + rice)
    ("IDN", "mandatory",  "no_program"),  # Indonesia
    ("VNM", "voluntary",  "no_program"),  # Vietnam
    ("MMR", "no_program", "no_program"),  # Myanmar
    ("KHM", "voluntary",  "no_program"),  # Cambodia
    ("THA", "voluntary",  "no_program"),  # Thailand
    ("LAO", "no_program", "no_program"),  # Lao PDR
    ("MNG", "mandatory",  "no_program"),  # Mongolia
    ("PNG", "mandatory",  "no_program"),  # Papua New Guinea
    ("AFG", "mandatory",  "no_program"),  # Afghanistan
    # ── Middle East & North Africa ────────────────────────────────────────────
    ("EGY", "mandatory",  "no_program"),  # Egypt
    ("MAR", "mandatory",  "no_program"),  # Morocco
    ("TUN", "mandatory",  "no_program"),  # Tunisia
    ("DZA", "mandatory",  "no_program"),  # Algeria
    ("LBY", "mandatory",  "no_program"),  # Libya
    ("JOR", "mandatory",  "no_program"),  # Jordan
    ("IRQ", "mandatory",  "no_program"),  # Iraq
    ("SAU", "mandatory",  "no_program"),  # Saudi Arabia
    ("ARE", "mandatory",  "no_program"),  # UAE
    ("KWT", "mandatory",  "no_program"),  # Kuwait
    ("BHR", "mandatory",  "no_program"),  # Bahrain
    ("QAT", "mandatory",  "no_program"),  # Qatar
    ("OMN", "mandatory",  "no_program"),  # Oman
    ("YEM", "mandatory",  "no_program"),  # Yemen
    ("LBN", "mandatory",  "no_program"),  # Lebanon
    ("SYR", "mandatory",  "no_program"),  # Syria
    ("IRN", "mandatory",  "no_program"),  # Iran
    ("TUR", "voluntary",  "no_program"),  # Turkey
    ("PSE", "mandatory",  "no_program"),  # West Bank and Gaza
    # ── Central Asia ─────────────────────────────────────────────────────────
    ("KAZ", "mandatory",  "no_program"),  # Kazakhstan
    ("UZB", "mandatory",  "no_program"),  # Uzbekistan
    ("KGZ", "mandatory",  "no_program"),  # Kyrgyzstan
    ("TJK", "mandatory",  "no_program"),  # Tajikistan
    ("TKM", "mandatory",  "no_program"),  # Turkmenistan
    ("AZE", "mandatory",  "no_program"),  # Azerbaijan
    ("ARM", "mandatory",  "no_program"),  # Armenia
    ("GEO", "voluntary",  "no_program"),  # Georgia
    # ── Latin America & Caribbean ─────────────────────────────────────────────
    ("MEX", "mandatory",  "mandatory"),   # Mexico
    ("GTM", "mandatory",  "mandatory"),   # Guatemala
    ("HND", "mandatory",  "mandatory"),   # Honduras
    ("SLV", "mandatory",  "mandatory"),   # El Salvador
    ("NIC", "mandatory",  "mandatory"),   # Nicaragua
    ("CRI", "mandatory",  "no_program"),  # Costa Rica
    ("PAN", "mandatory",  "no_program"),  # Panama
    ("COL", "mandatory",  "mandatory"),   # Colombia
    ("VEN", "mandatory",  "mandatory"),   # Venezuela
    ("PER", "mandatory",  "no_program"),  # Peru
    ("BOL", "mandatory",  "no_program"),  # Bolivia
    ("ECU", "mandatory",  "no_program"),  # Ecuador
    ("PRY", "mandatory",  "no_program"),  # Paraguay
    ("BRA", "mandatory",  "mandatory"),   # Brazil
    ("DOM", "mandatory",  "mandatory"),   # Dominican Republic
    ("HTI", "mandatory",  "mandatory"),   # Haiti
    ("JAM", "mandatory",  "no_program"),  # Jamaica
    ("TTO", "mandatory",  "no_program"),  # Trinidad and Tobago
    ("CHL", "mandatory",  "no_program"),  # Chile
    ("ARG", "mandatory",  "no_program"),  # Argentina
    ("URY", "mandatory",  "no_program"),  # Uruguay
    ("CUB", "mandatory",  "no_program"),  # Cuba
    # ── Europe ────────────────────────────────────────────────────────────────
    ("GBR", "mandatory",  "no_program"),  # United Kingdom
    ("UKR", "mandatory",  "no_program"),  # Ukraine
    ("MDA", "mandatory",  "no_program"),  # Moldova
    ("BGR", "voluntary",  "no_program"),  # Bulgaria
    ("ROU", "voluntary",  "no_program"),  # Romania
    ("POL", "no_program", "no_program"),  # Poland
    ("DEU", "no_program", "no_program"),  # Germany
    ("FRA", "no_program", "no_program"),  # France
    ("ESP", "no_program", "no_program"),  # Spain
    ("ITA", "no_program", "no_program"),  # Italy
    # ── Oceania ───────────────────────────────────────────────────────────────
    ("AUS", "mandatory",  "no_program"),  # Australia (folic acid in bread flour)
    ("NZL", "mandatory",  "no_program"),  # New Zealand
]

# Coverage proxy: estimated % of wheat flour that is fortified
COVERAGE_PROXY = {
    "mandatory":  75,   # Accounts for artisanal mill non-compliance in LMICs
    "voluntary":  20,   # Limited uptake typical for voluntary schemes
    "no_program":  0,
}


def build_lsff_dataset() -> pd.DataFrame:
    rows = []
    for iso3, wheat, maize in LSFF_STATUS:
        rows.append({
            "iso3": iso3,
            "wheat_flour_legislation": wheat,
            "maize_flour_legislation": maize,
            "lsff_any_mandatory": (wheat == "mandatory" or maize == "mandatory"),
            "lsff_any_program":   (wheat != "no_program" or maize != "no_program"),
            "lsff_wheat_coverage_proxy_pct": COVERAGE_PROXY[wheat],
            "lsff_maize_coverage_proxy_pct": COVERAGE_PROXY[maize],
            "lsff_coverage_proxy_pct": max(COVERAGE_PROXY[wheat], COVERAGE_PROXY[maize]),
            "data_year": 2023,
            "source": "FFI Global Progress Nov 2023 / GAIN GFDx",
        })
    return pd.DataFrame(rows)


def pull_all():
    out_path = OUTPUT_DIR / "ffi_country_status.csv"
    if out_path.exists():
        print("  [skip] LSFF country status already generated")
        return

    print("  [build] LSFF dataset from FFI 2023 curated data...")
    df = build_lsff_dataset()
    df.to_csv(out_path, index=False)

    n_mandatory = df["lsff_any_mandatory"].sum()
    n_voluntary = (~df["lsff_any_mandatory"] & df["lsff_any_program"]).sum()
    n_none      = (~df["lsff_any_program"]).sum()
    print(f"         {len(df)} countries: {n_mandatory} mandatory, {n_voluntary} voluntary-only, {n_none} no program")
    print(f"         saved → {out_path.name}")


if __name__ == "__main__":
    print("Building LSFF dataset...")
    pull_all()
    print("Done.")
