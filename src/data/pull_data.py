"""
Master data pull script — runs all source-specific pulls in sequence.
Usage: python src/data/pull_data.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.data.pull_who_gho import pull_all as pull_who
from src.data.pull_unicef import pull_all as pull_unicef
from src.data.pull_fao import pull_all as pull_fao
from src.data.pull_lsff import pull_all as pull_lsff
from src.data.pull_gbd import pull_all as pull_gbd
from src.data.pull_outcomes import pull_all as pull_outcomes


if __name__ == "__main__":
    print("=" * 60)
    print("Malnutrition Commons — Data Pull")
    print("=" * 60)

    print("\n[1/5] WHO Global Health Observatory")
    pull_who()

    print("\n[2/5] UNICEF / JME Malnutrition")
    pull_unicef()

    print("\n[3/5] FAO Food Security")
    pull_fao()

    print("\n[4/5] LSFF Coverage (FFI 2023)")
    pull_lsff()

    print("\n[5/5] GBD Micronutrient Deficiencies (OWID + optional manual GBD export)")
    pull_gbd()

    print("\n[6/6] Outcome & food-systems context indicators (World Bank)")
    pull_outcomes()

    print("\n" + "=" * 60)
    print("All pulls complete. Run src/data/harmonize.py next.")
    print("  Note: Iron deficiency requires a manual GBD download.")
    print("  See docs/gbd_download_guide.md for instructions.")
    print("=" * 60)
