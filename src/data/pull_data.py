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


if __name__ == "__main__":
    print("=" * 60)
    print("Malnutrition Commons — Data Pull")
    print("=" * 60)

    print("\n[1/4] WHO Global Health Observatory")
    pull_who()

    print("\n[2/4] UNICEF / JME Malnutrition")
    pull_unicef()

    print("\n[3/4] FAO Food Security")
    pull_fao()

    print("\n[4/4] LSFF Coverage (FFI 2023)")
    pull_lsff()

    print("\n" + "=" * 60)
    print("All pulls complete. Run src/data/harmonize.py next.")
    print("=" * 60)
