"""
Example of batch processing multiple PDB structures.

Useful for analyzing many complexes or comparing variants.
"""

from pathlib import Path
from molecular_entropy import BindingEntropyCalculator
import polars as pl

# Initialize calculator once (reuses rotamer library cache)
calc = BindingEntropyCalculator(
    rotlib_path="rotamer_library/simple_library.parquet"
)

# Directory containing PDB files
pdb_dir = Path("structures/")
output_dir = Path("results/")
output_dir.mkdir(exist_ok=True)

# Collect results
all_results = []

# Process each PDB file
for pdb_file in sorted(pdb_dir.glob("*.pdb")):
    print(f"Processing {pdb_file.name}...")

    try:
        result = calc.calculate(
            pdb_path=str(pdb_file),
            chain_a="A",
            chain_b="B",
        )

        # Save individual result
        calc.save_results(
            result,
            output_dir=str(output_dir),
            prefix=pdb_file.stem,
        )

        # Collect for summary
        all_results.append({
            "structure": pdb_file.stem,
            "sasa": result.negT_dS_sasa,
            "anm": result.negT_dS_vib,
            "rotamer": result.negT_dS_rotamer,
            "tr": result.negT_dS_TR,
            "total": result.total_negT_dS,
        })

        print(f"  Total: {result.total_negT_dS:.2f} kcal/mol")

    except Exception as e:
        print(f"  Error: {e}")

# Create summary DataFrame
if all_results:
    summary_df = pl.DataFrame(all_results)
    summary_df.write_csv(output_dir / "summary.csv")

    print("\nSummary Statistics:")
    print(summary_df.describe())
