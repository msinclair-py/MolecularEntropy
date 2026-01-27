"""
Basic usage example for Molecular Entropy.

This example shows the simplest way to calculate binding entropy
for a protein-protein complex from a PDB file.
"""

from molecular_entropy import BindingEntropyCalculator

# Initialize the calculator with default parameters
calc = BindingEntropyCalculator(
    rotlib_path="rotamer_library/simple_library.parquet"
)

# Calculate binding entropy for a complex
# Replace with your PDB file path
result = calc.calculate(
    pdb_path="data/static.pdb",
    chain_a="A",  # First binding partner
    chain_b="B",  # Second binding partner
)

# Print summary
print("Binding Entropy Results")
print("=" * 40)
print(f"Solvent (SASA):      {result.negT_dS_sasa:8.2f} kcal/mol")
print(f"Vibrational (ANM):   {result.negT_dS_vib:8.2f} kcal/mol")
print(f"Rotamer:             {result.negT_dS_rotamer:8.2f} kcal/mol")
print(f"Trans/Rot penalty:   {result.negT_dS_TR:8.2f} kcal/mol")
print("-" * 40)
print(f"Total -T*dS:         {result.total_negT_dS:8.2f} kcal/mol")

# Get results as a Polars DataFrame
df = result.to_dataframe()
print("\nAs DataFrame:")
print(df)

# Get results as a dictionary (useful for JSON serialization)
data = result.to_dict()
print("\nAs dictionary:")
print(data)
