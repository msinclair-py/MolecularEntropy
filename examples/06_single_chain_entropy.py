"""
Single-chain entropy calculation example.

This example shows how to calculate the absolute entropy of a single
protein chain, useful for:
- Comparing isolated vs bound protein states
- Protein-small molecule systems where the ligand has no rotamers
- Understanding the entropic properties of a protein in isolation
"""

from molecular_entropy import BindingEntropyCalculator

# Initialize calculator with rotamer library
calc = BindingEntropyCalculator(
    rotamer_library="simple",  # or "extended"
    temperature=298.15,
)

# Calculate entropy for a single chain
# This computes SASA, ANM vibrational entropy, and rotamer entropy
# without any binding partner
result = calc.calculate_chain(
    pdb_path="data/static.pdb",
    chain_id="A",  # Analyze only chain A
)

# Print summary
print("Single-Chain Entropy Results")
print("=" * 50)
print(f"Chain: {result.chain_id}")
print(f"Temperature: {result.temperature} K")
print()

# SASA information (absolute values, not deltas)
if result.sasa:
    print("SASA (Solvent Accessible Surface Area):")
    print(f"  Total:    {result.sasa.total:10.1f} A^2")
    print(f"  Polar:    {result.sasa.polar:10.1f} A^2")
    print(f"  Nonpolar: {result.sasa.nonpolar:10.1f} A^2")
    print()

# Vibrational entropy from ANM
if result.anm:
    print("Vibrational Entropy (ANM):")
    print(f"  S:     {result.S_vib_kcal_per_K:10.6f} kcal/mol/K")
    print(f"  -T*S:  {result.negT_S_vib:10.2f} kcal/mol")
    print(f"  Modes: {result.anm.n_modes}")
    print()

# Rotamer conformational entropy
if result.rotamer:
    print("Rotamer Entropy:")
    print(f"  S:     {result.S_rotamer_kcal_per_K:10.6f} kcal/mol/K")
    print(f"  -T*S:  {result.negT_S_rotamer:10.2f} kcal/mol")
    print(f"  Residues with rotamers: {len(result.rotamer.per_residue)}")
    print()

# Total entropy
print("-" * 50)
print(f"Total S:     {result.total_S_kcal_per_K:10.6f} kcal/mol/K")
print(f"Total -T*S:  {result.total_negT_S:10.2f} kcal/mol")

# As DataFrame
print("\nAs DataFrame:")
print(result.to_dataframe())

# ---------------------------------------------------------
# Comparing two states: isolated chains
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("Comparing Chain A vs Chain B")
print("=" * 50)

result_a = calc.calculate_chain("data/static.pdb", chain_id="A")
result_b = calc.calculate_chain("data/static.pdb", chain_id="B")

print(f"\nChain A:")
print(f"  Vibrational -T*S: {result_a.negT_S_vib:8.2f} kcal/mol")
print(f"  Rotamer -T*S:     {result_a.negT_S_rotamer:8.2f} kcal/mol")
print(f"  Total -T*S:       {result_a.total_negT_S:8.2f} kcal/mol")

print(f"\nChain B:")
print(f"  Vibrational -T*S: {result_b.negT_S_vib:8.2f} kcal/mol")
print(f"  Rotamer -T*S:     {result_b.negT_S_rotamer:8.2f} kcal/mol")
print(f"  Total -T*S:       {result_b.total_negT_S:8.2f} kcal/mol")

# ---------------------------------------------------------
# Per-residue rotamer entropy analysis
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("Per-Residue Rotamer Entropy (Top 10 by entropy)")
print("=" * 50)

if result.rotamer:
    # Get per-residue data as DataFrame
    df = calc.rotamer_calc.single_chain_to_dataframe(result.rotamer)

    # Sort by entropy (highest first)
    df = df.sort("S_kcal_per_K", descending=True)
    print(df.head(10))
