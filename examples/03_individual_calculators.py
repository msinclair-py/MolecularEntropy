"""
Example using individual entropy calculators.

For advanced users who want fine-grained control over each
entropy component or want to use only specific calculations.
"""

from molecular_entropy import (
    SASACalculator,
    ANMEntropyCalculator,
    RotamerEntropyCalculator,
)
from molecular_entropy.structure import StructureLoader

# Load structure
loader = StructureLoader()
traj = loader.load("your_complex.pdb")

# Define chain IDs
chain_a = "A"
chain_b = "B"

# 1. SASA-based solvent entropy
# -----------------------------
sasa_calc = SASACalculator(
    probe_radius=1.4,  # Water probe radius in Angstroms
    alpha_np=-0.007,   # Nonpolar coefficient (kcal/mol/A^2)
    beta_pol=0.003,    # Polar coefficient (kcal/mol/A^2)
)
sasa_result = sasa_calc.calculate_binding_delta(traj, chain_a, chain_b)
print(f"SASA entropy: {sasa_result.negT_dS:.2f} kcal/mol")
print(f"  Buried nonpolar area: {sasa_result.delta_nonpolar:.1f} A^2")
print(f"  Buried polar area: {sasa_result.delta_polar:.1f} A^2")

# 2. ANM vibrational entropy
# --------------------------
anm_calc = ANMEntropyCalculator(
    cutoff=15.0,       # Interaction cutoff in Angstroms
    gamma=1.0,         # Spring constant
    n_modes=20,        # Number of lowest modes to compute
    temperature=298.15,
)
anm_result = anm_calc.calculate_binding_delta(traj, chain_a, chain_b)
print(f"\nANM entropy: {anm_result.negT_dS:.2f} kcal/mol")
print(f"  Complex modes: {len(anm_result.eigenvalues_complex)}")

# 3. Rotamer conformational entropy
# ---------------------------------
rotamer_calc = RotamerEntropyCalculator(
    rotlib_path="rotamer_library/simple_library.parquet",
    temperature=298.15,
)
rotamer_result = rotamer_calc.calculate_binding_delta(traj, chain_a, chain_b)
print(f"\nRotamer entropy: {rotamer_result.negT_dS:.2f} kcal/mol")
print(f"  Residues with entropy loss: {rotamer_result.n_affected_residues}")

# Combine results manually
total = (
    sasa_result.negT_dS
    + anm_result.negT_dS
    + rotamer_result.negT_dS
    + 8.0  # Trans/rot penalty
)
print(f"\nTotal -T*dS: {total:.2f} kcal/mol")
