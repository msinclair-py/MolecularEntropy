"""
Protein-ligand binding entropy calculation.

Computes the entropy change (dS) upon ligand binding for:
1. A static structure (PDB)
2. A simulation trajectory (prmtop/dcd)

The observable is total -T*dS and its standard deviation across frames.
"""

from molecular_entropy import BindingEntropyCalculator

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
LIGAND_RESIDUES = [250]  # 0-based residue index of the ligand

calc = BindingEntropyCalculator(rotamer_library="simple")  # or "extended"

# ---------------------------------------------------------
# 1. Static Structure (PDB)
# ---------------------------------------------------------
print("=" * 50)
print("Static Structure Analysis")
print("=" * 50)

result = calc.calculate_ligand_binding(
    pdb_path="data/protein_ligand.pdb",
    topology_path="data/topology.prmtop"
    ligand_residues=LIGAND_RESIDUES,
)

print(f"Total -T*dS: {result.total_negT_dS:.2f} kcal/mol")

# ---------------------------------------------------------
# 2. Trajectory Analysis (prmtop/dcd)
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("Trajectory Analysis")
print("=" * 50)

result = calc.calculate_ligand_binding_trajectory(
    traj_path="data/trajectory.dcd",
    topology_path="data/topology.prmtop",
    ligand_residues=LIGAND_RESIDUES,
)

print(f"Frames: {result.n_frames}")
print(f"Total -T*dS: {result.mean_negT_dS:.2f} +/- {result.std_negT_dS:.2f} kcal/mol")

# Save per-frame results
result.to_dataframe().write_csv("ligand_binding_entropy.csv")
