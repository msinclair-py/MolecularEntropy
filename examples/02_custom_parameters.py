"""
Example showing how to customize calculation parameters.

Different parameters may be appropriate depending on your system
and the conditions you want to model.
"""

from molecular_entropy import BindingEntropyCalculator

# Initialize with custom parameters
calc = BindingEntropyCalculator(
    # Path to rotamer library (use extended for more rotamer states)
    rotlib_path="rotamer_library/extended_library.parquet",

    # Temperature in Kelvin (default: 298.15 K)
    # Use 310.15 K for physiological temperature
    temperature=310.15,

    # Translational/rotational entropy penalty in kcal/mol
    # Typical range: 5-15 kcal/mol depending on complex size
    trans_rot_penalty=10.0,

    # SASA probe radius in Angstroms (default: 1.4 A for water)
    probe_radius=1.4,

    # ANM cutoff distance in Angstroms (default: 15.0 A)
    # Shorter cutoff = faster but less accurate
    anm_cutoff=12.0,

    # ANM spring constant (default: 1.0)
    anm_gamma=1.0,
)

# Calculate with all components
result = calc.calculate("your_complex.pdb", chain_a="A", chain_b="B")
print(f"Full calculation: {result.total_negT_dS:.2f} kcal/mol")

# You can also skip specific components if not needed
calc_fast = BindingEntropyCalculator(
    rotlib_path="rotamer_library/simple_library.parquet",
    compute_sasa=True,
    compute_anm=False,      # Skip ANM (slowest component)
    compute_rotamer=False,  # Skip rotamer
)

result_fast = calc_fast.calculate("your_complex.pdb", chain_a="A", chain_b="B")
print(f"SASA only: {result_fast.negT_dS_sasa:.2f} kcal/mol")
