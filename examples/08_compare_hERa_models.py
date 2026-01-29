"""
Compare entropy between two hERa (human estrogen receptor alpha) models.

Calculates and compares the single-chain entropy of hERa bound to
estradiol vs genistein, highlighting differences in vibrational and
rotamer contributions.
"""

from molecular_entropy import BindingEntropyCalculator

calc = BindingEntropyCalculator(rotamer_library="simple")

# ---------------------------------------------------------
# Calculate entropy for each model
# ---------------------------------------------------------
result_estradiol = calc.calculate_chain("data/hERa_estradiol.pdb")
result_genistein = calc.calculate_chain("data/hERa_genistein.pdb")

# ---------------------------------------------------------
# Print individual results
# ---------------------------------------------------------
for label, result in [("hERa + Estradiol", result_estradiol),
                      ("hERa + Genistein", result_genistein)]:
    print("=" * 50)
    print(label)
    print("=" * 50)

    if result.sasa:
        print(f"  SASA total:       {result.sasa.total:10.1f} A^2")
        print(f"  SASA polar:       {result.sasa.polar:10.1f} A^2")
        print(f"  SASA nonpolar:    {result.sasa.nonpolar:10.1f} A^2")

    if result.anm:
        print(f"  Vibrational -T*S: {result.negT_S_vib:10.2f} kcal/mol")

    if result.rotamer:
        print(f"  Rotamer -T*S:     {result.negT_S_rotamer:10.2f} kcal/mol")

    print(f"  Total -T*S:       {result.total_negT_S:10.2f} kcal/mol")
    print()

# ---------------------------------------------------------
# Compare the two models
# ---------------------------------------------------------
print("=" * 50)
print("Comparison (Estradiol - Genistein)")
print("=" * 50)

d_vib = result_estradiol.negT_S_vib - result_genistein.negT_S_vib
d_rot = result_estradiol.negT_S_rotamer - result_genistein.negT_S_rotamer

d_total = d_vib + d_rot

print(f"  d(Vibrational -T*S): {d_vib:+10.2f} kcal/mol")
print(f"  d(Rotamer -T*S):     {d_rot:+10.2f} kcal/mol")
print(f"  d(Total -T*S):       {d_total:+10.2f} kcal/mol")
