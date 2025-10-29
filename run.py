# Re-run SASA with stricter interface metrics
python compute_sasa.py --pdb complex.pdb --chains A,B --interface-threshold 5.0 --out results/

# Vibrational entropy (ANM)
python anm_entropy.py --pdb complex.pdb --chains A,B --out results/anm_entropy.csv

# Dunbrack side-chain entropy with explicit rotamer feasibility
python rotamer_entropy_dunbrack.py --pdb complex.pdb --chains A,B \
        --rotlib dunbrack_bbdep.json --out results/
