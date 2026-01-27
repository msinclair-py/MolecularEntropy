"""
Molecular Entropy - Binding entropy calculations for protein complexes.

This package computes binding entropy changes decomposed into:
- Translational/rotational entropy (TR)
- Vibrational entropy from ANM (Anisotropic Network Model)
- Side-chain conformational entropy from rotamer analysis
- Solvent entropy from SASA changes

Usage:
    from molecular_entropy import BindingEntropyCalculator

    calc = BindingEntropyCalculator(
        rotlib_path="rotamer_library/simple_library.parquet"
    )
    result = calc.calculate("complex.pdb", chain_a="A", chain_b="B")
    print(result.to_dataframe())
"""

from .anm import ANMEntropyCalculator
from .constants import KB_KCAL, R_KCAL
from .driver import BindingEntropyCalculator, BindingEntropyResult
from .rotamer import RotamerEntropyCalculator
from .sasa import SASACalculator
from .structure import StructureLoader

__version__ = "0.1.0"
__all__ = [
    "BindingEntropyCalculator",
    "BindingEntropyResult",
    "SASACalculator",
    "ANMEntropyCalculator",
    "RotamerEntropyCalculator",
    "StructureLoader",
    "R_KCAL",
    "KB_KCAL",
]
