"""
Molecular Entropy - Binding entropy calculations for protein complexes.

This package computes binding entropy changes decomposed into:
- Translational/rotational entropy (TR)
- Vibrational entropy from ANM (Anisotropic Network Model)
- Side-chain conformational entropy from rotamer analysis
- Solvent entropy from SASA changes

Usage:
    from molecular_entropy import BindingEntropyCalculator

    calc = BindingEntropyCalculator(rotamer_library="simple")
    result = calc.calculate("complex.pdb", chain_a="A", chain_b="B")
    print(result.to_dataframe())

Available rotamer libraries:
    - "simple": Reduced library for faster calculations
    - "extended": Full Dunbrack backbone-dependent library
"""

from importlib.resources import files
from pathlib import Path
from typing import Literal

from .anm import ANMEntropyCalculator, ANMLigandResult
from .constants import KB_KCAL, R_KCAL
from .driver import (
    BindingEntropyCalculator,
    BindingEntropyResult,
    LigandBindingEntropyResult,
    LigandBindingTrajectoryResult,
    SingleChainEntropyResult,
)
from .rotamer import RotamerEntropyCalculator, SingleChainRotamerResult
from .sasa import SASACalculator
from .structure import StructureLoader

__version__ = "0.1.0"

RotamerLibrary = Literal["simple", "extended"]

_ROTLIB_FILES = {
    "simple": "simple_library.parquet",
    "extended": "extended_library.parquet",
}


def get_rotamer_library(name: RotamerLibrary) -> Path:
    """
    Get the path to a bundled rotamer library.

    Args:
        name: Library name ("simple" or "extended")

    Returns:
        Path to the parquet file

    Raises:
        ValueError: If library name is not recognized
    """
    if name not in _ROTLIB_FILES:
        valid = ", ".join(f'"{k}"' for k in _ROTLIB_FILES)
        raise ValueError(f"Unknown rotamer library: {name!r}. Valid options: {valid}")

    data_dir = files("molecular_entropy") / "data"
    return Path(str(data_dir / _ROTLIB_FILES[name]))


__all__ = [
    "BindingEntropyCalculator",
    "BindingEntropyResult",
    "SingleChainEntropyResult",
    "LigandBindingEntropyResult",
    "LigandBindingTrajectoryResult",
    "SASACalculator",
    "ANMEntropyCalculator",
    "ANMLigandResult",
    "RotamerEntropyCalculator",
    "SingleChainRotamerResult",
    "StructureLoader",
    "R_KCAL",
    "KB_KCAL",
    "get_rotamer_library",
    "RotamerLibrary",
]
