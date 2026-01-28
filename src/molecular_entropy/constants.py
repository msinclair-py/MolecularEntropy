"""
Physical constants for entropy calculations.

All units are in kcal/mol and Kelvin unless otherwise specified.
"""

__all__ = [
    "KB_KCAL",
    "R_KCAL",
    "ANGSTROM_PER_NM",
    "NM2_TO_ANGSTROM2",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_PROBE_RADIUS",
    "DEFAULT_N_SPHERE_POINTS",
    "DEFAULT_ANM_CUTOFF",
    "DEFAULT_ANM_GAMMA",
    "DEFAULT_ALPHA_NP",
    "DEFAULT_BETA_POL",
    "DEFAULT_TR_PENALTY",
    "CHI_RESIDUES",
    "CHI_DEFINITIONS",
    "POLAR_ELEMENTS",
    "SOLVENT_RESIDUES",
]

# Boltzmann constant in kcal/mol/K
KB_KCAL: float = 1.98720425864083e-3

# Gas constant (same as Boltzmann for per-mol quantities)
R_KCAL: float = 1.98720425864083e-3

# Unit conversions
ANGSTROM_PER_NM: float = 10.0
NM2_TO_ANGSTROM2: float = 100.0

# Default parameters for entropy calculations
DEFAULT_TEMPERATURE: float = 298.15  # K
DEFAULT_PROBE_RADIUS: float = 1.4  # Angstrom
DEFAULT_N_SPHERE_POINTS: int = 960

# ANM parameters
DEFAULT_ANM_CUTOFF: float = 15.0  # Angstrom
DEFAULT_ANM_GAMMA: float = 1.0  # Spring constant

# SASA-to-entropy coefficients (from Sharp et al. 1991)
# These relate SASA changes to solvent entropy
# For -T*dS_solv = alpha_np * dSASA_nonpolar + beta_pol * dSASA_polar
# Negative alpha_np means burying nonpolar surface is entropically favorable
DEFAULT_ALPHA_NP: float = -0.007  # kcal/mol per A^2 (nonpolar burial is favorable)
DEFAULT_BETA_POL: float = 0.003  # kcal/mol per A^2 (polar burial is unfavorable)

# Translational/rotational entropy penalty for binding
# This is the -T*dS_TR term at 298K, typically 6-12 kcal/mol
DEFAULT_TR_PENALTY: float = 8.0  # kcal/mol

# Residues with side-chain chi angles
CHI_RESIDUES: set[str] = {
    "ARG",
    "LYS",
    "GLU",
    "GLN",
    "MET",
    "ILE",
    "LEU",
    "VAL",
    "THR",
    "SER",
    "TYR",
    "PHE",
    "TRP",
    "ASN",
    "ASP",
    "HIS",
    "CYS",
    "PRO",
}

# Chi angle definitions per residue type
# Each chi is defined by 4 atom names
CHI_DEFINITIONS: dict[str, list[tuple[str, str, str, str]]] = {
    "ARG": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ"),
    ],
    "LYS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ"),
    ],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "VAL": [("N", "CA", "CB", "CG1")],
    "THR": [("N", "CA", "CB", "OG1")],
    "SER": [("N", "CA", "CB", "OG")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "ASN": [("N", "CA", "CB", "CG")],
    "ASP": [("N", "CA", "CB", "CG")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "PRO": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
}

# Polar elements for SASA classification
POLAR_ELEMENTS: set[str] = {"O", "N", "S"}

# Common solvent and ion residue names to exclude from protein selection
SOLVENT_RESIDUES: set[str] = {
    # Water models
    "WAT",
    "HOH",
    "TIP",
    "TIP3",
    "TIP4",
    "TIP5",
    "SPC",
    "SPCE",
    "OPC",
    "T3P",
    "T4P",
    "T4E",
    "T5P",
    "TP3",
    "TP4",
    "TP5",
    # Ions
    "NA",
    "Na+",
    "NA+",
    "SOD",
    "K",
    "K+",
    "POT",
    "CL",
    "Cl-",
    "CL-",
    "CLA",
    "MG",
    "MG2",
    "Mg2+",
    "CA",
    "CA2",
    "Ca2+",
    "ZN",
    "ZN2",
    "Zn2+",
    "FE",
    "FE2",
    "FE3",
    "CU",
    "CU1",
    "CU2",
    "MN",
    "CO",
    # Common buffer molecules
    "PO4",
    "SO4",
    "ACE",
    "NME",
    "NH2",
}
