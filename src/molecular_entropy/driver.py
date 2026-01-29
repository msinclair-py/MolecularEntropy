"""
Binding entropy driver - facade for complete entropy calculations.

Orchestrates SASA, ANM, and rotamer entropy calculations to provide
a complete binding entropy estimate.

Performance optimizations:
- Parallel computation of independent entropy terms
- Direct coordinate extraction (avoids redundant PDB parsing)
"""

__all__ = [
    "BindingEntropyCalculator",
    "BindingEntropyResult",
    "SingleChainEntropyResult",
    "LigandBindingEntropyResult",
    "LigandBindingTrajectoryResult",
    "main",
]

import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import mdtraj as md
import numpy as np
import polars as pl

from .anm import ANMBindingResult, ANMEntropyCalculator, ANMEntropyResult, ANMLigandResult
from .constants import (
    DEFAULT_ALPHA_NP,
    DEFAULT_ANM_CUTOFF,
    DEFAULT_ANM_GAMMA,
    DEFAULT_BETA_POL,
    DEFAULT_PROBE_RADIUS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TR_PENALTY,
    SOLVENT_RESIDUES,
)
from .rotamer import RotamerBindingResult, RotamerEntropyCalculator, SingleChainRotamerResult
from .sasa import SASABindingResult, SASACalculator, SASAResult
from .structure import StructureLoader

logger = logging.getLogger(__name__)


@dataclass
class BindingEntropyResult:
    """Complete binding entropy calculation result."""

    sasa: SASABindingResult | None
    anm: ANMBindingResult | None
    rotamer: RotamerBindingResult | None
    trans_rot_penalty: float
    temperature: float
    chain_a: str
    chain_b: str

    @property
    def negT_dS_sasa(self) -> float:
        """Solvent entropy contribution (-T*dS) in kcal/mol."""
        return self.sasa.negT_dS_solv if self.sasa else 0.0

    @property
    def negT_dS_vib(self) -> float:
        """Vibrational entropy contribution (-T*dS) in kcal/mol."""
        return self.anm.negT_dS_kcal if self.anm else 0.0

    @property
    def negT_dS_rotamer(self) -> float:
        """Side-chain entropy contribution (-T*dS) in kcal/mol."""
        return self.rotamer.total_negT_dS_kcal if self.rotamer else 0.0

    @property
    def negT_dS_TR(self) -> float:
        """Translational/rotational entropy penalty in kcal/mol."""
        return self.trans_rot_penalty

    @property
    def total_negT_dS(self) -> float:
        """Total -T*dS in kcal/mol."""
        return self.negT_dS_sasa + self.negT_dS_vib + self.negT_dS_rotamer + self.negT_dS_TR

    def to_dataframe(self) -> pl.DataFrame:
        """Convert summary to DataFrame."""
        rows = [
            {"term": "-T*dS_solvent", "value_kcal": self.negT_dS_sasa},
            {"term": "-T*dS_vibrational", "value_kcal": self.negT_dS_vib},
            {"term": "-T*dS_sidechain", "value_kcal": self.negT_dS_rotamer},
            {"term": "-T*dS_trans_rot", "value_kcal": self.negT_dS_TR},
            {"term": "TOTAL -T*dS", "value_kcal": self.total_negT_dS},
        ]
        return pl.DataFrame(rows)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "temperature_K": self.temperature,
            "chains": f"{self.chain_a},{self.chain_b}",
            "terms": {
                "negT_dS_solvent_kcal": self.negT_dS_sasa,
                "negT_dS_vibrational_kcal": self.negT_dS_vib,
                "negT_dS_sidechain_kcal": self.negT_dS_rotamer,
                "negT_dS_trans_rot_kcal": self.negT_dS_TR,
                "total_negT_dS_kcal": self.total_negT_dS,
            },
        }

        if self.sasa:
            result["sasa"] = {
                "delta_total_A2": self.sasa.delta_total,
                "delta_polar_A2": self.sasa.delta_polar,
                "delta_nonpolar_A2": self.sasa.delta_nonpolar,
            }

        if self.anm:
            result["anm"] = {
                "dS_kB": self.anm.dS_kB,
                "n_modes_complex": self.anm.complex_result.n_modes,
            }

        if self.rotamer:
            result["rotamer"] = {
                "n_residues": len(self.rotamer.per_residue),
                "total_dS_kcal_per_K": self.rotamer.total_dS_kcal_per_K,
            }

        return result


@dataclass
class SingleChainEntropyResult:
    """Complete single-chain entropy calculation result."""

    sasa: SASAResult | None
    anm: ANMEntropyResult | None
    rotamer: SingleChainRotamerResult | None
    temperature: float
    chain_id: str | None

    @property
    def S_sasa(self) -> float:
        """SASA entropy contribution (S) in kcal/mol/K."""
        # Note: For single chain, this is absolute SASA, not a delta
        # Entropy conversion would need empirical coefficients
        return 0.0  # Not directly comparable

    @property
    def S_vib_kcal_per_K(self) -> float:
        """Vibrational entropy (S) in kcal/mol/K."""
        return self.anm.S_kcal_per_K if self.anm else 0.0

    @property
    def S_rotamer_kcal_per_K(self) -> float:
        """Rotamer entropy (S) in kcal/mol/K."""
        return self.rotamer.total_S_kcal_per_K if self.rotamer else 0.0

    @property
    def total_S_kcal_per_K(self) -> float:
        """Total entropy (S) in kcal/mol/K (vibrational + rotamer)."""
        return self.S_vib_kcal_per_K + self.S_rotamer_kcal_per_K

    @property
    def negT_S_vib(self) -> float:
        """-T*S vibrational in kcal/mol."""
        return -self.temperature * self.S_vib_kcal_per_K

    @property
    def negT_S_rotamer(self) -> float:
        """-T*S rotamer in kcal/mol."""
        return self.rotamer.negT_S_kcal if self.rotamer else 0.0

    @property
    def total_negT_S(self) -> float:
        """Total -T*S in kcal/mol."""
        return self.negT_S_vib + self.negT_S_rotamer

    def to_dataframe(self) -> pl.DataFrame:
        """Convert summary to DataFrame."""
        rows = [
            {"term": "-T*S_vibrational", "value_kcal": self.negT_S_vib},
            {"term": "-T*S_sidechain", "value_kcal": self.negT_S_rotamer},
            {"term": "TOTAL -T*S", "value_kcal": self.total_negT_S},
        ]
        if self.sasa:
            rows.insert(0, {"term": "SASA_total_A2", "value_kcal": self.sasa.total})
        return pl.DataFrame(rows)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "temperature_K": self.temperature,
            "chain_id": self.chain_id,
            "terms": {
                "negT_S_vibrational_kcal": self.negT_S_vib,
                "negT_S_sidechain_kcal": self.negT_S_rotamer,
                "total_negT_S_kcal": self.total_negT_S,
            },
        }

        if self.sasa:
            result["sasa"] = {
                "total_A2": self.sasa.total,
                "polar_A2": self.sasa.polar,
                "nonpolar_A2": self.sasa.nonpolar,
            }

        if self.anm:
            result["anm"] = {
                "S_kB": self.anm.S_kB,
                "n_modes": self.anm.n_modes,
            }

        if self.rotamer:
            result["rotamer"] = {
                "n_residues": len(self.rotamer.per_residue),
                "total_S_kcal_per_K": self.rotamer.total_S_kcal_per_K,
            }

        return result


@dataclass
class LigandBindingEntropyResult:
    """Entropy change upon protein-small molecule binding."""

    # Complex state (protein + ligand)
    complex_sasa: SASAResult | None
    complex_anm: ANMLigandResult | None

    # Protein alone
    protein_alone_sasa: SASAResult | None
    protein_alone_anm: ANMEntropyResult | None
    protein_alone_rotamer: SingleChainRotamerResult | None

    # Protein in complex (for rotamer with ligand clashes)
    protein_complex_rotamer: SingleChainRotamerResult | None

    temperature: float
    ligand_residues: list[int] | None = None

    @property
    def delta_sasa_total(self) -> float:
        """Change in total SASA upon binding (A^2)."""
        if self.complex_sasa and self.protein_alone_sasa:
            # For protein-ligand: delta = protein_alone - complex_protein_portion
            # This is approximate since we don't separate protein SASA in complex
            return self.protein_alone_sasa.total - self.complex_sasa.total
        return 0.0

    @property
    def dS_vib_kcal_per_K(self) -> float:
        """Vibrational entropy change (dS) in kcal/mol/K."""
        if self.complex_anm and self.protein_alone_anm:
            return self.complex_anm.S_kcal_per_K - self.protein_alone_anm.S_kcal_per_K
        return 0.0

    @property
    def dS_rotamer_kcal_per_K(self) -> float:
        """Rotamer entropy change (dS) in kcal/mol/K."""
        if self.protein_complex_rotamer and self.protein_alone_rotamer:
            return (
                self.protein_complex_rotamer.total_S_kcal_per_K
                - self.protein_alone_rotamer.total_S_kcal_per_K
            )
        return 0.0

    @property
    def negT_dS_vib(self) -> float:
        """-T*dS vibrational in kcal/mol."""
        return -self.temperature * self.dS_vib_kcal_per_K

    @property
    def negT_dS_rotamer(self) -> float:
        """-T*dS rotamer in kcal/mol."""
        return -self.temperature * self.dS_rotamer_kcal_per_K

    @property
    def total_negT_dS(self) -> float:
        """Total -T*dS in kcal/mol (excludes translational/rotational)."""
        return self.negT_dS_vib + self.negT_dS_rotamer

    def to_dataframe(self) -> pl.DataFrame:
        """Convert summary to DataFrame."""
        rows = [
            {"term": "-T*dS_vibrational", "value_kcal": self.negT_dS_vib},
            {"term": "-T*dS_sidechain", "value_kcal": self.negT_dS_rotamer},
            {"term": "TOTAL -T*dS", "value_kcal": self.total_negT_dS},
        ]
        return pl.DataFrame(rows)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "temperature_K": self.temperature,
            "ligand_residues": self.ligand_residues,
            "terms": {
                "negT_dS_vibrational_kcal": self.negT_dS_vib,
                "negT_dS_sidechain_kcal": self.negT_dS_rotamer,
                "total_negT_dS_kcal": self.total_negT_dS,
            },
        }

        if self.complex_anm:
            result["complex_anm"] = {
                "n_protein_atoms": self.complex_anm.n_protein_atoms,
                "n_ligand_atoms": self.complex_anm.n_ligand_atoms,
                "n_modes": self.complex_anm.n_modes,
            }

        return result


@dataclass
class LigandBindingTrajectoryResult:
    """Trajectory statistics for protein-ligand binding entropy."""

    mean_negT_dS: float
    std_negT_dS: float
    n_frames: int
    per_frame_negT_dS: list[float] = field(repr=False)
    temperature: float = 298.15

    def to_dataframe(self) -> pl.DataFrame:
        """Return per-frame results as DataFrame."""
        return pl.DataFrame(
            {
                "frame": range(self.n_frames),
                "negT_dS_kcal": self.per_frame_negT_dS,
            }
        )


class BindingEntropyCalculator:
    """
    Facade for complete binding entropy calculations.

    Combines:
    - SASA-based solvent entropy
    - ANM vibrational entropy
    - Side-chain rotamer entropy
    - Translational/rotational entropy penalty

    Example:
        calc = BindingEntropyCalculator(rotamer_library="simple")
        result = calc.calculate("complex.pdb", chain_a="A", chain_b="B")
        print(result.total_negT_dS)
    """

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        trans_rot_penalty: float = DEFAULT_TR_PENALTY,
        # SASA options
        probe_radius: float = DEFAULT_PROBE_RADIUS,
        alpha_np: float = DEFAULT_ALPHA_NP,
        beta_pol: float = DEFAULT_BETA_POL,
        # ANM options
        anm_cutoff: float = DEFAULT_ANM_CUTOFF,
        anm_gamma: float = DEFAULT_ANM_GAMMA,
        # Rotamer options
        rotamer_library: str | None = None,
        rotlib_path: str | Path | None = None,
        # Component flags
        compute_sasa: bool = True,
        compute_anm: bool = True,
        compute_rotamer: bool = True,
    ):
        """
        Initialize binding entropy calculator.

        Args:
            temperature: Temperature in Kelvin
            trans_rot_penalty: -T*dS for translation/rotation loss (kcal/mol)
            probe_radius: Probe radius for SASA (Angstrom)
            alpha_np: SASA coefficient for nonpolar surface (kcal/mol/A^2)
            beta_pol: SASA coefficient for polar surface (kcal/mol/A^2)
            anm_cutoff: Distance cutoff for ANM springs (Angstrom)
            anm_gamma: ANM spring constant
            rotamer_library: Name of bundled library ("simple" or "extended")
            rotlib_path: Path to Dunbrack rotamer library (alternative to rotamer_library)
            compute_sasa: Whether to compute SASA entropy
            compute_anm: Whether to compute ANM entropy
            compute_rotamer: Whether to compute rotamer entropy
        """
        from . import get_rotamer_library

        self.temperature = temperature
        self.trans_rot_penalty = trans_rot_penalty

        self.compute_sasa_flag = compute_sasa
        self.compute_anm_flag = compute_anm
        self.compute_rotamer_flag = compute_rotamer

        # Initialize calculators
        if compute_sasa:
            self.sasa_calc = SASACalculator(
                probe_radius=probe_radius,
                alpha_np=alpha_np,
                beta_pol=beta_pol,
                temperature=temperature,
            )
        else:
            self.sasa_calc = None

        if compute_anm:
            self.anm_calc = ANMEntropyCalculator(
                cutoff=anm_cutoff,
                gamma=anm_gamma,
                temperature=temperature,
            )
        else:
            self.anm_calc = None

        # Resolve rotamer library path
        resolved_path = rotlib_path
        if rotamer_library is not None:
            resolved_path = get_rotamer_library(rotamer_library)

        if compute_rotamer and resolved_path:
            self.rotamer_calc = RotamerEntropyCalculator(
                rotlib_path=resolved_path,
                temperature=temperature,
            )
        else:
            self.rotamer_calc = None

    def calculate(
        self,
        pdb_path: str | Path,
        chain_a: str = "A",
        chain_b: str = "B",
        frame: int = 0,
        parallel: bool = False,  # Sequential is faster due to ANM's internal parallelism
        topology_path: str | Path | None = None,
        chain_a_residues: list[int] | None = None,
        chain_b_residues: list[int] | None = None,
    ) -> BindingEntropyResult:
        """
        Calculate complete binding entropy.

        Args:
            pdb_path: Path to PDB file of the complex
            chain_a: First chain ID (default "A")
            chain_b: Second chain ID (default "B")
            frame: Frame to use (for trajectories)
            parallel: Run entropy calculations in parallel (default True)
            topology_path: Optional topology file (e.g. AMBER prmtop) for
                          systems without chain IDs in the PDB.
            chain_a_residues: Optional list of 0-based residue indices to
                             assign to chain A. Use this for AMBER systems
                             that lack chain IDs. If None, chain_a is used
                             as a chain ID string for selection.
            chain_b_residues: Optional list of 0-based residue indices to
                             assign to chain B (e.g. ligand residues). Required
                             when chain_a_residues is provided.

        Returns:
            BindingEntropyResult with all entropy contributions
        """
        pdb_path = Path(pdb_path)
        logger.info(f"Calculating binding entropy for {pdb_path}")

        # Load structure once (shared by all calculations)
        traj = StructureLoader.load(pdb_path, topology=topology_path)

        # If residue-based chain assignment is requested, rebuild topology
        if chain_a_residues is not None or chain_b_residues is not None:
            if chain_a_residues is None or chain_b_residues is None:
                raise ValueError(
                    "Both chain_a_residues and chain_b_residues must be "
                    "provided together"
                )
            traj = StructureLoader.assign_chains(traj, chain_a_residues, chain_b_residues)
            chain_a = "A"
            chain_b = "B"

        logger.info(f"Chains: {chain_a} + {chain_b}, T = {self.temperature} K")

        if parallel:
            # Run all entropy calculations in parallel
            return self._calculate_parallel(traj, chain_a, chain_b, frame)
        else:
            # Sequential calculation
            return self._calculate_sequential(traj, chain_a, chain_b, frame)

    def calculate_chain(
        self,
        pdb_path: str | Path,
        chain_id: str | None = None,
        frame: int = 0,
    ) -> SingleChainEntropyResult:
        """
        Calculate entropy for a single protein chain.

        Useful for:
        - Protein-small molecule systems (ligand has no CA atoms/rotamers)
        - Comparing entropy of isolated vs bound protein

        Args:
            pdb_path: Path to PDB file
            chain_id: Chain ID to analyze (if None, uses all chains)
            frame: Frame index to use

        Returns:
            SingleChainEntropyResult with entropy components
        """
        pdb_path = Path(pdb_path)
        logger.info(f"Calculating single-chain entropy for {pdb_path}")
        logger.info(f"Chain: {chain_id or 'all'}, T = {self.temperature} K")

        # Load structure
        traj = StructureLoader.load(pdb_path)

        # If chain_id specified, select that chain
        if chain_id:
            traj = StructureLoader.select_chain(traj, chain_id)

        # SASA
        sasa_result = None
        if self.sasa_calc:
            logger.info("Computing SASA...")
            sasa_result = self.sasa_calc.calculate(traj, frame)

        # ANM vibrational entropy
        anm_result = None
        if self.anm_calc:
            logger.info("Computing ANM vibrational entropy...")
            anm_result = self.anm_calc.calculate(traj)

        # Rotamer entropy
        rotamer_result = None
        if self.rotamer_calc:
            logger.info("Computing rotamer entropy...")
            rotamer_result = self.rotamer_calc.calculate(traj, chain_id=None)

        result = SingleChainEntropyResult(
            sasa=sasa_result,
            anm=anm_result,
            rotamer=rotamer_result,
            temperature=self.temperature,
            chain_id=chain_id,
        )

        logger.info(f"Total -T*S = {result.total_negT_S:.3f} kcal/mol")

        return result

    def calculate_ligand_binding(
        self,
        pdb_path: str | Path,
        ligand_residues: list[int],
        protein_residues: list[int] | None = None,
        ligand_bonds: list[tuple[int, int]] | None = None,
        topology_path: str | Path | None = None,
        frame: int = 0,
    ) -> LigandBindingEntropyResult:
        """
        Calculate entropy change upon small molecule binding.

        Computes:
        - Protein entropy in complex (with ligand present for SASA/clash detection)
        - Protein entropy alone
        - Delta entropy

        Args:
            pdb_path: Path to PDB file of the protein-ligand complex
            ligand_residues: List of residue indices (0-based) that comprise the ligand.
            protein_residues: Optional list of residue indices for the protein.
                             If None, automatically detected by excluding ligand
                             and common solvent/ion residues (WAT, HOH, NA, CL, etc.)
            ligand_bonds: Optional list of (atom_idx, atom_idx) tuples defining
                         ligand internal connectivity for ANM. Indices are relative
                         to the ligand atoms. If None and topology_path is provided,
                         bonds will be extracted from the topology. Otherwise,
                         bonds will be inferred from distances.
            topology_path: Optional path to a topology file (e.g. AMBER prmtop)
                          that provides bond connectivity. Useful for static PDB
                          structures where ligand bonds cannot be reliably inferred
                          from distances alone.
            frame: Frame index to use

        Returns:
            LigandBindingEntropyResult with entropy changes
        """
        pdb_path = Path(pdb_path)
        logger.info(f"Calculating ligand binding entropy for {pdb_path}")
        logger.info(f"Ligand residues: {ligand_residues}")

        # Load full complex
        traj_complex = StructureLoader.load(pdb_path, topology=topology_path)

        ligand_residues = list(ligand_residues)
        ligand_set = set(ligand_residues)

        # Determine protein residue indices
        if protein_residues is None:
            # Auto-detect: exclude ligand and solvent/ions
            protein_residues = [
                res.index
                for res in traj_complex.topology.residues
                if res.index not in ligand_set and res.name not in SOLVENT_RESIDUES
            ]
            logger.info(f"Auto-detected {len(protein_residues)} protein residues")
        else:
            protein_residues = list(protein_residues)

        # Get protein-only trajectory
        traj_protein = StructureLoader.select_residues(traj_complex, protein_residues)

        # Get protein+ligand trajectory (excluding solvent for SASA/ANM)
        complex_residues = protein_residues + ligand_residues
        traj_protein_ligand = StructureLoader.select_residues(traj_complex, complex_residues)

        # Map original residue indices to new indices in sliced trajectory
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(complex_residues)}
        new_ligand_indices = [old_to_new[i] for i in ligand_residues]
        new_protein_indices = [old_to_new[i] for i in protein_residues]

        # Complex SASA (protein + ligand, no solvent)
        complex_sasa = None
        if self.sasa_calc:
            logger.info("Computing complex SASA...")
            complex_sasa = self.sasa_calc.calculate(traj_protein_ligand, frame)

        # Extract ligand bonds from topology if not explicitly provided
        if ligand_bonds is None and topology_path is not None:
            ligand_bonds = self._extract_ligand_bonds_from_topology(
                traj_complex, ligand_residues
            )

        # Complex ANM with ligand
        complex_anm = None
        if self.anm_calc:
            logger.info("Computing complex ANM (with ligand)...")
            complex_anm = self.anm_calc.calculate_with_ligand(
                traj_protein_ligand,
                ligand_residue_indices=new_ligand_indices,
                protein_residue_indices=new_protein_indices,
                ligand_bonds=ligand_bonds,
            )

        # Protein alone SASA
        protein_alone_sasa = None
        if self.sasa_calc:
            logger.info("Computing protein-alone SASA...")
            protein_alone_sasa = self.sasa_calc.calculate(traj_protein, frame)

        # Protein alone ANM
        protein_alone_anm = None
        if self.anm_calc:
            logger.info("Computing protein-alone ANM...")
            protein_alone_anm = self.anm_calc.calculate(traj_protein)

        # Protein alone rotamer entropy (all rotamers available)
        protein_alone_rotamer = None
        if self.rotamer_calc:
            logger.info("Computing protein-alone rotamer entropy...")
            protein_alone_rotamer = self.rotamer_calc.calculate(traj_protein)

        # Protein in complex rotamer entropy
        # For now, treat as same as alone (no clash detection with small molecule)
        # A more sophisticated approach would check rotamer clashes with ligand
        protein_complex_rotamer = protein_alone_rotamer

        result = LigandBindingEntropyResult(
            complex_sasa=complex_sasa,
            complex_anm=complex_anm,
            protein_alone_sasa=protein_alone_sasa,
            protein_alone_anm=protein_alone_anm,
            protein_alone_rotamer=protein_alone_rotamer,
            protein_complex_rotamer=protein_complex_rotamer,
            temperature=self.temperature,
            ligand_residues=ligand_residues,
        )

        logger.info(f"Total -T*dS = {result.total_negT_dS:.3f} kcal/mol")

        return result

    @staticmethod
    def _extract_ligand_bonds_from_topology(
        traj: "md.Trajectory",
        ligand_residues: list[int],
    ) -> list[tuple[int, int]] | None:
        """
        Extract ligand bond connectivity from a loaded topology.

        Finds all bonds where both atoms belong to ligand residues, then
        re-indexes them relative to the ligand atom list (0-based).

        Args:
            traj: Loaded trajectory with topology containing bond information.
            ligand_residues: Original residue indices of the ligand.

        Returns:
            List of (atom_i, atom_j) tuples with ligand-relative indices,
            or None if no bonds are found.
        """
        ligand_set = set(ligand_residues)

        # Collect atom indices belonging to ligand residues
        ligand_atoms = [
            atom.index
            for atom in traj.topology.atoms
            if atom.residue.index in ligand_set
        ]
        if not ligand_atoms:
            return None

        ligand_atom_set = set(ligand_atoms)
        # Map global atom index -> ligand-relative index
        global_to_local = {g: l for l, g in enumerate(sorted(ligand_atoms))}

        bonds = [
            (global_to_local[b[0].index], global_to_local[b[1].index])
            for b in traj.topology.bonds
            if b[0].index in ligand_atom_set and b[1].index in ligand_atom_set
        ]

        if bonds:
            logger.info(
                f"Extracted {len(bonds)} ligand bonds from topology"
            )
            return bonds

        logger.warning(
            "No ligand bonds found in topology; falling back to distance inference"
        )
        return None

    def calculate_ligand_binding_trajectory(
        self,
        traj_path: str | Path,
        topology_path: str | Path,
        ligand_residues: list[int],
        protein_residues: list[int] | None = None,
        ligand_bonds: list[tuple[int, int]] | None = None,
        frames: list[int] | None = None,
    ) -> LigandBindingTrajectoryResult:
        """
        Calculate ligand binding entropy statistics across a trajectory.

        Args:
            traj_path: Path to trajectory file (DCD, XTC, etc.)
            topology_path: Path to topology file (PDB or prmtop)
            ligand_residues: List of residue indices (0-based) that comprise the ligand
            protein_residues: Optional list of residue indices for the protein.
                             If None, automatically detected by excluding ligand
                             and common solvent/ion residues.
            ligand_bonds: Optional ligand bond connectivity for ANM
            frames: List of frame indices to analyze (default: all frames)

        Returns:
            LigandBindingTrajectoryResult with mean, std, and per-frame values
        """
        logger.info(f"Loading trajectory from {traj_path}")
        traj = StructureLoader.load(traj_path, topology=topology_path)

        # Auto-detect protein residues from first frame if not provided
        if protein_residues is None:
            ligand_set = set(ligand_residues)
            protein_residues = [
                res.index
                for res in traj.topology.residues
                if res.index not in ligand_set and res.name not in SOLVENT_RESIDUES
            ]
            logger.info(f"Auto-detected {len(protein_residues)} protein residues")

        if frames is None:
            frames = list(range(traj.n_frames))

        dS_values = []
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_pdb = Path(tmpdir) / "frame.pdb"

            for i, frame_idx in enumerate(frames):
                logger.info(f"Processing frame {frame_idx} ({i + 1}/{len(frames)})")
                traj[frame_idx].save_pdb(str(frame_pdb))

                result = self.calculate_ligand_binding(
                    pdb_path=frame_pdb,
                    ligand_residues=ligand_residues,
                    protein_residues=protein_residues,
                    ligand_bonds=ligand_bonds,
                )
                dS_values.append(result.total_negT_dS)

        dS_array = np.array(dS_values)

        return LigandBindingTrajectoryResult(
            mean_negT_dS=float(dS_array.mean()),
            std_negT_dS=float(dS_array.std()),
            n_frames=len(frames),
            per_frame_negT_dS=dS_values,
            temperature=self.temperature,
        )

    def _calculate_sequential(
        self,
        traj: md.Trajectory,
        chain_a: str,
        chain_b: str,
        frame: int = 0,
    ) -> BindingEntropyResult:
        """Calculate entropy terms sequentially."""
        # SASA entropy
        sasa_result = None
        if self.sasa_calc:
            logger.info("Computing SASA entropy...")
            sasa_result = self.sasa_calc.calculate_binding_delta(traj, chain_a, chain_b, frame)

        # ANM vibrational entropy (use trajectory directly, avoid re-parsing PDB)
        anm_result = None
        if self.anm_calc:
            logger.info("Computing ANM vibrational entropy...")
            anm_result = self.anm_calc.calculate_binding_delta(traj, chain_a, chain_b)

        # Rotamer entropy
        rotamer_result = None
        if self.rotamer_calc:
            logger.info("Computing rotamer entropy...")
            rotamer_result = self.rotamer_calc.calculate_binding_delta(traj, chain_a, chain_b)

        result = BindingEntropyResult(
            sasa=sasa_result,
            anm=anm_result,
            rotamer=rotamer_result,
            trans_rot_penalty=self.trans_rot_penalty,
            temperature=self.temperature,
            chain_a=chain_a,
            chain_b=chain_b,
        )

        logger.info(f"Total -T*dS = {result.total_negT_dS:.3f} kcal/mol")

        return result

    def _calculate_parallel(
        self,
        traj: md.Trajectory,
        chain_a: str,
        chain_b: str,
        frame: int = 0,
    ) -> BindingEntropyResult:
        """Calculate entropy terms in parallel."""
        results = {
            "sasa": None,
            "anm": None,
            "rotamer": None,
        }

        # Define calculation functions
        def calc_sasa():
            if self.sasa_calc:
                return self.sasa_calc.calculate_binding_delta(traj, chain_a, chain_b, frame)
            return None

        def calc_anm():
            if self.anm_calc:
                return self.anm_calc.calculate_binding_delta(traj, chain_a, chain_b)
            return None

        def calc_rotamer():
            if self.rotamer_calc:
                return self.rotamer_calc.calculate_binding_delta(traj, chain_a, chain_b)
            return None

        # Run calculations in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(calc_sasa): "sasa",
                executor.submit(calc_anm): "anm",
                executor.submit(calc_rotamer): "rotamer",
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error(f"Error in {name} calculation: {e}")
                    raise

        result = BindingEntropyResult(
            sasa=results["sasa"],
            anm=results["anm"],
            rotamer=results["rotamer"],
            trans_rot_penalty=self.trans_rot_penalty,
            temperature=self.temperature,
            chain_a=chain_a,
            chain_b=chain_b,
        )

        logger.info(f"Total -T*dS = {result.total_negT_dS:.3f} kcal/mol")

        return result

    def calculate_from_trajectory(
        self,
        traj_path: str | Path,
        topology_path: str | Path,
        chain_a: str = "A",
        chain_b: str = "B",
        frames: list[int] | None = None,
        chain_a_residues: list[int] | None = None,
        chain_b_residues: list[int] | None = None,
    ) -> list[BindingEntropyResult]:
        """
        Calculate binding entropy for multiple frames of a trajectory.

        Args:
            traj_path: Path to trajectory file (DCD, XTC, etc.)
            topology_path: Path to topology file (PDB or prmtop)
            chain_a: First chain ID (default "A")
            chain_b: Second chain ID (default "B")
            frames: List of frame indices to analyze (default: all frames)
            chain_a_residues: Optional list of 0-based residue indices to
                             assign to chain A. Use for AMBER systems without
                             chain IDs.
            chain_b_residues: Optional list of 0-based residue indices to
                             assign to chain B. Required when chain_a_residues
                             is provided.

        Returns:
            List of BindingEntropyResult for each frame
        """
        logger.info(f"Loading trajectory from {traj_path}")
        traj = StructureLoader.load(traj_path, topology=topology_path)

        # If residue-based chain assignment is requested, rebuild topology
        if chain_a_residues is not None or chain_b_residues is not None:
            if chain_a_residues is None or chain_b_residues is None:
                raise ValueError(
                    "Both chain_a_residues and chain_b_residues must be "
                    "provided together"
                )
            traj = StructureLoader.assign_chains(traj, chain_a_residues, chain_b_residues)
            chain_a = "A"
            chain_b = "B"

        if frames is None:
            frames = list(range(traj.n_frames))

        results = []
        for i, frame in enumerate(frames):
            logger.info(f"Processing frame {frame} ({i + 1}/{len(frames)})")

            # SASA for this frame
            sasa_result = None
            if self.sasa_calc:
                sasa_result = self.sasa_calc.calculate_binding_delta(traj, chain_a, chain_b, frame)

            # ANM vibrational entropy for this frame
            anm_result = None
            if self.anm_calc:
                anm_result = self.anm_calc.calculate_binding_delta(traj, chain_a, chain_b, frame)

            # Rotamer entropy for this frame
            rotamer_result = None
            if self.rotamer_calc:
                # Use single frame
                frame_traj = traj[frame]
                rotamer_result = self.rotamer_calc.calculate_binding_delta(
                    frame_traj, chain_a, chain_b
                )

            result = BindingEntropyResult(
                sasa=sasa_result,
                anm=anm_result,
                rotamer=rotamer_result,
                trans_rot_penalty=self.trans_rot_penalty,
                temperature=self.temperature,
                chain_a=chain_a,
                chain_b=chain_b,
            )
            results.append(result)

        return results

    def save_results(
        self,
        result: BindingEntropyResult,
        output_dir: str | Path,
        prefix: str = "binding_entropy",
    ) -> None:
        """
        Save results to files.

        Args:
            result: BindingEntropyResult
            output_dir: Directory for output files
            prefix: Prefix for output filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary CSV
        summary_df = result.to_dataframe()
        summary_df.write_csv(output_dir / f"{prefix}_summary.csv")

        # Full results JSON
        with open(output_dir / f"{prefix}_results.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Per-residue rotamer data if available
        if result.rotamer:
            rot_df = self.rotamer_calc.to_dataframe(result.rotamer)
            rot_df.write_csv(output_dir / f"{prefix}_rotamer.csv")

        # SASA interface data if available
        if result.sasa:
            sasa_df = pl.DataFrame(
                [
                    {
                        "SASA_complex_A2": result.sasa.complex_sasa.total,
                        "SASA_chainA_A2": result.sasa.chain_a_sasa.total,
                        "SASA_chainB_A2": result.sasa.chain_b_sasa.total,
                        "delta_SASA_A2": result.sasa.delta_total,
                        "delta_polar_A2": result.sasa.delta_polar,
                        "delta_nonpolar_A2": result.sasa.delta_nonpolar,
                    }
                ]
            )
            sasa_df.write_csv(output_dir / f"{prefix}_sasa.csv")

        logger.info(f"Results saved to {output_dir}")


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate binding entropy for protein complexes")
    parser.add_argument("--pdb", required=True, help="PDB file path")
    parser.add_argument("--chains", required=True, help="Chain IDs (e.g., A,B)")
    parser.add_argument("--rotlib", help="Path to rotamer library")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--T", type=float, default=DEFAULT_TEMPERATURE, help="Temperature (K)")
    parser.add_argument(
        "--tr-penalty",
        type=float,
        default=DEFAULT_TR_PENALTY,
        help="-T*dS for TR entropy (kcal/mol)",
    )
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=DEFAULT_PROBE_RADIUS,
        help="SASA probe radius (Angstrom)",
    )
    parser.add_argument(
        "--alpha-np", type=float, default=DEFAULT_ALPHA_NP, help="SASA nonpolar coefficient"
    )
    parser.add_argument(
        "--beta-pol", type=float, default=DEFAULT_BETA_POL, help="SASA polar coefficient"
    )
    parser.add_argument(
        "--anm-cutoff", type=float, default=DEFAULT_ANM_CUTOFF, help="ANM distance cutoff"
    )
    parser.add_argument(
        "--anm-gamma", type=float, default=DEFAULT_ANM_GAMMA, help="ANM spring constant"
    )
    parser.add_argument("--no-sasa", action="store_true", help="Skip SASA")
    parser.add_argument("--no-anm", action="store_true", help="Skip ANM")
    parser.add_argument("--no-rotamer", action="store_true", help="Skip rotamer")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s"
    )

    # Parse chains
    chain_a, chain_b = [c.strip() for c in args.chains.split(",")]

    # Create calculator
    calc = BindingEntropyCalculator(
        temperature=args.T,
        trans_rot_penalty=args.tr_penalty,
        probe_radius=args.probe_radius,
        alpha_np=args.alpha_np,
        beta_pol=args.beta_pol,
        anm_cutoff=args.anm_cutoff,
        anm_gamma=args.anm_gamma,
        rotlib_path=args.rotlib,
        compute_sasa=not args.no_sasa,
        compute_anm=not args.no_anm,
        compute_rotamer=not args.no_rotamer and args.rotlib,
    )

    # Calculate
    result = calc.calculate(args.pdb, chain_a, chain_b)

    # Save results
    calc.save_results(result, args.out)

    # Print summary
    print("\n=== Binding Entropy Summary ===")
    print(f"PDB: {args.pdb}")
    print(f"Chains: {chain_a} + {chain_b}")
    print(f"Temperature: {args.T} K")
    print()
    print(result.to_dataframe())
    print()
    print(f"Results saved to: {args.out}/")


if __name__ == "__main__":
    main()
