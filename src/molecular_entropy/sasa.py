"""
SASA (Solvent Accessible Surface Area) entropy calculations.

Computes SASA using the Shrake-Rupley algorithm via rust-simulation-tools,
then estimates the solvent entropy contribution based on burial of
polar/nonpolar surface.
"""

__all__ = [
    "SASACalculator",
    "SASAResult",
    "SASABindingResult",
]

import logging
from dataclasses import dataclass

import mdtraj as md
import numpy as np
import numpy.typing as npt
import polars as pl
from rust_simulation_tools import calculate_sasa, get_radii_array

from .constants import (
    ANGSTROM_PER_NM,
    DEFAULT_ALPHA_NP,
    DEFAULT_BETA_POL,
    DEFAULT_N_SPHERE_POINTS,
    DEFAULT_PROBE_RADIUS,
    DEFAULT_TEMPERATURE,
    POLAR_ELEMENTS,
)
from .structure import StructureLoader

logger = logging.getLogger(__name__)


@dataclass
class SASAResult:
    """Results from SASA calculation."""

    total: float  # Total SASA in A^2
    polar: float  # Polar SASA in A^2
    nonpolar: float  # Nonpolar SASA in A^2
    per_atom: npt.NDArray[np.float64] | None = None


@dataclass
class SASABindingResult:
    """Results from binding SASA change calculation."""

    delta_total: float  # SASA change (unbound - bound) in A^2
    delta_polar: float
    delta_nonpolar: float
    negT_dS_solv: float  # -T*dS_solv in kcal/mol
    complex_sasa: SASAResult
    chain_a_sasa: SASAResult
    chain_b_sasa: SASAResult


class SASACalculator:
    """
    Calculate SASA and solvent entropy contributions.

    Uses rust-simulation-tools' KD-tree optimized Shrake-Rupley algorithm
    for SASA computation. Converts SASA changes to entropy using empirical
    coefficients.
    """

    def __init__(
        self,
        probe_radius: float = DEFAULT_PROBE_RADIUS,
        n_sphere_points: int = DEFAULT_N_SPHERE_POINTS,
        alpha_np: float = DEFAULT_ALPHA_NP,
        beta_pol: float = DEFAULT_BETA_POL,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Initialize SASA calculator.

        Args:
            probe_radius: Probe radius in Angstrom (default 1.4 for water)
            n_sphere_points: Number of points on sphere for integration
            alpha_np: Coefficient for nonpolar SASA (kcal/mol/A^2)
                     Negative value means burying nonpolar surface is favorable
            beta_pol: Coefficient for polar SASA (kcal/mol/A^2)
                     Positive value means burying polar surface is unfavorable
            temperature: Temperature in Kelvin
        """
        self.probe_radius = probe_radius
        self.n_sphere_points = n_sphere_points
        self.alpha_np = alpha_np
        self.beta_pol = beta_pol
        self.temperature = temperature

    def compute_sasa(self, traj: md.Trajectory, frame: int = 0) -> npt.NDArray[np.float64]:
        """
        Compute per-atom SASA in Angstrom^2.

        Args:
            traj: MDTraj trajectory
            frame: Frame index to use (default 0)

        Returns:
            Array of per-atom SASA values in Angstrom^2
        """
        # Extract data from mdtraj trajectory
        # mdtraj uses nm, rust-simulation-tools uses Angstrom
        # Ensure float64 dtype for Rust interop
        coords = np.asarray(traj.xyz[frame] * ANGSTROM_PER_NM, dtype=np.float64)

        # Get element symbols and VDW radii
        elements = [atom.element.symbol if atom.element else "C" for atom in traj.topology.atoms]
        radii = np.asarray(get_radii_array(elements), dtype=np.float64)

        # Get residue indices
        residue_indices = np.array(
            [atom.residue.index for atom in traj.topology.atoms], dtype=np.int64
        )

        # Calculate SASA using Rust backend (returns Angstrom^2)
        result = calculate_sasa(
            coords,
            radii,
            residue_indices,
            probe_radius=self.probe_radius,
            n_sphere_points=self.n_sphere_points,
        )

        return result["per_atom"]

    def split_polar_nonpolar(
        self, traj: md.Trajectory, sasa: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Split SASA into polar and nonpolar components.

        Args:
            traj: MDTraj trajectory
            sasa: Per-atom SASA array

        Returns:
            Tuple of (polar_sasa, nonpolar_sasa) arrays
        """
        polar = np.zeros_like(sasa)
        nonpolar = np.zeros_like(sasa)

        for atom in traj.topology.atoms:
            elem = atom.element.symbol if atom.element else ""
            if elem in POLAR_ELEMENTS:
                polar[atom.index] = sasa[atom.index]
            else:
                nonpolar[atom.index] = sasa[atom.index]

        return polar, nonpolar

    def calculate(self, traj: md.Trajectory, frame: int = 0) -> SASAResult:
        """
        Calculate SASA for a structure.

        Args:
            traj: MDTraj trajectory
            frame: Frame index (default 0)

        Returns:
            SASAResult with total, polar, and nonpolar SASA
        """
        sasa = self.compute_sasa(traj, frame)
        polar, nonpolar = self.split_polar_nonpolar(traj, sasa)

        return SASAResult(
            total=float(sasa.sum()),
            polar=float(polar.sum()),
            nonpolar=float(nonpolar.sum()),
            per_atom=sasa,
        )

    def calculate_binding_delta(
        self,
        traj: md.Trajectory,
        chain_a: str,
        chain_b: str,
        frame: int = 0,
    ) -> SASABindingResult:
        """
        Calculate SASA change upon binding.

        The change is computed as: delta = (SASA_A_alone + SASA_B_alone) - SASA_complex
        Positive delta means surface was buried upon binding.

        Args:
            traj: MDTraj trajectory of the complex
            chain_a: First chain ID
            chain_b: Second chain ID
            frame: Frame index (default 0)

        Returns:
            SASABindingResult with SASA changes and entropy contribution
        """
        # Get trajectories for each state
        traj_a = StructureLoader.select_chain(traj, chain_a)
        traj_b = StructureLoader.select_chain(traj, chain_b)

        # Compute SASA for each state
        complex_sasa = self.calculate(traj, frame)
        sasa_a = self.calculate(traj_a, frame)
        sasa_b = self.calculate(traj_b, frame)

        # Delta = unbound - bound (positive = buried)
        delta_total = (sasa_a.total + sasa_b.total) - complex_sasa.total
        delta_polar = (sasa_a.polar + sasa_b.polar) - complex_sasa.polar
        delta_nonpolar = (sasa_a.nonpolar + sasa_b.nonpolar) - complex_sasa.nonpolar

        # Solvent entropy contribution: -T*dS_solv
        # alpha_np < 0 means burying nonpolar surface releases entropy (favorable)
        # beta_pol > 0 means burying polar surface is entropically unfavorable
        # delta > 0 when surface is buried, so:
        # -T*dS_solv = alpha_np * delta_nonpolar + beta_pol * delta_polar
        negT_dS_solv = self.alpha_np * delta_nonpolar + self.beta_pol * delta_polar

        logger.info(
            f"SASA change: total={delta_total:.1f} A^2, "
            f"polar={delta_polar:.1f} A^2, nonpolar={delta_nonpolar:.1f} A^2"
        )
        logger.info(f"-T*dS_solv = {negT_dS_solv:.3f} kcal/mol")

        return SASABindingResult(
            delta_total=delta_total,
            delta_polar=delta_polar,
            delta_nonpolar=delta_nonpolar,
            negT_dS_solv=negT_dS_solv,
            complex_sasa=complex_sasa,
            chain_a_sasa=sasa_a,
            chain_b_sasa=sasa_b,
        )

    def per_residue_sasa(self, traj: md.Trajectory, frame: int = 0) -> pl.DataFrame:
        """
        Calculate per-residue SASA.

        Args:
            traj: MDTraj trajectory
            frame: Frame index

        Returns:
            DataFrame with columns: resSeq, resName, chain_id, sasa_A2
        """
        sasa = self.compute_sasa(traj, frame)

        # Sum SASA per residue
        res_sasa = {}
        for atom in traj.topology.atoms:
            res = atom.residue
            key = (res.chain.chain_id, res.resSeq, res.name)
            res_sasa[key] = res_sasa.get(key, 0.0) + sasa[atom.index]

        rows = [
            {
                "chain_id": chain_id,
                "resSeq": resSeq,
                "resName": resName,
                "sasa_A2": sasa_val,
            }
            for (chain_id, resSeq, resName), sasa_val in res_sasa.items()
        ]

        df = pl.DataFrame(rows)
        return df.sort(["chain_id", "resSeq"])

    def detect_interface(
        self,
        traj: md.Trajectory,
        chain_a: str,
        chain_b: str,
        threshold: float = 1.0,
        frame: int = 0,
    ) -> pl.DataFrame:
        """
        Detect interface residues based on SASA change.

        Args:
            traj: MDTraj trajectory of complex
            chain_a: First chain ID
            chain_b: Second chain ID
            threshold: Minimum SASA change (A^2) to be considered interface
            frame: Frame index

        Returns:
            DataFrame of interface residues with delta_sasa_A2 column
        """
        # Get per-residue SASA for each state
        traj_a = StructureLoader.select_chain(traj, chain_a)
        traj_b = StructureLoader.select_chain(traj, chain_b)

        sasa_complex = self.per_residue_sasa(traj, frame)
        sasa_a = self.per_residue_sasa(traj_a, frame).with_columns(
            pl.lit(chain_a).alias("chain_id")
        )
        sasa_b = self.per_residue_sasa(traj_b, frame).with_columns(
            pl.lit(chain_b).alias("chain_id")
        )

        # Combine unbound
        sasa_unbound = pl.concat([sasa_a, sasa_b])

        # Merge and compute delta
        merged = sasa_unbound.join(
            sasa_complex, on=["chain_id", "resSeq", "resName"], suffix="_bound", how="full"
        ).fill_null(0.0)

        # Rename sasa_A2 to sasa_A2_unbound for clarity
        merged = merged.rename({"sasa_A2": "sasa_A2_unbound"})

        merged = merged.with_columns(
            (pl.col("sasa_A2_unbound") - pl.col("sasa_A2_bound")).alias("delta_sasa_A2")
        )

        # Filter by threshold
        interface = merged.filter(pl.col("delta_sasa_A2") >= threshold)
        return interface.sort(["chain_id", "resSeq"])
