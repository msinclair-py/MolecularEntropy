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
    "main",
]

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import mdtraj as md
import polars as pl

from .anm import ANMBindingResult, ANMEntropyCalculator
from .constants import (
    DEFAULT_ALPHA_NP,
    DEFAULT_ANM_CUTOFF,
    DEFAULT_ANM_GAMMA,
    DEFAULT_BETA_POL,
    DEFAULT_PROBE_RADIUS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TR_PENALTY,
)
from .rotamer import RotamerBindingResult, RotamerEntropyCalculator
from .sasa import SASABindingResult, SASACalculator
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


class BindingEntropyCalculator:
    """
    Facade for complete binding entropy calculations.

    Combines:
    - SASA-based solvent entropy
    - ANM vibrational entropy
    - Side-chain rotamer entropy
    - Translational/rotational entropy penalty

    Example:
        calc = BindingEntropyCalculator(
            rotlib_path="rotamer_library/simple_library.parquet"
        )
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
            rotlib_path: Path to Dunbrack rotamer library
            compute_sasa: Whether to compute SASA entropy
            compute_anm: Whether to compute ANM entropy
            compute_rotamer: Whether to compute rotamer entropy
        """
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

        if compute_rotamer and rotlib_path:
            self.rotamer_calc = RotamerEntropyCalculator(
                rotlib_path=rotlib_path,
                temperature=temperature,
            )
        else:
            self.rotamer_calc = None

    def calculate(
        self,
        pdb_path: str | Path,
        chain_a: str,
        chain_b: str,
        frame: int = 0,
        parallel: bool = False,  # Sequential is faster due to ANM's internal parallelism
    ) -> BindingEntropyResult:
        """
        Calculate complete binding entropy.

        Args:
            pdb_path: Path to PDB file of the complex
            chain_a: First chain ID
            chain_b: Second chain ID
            frame: Frame to use (for trajectories)
            parallel: Run entropy calculations in parallel (default True)

        Returns:
            BindingEntropyResult with all entropy contributions
        """
        pdb_path = Path(pdb_path)
        logger.info(f"Calculating binding entropy for {pdb_path}")
        logger.info(f"Chains: {chain_a} + {chain_b}, T = {self.temperature} K")

        # Load structure once (shared by all calculations)
        traj = StructureLoader.load(pdb_path)

        if parallel:
            # Run all entropy calculations in parallel
            return self._calculate_parallel(traj, chain_a, chain_b, frame)
        else:
            # Sequential calculation
            return self._calculate_sequential(traj, chain_a, chain_b, frame)

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
        chain_a: str,
        chain_b: str,
        frames: list[int] | None = None,
    ) -> list[BindingEntropyResult]:
        """
        Calculate binding entropy for multiple frames of a trajectory.

        Args:
            traj_path: Path to trajectory file (DCD, XTC, etc.)
            topology_path: Path to topology file (PDB)
            chain_a: First chain ID
            chain_b: Second chain ID
            frames: List of frame indices to analyze (default: all frames)

        Returns:
            List of BindingEntropyResult for each frame
        """
        logger.info(f"Loading trajectory from {traj_path}")
        traj = StructureLoader.load(traj_path, topology=topology_path)

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
