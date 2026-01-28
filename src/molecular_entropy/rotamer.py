"""
Rotamer entropy calculations using Dunbrack backbone-dependent library.

Calculates side-chain conformational entropy based on rotamer probabilities.
Interface effects are estimated by checking for steric clashes with the
partner chain.

Performance optimizations:
- Binary cache for rotamer library index (10-20x faster loading)
- Rust batch processing for clash detection and entropy calculation
- Backbone angle caching for repeated structures
"""

__all__ = [
    "RotamerEntropyCalculator",
    "RotamerBindingResult",
    "ResidueRotamerResult",
    "SingleChainRotamerResult",
    "SingleResidueRotamerResult",
    "clear_backbone_cache",
]

import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mdtraj as md
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from .constants import (
    CHI_RESIDUES,
    DEFAULT_TEMPERATURE,
    R_KCAL,
)
from .structure import StructureLoader

logger = logging.getLogger(__name__)

# Try to import Rust extension for performance
try:
    from molecular_entropy._core import (
        check_clashes_batch as _rust_check_clashes_batch,
    )
    from molecular_entropy._core import (
        compute_rotamer_entropies_batch as _rust_compute_entropies_batch,
    )

    _USE_RUST = True
    logger.debug("Using Rust extension for performance")
except ImportError:
    _USE_RUST = False
    logger.debug("Rust extension not available, using Python fallback")

# Module-level cache for backbone bins
_backbone_bins_cache: dict[tuple, dict[int, tuple[int, int]]] = {}


def _traj_hash_key(traj: md.Trajectory) -> tuple:
    """Generate a hash key for trajectory caching."""
    xyz_bytes = traj.xyz[0].tobytes()
    return (traj.n_atoms, traj.n_residues, hash(xyz_bytes))


def clear_backbone_cache() -> None:
    """Clear the backbone bins cache."""
    global _backbone_bins_cache
    _backbone_bins_cache = {}


@dataclass
class ResidueRotamerResult:
    """Rotamer entropy for a single residue."""

    chain_id: str
    resSeq: int
    resName: str
    S_unbound_kcal_per_K: float  # Entropy in unbound state
    S_bound_kcal_per_K: float  # Entropy in bound state
    dS_kcal_per_K: float  # Change in entropy (bound - unbound)
    negT_dS_kcal: float  # -T*dS in kcal/mol
    n_rotamers_unbound: int
    n_rotamers_bound: int


@dataclass
class RotamerBindingResult:
    """Results from binding rotamer entropy calculation."""

    total_dS_kcal_per_K: float  # Sum of dS for all residues
    total_negT_dS_kcal: float  # Sum of -T*dS in kcal/mol
    per_residue: list[ResidueRotamerResult]
    temperature: float


@dataclass
class SingleResidueRotamerResult:
    """Rotamer entropy for a single residue (no binding partner)."""

    chain_id: str
    resSeq: int
    resName: str
    S_kcal_per_K: float  # Entropy in kcal/mol/K
    negT_S_kcal: float  # -T*S in kcal/mol
    n_rotamers: int  # Number of available rotamers


@dataclass
class SingleChainRotamerResult:
    """Rotamer entropy for a single chain (no binding partner)."""

    total_S_kcal_per_K: float  # Sum of entropy for all residues
    negT_S_kcal: float  # -T*S in kcal/mol (useful for comparisons)
    per_residue: list[SingleResidueRotamerResult]
    temperature: float
    chain_id: str | None = None


class RotamerEntropyCalculator:
    """
    Calculate side-chain rotamer entropy from Dunbrack library.

    Uses backbone-dependent rotamer probabilities to estimate the
    conformational entropy of side chains. Binding effects are estimated
    by eliminating rotamers that clash with the partner chain.

    The entropy is computed as:
        S = -R * sum(p * ln(p))  [kcal/mol/K]

    where p are the rotamer probabilities (normalized to sum to 1).
    """

    def __init__(
        self,
        rotlib_path: str | Path | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        clash_distance: float = 2.5,  # Angstrom
        use_cache: bool = True,
    ):
        """
        Initialize rotamer entropy calculator.

        Args:
            rotlib_path: Path to Dunbrack rotamer library (parquet or JSON)
            temperature: Temperature in Kelvin
            clash_distance: Distance threshold for steric clashes (Angstrom)
            use_cache: Use binary cache for faster library loading (default True)
        """
        self.temperature = temperature
        self.clash_distance = clash_distance
        self.rotlib = None
        self.rotlib_index = None
        self.use_cache = use_cache

        if rotlib_path:
            self.load_rotamer_library(rotlib_path, use_cache=use_cache)

    def load_rotamer_library(self, path: str | Path, use_cache: bool = True) -> None:
        """
        Load Dunbrack backbone-dependent rotamer library.

        Supports parquet format (from make_parquet.py) or JSON format.
        Uses binary cache for faster subsequent loads.

        Args:
            path: Path to rotamer library file
            use_cache: Whether to use/create binary cache (default True)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Rotamer library not found: {path}")

        # Try to load from binary cache first
        if use_cache and path.suffix == ".parquet":
            cache_path = self._get_cache_path(path)
            if self._try_load_from_cache(path, cache_path):
                logger.info(f"Loaded rotamer library from cache ({len(self.rotlib_index)} entries)")
                return

        # Load from source file
        if path.suffix == ".parquet":
            self.rotlib = pl.read_parquet(path)
            self._build_index_from_parquet()

            # Save to cache for next time
            if use_cache:
                self._save_to_cache(path, cache_path)

        elif path.suffix == ".json":
            import json

            with open(path) as f:
                data = json.load(f)
            self._build_index_from_json(data)
        else:
            raise ValueError(f"Unsupported rotamer library format: {path.suffix}")

        logger.info(f"Loaded rotamer library with {len(self.rotlib_index)} entries")

    def _get_cache_path(self, source_path: Path) -> Path:
        """Get path to binary cache file for a rotamer library."""
        return source_path.with_suffix(".index.pkl")

    def _try_load_from_cache(self, source_path: Path, cache_path: Path) -> bool:
        """
        Try to load rotamer index from binary cache.

        Returns True if cache was loaded successfully, False otherwise.
        Cache is invalidated if source file is newer than cache.
        """
        if not cache_path.exists():
            return False

        # Check if cache is stale (source file modified after cache)
        if source_path.stat().st_mtime > cache_path.stat().st_mtime:
            logger.debug("Cache is stale, rebuilding from source")
            return False

        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # Validate cache version/format
            if not isinstance(cache_data, dict) or "version" not in cache_data:
                logger.debug("Cache format invalid, rebuilding")
                return False

            if cache_data["version"] != 1:
                logger.debug("Cache version mismatch, rebuilding")
                return False

            self.rotlib_index = cache_data["index"]
            self.rotlib = None  # Don't need raw dataframe when using cache
            return True

        except (pickle.UnpicklingError, KeyError, EOFError) as e:
            logger.debug(f"Cache load failed: {e}, rebuilding")
            return False

    def _save_to_cache(self, source_path: Path, cache_path: Path) -> None:
        """Save rotamer index to binary cache."""
        try:
            cache_data = {
                "version": 1,
                "source": str(source_path),
                "index": self.rotlib_index,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Saved rotamer index cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _build_index_from_parquet(self) -> None:
        """Build lookup index from parquet dataframe using vectorized operations."""
        # Cast and filter in polars (vectorized)
        df = self.rotlib.with_columns(
            [
                pl.col("prob").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("chi1").cast(pl.Float64, strict=False),
                pl.col("chi2").cast(pl.Float64, strict=False),
                pl.col("chi3").cast(pl.Float64, strict=False),
                pl.col("chi4").cast(pl.Float64, strict=False),
            ]
        ).filter(pl.col("prob") > 0)

        # Single-pass grouping using to_dicts()
        rows = df.to_dicts()
        groups = defaultdict(list)
        for row in rows:
            key = (row["resname"].upper(), int(row["phi"]), int(row["psi"]))
            chi_vals = [
                row[col] for col in ["chi1", "chi2", "chi3", "chi4"] if row[col] is not None
            ]
            groups[key].append({"chi": chi_vals, "p": row["prob"]})

        self.rotlib_index = dict(groups)

    def _build_index_from_json(self, data: list) -> None:
        """Build lookup index from JSON data."""
        self.rotlib_index = {}
        for row in data:
            key = (row["resname"].upper(), int(row["phi_bin"]), int(row["psi_bin"]))
            self.rotlib_index[key] = row["rotamers"]

    @staticmethod
    def angle_to_bin(angle_deg: float, bin_size: int = 60) -> int:
        """
        Assign dihedral angle to nearest bin center using O(1) arithmetic.

        Args:
            angle_deg: Angle in degrees
            bin_size: Bin size in degrees (default 60 for Dunbrack)

        Returns:
            Bin center in degrees
        """
        # Normalize to [-180, 180)
        angle_normalized = ((angle_deg + 180) % 360) - 180
        # Round to nearest bin center
        half_bin = bin_size / 2
        bin_center = int((angle_normalized + half_bin) // bin_size) * bin_size
        # Handle edge: 180 -> -180
        if bin_center >= 180:
            bin_center = -180
        return bin_center

    def compute_backbone_bins(
        self, traj: md.Trajectory, use_cache: bool = True
    ) -> dict[int, tuple[int, int]]:
        """
        Compute backbone phi/psi bins for all residues.

        Args:
            traj: MDTraj trajectory
            use_cache: Whether to use caching for repeated calls

        Returns:
            Dict mapping residue index to (phi_bin, psi_bin)
        """
        global _backbone_bins_cache

        if use_cache:
            cache_key = _traj_hash_key(traj)
            if cache_key in _backbone_bins_cache:
                return _backbone_bins_cache[cache_key]

        # Delegate to StructureLoader (eliminates code duplication)
        phi_map, psi_map = StructureLoader.compute_backbone_dihedrals(traj)

        # Bin the angles
        bins = {}
        for res in traj.topology.residues:
            phi = phi_map.get(res.index)
            psi = psi_map.get(res.index)
            if phi is not None and psi is not None:
                bins[res.index] = (self.angle_to_bin(phi), self.angle_to_bin(psi))
            else:
                bins[res.index] = (None, None)

        if use_cache:
            _backbone_bins_cache[cache_key] = bins

        return bins

    @staticmethod
    def compute_entropy_from_probs(probs: np.ndarray) -> float:
        """
        Compute Shannon entropy from probabilities.

        S = -R * sum(p * ln(p))  [kcal/mol/K]

        Note: This is the CORRECT formula for entropy.
        The original code had -R*T*sum(p*ln(p)) which gives T*S, not S.

        Args:
            probs: Array of probabilities (will be normalized)

        Returns:
            Entropy in kcal/mol/K
        """
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 1e-12, None)  # Avoid log(0)
        probs = probs / probs.sum()  # Normalize

        # Shannon entropy: S = -R * sum(p * ln(p))
        S = -R_KCAL * float(np.sum(probs * np.log(probs)))
        return S

    def _build_partner_kdtree(self, partner_traj: md.Trajectory) -> cKDTree:
        """
        Build a KD-tree from partner chain heavy atoms.

        Args:
            partner_traj: Partner chain trajectory

        Returns:
            cKDTree for spatial queries
        """
        partner_heavy = [
            a.index
            for a in partner_traj.topology.atoms
            if a.element is not None and a.element.symbol != "H"
        ]
        return cKDTree(partner_traj.xyz[0][partner_heavy])

    def check_clashes(
        self,
        traj: md.Trajectory,
        residue,
        partner_traj: md.Trajectory,
        _partner_kdtree: cKDTree | None = None,
    ) -> bool:
        """
        Check if residue side chain clashes with partner chain using KD-tree.

        Args:
            traj: Complex trajectory
            residue: Residue object to check
            partner_traj: Partner chain trajectory
            _partner_kdtree: Pre-built KD-tree for partner (optional, for performance)

        Returns:
            True if there are close contacts (clashes likely for some rotamers)
        """
        # Get side-chain heavy atom indices for this residue
        backbone_names = {"N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3"}
        sc_atoms = [a.index for a in residue.atoms if a.name not in backbone_names]

        if not sc_atoms:
            return False

        # Build KD-tree if not provided
        if _partner_kdtree is None:
            _partner_kdtree = self._build_partner_kdtree(partner_traj)

        # Check distances using KD-tree (mdtraj uses nm)
        threshold_nm = self.clash_distance / 10.0
        sc_coords = traj.xyz[0][sc_atoms]
        neighbors = _partner_kdtree.query_ball_point(sc_coords, r=threshold_nm)
        return any(len(n) > 0 for n in neighbors)

    def heuristic_feasible_mask(
        self,
        has_clash: bool,
        rotamers: list[dict],
        n_keep: int = 2,
    ) -> np.ndarray:
        """
        Return mask of feasible rotamers based on clash detection.

        If there are clashes, conservatively keep only the top N rotamers
        by Dunbrack probability, assuming some will be eliminated.

        Args:
            has_clash: Whether there are potential clashes
            rotamers: List of rotamer dicts with 'p' key
            n_keep: Number of top rotamers to keep when clash detected

        Returns:
            Boolean mask array
        """
        n = len(rotamers)
        if not has_clash:
            return np.ones(n, dtype=bool)

        # Sort by probability, keep top n_keep
        probs = [r["p"] for r in rotamers]
        order = np.argsort(probs)[::-1]

        mask = np.zeros(n, dtype=bool)
        mask[order[: min(n_keep, n)]] = True
        return mask

    def calculate_residue_entropy(
        self,
        traj: md.Trajectory,
        residue,
        partner_traj: md.Trajectory | None,
        backbone_bins: dict[int, tuple[int, int]],
        _partner_kdtree: cKDTree | None = None,
    ) -> ResidueRotamerResult | None:
        """
        Calculate rotamer entropy for a single residue.

        Args:
            traj: Complex trajectory
            residue: Residue object
            partner_traj: Partner chain trajectory (for clash detection)
            backbone_bins: Dict mapping residue index to (phi_bin, psi_bin)
            _partner_kdtree: Pre-built KD-tree for partner (optional, for performance)

        Returns:
            ResidueRotamerResult or None if residue has no rotamers
        """
        if self.rotlib_index is None:
            raise RuntimeError("Rotamer library not loaded")

        if residue.name not in CHI_RESIDUES:
            return None

        phi_bin, psi_bin = backbone_bins.get(residue.index, (None, None))
        if phi_bin is None or psi_bin is None:
            return None

        key = (residue.name, phi_bin, psi_bin)
        rotamers = self.rotlib_index.get(key)
        if not rotamers:
            return None

        # Unbound entropy (all rotamers available)
        p_unbound = np.array([r["p"] for r in rotamers])
        S_unbound = self.compute_entropy_from_probs(p_unbound)

        # Bound entropy (some rotamers may be excluded due to clashes)
        if partner_traj is not None:
            has_clash = self.check_clashes(traj, residue, partner_traj, _partner_kdtree)
            feasible = self.heuristic_feasible_mask(has_clash, rotamers)
            p_bound = p_unbound * feasible.astype(float)

            # If all rotamers filtered, keep the top one
            if p_bound.sum() == 0:
                top_idx = np.argmax(p_unbound)
                p_bound[top_idx] = 1.0
        else:
            p_bound = p_unbound.copy()

        S_bound = self.compute_entropy_from_probs(p_bound)

        # Change in entropy: dS = S_bound - S_unbound
        # Typically negative (entropy loss upon binding)
        dS = S_bound - S_unbound
        negT_dS = -self.temperature * dS

        return ResidueRotamerResult(
            chain_id=residue.chain.chain_id,
            resSeq=residue.resSeq,
            resName=residue.name,
            S_unbound_kcal_per_K=S_unbound,
            S_bound_kcal_per_K=S_bound,
            dS_kcal_per_K=dS,
            negT_dS_kcal=negT_dS,
            n_rotamers_unbound=len(p_unbound),
            n_rotamers_bound=int(np.sum(p_bound > 0)),
        )

    def _calculate_binding_delta_rust(
        self,
        traj: md.Trajectory,
        chain_a: str,
        chain_b: str,
    ) -> RotamerBindingResult:
        """
        Calculate binding entropy using Rust batch functions for performance.

        Uses rayon-parallelized Rust functions for clash detection and entropy
        computation. Falls back to Python if Rust extension is unavailable.
        """
        # Get partner trajectories and their heavy atom coordinates
        traj_a = StructureLoader.select_chain(traj, chain_a)
        traj_b = StructureLoader.select_chain(traj, chain_b)

        # Get heavy atom coordinates for each partner chain
        heavy_a = [
            a.index
            for a in traj_a.topology.atoms
            if a.element is not None and a.element.symbol != "H"
        ]
        heavy_b = [
            a.index
            for a in traj_b.topology.atoms
            if a.element is not None and a.element.symbol != "H"
        ]
        partner_coords_a = np.ascontiguousarray(traj_a.xyz[0][heavy_a], dtype=np.float64)
        partner_coords_b = np.ascontiguousarray(traj_b.xyz[0][heavy_b], dtype=np.float64)

        # Compute backbone bins
        backbone_bins = self.compute_backbone_bins(traj)

        # Collect residue data for batch processing
        backbone_names = {"N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3"}
        residue_data_a = []  # Residues in chain A (partner is B)
        residue_data_b = []  # Residues in chain B (partner is A)

        for res in traj.topology.residues:
            if res.name not in CHI_RESIDUES:
                continue
            if res.chain.chain_id not in {chain_a, chain_b}:
                continue

            phi_bin, psi_bin = backbone_bins.get(res.index, (None, None))
            if phi_bin is None or psi_bin is None:
                continue

            key = (res.name, phi_bin, psi_bin)
            rotamers = self.rotlib_index.get(key)
            if not rotamers:
                continue

            # Get side-chain atom indices
            sc_atoms = [a.index for a in res.atoms if a.name not in backbone_names]
            if not sc_atoms:
                continue

            data = {
                "res": res,
                "sc_atom_range": (min(sc_atoms), max(sc_atoms) + 1),
                "rotamers": rotamers,
                "probs": np.array([r["p"] for r in rotamers], dtype=np.float64),
            }

            if res.chain.chain_id == chain_a:
                residue_data_a.append(data)
            else:
                residue_data_b.append(data)

        # Get complex coordinates (ensure float64 and contiguous for Rust)
        complex_coords = np.ascontiguousarray(traj.xyz[0], dtype=np.float64)
        threshold_nm = self.clash_distance / 10.0

        # Batch clash detection using Rust
        results = []

        for residue_data, partner_coords in [
            (residue_data_a, partner_coords_b),
            (residue_data_b, partner_coords_a),
        ]:
            if not residue_data:
                continue

            atom_ranges = [d["sc_atom_range"] for d in residue_data]

            # Use Rust batch clash detection
            clash_results = _rust_check_clashes_batch(
                complex_coords, partner_coords, atom_ranges, threshold_nm
            )

            # Build feasible masks and probability arrays
            prob_arrays = []
            feasible_masks = []
            for data, has_clash in zip(residue_data, clash_results):
                probs = data["probs"]
                mask = self.heuristic_feasible_mask(has_clash, data["rotamers"])
                prob_arrays.append(probs)
                feasible_masks.append(mask)

            # Use Rust batch entropy computation
            s_unbound_list, s_bound_list = _rust_compute_entropies_batch(
                prob_arrays, feasible_masks, R_KCAL
            )

            # Build results
            for data, s_unbound, s_bound in zip(residue_data, s_unbound_list, s_bound_list):
                res = data["res"]
                probs = data["probs"]
                dS = s_bound - s_unbound
                negT_dS = -self.temperature * dS

                results.append(
                    ResidueRotamerResult(
                        chain_id=res.chain.chain_id,
                        resSeq=res.resSeq,
                        resName=res.name,
                        S_unbound_kcal_per_K=s_unbound,
                        S_bound_kcal_per_K=s_bound,
                        dS_kcal_per_K=dS,
                        negT_dS_kcal=negT_dS,
                        n_rotamers_unbound=len(probs),
                        n_rotamers_bound=int(np.sum(data["probs"] > 0)),
                    )
                )

        # Sum up contributions
        total_dS = sum(r.dS_kcal_per_K for r in results)
        total_negT_dS = sum(r.negT_dS_kcal for r in results)

        logger.info(
            f"Rotamer entropy (Rust): {len(results)} residues, "
            f"total dS = {total_dS:.4f} kcal/mol/K, "
            f"-T*dS = {total_negT_dS:.3f} kcal/mol"
        )

        return RotamerBindingResult(
            total_dS_kcal_per_K=total_dS,
            total_negT_dS_kcal=total_negT_dS,
            per_residue=results,
            temperature=self.temperature,
        )

    def _calculate_binding_delta_python(
        self,
        traj: md.Trajectory,
        chain_a: str,
        chain_b: str,
    ) -> RotamerBindingResult:
        """
        Calculate binding entropy using pure Python (fallback).
        """
        # Get partner trajectories
        traj_a = StructureLoader.select_chain(traj, chain_a)
        traj_b = StructureLoader.select_chain(traj, chain_b)

        # Build KD-trees once per partner chain for O(N*log(M)) clash detection
        kdtree_a = self._build_partner_kdtree(traj_a)
        kdtree_b = self._build_partner_kdtree(traj_b)

        # Compute backbone bins for the complex
        backbone_bins = self.compute_backbone_bins(traj)

        results = []

        for res in traj.topology.residues:
            if res.name not in CHI_RESIDUES:
                continue

            if res.chain.chain_id not in {chain_a, chain_b}:
                continue

            # Partner is the opposite chain
            if res.chain.chain_id == chain_a:
                partner = traj_b
                kdtree = kdtree_b
            else:
                partner = traj_a
                kdtree = kdtree_a

            result = self.calculate_residue_entropy(
                traj, res, partner, backbone_bins, _partner_kdtree=kdtree
            )
            if result:
                results.append(result)

        # Sum up contributions
        total_dS = sum(r.dS_kcal_per_K for r in results)
        total_negT_dS = sum(r.negT_dS_kcal for r in results)

        logger.info(
            f"Rotamer entropy (Python): {len(results)} residues, "
            f"total dS = {total_dS:.4f} kcal/mol/K, "
            f"-T*dS = {total_negT_dS:.3f} kcal/mol"
        )

        return RotamerBindingResult(
            total_dS_kcal_per_K=total_dS,
            total_negT_dS_kcal=total_negT_dS,
            per_residue=results,
            temperature=self.temperature,
        )

    def calculate_binding_delta(
        self,
        traj: md.Trajectory,
        chain_a: str,
        chain_b: str,
    ) -> RotamerBindingResult:
        """
        Calculate side-chain entropy change upon binding.

        Uses Rust extension when available for better performance via
        rayon parallelization. Falls back to pure Python otherwise.

        Args:
            traj: MDTraj trajectory of the complex
            chain_a: First chain ID
            chain_b: Second chain ID

        Returns:
            RotamerBindingResult with per-residue and total entropy changes
        """
        if self.rotlib_index is None:
            raise RuntimeError("Rotamer library not loaded")

        if _USE_RUST:
            return self._calculate_binding_delta_rust(traj, chain_a, chain_b)
        else:
            return self._calculate_binding_delta_python(traj, chain_a, chain_b)

    def calculate(
        self,
        traj: md.Trajectory,
        chain_id: str | None = None,
    ) -> SingleChainRotamerResult:
        """
        Calculate rotamer entropy for a single chain (no binding partner).

        All rotamers are available since there's no partner to clash with.
        This gives the maximum conformational entropy for the side chains.

        Args:
            traj: MDTraj trajectory
            chain_id: Optional chain ID to filter (if None, uses all residues)

        Returns:
            SingleChainRotamerResult with per-residue and total entropy
        """
        if self.rotlib_index is None:
            raise RuntimeError("Rotamer library not loaded")

        # Compute backbone bins
        backbone_bins = self.compute_backbone_bins(traj)

        results = []

        for res in traj.topology.residues:
            if res.name not in CHI_RESIDUES:
                continue

            # Filter by chain if specified
            if chain_id is not None and res.chain.chain_id != chain_id:
                continue

            phi_bin, psi_bin = backbone_bins.get(res.index, (None, None))
            if phi_bin is None or psi_bin is None:
                continue

            key = (res.name, phi_bin, psi_bin)
            rotamers = self.rotlib_index.get(key)
            if not rotamers:
                continue

            # Calculate entropy with all rotamers available
            probs = np.array([r["p"] for r in rotamers])
            S = self.compute_entropy_from_probs(probs)
            negT_S = -self.temperature * S

            results.append(
                SingleResidueRotamerResult(
                    chain_id=res.chain.chain_id,
                    resSeq=res.resSeq,
                    resName=res.name,
                    S_kcal_per_K=S,
                    negT_S_kcal=negT_S,
                    n_rotamers=len(probs),
                )
            )

        # Sum up contributions
        total_S = sum(r.S_kcal_per_K for r in results)
        total_negT_S = -self.temperature * total_S

        logger.info(
            f"Single-chain rotamer entropy: {len(results)} residues, "
            f"total S = {total_S:.4f} kcal/mol/K, "
            f"-T*S = {total_negT_S:.3f} kcal/mol"
        )

        return SingleChainRotamerResult(
            total_S_kcal_per_K=total_S,
            negT_S_kcal=total_negT_S,
            per_residue=results,
            temperature=self.temperature,
            chain_id=chain_id,
        )

    def to_dataframe(self, result: RotamerBindingResult) -> pl.DataFrame:
        """
        Convert results to a polars DataFrame.

        Args:
            result: RotamerBindingResult

        Returns:
            DataFrame with per-residue entropy data
        """
        rows = []
        for r in result.per_residue:
            rows.append(
                {
                    "chain_id": r.chain_id,
                    "resSeq": r.resSeq,
                    "resName": r.resName,
                    "S_unbound_kcal_per_K": r.S_unbound_kcal_per_K,
                    "S_bound_kcal_per_K": r.S_bound_kcal_per_K,
                    "dS_kcal_per_K": r.dS_kcal_per_K,
                    "negT_dS_kcal": r.negT_dS_kcal,
                    "n_rotamers_unbound": r.n_rotamers_unbound,
                    "n_rotamers_bound": r.n_rotamers_bound,
                }
            )

        df = pl.DataFrame(rows)
        if len(df) > 0:
            df = df.sort(["chain_id", "resSeq"])
        return df

    def single_chain_to_dataframe(self, result: SingleChainRotamerResult) -> pl.DataFrame:
        """
        Convert single-chain results to a polars DataFrame.

        Args:
            result: SingleChainRotamerResult

        Returns:
            DataFrame with per-residue entropy data
        """
        rows = []
        for r in result.per_residue:
            rows.append(
                {
                    "chain_id": r.chain_id,
                    "resSeq": r.resSeq,
                    "resName": r.resName,
                    "S_kcal_per_K": r.S_kcal_per_K,
                    "negT_S_kcal": r.negT_S_kcal,
                    "n_rotamers": r.n_rotamers,
                }
            )

        df = pl.DataFrame(rows)
        if len(df) > 0:
            df = df.sort(["chain_id", "resSeq"])
        return df
