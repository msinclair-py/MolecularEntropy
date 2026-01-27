"""
ANM (Anisotropic Network Model) vibrational entropy calculations.

Uses sparse eigensolvers for efficient computation of vibrational entropy
changes upon binding. The entropy is calculated from the normal mode eigenvalues.

Performance optimizations:
- Sparse Hessian construction (O(N) memory vs O(N²) for dense)
- Iterative eigensolver (only computes needed eigenvalues)
- Parallel computation of independent ANM calculations
"""

__all__ = [
    "ANMEntropyCalculator",
    "ANMEntropyResult",
    "ANMBindingResult",
    "build_sparse_hessian",
    "compute_anm_eigenvalues_sparse",
]

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
import mdtraj as md

from .constants import (
    KB_KCAL,
    DEFAULT_ANM_CUTOFF,
    DEFAULT_ANM_GAMMA,
    DEFAULT_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Try to import Rust backend
try:
    from molecular_entropy._core import (
        build_anm_hessian_coo as rust_build_hessian_coo,
    )
    _RUST_ANM_AVAILABLE = True
except ImportError:
    _RUST_ANM_AVAILABLE = False
    logger.debug("Rust ANM backend not available, using Python")


def build_sparse_hessian(
    coords: np.ndarray,
    cutoff: float,
    gamma: float = 1.0,
) -> csr_matrix:
    """
    Build sparse ANM Hessian matrix using fully vectorized operations.

    The Hessian H is a 3N x 3N matrix where N is the number of atoms.
    For atoms i and j within cutoff distance:
        H[3i:3i+3, 3j:3j+3] = -gamma * (r_ij ⊗ r_ij) / |r_ij|²

    Diagonal blocks are computed to satisfy sum rule: H[ii] = -sum(H[ij], j≠i)

    Args:
        coords: Nx3 array of atom coordinates (Angstrom)
        cutoff: Distance cutoff for springs (Angstrom)
        gamma: Spring constant

    Returns:
        Sparse CSR Hessian matrix (3N x 3N)
    """
    n_atoms = len(coords)
    n_dof = 3 * n_atoms

    # Build KD-tree for efficient neighbor finding
    tree = cKDTree(coords)

    # Find all pairs within cutoff
    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    n_pairs = len(pairs)

    if n_pairs == 0:
        return csr_matrix((n_dof, n_dof), dtype=np.float64)

    # Vectorized computation of all pair interactions
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    # Distance vectors for all pairs
    r_ij = coords[j_idx] - coords[i_idx]  # Shape: (n_pairs, 3)
    dist_sq = np.sum(r_ij * r_ij, axis=1)  # Shape: (n_pairs,)

    # Filter out zero distances
    valid = dist_sq > 1e-10
    i_idx, j_idx = i_idx[valid], j_idx[valid]
    r_ij, dist_sq = r_ij[valid], dist_sq[valid]
    n_valid = len(i_idx)

    if n_valid == 0:
        return csr_matrix((n_dof, n_dof), dtype=np.float64)

    # Compute outer products: -gamma * (r_ij ⊗ r_ij) / |r_ij|²
    # Shape: (n_valid, 3, 3)
    blocks = -gamma * np.einsum('ij,ik->ijk', r_ij, r_ij) / dist_sq[:, None, None]
    blocks_flat = blocks.reshape(n_valid, 9)  # Shape: (n_valid, 9)

    # Build COO format data - fully vectorized
    # Local indices within 3x3 block
    local_i = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    local_j = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)

    # Off-diagonal block (i, j) indices
    row_ij = (3 * i_idx[:, None] + local_i[None, :]).ravel()  # Shape: (n_valid * 9,)
    col_ij = (3 * j_idx[:, None] + local_j[None, :]).ravel()

    # Off-diagonal block (j, i) indices - symmetric
    row_ji = (3 * j_idx[:, None] + local_i[None, :]).ravel()
    col_ji = (3 * i_idx[:, None] + local_j[None, :]).ravel()

    # Combine all off-diagonal entries
    row_offdiag = np.concatenate([row_ij, row_ji])
    col_offdiag = np.concatenate([col_ij, col_ji])
    data_offdiag = np.concatenate([blocks_flat.ravel(), blocks_flat.ravel()])

    # Create sparse matrix from off-diagonal entries
    hessian = csr_matrix(
        (data_offdiag, (row_offdiag, col_offdiag)),
        shape=(n_dof, n_dof),
        dtype=np.float64
    )

    # Diagonal blocks: sum of negative off-diagonal blocks for each atom
    # H[ii] = -sum(H[ij], j≠i)
    # Use np.add.at for efficient accumulation
    diag_blocks = np.zeros((n_atoms, 3, 3), dtype=np.float64)
    np.add.at(diag_blocks, i_idx, -blocks)
    np.add.at(diag_blocks, j_idx, -blocks)

    # Build diagonal entries - vectorized
    atom_indices = np.arange(n_atoms, dtype=np.int64)
    diag_row = (3 * atom_indices[:, None, None] + local_i.reshape(3, 3)[None, :, :]).ravel()
    diag_col = (3 * atom_indices[:, None, None] + local_j.reshape(3, 3)[None, :, :]).ravel()
    diag_data = diag_blocks.ravel()

    diag_matrix = csr_matrix(
        (diag_data, (diag_row, diag_col)),
        shape=(n_dof, n_dof),
        dtype=np.float64
    )

    return hessian + diag_matrix


def compute_anm_eigenvalues_sparse(
    coords: np.ndarray,
    cutoff: float,
    gamma: float = 1.0,
    n_modes: int = 20,
) -> np.ndarray:
    """
    Compute lowest ANM eigenvalues using sparse iterative solver.

    Uses ARPACK (via scipy.sparse.linalg.eigsh) with shift-invert mode
    to efficiently find the smallest non-zero eigenvalues.

    Args:
        coords: Nx3 array of atom coordinates (Angstrom)
        cutoff: Distance cutoff for ANM springs (Angstrom)
        gamma: Spring constant
        n_modes: Number of lowest modes to compute (excluding 6 trivial)

    Returns:
        Array of non-trivial eigenvalues (smallest n_modes)
    """
    n_atoms = len(coords)

    if n_atoms < 4:
        return np.array([])

    # Build sparse Hessian
    hessian = build_sparse_hessian(coords, cutoff, gamma)

    # Number of eigenvalues to request (6 trivial + n_modes non-trivial)
    # Request a few extra to ensure we get enough valid ones
    k = min(6 + n_modes + 6, 3 * n_atoms - 1)

    try:
        # Use shift-invert mode for much faster convergence on small eigenvalues
        # sigma should be slightly larger than the trivial modes (which are ~0)
        # but smaller than the non-trivial modes we want
        # Typical first non-trivial eigenvalue is O(0.01-0.1) for proteins
        eigenvalues, _ = eigsh(
            hessian,
            k=k,
            sigma=1e-4,  # Shift-invert: find eigenvalues closest to this value
            which='LM',  # Largest magnitude of (H - sigma*I)^-1 = closest to sigma
            tol=1e-5,
            maxiter=500,
        )
    except Exception as e:
        logger.warning(f"Sparse eigensolver failed: {e}, falling back to dense")
        # Fallback to dense solver
        dense_hessian = hessian.toarray()
        eigenvalues = np.linalg.eigvalsh(dense_hessian)

    # Sort eigenvalues and filter out the 6 trivial modes (near-zero)
    eigenvalues = np.sort(np.abs(eigenvalues))  # abs because shift-invert can give small negatives
    non_trivial = eigenvalues[eigenvalues > 1e-6]

    return non_trivial[:n_modes]


@dataclass
class ANMEntropyResult:
    """Results from ANM entropy calculation."""
    S_kB: float  # Entropy in k_B units
    S_kcal_per_K: float  # Entropy in kcal/mol/K
    n_modes: int  # Number of non-trivial modes used


@dataclass
class ANMBindingResult:
    """Results from binding ANM entropy calculation."""
    dS_kB: float  # Delta entropy in k_B units
    dS_kcal_per_K: float  # Delta entropy in kcal/mol/K
    negT_dS_kcal: float  # -T*dS in kcal/mol
    complex_result: ANMEntropyResult
    chain_a_result: ANMEntropyResult
    chain_b_result: ANMEntropyResult


class ANMEntropyCalculator:
    """
    Calculate vibrational entropy from ANM (Anisotropic Network Model).

    The ANM uses a coarse-grained elastic network on C-alpha atoms.
    Entropy is estimated from the eigenvalues using:
        S* = sum[-0.5 * ln(eigenvalue)]

    Note: This gives a relative measure. Constants that are the same for
    all systems cancel when computing binding entropy changes (delta S).

    Performance features:
    - Sparse Hessian construction (scales O(N) vs O(N²))
    - Iterative eigensolver (computes only needed modes)
    - Parallel computation for binding calculations
    """

    def __init__(
        self,
        cutoff: float = DEFAULT_ANM_CUTOFF,
        gamma: float = DEFAULT_ANM_GAMMA,
        temperature: float = DEFAULT_TEMPERATURE,
        n_modes: int = 20,
        parallel: bool = True,
        use_rust: bool = False,
    ):
        """
        Initialize ANM entropy calculator.

        Args:
            cutoff: Distance cutoff for ANM springs in Angstrom
            gamma: Spring constant (arbitrary units, cancels in delta)
            temperature: Temperature in Kelvin
            n_modes: Number of modes to compute (default 20)
            parallel: Run independent ANM calculations in parallel
            use_rust: Use Rust Hessian builder (experimental, not faster for typical sizes)
        """
        self.cutoff = cutoff
        self.gamma = gamma
        self.temperature = temperature
        self.n_modes = n_modes
        self.parallel = parallel
        # Rust ANM disabled by default - Python scipy is faster due to
        # optimized vectorized Hessian + shift-invert eigensolver
        self.use_rust = use_rust and _RUST_ANM_AVAILABLE

    def _compute_entropy_rust(
        self,
        coords: np.ndarray,
    ) -> ANMEntropyResult:
        """
        Compute vibrational entropy using Rust Hessian + scipy eigsh.

        This hybrid approach uses:
        - Rust for fast Hessian construction (KD-tree based)
        - scipy shift-invert eigensolver for accurate eigenvalues

        Args:
            coords: Nx3 array of C-alpha coordinates (Angstrom)

        Returns:
            ANMEntropyResult with entropy values
        """
        if len(coords) < 4:
            logger.warning(f"Too few atoms ({len(coords)}) for ANM")
            return ANMEntropyResult(S_kB=0.0, S_kcal_per_K=0.0, n_modes=0)

        # Ensure contiguous float64 array
        coords_arr = np.ascontiguousarray(coords, dtype=np.float64)

        # Build Hessian in Rust (fast KD-tree based)
        rows, cols, data, n_dof = rust_build_hessian_coo(
            coords_arr,
            self.cutoff,
            self.gamma,
        )

        # Convert to scipy sparse matrix
        from scipy.sparse import coo_matrix
        hessian = coo_matrix(
            (data, (rows, cols)),
            shape=(n_dof, n_dof),
            dtype=np.float64
        ).tocsr()

        # Use scipy shift-invert eigensolver (accurate for smallest eigenvalues)
        k = min(6 + self.n_modes + 6, n_dof - 1)
        try:
            eigenvalues, _ = eigsh(
                hessian,
                k=k,
                sigma=1e-4,
                which='LM',
                tol=1e-5,
                maxiter=500,
            )
        except Exception as e:
            logger.warning(f"Sparse eigensolver failed: {e}, falling back to dense")
            eigenvalues = np.linalg.eigvalsh(hessian.toarray())

        # Sort and filter out trivial modes
        eigenvalues = np.sort(np.abs(eigenvalues))
        non_trivial = eigenvalues[eigenvalues > 1e-6]

        return self._entropy_from_eigenvalues(non_trivial[:self.n_modes])

    def _compute_entropy_sparse(
        self,
        coords: np.ndarray,
    ) -> ANMEntropyResult:
        """
        Compute vibrational entropy using sparse eigensolver.

        Args:
            coords: Nx3 array of C-alpha coordinates (Angstrom)

        Returns:
            ANMEntropyResult with entropy values
        """
        if len(coords) < 4:
            logger.warning(f"Too few atoms ({len(coords)}) for ANM")
            return ANMEntropyResult(S_kB=0.0, S_kcal_per_K=0.0, n_modes=0)

        # Use Rust backend if available
        if self.use_rust:
            return self._compute_entropy_rust(coords)

        eigenvalues = compute_anm_eigenvalues_sparse(
            coords,
            self.cutoff,
            self.gamma,
            self.n_modes,
        )

        return self._entropy_from_eigenvalues(eigenvalues)

    def _entropy_from_eigenvalues(
        self,
        eigenvalues: np.ndarray
    ) -> ANMEntropyResult:
        """
        Calculate entropy from ANM eigenvalues.

        The formula S* = sum[-0.5 * ln(eigenvalue)] comes from the
        classical harmonic oscillator partition function.

        Args:
            eigenvalues: ANM eigenvalues (frequencies squared)

        Returns:
            ANMEntropyResult with S_kB, S_kcal_per_K, n_modes
        """
        # Filter out numerical noise (near-zero eigenvalues)
        evals = eigenvalues[eigenvalues > 1e-12]

        if len(evals) == 0:
            logger.warning("No valid eigenvalues found")
            return ANMEntropyResult(S_kB=0.0, S_kcal_per_K=0.0, n_modes=0)

        # Relative entropy in k_B units
        S_kB = float(np.sum(-0.5 * np.log(evals)))
        S_kcal_per_K = S_kB * KB_KCAL

        return ANMEntropyResult(
            S_kB=S_kB,
            S_kcal_per_K=S_kcal_per_K,
            n_modes=len(evals),
        )

    def _extract_ca_coords(
        self,
        traj: md.Trajectory,
        chain_id: Optional[str] = None,
        frame: int = 0,
    ) -> np.ndarray:
        """
        Extract C-alpha coordinates from MDTraj trajectory.

        Args:
            traj: MDTraj trajectory
            chain_id: Optional chain ID to filter
            frame: Frame index to extract coordinates from

        Returns:
            Nx3 array of C-alpha coordinates in Angstrom
        """
        ca_indices = []
        for atom in traj.topology.atoms:
            if atom.name == 'CA':
                if chain_id is None or atom.residue.chain.chain_id == chain_id:
                    ca_indices.append(atom.index)

        if not ca_indices:
            return np.array([])

        # mdtraj uses nm, ANM expects Angstrom
        coords = traj.xyz[frame, ca_indices] * 10.0
        return np.ascontiguousarray(coords, dtype=np.float64)

    def calculate(self, traj: md.Trajectory) -> ANMEntropyResult:
        """
        Calculate vibrational entropy for a structure.

        Args:
            traj: MDTraj trajectory (uses first frame)

        Returns:
            ANMEntropyResult with entropy values
        """
        coords = self._extract_ca_coords(traj)

        if len(coords) < 4:
            logger.warning(f"Too few CA atoms ({len(coords)}) for ANM")
            return ANMEntropyResult(S_kB=0.0, S_kcal_per_K=0.0, n_modes=0)

        result = self._compute_entropy_sparse(coords)
        logger.info(f"ANM: {result.n_modes} modes, S = {result.S_kB:.2f} k_B")

        return result

    def calculate_from_pdb(
        self,
        pdb_path: Union[str, Path],
        chain_id: Optional[str] = None
    ) -> ANMEntropyResult:
        """
        Calculate vibrational entropy directly from PDB file.

        Args:
            pdb_path: Path to PDB file
            chain_id: Optional chain to select

        Returns:
            ANMEntropyResult with entropy values
        """
        traj = md.load(str(pdb_path))

        if chain_id:
            coords = self._extract_ca_coords(traj, chain_id)
        else:
            coords = self._extract_ca_coords(traj)

        if len(coords) < 4:
            logger.warning(f"Too few CA atoms ({len(coords)}) for ANM")
            return ANMEntropyResult(S_kB=0.0, S_kcal_per_K=0.0, n_modes=0)

        return self._compute_entropy_sparse(coords)

    def _extract_ca_coords_by_chains(
        self,
        traj: md.Trajectory,
        chain_a: str,
        chain_b: str,
        frame: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract C-alpha coordinates for complex and individual chains.

        Args:
            traj: MDTraj trajectory
            chain_a: First chain ID
            chain_b: Second chain ID
            frame: Frame index to extract coordinates from

        Returns:
            Tuple of (coords_complex, coords_a, coords_b) in Angstrom
        """
        ca_indices_complex = []
        ca_indices_a = []
        ca_indices_b = []

        for atom in traj.topology.atoms:
            if atom.name == 'CA':
                ca_indices_complex.append(atom.index)
                chain_id = atom.residue.chain.chain_id
                if chain_id == chain_a:
                    ca_indices_a.append(atom.index)
                elif chain_id == chain_b:
                    ca_indices_b.append(atom.index)

        if not ca_indices_a or not ca_indices_b:
            raise ValueError(f"Could not find CA atoms for chains {chain_a}, {chain_b}")

        # mdtraj uses nm, ANM expects Angstrom
        coords = traj.xyz[frame] * 10.0

        coords_complex = np.ascontiguousarray(coords[ca_indices_complex], dtype=np.float64)
        coords_a = np.ascontiguousarray(coords[ca_indices_a], dtype=np.float64)
        coords_b = np.ascontiguousarray(coords[ca_indices_b], dtype=np.float64)

        return coords_complex, coords_a, coords_b

    def calculate_binding_delta(
        self,
        traj: md.Trajectory,
        chain_a: str,
        chain_b: str,
        frame: int = 0,
    ) -> ANMBindingResult:
        """
        Calculate vibrational entropy change upon binding.

        dS = S_complex - S_A - S_B

        Uses sparse eigensolver and parallel computation for efficiency.

        Args:
            traj: MDTraj trajectory of the complex
            chain_a: First chain ID
            chain_b: Second chain ID
            frame: Frame index to use for coordinates

        Returns:
            ANMBindingResult with entropy changes
        """
        # Extract CA coordinates directly from mdtraj (fast)
        coords_complex, coords_a, coords_b = self._extract_ca_coords_by_chains(
            traj, chain_a, chain_b, frame
        )

        if self.parallel:
            result = self._compute_binding_parallel(
                coords_complex, coords_a, coords_b
            )
        else:
            result = self._compute_binding_sequential(
                coords_complex, coords_a, coords_b
            )

        logger.info(
            f"ANM binding: dS = {result.dS_kB:.2f} k_B, "
            f"-T*dS = {result.negT_dS_kcal:.3f} kcal/mol"
        )

        return result

    def calculate_binding_delta_from_pdb(
        self,
        pdb_path: Union[str, Path],
        chain_a: str,
        chain_b: str,
    ) -> ANMBindingResult:
        """
        Calculate vibrational entropy change from PDB file.

        Args:
            pdb_path: Path to PDB file
            chain_a: First chain ID
            chain_b: Second chain ID

        Returns:
            ANMBindingResult with entropy changes
        """
        traj = md.load(str(pdb_path))
        return self.calculate_binding_delta(traj, chain_a, chain_b)

    def _compute_binding_parallel(
        self,
        coords_complex: np.ndarray,
        coords_a: np.ndarray,
        coords_b: np.ndarray,
    ) -> ANMBindingResult:
        """Compute binding entropy with parallel eigensolvers."""
        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._compute_entropy_sparse, coords_complex): 'complex',
                executor.submit(self._compute_entropy_sparse, coords_a): 'a',
                executor.submit(self._compute_entropy_sparse, coords_b): 'b',
            }

            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()

        return self._build_binding_result(
            results['complex'], results['a'], results['b']
        )

    def _compute_binding_sequential(
        self,
        coords_complex: np.ndarray,
        coords_a: np.ndarray,
        coords_b: np.ndarray,
    ) -> ANMBindingResult:
        """Compute binding entropy sequentially."""
        result_complex = self._compute_entropy_sparse(coords_complex)
        result_a = self._compute_entropy_sparse(coords_a)
        result_b = self._compute_entropy_sparse(coords_b)

        return self._build_binding_result(result_complex, result_a, result_b)

    def _build_binding_result(
        self,
        complex_result: ANMEntropyResult,
        chain_a_result: ANMEntropyResult,
        chain_b_result: ANMEntropyResult,
    ) -> ANMBindingResult:
        """Build ANMBindingResult from individual chain results."""
        dS_kB = complex_result.S_kB - chain_a_result.S_kB - chain_b_result.S_kB
        dS_kcal_per_K = complex_result.S_kcal_per_K - chain_a_result.S_kcal_per_K - chain_b_result.S_kcal_per_K
        negT_dS_kcal = -self.temperature * dS_kcal_per_K

        return ANMBindingResult(
            dS_kB=dS_kB,
            dS_kcal_per_K=dS_kcal_per_K,
            negT_dS_kcal=negT_dS_kcal,
            complex_result=complex_result,
            chain_a_result=chain_a_result,
            chain_b_result=chain_b_result,
        )
