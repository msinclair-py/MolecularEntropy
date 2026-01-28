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
    "ANMLigandResult",
    "build_sparse_hessian",
    "build_sparse_hessian_with_ligand",
    "compute_anm_eigenvalues_sparse",
]

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import mdtraj as md
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

from .constants import (
    DEFAULT_ANM_CUTOFF,
    DEFAULT_ANM_GAMMA,
    DEFAULT_TEMPERATURE,
    KB_KCAL,
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
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
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
    blocks = -gamma * np.einsum("ij,ik->ijk", r_ij, r_ij) / dist_sq[:, None, None]
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
        (data_offdiag, (row_offdiag, col_offdiag)), shape=(n_dof, n_dof), dtype=np.float64
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
        (diag_data, (diag_row, diag_col)), shape=(n_dof, n_dof), dtype=np.float64
    )

    return hessian + diag_matrix


def build_sparse_hessian_with_ligand(
    ca_coords: np.ndarray,
    ligand_coords: np.ndarray,
    ligand_bonds: list[tuple[int, int]],
    cutoff: float,
    gamma_protein: float = 1.0,
    gamma_ligand_bond: float = 10.0,
    gamma_interface: float = 1.0,
) -> csr_matrix:
    """
    Build ANM Hessian matrix including ligand atoms with explicit bond connectivity.

    The Hessian includes:
    1. Protein CA-CA springs (distance cutoff, gamma_protein)
    2. Ligand internal bonds (explicit connectivity, gamma_ligand_bond)
    3. Protein-ligand interface springs (distance cutoff, gamma_interface)

    This allows the ligand to contribute vibrational modes while respecting
    its covalent bond geometry.

    Args:
        ca_coords: Nx3 array of protein C-alpha coordinates (Angstrom)
        ligand_coords: Mx3 array of ligand atom coordinates (Angstrom)
        ligand_bonds: List of (local_atom_i, local_atom_j) tuples defining ligand bonds.
                     Indices are 0-based relative to ligand_coords.
        cutoff: Distance cutoff for non-bonded springs (Angstrom)
        gamma_protein: Spring constant for protein CA-CA interactions
        gamma_ligand_bond: Spring constant for ligand covalent bonds (stiffer)
        gamma_interface: Spring constant for protein-ligand interface springs

    Returns:
        Sparse CSR Hessian matrix ((3*(N+M)) x (3*(N+M)))
    """
    n_protein = len(ca_coords)
    n_ligand = len(ligand_coords)
    n_total = n_protein + n_ligand
    n_dof = 3 * n_total

    # Combine all coordinates
    all_coords = np.vstack([ca_coords, ligand_coords])

    # Build protein-protein Hessian using existing function
    # This handles protein CA-CA interactions efficiently
    protein_hessian = build_sparse_hessian(ca_coords, cutoff, gamma_protein)

    # Embed protein Hessian in larger matrix
    # We'll build the full matrix and add contributions

    # Initialize data structures for COO format
    rows_list = []
    cols_list = []
    data_list = []

    # Copy protein Hessian entries
    protein_coo = protein_hessian.tocoo()
    rows_list.append(protein_coo.row)
    cols_list.append(protein_coo.col)
    data_list.append(protein_coo.data)

    # Helper function to add a spring between two atoms
    def add_spring(atom_i: int, atom_j: int, gamma: float, coords: np.ndarray):
        """Add a spring contribution to the Hessian."""
        r_ij = coords[atom_j] - coords[atom_i]
        dist_sq = np.sum(r_ij * r_ij)
        if dist_sq < 1e-10:
            return [], [], []

        # Off-diagonal block: H[i,j] = -gamma * (r_ij ⊗ r_ij) / |r_ij|²
        block = -gamma * np.outer(r_ij, r_ij) / dist_sq

        rows = []
        cols = []
        data = []

        # Off-diagonal block (i, j)
        for di in range(3):
            for dj in range(3):
                rows.append(3 * atom_i + di)
                cols.append(3 * atom_j + dj)
                data.append(block[di, dj])
                # Symmetric (j, i)
                rows.append(3 * atom_j + di)
                cols.append(3 * atom_i + dj)
                data.append(block[di, dj])

        return rows, cols, data

    # Add ligand internal bonds (covalent bonds with stiffer springs)
    ligand_diag = np.zeros((n_ligand, 3, 3), dtype=np.float64)

    for local_i, local_j in ligand_bonds:
        # Convert to global indices
        global_i = n_protein + local_i
        global_j = n_protein + local_j

        r_ij = all_coords[global_j] - all_coords[global_i]
        dist_sq = np.sum(r_ij * r_ij)
        if dist_sq < 1e-10:
            continue

        block = -gamma_ligand_bond * np.outer(r_ij, r_ij) / dist_sq

        # Add off-diagonal entries
        for di in range(3):
            for dj in range(3):
                rows_list.append(np.array([3 * global_i + di]))
                cols_list.append(np.array([3 * global_j + dj]))
                data_list.append(np.array([block[di, dj]]))
                rows_list.append(np.array([3 * global_j + di]))
                cols_list.append(np.array([3 * global_i + dj]))
                data_list.append(np.array([block[di, dj]]))

        # Accumulate diagonal contributions
        ligand_diag[local_i] -= block
        ligand_diag[local_j] -= block

    # Add protein-ligand interface springs (based on distance cutoff)
    protein_tree = cKDTree(ca_coords)
    ligand_tree = cKDTree(ligand_coords)

    # Find protein-ligand contacts
    contacts = protein_tree.query_ball_tree(ligand_tree, r=cutoff)

    for protein_idx, ligand_indices in enumerate(contacts):
        for ligand_local_idx in ligand_indices:
            global_ligand_idx = n_protein + ligand_local_idx

            r_ij = all_coords[global_ligand_idx] - all_coords[protein_idx]
            dist_sq = np.sum(r_ij * r_ij)
            if dist_sq < 1e-10:
                continue

            block = -gamma_interface * np.outer(r_ij, r_ij) / dist_sq

            # Add off-diagonal entries
            for di in range(3):
                for dj in range(3):
                    rows_list.append(np.array([3 * protein_idx + di]))
                    cols_list.append(np.array([3 * global_ligand_idx + dj]))
                    data_list.append(np.array([block[di, dj]]))
                    rows_list.append(np.array([3 * global_ligand_idx + di]))
                    cols_list.append(np.array([3 * protein_idx + dj]))
                    data_list.append(np.array([block[di, dj]]))

            # Accumulate diagonal contributions for ligand
            ligand_diag[ligand_local_idx] -= block

            # Need to add to protein diagonal too - but protein_hessian already
            # has its internal diagonal. We need to add interface contributions.
            # Build sparse update for protein diagonal
            for di in range(3):
                for dj in range(3):
                    rows_list.append(np.array([3 * protein_idx + di]))
                    cols_list.append(np.array([3 * protein_idx + dj]))
                    data_list.append(np.array([-block[di, dj]]))

    # Add ligand diagonal blocks
    local_i = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    local_j = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)

    for ligand_local_idx in range(n_ligand):
        global_idx = n_protein + ligand_local_idx
        diag_block = ligand_diag[ligand_local_idx]

        diag_row = 3 * global_idx + local_i
        diag_col = 3 * global_idx + local_j
        rows_list.append(diag_row)
        cols_list.append(diag_col)
        data_list.append(diag_block.ravel())

    # Combine all entries
    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_data = np.concatenate(data_list)

    # Build sparse matrix (duplicate entries are summed)
    hessian = csr_matrix(
        (all_data, (all_rows, all_cols)), shape=(n_dof, n_dof), dtype=np.float64
    )

    return hessian


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
            which="LM",  # Largest magnitude of (H - sigma*I)^-1 = closest to sigma
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


@dataclass
class ANMLigandResult:
    """Results from ANM calculation with ligand atoms included."""

    S_kB: float  # Entropy in k_B units
    S_kcal_per_K: float  # Entropy in kcal/mol/K
    n_modes: int  # Number of non-trivial modes used
    n_protein_atoms: int  # Number of protein CA atoms
    n_ligand_atoms: int  # Number of ligand atoms


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

        hessian = coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof), dtype=np.float64).tocsr()

        # Use scipy shift-invert eigensolver (accurate for smallest eigenvalues)
        k = min(6 + self.n_modes + 6, n_dof - 1)
        try:
            eigenvalues, _ = eigsh(
                hessian,
                k=k,
                sigma=1e-4,
                which="LM",
                tol=1e-5,
                maxiter=500,
            )
        except Exception as e:
            logger.warning(f"Sparse eigensolver failed: {e}, falling back to dense")
            eigenvalues = np.linalg.eigvalsh(hessian.toarray())

        # Sort and filter out trivial modes
        eigenvalues = np.sort(np.abs(eigenvalues))
        non_trivial = eigenvalues[eigenvalues > 1e-6]

        return self._entropy_from_eigenvalues(non_trivial[: self.n_modes])

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

    def _entropy_from_eigenvalues(self, eigenvalues: np.ndarray) -> ANMEntropyResult:
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
        chain_id: str | None = None,
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
            if atom.name == "CA":
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
        self, pdb_path: str | Path, chain_id: str | None = None
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
            if atom.name == "CA":
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
            result = self._compute_binding_parallel(coords_complex, coords_a, coords_b)
        else:
            result = self._compute_binding_sequential(coords_complex, coords_a, coords_b)

        logger.info(
            f"ANM binding: dS = {result.dS_kB:.2f} k_B, -T*dS = {result.negT_dS_kcal:.3f} kcal/mol"
        )

        return result

    def calculate_binding_delta_from_pdb(
        self,
        pdb_path: str | Path,
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
                executor.submit(self._compute_entropy_sparse, coords_complex): "complex",
                executor.submit(self._compute_entropy_sparse, coords_a): "a",
                executor.submit(self._compute_entropy_sparse, coords_b): "b",
            }

            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()

        return self._build_binding_result(results["complex"], results["a"], results["b"])

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
        dS_kcal_per_K = (
            complex_result.S_kcal_per_K - chain_a_result.S_kcal_per_K - chain_b_result.S_kcal_per_K
        )
        negT_dS_kcal = -self.temperature * dS_kcal_per_K

        return ANMBindingResult(
            dS_kB=dS_kB,
            dS_kcal_per_K=dS_kcal_per_K,
            negT_dS_kcal=negT_dS_kcal,
            complex_result=complex_result,
            chain_a_result=chain_a_result,
            chain_b_result=chain_b_result,
        )

    def calculate_with_ligand(
        self,
        traj: md.Trajectory,
        ligand_residue_indices: list[int],
        protein_residue_indices: list[int] | None = None,
        ligand_bonds: list[tuple[int, int]] | None = None,
        gamma_ligand_bond: float = 10.0,
        gamma_interface: float = 1.0,
    ) -> ANMLigandResult:
        """
        Calculate vibrational entropy including ligand atoms.

        Builds an elastic network that includes:
        - Protein CA atoms with distance-based springs
        - Ligand atoms with explicit bond connectivity
        - Protein-ligand interface springs

        Args:
            traj: MDTraj trajectory
            ligand_residue_indices: List of residue indices (0-based) that comprise
                                   the ligand.
            protein_residue_indices: List of residue indices for the protein.
                                    If None, uses all non-ligand residues (may include solvent).
            ligand_bonds: List of (atom_i, atom_j) tuples defining ligand bonds.
                         Indices are relative to the ligand atoms in the trajectory.
                         If None, will attempt to infer bonds from distances.
            gamma_ligand_bond: Spring constant for ligand covalent bonds
            gamma_interface: Spring constant for protein-ligand interface

        Returns:
            ANMLigandResult with entropy including ligand contribution
        """
        ligand_res_set = set(ligand_residue_indices) if ligand_residue_indices else set()
        protein_res_set = set(protein_residue_indices) if protein_residue_indices else None

        # Extract protein CA coordinates
        ca_indices = []
        for atom in traj.topology.atoms:
            if atom.name == "CA":
                if protein_res_set is not None:
                    # Use explicit protein residue list
                    if atom.residue.index in protein_res_set:
                        ca_indices.append(atom.index)
                else:
                    # Fallback: exclude ligand residues
                    if atom.residue.index not in ligand_res_set:
                        ca_indices.append(atom.index)

        if len(ca_indices) < 4:
            logger.warning(f"Too few CA atoms ({len(ca_indices)}) for ANM")
            return ANMLigandResult(
                S_kB=0.0, S_kcal_per_K=0.0, n_modes=0, n_protein_atoms=len(ca_indices), n_ligand_atoms=0
            )

        # mdtraj uses nm, ANM expects Angstrom
        ca_coords = np.ascontiguousarray(traj.xyz[0, ca_indices] * 10.0, dtype=np.float64)

        # Extract ligand atom coordinates
        ligand_atom_indices = []
        if ligand_residue_indices:
            for atom in traj.topology.atoms:
                if atom.residue.index in ligand_res_set:
                    ligand_atom_indices.append(atom.index)

        if not ligand_atom_indices:
            # No ligand found, fall back to protein-only ANM
            logger.info("No ligand atoms found, computing protein-only ANM")
            result = self._compute_entropy_sparse(ca_coords)
            return ANMLigandResult(
                S_kB=result.S_kB,
                S_kcal_per_K=result.S_kcal_per_K,
                n_modes=result.n_modes,
                n_protein_atoms=len(ca_indices),
                n_ligand_atoms=0,
            )

        ligand_coords = np.ascontiguousarray(
            traj.xyz[0, ligand_atom_indices] * 10.0, dtype=np.float64
        )

        # Handle ligand bonds
        if ligand_bonds is None:
            # Infer bonds from distances (simple heuristic: bond if < 2.0 Angstrom)
            ligand_bonds = self._infer_ligand_bonds(ligand_coords)
            logger.info(f"Inferred {len(ligand_bonds)} ligand bonds from distances")

        # Build Hessian with ligand
        n_protein = len(ca_coords)
        n_ligand = len(ligand_coords)
        n_total = n_protein + n_ligand
        n_dof = 3 * n_total

        hessian = build_sparse_hessian_with_ligand(
            ca_coords,
            ligand_coords,
            ligand_bonds,
            self.cutoff,
            self.gamma,
            gamma_ligand_bond,
            gamma_interface,
        )

        # Compute eigenvalues
        k = min(6 + self.n_modes + 6, n_dof - 1)

        try:
            eigenvalues, _ = eigsh(
                hessian,
                k=k,
                sigma=1e-4,
                which="LM",
                tol=1e-5,
                maxiter=500,
            )
        except Exception as e:
            logger.warning(f"Sparse eigensolver failed: {e}, falling back to dense")
            eigenvalues = np.linalg.eigvalsh(hessian.toarray())

        # Sort and filter trivial modes
        eigenvalues = np.sort(np.abs(eigenvalues))
        non_trivial = eigenvalues[eigenvalues > 1e-6]
        evals = non_trivial[: self.n_modes]

        if len(evals) == 0:
            logger.warning("No valid eigenvalues found")
            return ANMLigandResult(
                S_kB=0.0, S_kcal_per_K=0.0, n_modes=0, n_protein_atoms=n_protein, n_ligand_atoms=n_ligand
            )

        # Calculate entropy
        S_kB = float(np.sum(-0.5 * np.log(evals)))
        S_kcal_per_K = S_kB * KB_KCAL

        logger.info(
            f"ANM with ligand: {n_protein} CA + {n_ligand} ligand atoms, "
            f"{len(evals)} modes, S = {S_kB:.2f} k_B"
        )

        return ANMLigandResult(
            S_kB=S_kB,
            S_kcal_per_K=S_kcal_per_K,
            n_modes=len(evals),
            n_protein_atoms=n_protein,
            n_ligand_atoms=n_ligand,
        )

    def _infer_ligand_bonds(
        self,
        coords: np.ndarray,
        bond_threshold: float = 2.0,
    ) -> list[tuple[int, int]]:
        """
        Infer ligand bonds from distances.

        Simple heuristic: atoms closer than bond_threshold are bonded.

        Args:
            coords: Nx3 array of ligand coordinates (Angstrom)
            bond_threshold: Maximum bond distance (Angstrom)

        Returns:
            List of (atom_i, atom_j) tuples representing bonds
        """
        n_atoms = len(coords)
        bonds = []

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[j] - coords[i])
                if dist < bond_threshold:
                    bonds.append((i, j))

        return bonds
