"""Type stubs for Rust extension module."""

import numpy as np
from numpy.typing import NDArray

def check_clashes_batch(
    complex_coords: NDArray[np.float64],
    partner_coords: NDArray[np.float64],
    residue_atom_ranges: list[tuple[int, int]],
    threshold_nm: float,
) -> list[bool]:
    """
    Batch clash detection using KD-tree and rayon parallelization.

    Args:
        complex_coords: Coordinates of the complex atoms, shape (n_atoms, 3), in nanometers.
        partner_coords: Coordinates of the partner atoms, shape (m_atoms, 3), in nanometers.
        residue_atom_ranges: List of (start, end) tuples defining atom index ranges for each residue.
        threshold_nm: Distance threshold in nanometers for clash detection.

    Returns:
        List of booleans, one per residue, indicating whether that residue clashes with the partner.
    """
    ...

def compute_rotamer_entropies_batch(
    prob_arrays: list[NDArray[np.float64]],
    feasible_masks: list[NDArray[np.bool_]],
    r_kcal: float,
) -> tuple[list[float], list[float]]:
    """
    Batch entropy calculation with rayon parallelization.

    Args:
        prob_arrays: List of probability arrays, one per residue.
        feasible_masks: Boolean masks indicating which rotamers are feasible in bound state.
        r_kcal: Gas constant in kcal/mol/K (typically 0.001987204).

    Returns:
        Tuple of (S_unbound, S_bound) lists containing entropies for each residue.
    """
    ...

def build_kdtree_and_query_batch(
    tree_coords: NDArray[np.float64],
    query_coords: NDArray[np.float64],
    radius: float,
) -> list[list[int]]:
    """
    Build a KD-tree and perform batch radius queries.

    Args:
        tree_coords: Points to build tree from, shape (n, 3).
        query_coords: Points to query, shape (m, 3).
        radius: Search radius.

    Returns:
        For each query point, a list of indices of neighbors within the radius.
    """
    ...
