//! KD-tree operations for spatial queries and clash detection.

use kiddo::{KdTree, SquaredEuclidean};
use ndarray::ArrayView1;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Error messages for validation
pub const ERR_INVALID_SHAPE: &str = "Coordinate array must have shape (n, 3)";
pub const ERR_INVALID_RANGE: &str = "Invalid atom range: start > end or indices out of bounds";

/// Build a 3D KD-tree from coordinate data.
///
/// # Arguments
/// * `coords` - Flattened Nx3 array of 3D coordinates
/// * `n_points` - Number of points
///
/// # Returns
/// A KD-tree populated with the coordinate points
pub fn build_kdtree(coords: &ArrayView1<f64>, n_points: usize) -> KdTree<f64, 3> {
    let mut tree: KdTree<f64, 3> = KdTree::new();

    for i in 0..n_points {
        let point: [f64; 3] = [coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]];
        tree.add(&point, i as u64);
    }

    tree
}

/// Check for steric clashes between complex atoms and partner atoms.
///
/// Uses a KD-tree for efficient spatial queries to determine which residues
/// have atoms within a threshold distance of the binding partner.
///
/// # Arguments
/// * `complex_coords` - Coordinates of the complex atoms, shape `(n_atoms, 3)`, in nanometers
/// * `partner_coords` - Coordinates of the partner atoms, shape `(m_atoms, 3)`, in nanometers
/// * `residue_atom_ranges` - List of (start, end) tuples defining atom index ranges for each residue
/// * `threshold_nm` - Distance threshold in nanometers for clash detection
///
/// # Returns
/// A list of booleans, one per residue, indicating whether that residue clashes with the partner
#[pyfunction]
#[pyo3(signature = (complex_coords, partner_coords, residue_atom_ranges, threshold_nm))]
pub fn check_clashes_batch(
    py: Python<'_>,
    complex_coords: PyReadonlyArray2<f64>,
    partner_coords: PyReadonlyArray2<f64>,
    residue_atom_ranges: Vec<(usize, usize)>,
    threshold_nm: f64,
) -> PyResult<Vec<bool>> {
    // Validate inputs
    let complex_shape = complex_coords.shape();
    let partner_shape = partner_coords.shape();

    if complex_shape.len() != 2 || complex_shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(ERR_INVALID_SHAPE));
    }
    if partner_shape.len() != 2 || partner_shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(ERR_INVALID_SHAPE));
    }

    let n_complex = complex_shape[0];
    let n_partner = partner_shape[0];

    // Handle edge cases
    if residue_atom_ranges.is_empty() {
        return Ok(Vec::new());
    }
    if n_partner == 0 {
        // No partner atoms means no clashes possible
        return Ok(vec![false; residue_atom_ranges.len()]);
    }

    // Validate residue ranges
    for (start, end) in &residue_atom_ranges {
        if start > end || *end > n_complex {
            return Err(pyo3::exceptions::PyValueError::new_err(ERR_INVALID_RANGE));
        }
    }

    // Get array data as contiguous slices
    let complex_array = complex_coords.as_array();
    let partner_array = partner_coords.as_array();

    // Flatten arrays for efficient access
    let complex_flat: Vec<f64> = complex_array.iter().copied().collect();
    let partner_flat: Vec<f64> = partner_array.iter().copied().collect();

    // Build KD-tree from partner coordinates
    let partner_view = ArrayView1::from(&partner_flat);
    let tree = build_kdtree(&partner_view, n_partner);

    // Squared distance threshold for comparison (kiddo uses squared distances)
    let threshold_sq = threshold_nm * threshold_nm;

    // Release GIL during parallel computation
    let results: Vec<bool> = py.allow_threads(|| {
        // Process residues in parallel
        residue_atom_ranges
            .par_iter()
            .map(|(start, end)| {
                // Check if any atom in this residue clashes with partner
                for atom_idx in *start..*end {
                    let query_point: [f64; 3] = [
                        complex_flat[atom_idx * 3],
                        complex_flat[atom_idx * 3 + 1],
                        complex_flat[atom_idx * 3 + 2],
                    ];

                    // Query for nearest neighbor within threshold
                    let nearest = tree.nearest_one::<SquaredEuclidean>(&query_point);
                    if nearest.distance <= threshold_sq {
                        return true;
                    }
                }
                false
            })
            .collect()
    });

    Ok(results)
}

/// Build a KD-tree and perform batch radius queries.
///
/// A lower-level function for flexible KD-tree usage.
///
/// # Arguments
/// * `tree_coords` - Points to build tree from, shape (n, 3)
/// * `query_coords` - Points to query, shape (m, 3)
/// * `radius` - Search radius
///
/// # Returns
/// For each query point, a list of indices of neighbors within the radius
#[pyfunction]
#[pyo3(signature = (tree_coords, query_coords, radius))]
pub fn build_kdtree_and_query_batch(
    py: Python<'_>,
    tree_coords: PyReadonlyArray2<f64>,
    query_coords: PyReadonlyArray2<f64>,
    radius: f64,
) -> PyResult<Vec<Vec<usize>>> {
    // Validate inputs
    let tree_shape = tree_coords.shape();
    let query_shape = query_coords.shape();

    if tree_shape.len() != 2 || tree_shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(ERR_INVALID_SHAPE));
    }
    if query_shape.len() != 2 || query_shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(ERR_INVALID_SHAPE));
    }

    let n_tree = tree_shape[0];
    let n_query = query_shape[0];

    // Handle edge cases
    if n_query == 0 {
        return Ok(Vec::new());
    }
    if n_tree == 0 {
        // No tree points means empty results for all queries
        return Ok(vec![Vec::new(); n_query]);
    }

    // Get array data
    let tree_array = tree_coords.as_array();
    let query_array = query_coords.as_array();

    // Flatten arrays for efficient access
    let tree_flat: Vec<f64> = tree_array.iter().copied().collect();
    let query_flat: Vec<f64> = query_array.iter().copied().collect();

    // Build KD-tree
    let tree_view = ArrayView1::from(&tree_flat);
    let tree = build_kdtree(&tree_view, n_tree);

    // Squared radius for comparison
    let radius_sq = radius * radius;

    // Release GIL during parallel computation
    let results: Vec<Vec<usize>> = py.allow_threads(|| {
        (0..n_query)
            .into_par_iter()
            .map(|i| {
                let query_point: [f64; 3] = [
                    query_flat[i * 3],
                    query_flat[i * 3 + 1],
                    query_flat[i * 3 + 2],
                ];

                // Query all points within radius
                let neighbors = tree.within::<SquaredEuclidean>(&query_point, radius_sq);

                // Extract indices as usize
                neighbors
                    .iter()
                    .map(|n| n.item as usize)
                    .collect::<Vec<usize>>()
            })
            .collect()
    });

    Ok(results)
}
