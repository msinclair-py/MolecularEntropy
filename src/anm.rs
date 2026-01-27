//! ANM (Anisotropic Network Model) Hessian and eigenvalue computation.

use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::{DMatrix, SymmetricEigen};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use sprs::{CsMat, TriMat};

use crate::kdtree::ERR_INVALID_SHAPE;

/// Build sparse ANM Hessian matrix from C-alpha coordinates.
///
/// The Hessian is a 3N x 3N matrix where N is the number of atoms.
/// For atoms i and j within cutoff distance:
///     H[3i:3i+3, 3j:3j+3] = -gamma * (r_ij x r_ij) / |r_ij|^2
///
/// Diagonal blocks satisfy: H[ii] = -sum(H[ij], j != i)
pub fn build_anm_hessian_sparse(coords: &[[f64; 3]], cutoff: f64, gamma: f64) -> CsMat<f64> {
    let n_atoms = coords.len();
    let n_dof = 3 * n_atoms;

    if n_atoms == 0 {
        return CsMat::zero((n_dof, n_dof));
    }

    // Build KD-tree for neighbor finding
    let mut tree: KdTree<f64, 3> = KdTree::new();
    for (i, coord) in coords.iter().enumerate() {
        tree.add(coord, i as u64);
    }

    let cutoff_sq = cutoff * cutoff;

    // Use triplet format for efficient construction
    let mut triplets = TriMat::new((n_dof, n_dof));

    // Diagonal block accumulators
    let mut diag_blocks = vec![[0.0f64; 9]; n_atoms];

    // Find all pairs within cutoff and compute off-diagonal blocks
    for i in 0..n_atoms {
        let neighbors = tree.within::<SquaredEuclidean>(&coords[i], cutoff_sq);

        for neighbor in neighbors {
            let j = neighbor.item as usize;
            if j <= i {
                continue; // Only process each pair once (i < j)
            }

            let dist_sq = neighbor.distance;
            if dist_sq < 1e-10 {
                continue;
            }

            // Distance vector
            let r = [
                coords[j][0] - coords[i][0],
                coords[j][1] - coords[i][1],
                coords[j][2] - coords[i][2],
            ];

            // Compute 3x3 block: -gamma * outer(r, r) / |r|^2
            let factor = -gamma / dist_sq;
            let mut block = [0.0f64; 9];
            for di in 0..3 {
                for dj in 0..3 {
                    block[di * 3 + dj] = factor * r[di] * r[dj];
                }
            }

            // Add off-diagonal blocks (both (i,j) and (j,i))
            for di in 0..3 {
                for dj in 0..3 {
                    let val = block[di * 3 + dj];
                    triplets.add_triplet(3 * i + di, 3 * j + dj, val);
                    triplets.add_triplet(3 * j + di, 3 * i + dj, val);
                }
            }

            // Accumulate diagonal blocks (negative of off-diagonal)
            for k in 0..9 {
                diag_blocks[i][k] -= block[k];
                diag_blocks[j][k] -= block[k];
            }
        }
    }

    // Add diagonal blocks
    for i in 0..n_atoms {
        for di in 0..3 {
            for dj in 0..3 {
                let val = diag_blocks[i][di * 3 + dj];
                if val.abs() > 1e-15 {
                    triplets.add_triplet(3 * i + di, 3 * i + dj, val);
                }
            }
        }
    }

    triplets.to_csr()
}

/// Sparse matrix-vector multiplication: y = A * x
fn spmv(a: &CsMat<f64>, x: &[f64], y: &mut [f64]) {
    y.fill(0.0);
    for (row_idx, row) in a.outer_iterator().enumerate() {
        for (col_idx, &val) in row.iter() {
            y[row_idx] += val * x[col_idx];
        }
    }
}

/// Lanczos iteration to compute smallest eigenvalues of a sparse symmetric matrix.
///
/// Uses shift-invert mode: finds eigenvalues of (A - sigma*I)^-1 closest to 0,
/// which correspond to eigenvalues of A closest to sigma.
///
/// For finding smallest eigenvalues, we use sigma close to 0.
pub fn lanczos_smallest_eigenvalues(
    hessian: &CsMat<f64>,
    n_modes: usize,
    _max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = hessian.rows();
    if n == 0 {
        return Vec::new();
    }

    // Number of Lanczos vectors to compute (need many more for accurate smallest eigenvalues)
    // For finding smallest eigenvalues, Lanczos needs a larger subspace
    let m = (n_modes * 5 + 50).min(n / 2).min(300).max(n_modes + 6);

    // Initialize with random starting vector
    let mut v: Vec<f64> = (0..n)
        .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0 - 0.5)
        .collect();

    // Normalize
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return Vec::new();
    }
    for x in &mut v {
        *x /= norm;
    }

    // Lanczos vectors
    let mut lanczos_vecs: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    lanczos_vecs.push(v);

    // Tridiagonal matrix elements
    let mut alpha: Vec<f64> = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m);

    let mut w = vec![0.0; n];

    for j in 0..m {
        // w = A * v_j
        spmv(hessian, &lanczos_vecs[j], &mut w);

        // alpha_j = v_j^T * w
        let alpha_j: f64 = lanczos_vecs[j].iter().zip(&w).map(|(a, b)| a * b).sum();
        alpha.push(alpha_j);

        // w = w - alpha_j * v_j
        for (w_i, v_i) in w.iter_mut().zip(&lanczos_vecs[j]) {
            *w_i -= alpha_j * v_i;
        }

        // w = w - beta_{j-1} * v_{j-1}  (if j > 0)
        if j > 0 {
            for (w_i, v_i) in w.iter_mut().zip(&lanczos_vecs[j - 1]) {
                *w_i -= beta[j - 1] * v_i;
            }
        }

        // Reorthogonalization (full, for numerical stability)
        for k in 0..=j {
            let proj: f64 = lanczos_vecs[k].iter().zip(&w).map(|(a, b)| a * b).sum();
            for (w_i, v_i) in w.iter_mut().zip(&lanczos_vecs[k]) {
                *w_i -= proj * v_i;
            }
        }

        // beta_j = ||w||
        let beta_j = w.iter().map(|x| x * x).sum::<f64>().sqrt();

        if beta_j < tol {
            // Invariant subspace found
            break;
        }

        beta.push(beta_j);

        // v_{j+1} = w / beta_j
        let v_next: Vec<f64> = w.iter().map(|x| x / beta_j).collect();
        lanczos_vecs.push(v_next);
    }

    // Build tridiagonal matrix and compute its eigenvalues
    let k = alpha.len();
    if k == 0 {
        return Vec::new();
    }

    let mut tridiag = DMatrix::zeros(k, k);
    for i in 0..k {
        tridiag[(i, i)] = alpha[i];
        if i > 0 {
            tridiag[(i, i - 1)] = beta[i - 1];
            tridiag[(i - 1, i)] = beta[i - 1];
        }
    }

    // Compute eigenvalues of tridiagonal matrix
    let eigen = SymmetricEigen::new(tridiag);
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(|a, b| a.total_cmp(b));

    // Return smallest non-trivial eigenvalues (skip near-zero ones)
    eigenvalues
        .into_iter()
        .filter(|&e| e > 1e-6)
        .take(n_modes)
        .collect()
}

/// Build ANM Hessian matrix and return in COO format for scipy.
///
/// Returns (row_indices, col_indices, data, n_dof) for constructing
/// scipy.sparse.coo_matrix.
///
/// # Arguments
/// * `coords` - Nx3 array of C-alpha coordinates in Angstrom
/// * `cutoff` - Distance cutoff for ANM springs (Angstrom)
/// * `gamma` - Spring constant
///
/// # Returns
/// Tuple of (row_indices, col_indices, data, n_dof)
#[pyfunction]
#[pyo3(signature = (coords, cutoff, gamma))]
pub fn build_anm_hessian_coo(
    py: Python<'_>,
    coords: PyReadonlyArray2<f64>,
    cutoff: f64,
    gamma: f64,
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>, usize)> {
    let shape = coords.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(ERR_INVALID_SHAPE));
    }

    let n_atoms = shape[0];
    let n_dof = 3 * n_atoms;

    // Convert to Vec of [f64; 3]
    let array = coords.as_array();
    let coords_vec: Vec<[f64; 3]> = (0..n_atoms)
        .map(|i| [array[[i, 0]], array[[i, 1]], array[[i, 2]]])
        .collect();

    // Build sparse Hessian (release GIL)
    let (rows, cols, data) = py.allow_threads(|| {
        let hessian = build_anm_hessian_sparse(&coords_vec, cutoff, gamma);

        // Convert CSR to COO format
        let mut rows = Vec::with_capacity(hessian.nnz());
        let mut cols = Vec::with_capacity(hessian.nnz());
        let mut data = Vec::with_capacity(hessian.nnz());

        for (row_idx, row) in hessian.outer_iterator().enumerate() {
            for (col_idx, &val) in row.iter() {
                rows.push(row_idx);
                cols.push(col_idx);
                data.push(val);
            }
        }

        (rows, cols, data)
    });

    Ok((rows, cols, data, n_dof))
}

/// Build ANM Hessian matrix and return numpy arrays (faster than Vec).
///
/// Returns (row_indices, col_indices, data, n_dof) as numpy arrays for
/// efficient scipy.sparse.coo_matrix construction.
#[pyfunction]
#[pyo3(signature = (coords, cutoff, gamma))]
pub fn build_anm_hessian_numpy<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    cutoff: f64,
    gamma: f64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
)> {
    let shape = coords.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(ERR_INVALID_SHAPE));
    }

    let n_atoms = shape[0];
    let n_dof = 3 * n_atoms;

    // Convert to Vec of [f64; 3]
    let array = coords.as_array();
    let coords_vec: Vec<[f64; 3]> = (0..n_atoms)
        .map(|i| [array[[i, 0]], array[[i, 1]], array[[i, 2]]])
        .collect();

    // Build sparse Hessian (release GIL)
    let (rows, cols, data) = py.allow_threads(|| {
        let hessian = build_anm_hessian_sparse(&coords_vec, cutoff, gamma);

        // Convert CSR to COO format with i64 indices for numpy
        let mut rows: Vec<i64> = Vec::with_capacity(hessian.nnz());
        let mut cols: Vec<i64> = Vec::with_capacity(hessian.nnz());
        let mut data: Vec<f64> = Vec::with_capacity(hessian.nnz());

        for (row_idx, row) in hessian.outer_iterator().enumerate() {
            for (col_idx, &val) in row.iter() {
                rows.push(row_idx as i64);
                cols.push(col_idx as i64);
                data.push(val);
            }
        }

        (rows, cols, data)
    });

    // Convert to numpy arrays (zero-copy, transfers ownership)
    let rows_arr = rows.into_pyarray_bound(py);
    let cols_arr = cols.into_pyarray_bound(py);
    let data_arr = data.into_pyarray_bound(py);

    Ok((rows_arr, cols_arr, data_arr, n_dof))
}

/// Compute ANM eigenvalues for a set of C-alpha coordinates.
///
/// # Arguments
/// * `coords` - Nx3 array of C-alpha coordinates in Angstrom
/// * `cutoff` - Distance cutoff for ANM springs (Angstrom)
/// * `gamma` - Spring constant
/// * `n_modes` - Number of modes to compute
///
/// # Returns
/// Vector of smallest non-trivial eigenvalues
#[pyfunction]
#[pyo3(signature = (coords, cutoff, gamma, n_modes))]
pub fn compute_anm_eigenvalues(
    py: Python<'_>,
    coords: PyReadonlyArray2<f64>,
    cutoff: f64,
    gamma: f64,
    n_modes: usize,
) -> PyResult<Vec<f64>> {
    let shape = coords.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(ERR_INVALID_SHAPE));
    }

    let n_atoms = shape[0];
    if n_atoms < 4 {
        return Ok(Vec::new());
    }

    // Convert to Vec of [f64; 3]
    let array = coords.as_array();
    let coords_vec: Vec<[f64; 3]> = (0..n_atoms)
        .map(|i| [array[[i, 0]], array[[i, 1]], array[[i, 2]]])
        .collect();

    // Release GIL during computation
    let eigenvalues = py.allow_threads(|| {
        // Build sparse Hessian
        let hessian = build_anm_hessian_sparse(&coords_vec, cutoff, gamma);

        // Compute eigenvalues using Lanczos
        let max_iter = n_modes * 3 + 50;
        lanczos_smallest_eigenvalues(&hessian, n_modes, max_iter, 1e-10)
    });

    Ok(eigenvalues)
}

/// Compute ANM binding entropy for complex and individual chains.
///
/// Calculates eigenvalues for complex, chain A, and chain B in parallel,
/// then computes the entropy difference.
///
/// # Arguments
/// * `coords_complex` - Complex C-alpha coordinates (Nx3)
/// * `coords_a` - Chain A C-alpha coordinates (Mx3)
/// * `coords_b` - Chain B C-alpha coordinates (Kx3)
/// * `cutoff` - Distance cutoff for ANM springs (Angstrom)
/// * `gamma` - Spring constant
/// * `n_modes` - Number of modes to compute
///
/// # Returns
/// Tuple of (eigenvalues_complex, eigenvalues_a, eigenvalues_b)
#[pyfunction]
#[pyo3(signature = (coords_complex, coords_a, coords_b, cutoff, gamma, n_modes))]
pub fn compute_anm_binding_eigenvalues(
    py: Python<'_>,
    coords_complex: PyReadonlyArray2<f64>,
    coords_a: PyReadonlyArray2<f64>,
    coords_b: PyReadonlyArray2<f64>,
    cutoff: f64,
    gamma: f64,
    n_modes: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    // Validate shapes
    for (name, arr) in [
        ("complex", &coords_complex),
        ("chain_a", &coords_a),
        ("chain_b", &coords_b),
    ] {
        let shape = arr.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} coordinates must have shape (n, 3)",
                name
            )));
        }
    }

    // Convert to Vec<[f64; 3]>
    fn to_coords(arr: &PyReadonlyArray2<f64>) -> Vec<[f64; 3]> {
        let array = arr.as_array();
        let n = arr.shape()[0];
        (0..n)
            .map(|i| [array[[i, 0]], array[[i, 1]], array[[i, 2]]])
            .collect()
    }

    let complex_coords = to_coords(&coords_complex);
    let a_coords = to_coords(&coords_a);
    let b_coords = to_coords(&coords_b);

    // Release GIL and compute in parallel
    let (eigs_complex, (eigs_a, eigs_b)) = py.allow_threads(|| {
        let max_iter = n_modes * 3 + 50;
        let tol = 1e-10;

        // Parallel computation of all three
        rayon::join(
            || {
                let h = build_anm_hessian_sparse(&complex_coords, cutoff, gamma);
                lanczos_smallest_eigenvalues(&h, n_modes, max_iter, tol)
            },
            || {
                rayon::join(
                    || {
                        let h = build_anm_hessian_sparse(&a_coords, cutoff, gamma);
                        lanczos_smallest_eigenvalues(&h, n_modes, max_iter, tol)
                    },
                    || {
                        let h = build_anm_hessian_sparse(&b_coords, cutoff, gamma);
                        lanczos_smallest_eigenvalues(&h, n_modes, max_iter, tol)
                    },
                )
            },
        )
    });

    Ok((eigs_complex, eigs_a, eigs_b))
}
