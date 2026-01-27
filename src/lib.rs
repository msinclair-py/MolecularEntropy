//! High-performance Rust extensions for molecular entropy calculations.
//!
//! This module provides `PyO3` bindings for computationally intensive operations
//! in molecular dynamics analysis, including:
//! - Clash detection using KD-trees
//! - Shannon entropy calculations for rotamer distributions
//! - Batch spatial queries
//! - ANM (Anisotropic Network Model) eigenvalue computation

// PyO3 bindings require pass-by-value for Python objects and collections
#![allow(clippy::useless_conversion)]
#![allow(clippy::needless_pass_by_value)]
// Truncation from u64 to usize is acceptable for index values on 64-bit systems
#![allow(clippy::cast_possible_truncation)]

mod anm;
mod entropy;
mod kdtree;

use pyo3::prelude::*;

// Re-export functions for use in tests
pub use anm::{build_anm_hessian_sparse, lanczos_smallest_eigenvalues};
pub use entropy::{compute_masked_entropy, compute_shannon_entropy};
pub use kdtree::build_kdtree;

/// A Python module implemented in Rust for molecular entropy calculations.
///
/// This module provides high-performance implementations of:
/// - `check_clashes_batch`: Detect steric clashes using KD-trees
/// - `compute_rotamer_entropies_batch`: Calculate Shannon entropies for rotamer distributions
/// - `build_kdtree_and_query_batch`: Flexible KD-tree construction and batch queries
/// - `build_anm_hessian_coo`: Build ANM Hessian in COO format for scipy
/// - `compute_anm_eigenvalues`: ANM eigenvalue computation using sparse Lanczos
/// - `compute_anm_binding_eigenvalues`: Parallel ANM computation for binding calculations
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // KD-tree operations
    m.add_function(wrap_pyfunction!(kdtree::check_clashes_batch, m)?)?;
    m.add_function(wrap_pyfunction!(kdtree::build_kdtree_and_query_batch, m)?)?;

    // Entropy calculations
    m.add_function(wrap_pyfunction!(
        entropy::compute_rotamer_entropies_batch,
        m
    )?)?;

    // ANM operations
    m.add_function(wrap_pyfunction!(anm::build_anm_hessian_coo, m)?)?;
    m.add_function(wrap_pyfunction!(anm::build_anm_hessian_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(anm::compute_anm_eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(anm::compute_anm_binding_eigenvalues, m)?)?;

    Ok(())
}
