//! Shannon entropy calculations for rotamer distributions.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute Shannon entropy for a probability distribution.
///
/// S = -R * sum(p * ln(p))
///
/// Handles p=0 correctly (0 * ln(0) = 0 by convention).
#[inline]
pub fn compute_shannon_entropy(probs: &[f64], r: f64) -> f64 {
    let mut entropy = 0.0;

    for &p in probs {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    r * entropy
}

/// Compute Shannon entropy for a masked and renormalized probability distribution.
///
/// Only considers probabilities where the mask is true, renormalizes them,
/// then computes the entropy.
#[inline]
pub fn compute_masked_entropy(probs: &[f64], masks: &[bool], r: f64) -> f64 {
    // Sum probabilities for feasible rotamers
    let mut sum_feasible = 0.0;
    for (&p, &m) in probs.iter().zip(masks) {
        if m {
            sum_feasible += p;
        }
    }

    // Handle edge case: no feasible rotamers or zero probability sum
    if sum_feasible <= 0.0 {
        return 0.0;
    }

    // Compute entropy with renormalized probabilities
    let mut entropy = 0.0;
    for (&p, &m) in probs.iter().zip(masks) {
        if m && p > 0.0 {
            let p_norm = p / sum_feasible;
            entropy -= p_norm * p_norm.ln();
        }
    }

    r * entropy
}

/// Compute Shannon entropies for rotamer probability distributions.
///
/// Calculates both unbound (full distribution) and bound (masked/renormalized)
/// entropies for each residue.
///
/// # Arguments
/// * `prob_arrays` - List of probability arrays, one per residue
/// * `feasible_masks` - Boolean masks indicating which rotamers are feasible in bound state
/// * `r_kcal` - Gas constant in kcal/mol/K (typically 0.001987204)
///
/// # Returns
/// Tuple of `(S_unbound, S_bound)` vectors containing entropies for each residue
#[pyfunction]
#[pyo3(signature = (prob_arrays, feasible_masks, r_kcal))]
pub fn compute_rotamer_entropies_batch(
    py: Python<'_>,
    prob_arrays: Vec<PyReadonlyArray1<f64>>,
    feasible_masks: Vec<PyReadonlyArray1<bool>>,
    r_kcal: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let n_residues = prob_arrays.len();

    if feasible_masks.len() != n_residues {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prob_arrays and feasible_masks must have the same length",
        ));
    }

    // Handle empty input
    if n_residues == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // Convert to owned vectors for parallel processing
    let prob_vecs: Vec<Vec<f64>> = prob_arrays
        .iter()
        .map(|arr| arr.as_array().iter().copied().collect())
        .collect();

    let mask_vecs: Vec<Vec<bool>> = feasible_masks
        .iter()
        .map(|arr| arr.as_array().iter().copied().collect())
        .collect();

    // Validate that each pair has matching lengths
    for (i, (probs, masks)) in prob_vecs.iter().zip(&mask_vecs).enumerate() {
        if probs.len() != masks.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Probability array and mask at index {} have different lengths ({} vs {})",
                i,
                probs.len(),
                masks.len()
            )));
        }
    }

    // Release GIL during parallel computation
    let (s_unbound, s_bound) = py.allow_threads(|| {
        prob_vecs
            .par_iter()
            .zip(&mask_vecs)
            .map(|(probs, masks)| {
                // Compute unbound entropy: S = -R * sum(p * ln(p))
                let s_unbound = compute_shannon_entropy(probs, r_kcal);

                // Compute bound entropy with masked/renormalized distribution
                let s_bound = compute_masked_entropy(probs, masks, r_kcal);

                (s_unbound, s_bound)
            })
            .unzip()
    });

    Ok((s_unbound, s_bound))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution over 4 states: S = R * ln(4)
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let r = 1.0; // Use R=1 for easy verification
        let s = compute_shannon_entropy(&probs, r);
        let expected = 4.0_f64.ln();
        assert!((s - expected).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_single_state() {
        // Single state (p=1): S = 0
        let probs = vec![1.0];
        let r = 1.0;
        let s = compute_shannon_entropy(&probs, r);
        assert!((s - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_shannon_entropy_with_zeros() {
        // Distribution with zeros
        let probs = vec![0.5, 0.0, 0.5, 0.0];
        let r = 1.0;
        let s = compute_shannon_entropy(&probs, r);
        let expected = 2.0_f64.ln(); // Only 2 non-zero states
        assert!((s - expected).abs() < 1e-10);
    }

    #[test]
    fn test_masked_entropy_full_mask() {
        // All rotamers feasible - should equal unbound entropy
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let masks = vec![true, true, true, true];
        let r = 1.0;
        let s_masked = compute_masked_entropy(&probs, &masks, r);
        let s_full = compute_shannon_entropy(&probs, r);
        assert!((s_masked - s_full).abs() < 1e-10);
    }

    #[test]
    fn test_masked_entropy_partial_mask() {
        // Only 2 of 4 rotamers feasible
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let masks = vec![true, true, false, false];
        let r = 1.0;
        let s = compute_masked_entropy(&probs, &masks, r);
        // After renormalization: [0.5, 0.5, -, -]
        let expected = 2.0_f64.ln();
        assert!((s - expected).abs() < 1e-10);
    }

    #[test]
    fn test_masked_entropy_no_feasible() {
        // No feasible rotamers
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let masks = vec![false, false, false, false];
        let r = 1.0;
        let s = compute_masked_entropy(&probs, &masks, r);
        assert!((s - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_masked_entropy_unequal_probs() {
        // Unequal probabilities with partial mask
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let masks = vec![true, false, true, false];
        let r = 1.0;
        let s = compute_masked_entropy(&probs, &masks, r);
        // After renormalization: p1 = 0.1/0.4 = 0.25, p3 = 0.3/0.4 = 0.75
        let p1: f64 = 0.25;
        let p3: f64 = 0.75;
        let expected = -(p1 * p1.ln() + p3 * p3.ln());
        assert!((s - expected).abs() < 1e-10);
    }
}
