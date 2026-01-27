"""
Tests for molecular_entropy package.

Tests both static structures and dynamic trajectories using the test data
in the data/ directory.
"""

import pytest
from pathlib import Path
import numpy as np
import polars as pl

# Get paths to test data
TEST_DIR = Path(__file__).parent.parent
DATA_DIR = TEST_DIR / "data"
ROTLIB_DIR = TEST_DIR / "rotamer_library"

STATIC_PDB = DATA_DIR / "static.pdb"
DYNAMIC_PDB = DATA_DIR / "dynamic.pdb"
DYNAMIC_DCD = DATA_DIR / "dynamic.dcd"
ROTLIB_SIMPLE = ROTLIB_DIR / "simple_library.parquet"


# Skip tests if test data not available
requires_static = pytest.mark.skipif(
    not STATIC_PDB.exists(),
    reason="static.pdb not found"
)

requires_dynamic = pytest.mark.skipif(
    not DYNAMIC_PDB.exists(),
    reason="dynamic.pdb not found"
)

requires_rotlib = pytest.mark.skipif(
    not ROTLIB_SIMPLE.exists(),
    reason="rotamer library not found"
)


class TestStructureLoader:
    """Tests for structure loading module."""

    @requires_static
    def test_load_static_pdb(self):
        """Test loading static PDB file."""
        from molecular_entropy.structure import StructureLoader

        traj = StructureLoader.load(STATIC_PDB)
        assert traj.n_frames == 1
        assert traj.n_atoms > 0
        assert traj.n_residues > 0

    @requires_static
    def test_get_chain_ids(self):
        """Test getting chain IDs from structure."""
        from molecular_entropy.structure import StructureLoader

        traj = StructureLoader.load(STATIC_PDB)
        chains = StructureLoader.get_chain_ids(traj)
        assert 'A' in chains
        assert 'B' in chains

    @requires_static
    def test_select_chain(self):
        """Test selecting individual chains."""
        from molecular_entropy.structure import StructureLoader

        traj = StructureLoader.load(STATIC_PDB)
        traj_a = StructureLoader.select_chain(traj, 'A')
        traj_b = StructureLoader.select_chain(traj, 'B')

        assert traj_a.n_atoms > 0
        assert traj_b.n_atoms > 0
        assert traj_a.n_atoms + traj_b.n_atoms == traj.n_atoms

    @requires_static
    def test_select_chain_not_found(self):
        """Test error when chain not found."""
        from molecular_entropy.structure import StructureLoader, ChainNotFoundError

        traj = StructureLoader.load(STATIC_PDB)
        with pytest.raises(ChainNotFoundError):
            StructureLoader.select_chain(traj, 'Z')

    @requires_static
    def test_compute_backbone_dihedrals(self):
        """Test computing phi/psi angles."""
        from molecular_entropy.structure import StructureLoader

        traj = StructureLoader.load(STATIC_PDB)
        phi_map, psi_map = StructureLoader.compute_backbone_dihedrals(traj)

        # Should have values for most residues (not termini)
        assert len(phi_map) > 0
        assert len(psi_map) > 0

        # Values should be in valid range
        for angle in phi_map.values():
            assert -180 <= angle <= 180
        for angle in psi_map.values():
            assert -180 <= angle <= 180


class TestSASACalculator:
    """Tests for SASA entropy calculations."""

    @requires_static
    def test_compute_sasa(self):
        """Test SASA computation."""
        from molecular_entropy.structure import StructureLoader
        from molecular_entropy.sasa import SASACalculator

        traj = StructureLoader.load(STATIC_PDB)
        calc = SASACalculator()
        sasa = calc.compute_sasa(traj)

        # SASA should be positive array
        assert len(sasa) == traj.n_atoms
        assert sasa.sum() > 0
        assert np.all(sasa >= 0)

    @requires_static
    def test_sasa_result(self):
        """Test SASA result dataclass."""
        from molecular_entropy.structure import StructureLoader
        from molecular_entropy.sasa import SASACalculator

        traj = StructureLoader.load(STATIC_PDB)
        calc = SASACalculator()
        result = calc.calculate(traj)

        assert result.total > 0
        assert result.polar >= 0
        assert result.nonpolar >= 0
        assert abs(result.polar + result.nonpolar - result.total) < 0.1

    @requires_static
    def test_sasa_binding_delta(self):
        """Test SASA change upon binding."""
        from molecular_entropy.structure import StructureLoader
        from molecular_entropy.sasa import SASACalculator

        traj = StructureLoader.load(STATIC_PDB)
        calc = SASACalculator()
        result = calc.calculate_binding_delta(traj, 'A', 'B')

        # Binding should bury surface (positive delta)
        assert result.delta_total > 0
        # Entropy contribution should be computed
        assert isinstance(result.negT_dS_solv, float)

    @requires_static
    def test_detect_interface(self):
        """Test interface residue detection."""
        from molecular_entropy.structure import StructureLoader
        from molecular_entropy.sasa import SASACalculator

        traj = StructureLoader.load(STATIC_PDB)
        calc = SASACalculator()
        interface = calc.detect_interface(traj, 'A', 'B', threshold=1.0)

        # Should find some interface residues
        assert isinstance(interface, pl.DataFrame)
        assert len(interface) > 0
        assert 'delta_sasa_A2' in interface.columns


class TestANMEntropyCalculator:
    """Tests for ANM vibrational entropy calculations."""

    @requires_static
    def test_anm_entropy(self):
        """Test ANM entropy calculation."""
        from molecular_entropy.anm import ANMEntropyCalculator

        calc = ANMEntropyCalculator()
        result = calc.calculate_from_pdb(STATIC_PDB, chain_id='A')

        # Should have non-zero entropy with positive mode count
        assert result.n_modes > 0
        assert isinstance(result.S_kB, float)
        assert isinstance(result.S_kcal_per_K, float)

    @requires_static
    def test_anm_binding_delta(self):
        """Test ANM binding entropy change."""
        from molecular_entropy.anm import ANMEntropyCalculator

        calc = ANMEntropyCalculator()
        result = calc.calculate_binding_delta_from_pdb(STATIC_PDB, 'A', 'B')

        # Should have results for all states
        assert result.complex_result.n_modes > 0
        assert result.chain_a_result.n_modes > 0
        assert result.chain_b_result.n_modes > 0

        # Delta should be computed
        assert isinstance(result.dS_kB, float)
        assert isinstance(result.negT_dS_kcal, float)


class TestRotamerEntropyCalculator:
    """Tests for rotamer entropy calculations."""

    @requires_rotlib
    def test_load_rotamer_library(self):
        """Test loading rotamer library."""
        from molecular_entropy.rotamer import RotamerEntropyCalculator

        calc = RotamerEntropyCalculator(rotlib_path=ROTLIB_SIMPLE)
        assert calc.rotlib_index is not None
        assert len(calc.rotlib_index) > 0

    @requires_rotlib
    def test_vectorized_rotamer_loading(self):
        """Test that vectorized rotamer loading produces correct index structure."""
        from molecular_entropy.rotamer import RotamerEntropyCalculator

        calc = RotamerEntropyCalculator(rotlib_path=ROTLIB_SIMPLE)

        # Check that index entries have the correct structure
        for key, rotamers in list(calc.rotlib_index.items())[:5]:
            # Key should be (resname, phi_bin, psi_bin)
            assert len(key) == 3
            resname, phi_bin, psi_bin = key
            assert isinstance(resname, str)
            assert resname.isupper()
            assert isinstance(phi_bin, int)
            assert isinstance(psi_bin, int)

            # Rotamers should be list of dicts with 'chi' and 'p'
            assert isinstance(rotamers, list)
            assert len(rotamers) > 0
            for r in rotamers:
                assert 'chi' in r
                assert 'p' in r
                assert isinstance(r['chi'], list)
                assert r['p'] > 0  # Filtered out zero probs

    def test_angle_to_bin(self):
        """Test angle binning."""
        from molecular_entropy.rotamer import RotamerEntropyCalculator

        # Test standard angles
        assert RotamerEntropyCalculator.angle_to_bin(0) == 0
        assert RotamerEntropyCalculator.angle_to_bin(-60) == -60
        assert RotamerEntropyCalculator.angle_to_bin(-65) == -60
        assert RotamerEntropyCalculator.angle_to_bin(175) in [180, -180]

    def test_angle_to_bin_comprehensive(self):
        """Test O(1) angle binning across full range."""
        from molecular_entropy.rotamer import RotamerEntropyCalculator

        # Test all bin centers
        for center in [-180, -120, -60, 0, 60, 120]:
            assert RotamerEntropyCalculator.angle_to_bin(center) == center

        # Test bin edges (should round to nearest center)
        # At exact midpoints (like 30, -150), either adjacent bin is valid
        assert RotamerEntropyCalculator.angle_to_bin(-150) in [-120, -180]  # midpoint
        assert RotamerEntropyCalculator.angle_to_bin(-151) == -180
        assert RotamerEntropyCalculator.angle_to_bin(-149) == -120
        assert RotamerEntropyCalculator.angle_to_bin(30) in [0, 60]  # midpoint
        assert RotamerEntropyCalculator.angle_to_bin(31) == 60
        assert RotamerEntropyCalculator.angle_to_bin(29) == 0

        # Test periodicity (angles outside -180 to 180)
        assert RotamerEntropyCalculator.angle_to_bin(360) == 0
        assert RotamerEntropyCalculator.angle_to_bin(-240) == 120
        assert RotamerEntropyCalculator.angle_to_bin(400) in [0, 60]  # 400 % 360 = 40

    def test_angle_to_bin_performance(self):
        """Test that O(1) angle binning is fast."""
        from molecular_entropy.rotamer import RotamerEntropyCalculator
        import time

        # Run many iterations to verify performance
        start = time.perf_counter()
        for _ in range(10000):
            RotamerEntropyCalculator.angle_to_bin(-123.5)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 0.1s for 10K calls)
        assert elapsed < 0.1, f"Angle binning too slow: {elapsed:.3f}s for 10K calls"

    def test_entropy_formula(self):
        """Test entropy calculation from probabilities."""
        from molecular_entropy.rotamer import RotamerEntropyCalculator
        from molecular_entropy.constants import R_KCAL

        # Uniform distribution: S = R * ln(n)
        probs_uniform = [0.25, 0.25, 0.25, 0.25]
        S = RotamerEntropyCalculator.compute_entropy_from_probs(probs_uniform)
        expected = R_KCAL * np.log(4)
        assert abs(S - expected) < 1e-10

        # Single state: S = 0
        probs_single = [1.0]
        S = RotamerEntropyCalculator.compute_entropy_from_probs(probs_single)
        assert abs(S) < 1e-10

        # Non-uniform: 0 < S < max
        probs_biased = [0.9, 0.1]
        S = RotamerEntropyCalculator.compute_entropy_from_probs(probs_biased)
        max_S = R_KCAL * np.log(2)
        assert 0 < S < max_S

    @requires_static
    @requires_rotlib
    def test_rotamer_binding_delta(self):
        """Test rotamer entropy change upon binding."""
        from molecular_entropy.structure import StructureLoader
        from molecular_entropy.rotamer import RotamerEntropyCalculator

        traj = StructureLoader.load(STATIC_PDB)
        calc = RotamerEntropyCalculator(rotlib_path=ROTLIB_SIMPLE)
        result = calc.calculate_binding_delta(traj, 'A', 'B')

        # Should have results for some residues
        assert len(result.per_residue) > 0

        # Total should be sum of per-residue
        total = sum(r.negT_dS_kcal for r in result.per_residue)
        assert abs(total - result.total_negT_dS_kcal) < 1e-10

    @requires_static
    @requires_rotlib
    def test_backbone_caching(self):
        """Test that backbone bin caching works correctly."""
        from molecular_entropy.structure import StructureLoader
        from molecular_entropy.rotamer import (
            RotamerEntropyCalculator,
            clear_backbone_cache
        )
        import time

        traj = StructureLoader.load(STATIC_PDB)
        calc = RotamerEntropyCalculator(rotlib_path=ROTLIB_SIMPLE)

        # Clear cache first
        clear_backbone_cache()

        # First call (uncached)
        start = time.perf_counter()
        bins1 = calc.compute_backbone_bins(traj, use_cache=True)
        first_call_time = time.perf_counter() - start

        # Second call (cached)
        start = time.perf_counter()
        bins2 = calc.compute_backbone_bins(traj, use_cache=True)
        second_call_time = time.perf_counter() - start

        # Results should be identical
        assert bins1 == bins2

        # Cached call should be faster (at least 2x)
        assert second_call_time < first_call_time * 0.5, \
            f"Cache not working: first={first_call_time:.4f}s, second={second_call_time:.4f}s"

        # Test cache bypass
        bins3 = calc.compute_backbone_bins(traj, use_cache=False)
        assert bins1 == bins3

        # Clean up
        clear_backbone_cache()

    @requires_static
    @requires_rotlib
    def test_kdtree_clash_detection(self):
        """Test KD-tree based clash detection."""
        from molecular_entropy.structure import StructureLoader
        from molecular_entropy.rotamer import RotamerEntropyCalculator

        traj = StructureLoader.load(STATIC_PDB)
        traj_a = StructureLoader.select_chain(traj, 'A')
        traj_b = StructureLoader.select_chain(traj, 'B')

        calc = RotamerEntropyCalculator(rotlib_path=ROTLIB_SIMPLE)

        # Build KD-tree once
        kdtree_b = calc._build_partner_kdtree(traj_b)

        # Test clash detection with and without pre-built KD-tree
        for res in traj.topology.residues:
            if res.chain.chain_id == 'A':
                # Both methods should return same result
                clash_with_tree = calc.check_clashes(
                    traj, res, traj_b, _partner_kdtree=kdtree_b
                )
                clash_without_tree = calc.check_clashes(
                    traj, res, traj_b, _partner_kdtree=None
                )
                assert clash_with_tree == clash_without_tree
                break  # Just test one residue


class TestBindingEntropyCalculator:
    """Tests for complete binding entropy calculations."""

    @requires_static
    def test_calculate_no_rotlib(self):
        """Test calculation without rotamer library."""
        from molecular_entropy import BindingEntropyCalculator

        calc = BindingEntropyCalculator(compute_rotamer=False)
        result = calc.calculate(STATIC_PDB, 'A', 'B')

        # Should have SASA and ANM results
        assert result.sasa is not None
        assert result.anm is not None
        assert result.rotamer is None

        # Total should be computed
        assert isinstance(result.total_negT_dS, float)

    @requires_static
    @requires_rotlib
    def test_calculate_full(self):
        """Test full calculation with all components."""
        from molecular_entropy import BindingEntropyCalculator

        calc = BindingEntropyCalculator(rotlib_path=ROTLIB_SIMPLE)
        result = calc.calculate(STATIC_PDB, 'A', 'B')

        # All components should be present
        assert result.sasa is not None
        assert result.anm is not None
        assert result.rotamer is not None

        # Total should be sum of components
        total = (
            result.negT_dS_sasa +
            result.negT_dS_vib +
            result.negT_dS_rotamer +
            result.negT_dS_TR
        )
        assert abs(total - result.total_negT_dS) < 1e-10

    @requires_static
    def test_to_dataframe(self):
        """Test converting results to DataFrame."""
        from molecular_entropy import BindingEntropyCalculator

        calc = BindingEntropyCalculator(compute_rotamer=False)
        result = calc.calculate(STATIC_PDB, 'A', 'B')
        df = result.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5  # 4 terms + total
        assert 'term' in df.columns
        assert 'value_kcal' in df.columns

    @requires_static
    def test_to_dict(self):
        """Test converting results to dict."""
        from molecular_entropy import BindingEntropyCalculator

        calc = BindingEntropyCalculator(compute_rotamer=False)
        result = calc.calculate(STATIC_PDB, 'A', 'B')
        d = result.to_dict()

        assert isinstance(d, dict)
        assert 'temperature_K' in d
        assert 'terms' in d
        assert 'total_negT_dS_kcal' in d['terms']


class TestDynamicModels:
    """Tests for dynamic trajectory analysis."""

    @requires_dynamic
    def test_load_dynamic_pdb(self):
        """Test loading dynamic PDB (multi-model)."""
        from molecular_entropy.structure import StructureLoader

        traj = StructureLoader.load(DYNAMIC_PDB)
        assert traj.n_frames >= 1
        assert traj.n_atoms > 0

    @requires_dynamic
    def test_sasa_dynamic(self):
        """Test SASA on dynamic structure."""
        from molecular_entropy.structure import StructureLoader
        from molecular_entropy.sasa import SASACalculator

        traj = StructureLoader.load(DYNAMIC_PDB)
        calc = SASACalculator()
        result = calc.calculate(traj)

        assert result.total > 0

    @requires_dynamic
    def test_anm_dynamic(self):
        """Test ANM on dynamic structure."""
        from molecular_entropy.anm import ANMEntropyCalculator

        calc = ANMEntropyCalculator()
        result = calc.calculate_from_pdb(DYNAMIC_PDB)

        assert result.n_modes > 0

    @pytest.mark.skipif(
        not (DYNAMIC_PDB.exists() and DYNAMIC_DCD.exists()),
        reason="Dynamic test data not found"
    )
    def test_load_trajectory(self):
        """Test loading DCD trajectory with topology."""
        from molecular_entropy.structure import StructureLoader

        traj = StructureLoader.load(DYNAMIC_DCD, topology=DYNAMIC_PDB)
        assert traj.n_frames > 0
        assert traj.n_atoms > 0


class TestIntegration:
    """Integration tests combining multiple components."""

    @requires_static
    @requires_rotlib
    def test_full_workflow_static(self):
        """Test complete workflow on static structure."""
        from molecular_entropy import BindingEntropyCalculator
        import tempfile

        # Calculate binding entropy
        calc = BindingEntropyCalculator(rotlib_path=ROTLIB_SIMPLE)
        result = calc.calculate(STATIC_PDB, 'A', 'B')

        # Verify reasonable values
        # Total -T*dS should typically be positive (entropy loss upon binding)
        # Components should have expected signs:
        # - solvent: negative (entropy gain from releasing water) with alpha_np < 0
        # - vibrational: typically positive (entropy loss)
        # - rotamer: typically positive (entropy loss)
        # - TR: positive (entropy loss)
        assert result.negT_dS_TR > 0  # Always positive by default

        # Save and verify files
        with tempfile.TemporaryDirectory() as tmpdir:
            calc.save_results(result, tmpdir)

            # Check files exist
            assert (Path(tmpdir) / "binding_entropy_summary.csv").exists()
            assert (Path(tmpdir) / "binding_entropy_results.json").exists()
            assert (Path(tmpdir) / "binding_entropy_rotamer.csv").exists()
            assert (Path(tmpdir) / "binding_entropy_sasa.csv").exists()

    @requires_static
    def test_reproducibility(self):
        """Test that calculations are reproducible."""
        from molecular_entropy import BindingEntropyCalculator

        calc = BindingEntropyCalculator(compute_rotamer=False)

        result1 = calc.calculate(STATIC_PDB, 'A', 'B')
        result2 = calc.calculate(STATIC_PDB, 'A', 'B')

        assert abs(result1.total_negT_dS - result2.total_negT_dS) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
