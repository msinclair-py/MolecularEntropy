#!/usr/bin/env python
"""
Performance profiling script for molecular_entropy package.

Provides multiple profiling approaches:
1. Component-level timing breakdown
2. cProfile statistical profiling
3. Detailed hot-path analysis

Usage:
    python scripts/profile_entropy.py --pdb data/static.pdb --chains A,B
    python scripts/profile_entropy.py --pdb data/static.pdb --chains A,B --detailed
    python scripts/profile_entropy.py --pdb data/static.pdb --chains A,B --cprofile
"""

import argparse
import cProfile
import pstats
import io
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from contextlib import contextmanager

import numpy as np


@dataclass
class TimingResult:
    """Stores timing for a single operation."""
    name: str
    elapsed_ms: float
    calls: int = 1

    @property
    def per_call_ms(self) -> float:
        return self.elapsed_ms / self.calls if self.calls > 0 else 0


@dataclass
class ProfileReport:
    """Complete profiling report."""
    total_time_ms: float
    timings: list[TimingResult] = field(default_factory=list)

    def add(self, name: str, elapsed_ms: float, calls: int = 1):
        self.timings.append(TimingResult(name, elapsed_ms, calls))

    def print_report(self):
        print("\n" + "=" * 70)
        print("PERFORMANCE PROFILE REPORT")
        print("=" * 70)

        # Sort by time descending
        sorted_timings = sorted(self.timings, key=lambda x: x.elapsed_ms, reverse=True)

        print(f"\n{'Component':<45} {'Time (ms)':>10} {'%':>8} {'Calls':>6}")
        print("-" * 70)

        for t in sorted_timings:
            pct = (t.elapsed_ms / self.total_time_ms * 100) if self.total_time_ms > 0 else 0
            print(f"{t.name:<45} {t.elapsed_ms:>10.2f} {pct:>7.1f}% {t.calls:>6}")

        print("-" * 70)
        print(f"{'TOTAL':<45} {self.total_time_ms:>10.2f} {'100.0':>7}%")
        print("=" * 70)

        # Identify bottlenecks
        print("\nBOTTLENECK ANALYSIS:")
        for t in sorted_timings[:3]:
            pct = (t.elapsed_ms / self.total_time_ms * 100) if self.total_time_ms > 0 else 0
            if pct > 10:
                print(f"  - {t.name}: {pct:.1f}% of total time")


@contextmanager
def timer(name: str, report: ProfileReport):
    """Context manager for timing a code block."""
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    report.add(name, elapsed_ms)


def profile_component_timing(
    pdb_path: Path,
    chain_a: str,
    chain_b: str,
    rotlib_path: Optional[Path] = None,
    n_iterations: int = 1,
) -> ProfileReport:
    """
    Profile each component of the binding entropy calculation.

    Returns detailed timing breakdown.
    """
    from molecular_entropy.structure import StructureLoader
    from molecular_entropy.sasa import SASACalculator
    from molecular_entropy.anm import ANMEntropyCalculator
    from molecular_entropy.rotamer import RotamerEntropyCalculator, clear_backbone_cache

    report = ProfileReport(total_time_ms=0)
    total_start = time.perf_counter()

    # Structure loading
    with timer("Structure loading (mdtraj)", report):
        traj = StructureLoader.load(pdb_path)

    # Chain selection
    with timer("Chain selection", report):
        traj_a = StructureLoader.select_chain(traj, chain_a)
        traj_b = StructureLoader.select_chain(traj, chain_b)

    # SASA calculations
    sasa_calc = SASACalculator()

    with timer("SASA: compute_sasa (complex)", report):
        for _ in range(n_iterations):
            sasa_complex = sasa_calc.compute_sasa(traj)

    with timer("SASA: compute_sasa (chain A)", report):
        for _ in range(n_iterations):
            sasa_a = sasa_calc.compute_sasa(traj_a)

    with timer("SASA: compute_sasa (chain B)", report):
        for _ in range(n_iterations):
            sasa_b = sasa_calc.compute_sasa(traj_b)

    with timer("SASA: split_polar_nonpolar", report):
        for _ in range(n_iterations):
            polar, nonpolar = sasa_calc.split_polar_nonpolar(traj, sasa_complex)

    with timer("SASA: calculate_binding_delta (full)", report):
        sasa_result = sasa_calc.calculate_binding_delta(traj, chain_a, chain_b)

    # ANM calculations
    anm_calc = ANMEntropyCalculator()

    with timer("ANM: calculate_binding_delta_from_pdb", report):
        anm_result = anm_calc.calculate_binding_delta_from_pdb(pdb_path, chain_a, chain_b)

    # Rotamer calculations (if library available)
    if rotlib_path and rotlib_path.exists():
        # Clear cache for accurate timing
        clear_backbone_cache()

        with timer("Rotamer: load_rotamer_library", report):
            rotamer_calc = RotamerEntropyCalculator(rotlib_path=rotlib_path)

        with timer("Rotamer: compute_backbone_bins (uncached)", report):
            clear_backbone_cache()
            bins = rotamer_calc.compute_backbone_bins(traj, use_cache=True)

        with timer("Rotamer: compute_backbone_bins (cached)", report):
            bins_cached = rotamer_calc.compute_backbone_bins(traj, use_cache=True)

        # Profile the full calculation
        clear_backbone_cache()
        with timer("Rotamer: calculate_binding_delta (full)", report):
            rotamer_result = rotamer_calc.calculate_binding_delta(traj, chain_a, chain_b)

        # Profile sub-components
        profile_rotamer_details(traj, chain_a, chain_b, rotamer_calc, report)

    report.total_time_ms = (time.perf_counter() - total_start) * 1000
    return report


def profile_rotamer_details(traj, chain_a, chain_b, calc, report: ProfileReport):
    """Profile detailed rotamer calculation sub-components."""
    from molecular_entropy.structure import StructureLoader
    from molecular_entropy.rotamer import _USE_RUST, clear_backbone_cache
    from molecular_entropy.constants import CHI_RESIDUES, R_KCAL

    clear_backbone_cache()

    # Get trajectories
    traj_a = StructureLoader.select_chain(traj, chain_a)
    traj_b = StructureLoader.select_chain(traj, chain_b)

    # Heavy atom extraction
    with timer("Rotamer: extract heavy atoms", report):
        heavy_a = [a.index for a in traj_a.topology.atoms
                   if a.element is not None and a.element.symbol != 'H']
        heavy_b = [a.index for a in traj_b.topology.atoms
                   if a.element is not None and a.element.symbol != 'H']
        partner_coords_a = np.ascontiguousarray(traj_a.xyz[0][heavy_a], dtype=np.float64)
        partner_coords_b = np.ascontiguousarray(traj_b.xyz[0][heavy_b], dtype=np.float64)

    # Backbone bins
    with timer("Rotamer: backbone bins computation", report):
        backbone_bins = calc.compute_backbone_bins(traj)

    # Collect residue data
    with timer("Rotamer: collect residue data", report):
        backbone_names = {'N', 'CA', 'C', 'O', 'OXT', 'H', 'H1', 'H2', 'H3'}
        residue_data = []

        for res in traj.topology.residues:
            if res.name not in CHI_RESIDUES:
                continue
            if res.chain.chain_id not in {chain_a, chain_b}:
                continue

            phi_bin, psi_bin = backbone_bins.get(res.index, (None, None))
            if phi_bin is None or psi_bin is None:
                continue

            key = (res.name, phi_bin, psi_bin)
            rotamers = calc.rotlib_index.get(key)
            if not rotamers:
                continue

            sc_atoms = [a.index for a in res.atoms if a.name not in backbone_names]
            if not sc_atoms:
                continue

            residue_data.append({
                'res': res,
                'sc_atom_range': (min(sc_atoms), max(sc_atoms) + 1),
                'rotamers': rotamers,
                'probs': np.array([r['p'] for r in rotamers], dtype=np.float64),
            })

    print(f"\n  [INFO] Found {len(residue_data)} residues with rotamers")

    # Clash detection
    if _USE_RUST:
        from molecular_entropy._core import check_clashes_batch, compute_rotamer_entropies_batch

        complex_coords = np.ascontiguousarray(traj.xyz[0], dtype=np.float64)
        threshold_nm = calc.clash_distance / 10.0

        # Split by chain
        data_a = [d for d in residue_data if d['res'].chain.chain_id == chain_a]
        data_b = [d for d in residue_data if d['res'].chain.chain_id == chain_b]

        with timer("Rotamer: Rust check_clashes_batch (chain A)", report):
            if data_a:
                ranges_a = [d['sc_atom_range'] for d in data_a]
                clashes_a = check_clashes_batch(complex_coords, partner_coords_b, ranges_a, threshold_nm)

        with timer("Rotamer: Rust check_clashes_batch (chain B)", report):
            if data_b:
                ranges_b = [d['sc_atom_range'] for d in data_b]
                clashes_b = check_clashes_batch(complex_coords, partner_coords_a, ranges_b, threshold_nm)

        # Entropy computation
        with timer("Rotamer: build feasible masks", report):
            prob_arrays = []
            feasible_masks = []
            all_clashes = (list(clashes_a) if data_a else []) + (list(clashes_b) if data_b else [])
            all_data = data_a + data_b
            for data, has_clash in zip(all_data, all_clashes):
                probs = data['probs']
                mask = calc.heuristic_feasible_mask(has_clash, data['rotamers'])
                prob_arrays.append(probs)
                feasible_masks.append(mask)

        with timer("Rotamer: Rust compute_rotamer_entropies_batch", report):
            if prob_arrays:
                s_unbound, s_bound = compute_rotamer_entropies_batch(prob_arrays, feasible_masks, R_KCAL)
    else:
        print("  [INFO] Rust extension not available, using Python fallback")
        with timer("Rotamer: Python clash detection (KD-tree)", report):
            kdtree_b = calc._build_partner_kdtree(traj_b)
            for res in traj.topology.residues:
                if res.name in CHI_RESIDUES and res.chain.chain_id == chain_a:
                    calc.check_clashes(traj, res, traj_b, _partner_kdtree=kdtree_b)


def run_cprofile(
    pdb_path: Path,
    chain_a: str,
    chain_b: str,
    rotlib_path: Optional[Path] = None,
    top_n: int = 30,
):
    """Run cProfile statistical profiler."""
    from molecular_entropy import BindingEntropyCalculator

    print("\n" + "=" * 70)
    print("cProfile STATISTICAL PROFILING")
    print("=" * 70)

    profiler = cProfile.Profile()

    profiler.enable()
    calc = BindingEntropyCalculator(
        rotlib_path=rotlib_path,
        compute_rotamer=rotlib_path is not None,
    )
    result = calc.calculate(pdb_path, chain_a, chain_b)
    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(top_n)
    print(s.getvalue())

    print("\nSorted by time spent in function (tottime):")
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
    ps2.print_stats(top_n)
    print(s2.getvalue())

    return result


def profile_rust_vs_python(
    pdb_path: Path,
    chain_a: str,
    chain_b: str,
    rotlib_path: Path,
):
    """Compare Rust vs Python implementation speeds."""
    from molecular_entropy.structure import StructureLoader
    from molecular_entropy.rotamer import RotamerEntropyCalculator, clear_backbone_cache, _USE_RUST

    print("\n" + "=" * 70)
    print("RUST vs PYTHON COMPARISON")
    print("=" * 70)

    if not _USE_RUST:
        print("Rust extension not available - cannot compare")
        return

    traj = StructureLoader.load(pdb_path)
    calc = RotamerEntropyCalculator(rotlib_path=rotlib_path)

    # Force Python path
    import molecular_entropy.rotamer as rotamer_module
    original_use_rust = rotamer_module._USE_RUST

    # Python timing
    rotamer_module._USE_RUST = False
    clear_backbone_cache()
    start = time.perf_counter()
    result_py = calc.calculate_binding_delta(traj, chain_a, chain_b)
    python_time = (time.perf_counter() - start) * 1000

    # Rust timing
    rotamer_module._USE_RUST = True
    clear_backbone_cache()
    start = time.perf_counter()
    result_rust = calc.calculate_binding_delta(traj, chain_a, chain_b)
    rust_time = (time.perf_counter() - start) * 1000

    # Restore
    rotamer_module._USE_RUST = original_use_rust

    print(f"\nPython implementation: {python_time:.2f} ms")
    print(f"Rust implementation:   {rust_time:.2f} ms")
    print(f"Speedup:               {python_time / rust_time:.1f}x")

    # Verify results match
    if abs(result_py.total_negT_dS_kcal - result_rust.total_negT_dS_kcal) < 1e-6:
        print("Results match: YES")
    else:
        print(f"Results differ: Python={result_py.total_negT_dS_kcal:.6f}, Rust={result_rust.total_negT_dS_kcal:.6f}")


def profile_memory(
    pdb_path: Path,
    chain_a: str,
    chain_b: str,
    rotlib_path: Optional[Path] = None,
):
    """Profile memory usage (requires memory_profiler)."""
    try:
        from memory_profiler import memory_usage
    except ImportError:
        print("memory_profiler not installed. Install with: pip install memory-profiler")
        return

    from molecular_entropy import BindingEntropyCalculator

    print("\n" + "=" * 70)
    print("MEMORY PROFILING")
    print("=" * 70)

    def run_calculation():
        calc = BindingEntropyCalculator(
            rotlib_path=rotlib_path,
            compute_rotamer=rotlib_path is not None,
        )
        return calc.calculate(pdb_path, chain_a, chain_b)

    mem_usage = memory_usage((run_calculation,), interval=0.1, max_iterations=1)

    print(f"\nPeak memory usage: {max(mem_usage):.1f} MB")
    print(f"Memory at start:   {mem_usage[0]:.1f} MB")
    print(f"Memory increase:   {max(mem_usage) - mem_usage[0]:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Profile molecular_entropy performance'
    )
    parser.add_argument('--pdb', required=True, help='PDB file path')
    parser.add_argument('--chains', required=True, help='Chain IDs (e.g., A,B)')
    parser.add_argument('--rotlib', help='Path to rotamer library')
    parser.add_argument('--cprofile', action='store_true', help='Run cProfile')
    parser.add_argument('--detailed', action='store_true', help='Detailed timing')
    parser.add_argument('--compare', action='store_true', help='Compare Rust vs Python')
    parser.add_argument('--memory', action='store_true', help='Profile memory usage')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for timing')

    args = parser.parse_args()

    pdb_path = Path(args.pdb)
    chain_a, chain_b = [c.strip() for c in args.chains.split(',')]
    rotlib_path = Path(args.rotlib) if args.rotlib else None

    # Auto-detect rotamer library
    if rotlib_path is None:
        default_rotlib = Path(__file__).parent.parent / "rotamer_library" / "simple_library.parquet"
        if default_rotlib.exists():
            rotlib_path = default_rotlib
            print(f"Using default rotamer library: {rotlib_path}")

    print(f"\nProfiling: {pdb_path}")
    print(f"Chains: {chain_a}, {chain_b}")
    print(f"Rotamer library: {rotlib_path or 'None'}")

    # Run requested profiles
    if args.cprofile:
        run_cprofile(pdb_path, chain_a, chain_b, rotlib_path)

    if args.compare and rotlib_path:
        profile_rust_vs_python(pdb_path, chain_a, chain_b, rotlib_path)

    if args.memory:
        profile_memory(pdb_path, chain_a, chain_b, rotlib_path)

    # Always run component timing
    report = profile_component_timing(
        pdb_path, chain_a, chain_b, rotlib_path, args.iterations
    )
    report.print_report()

    # Additional recommendations
    print("\nPOTENTIAL OPTIMIZATION OPPORTUNITIES:")
    print("-" * 70)

    recommendations = []
    for t in sorted(report.timings, key=lambda x: x.elapsed_ms, reverse=True):
        pct = (t.elapsed_ms / report.total_time_ms * 100) if report.total_time_ms > 0 else 0

        if "ANM" in t.name and pct > 20:
            recommendations.append(
                f"- ANM ({pct:.1f}%): Consider pre-computing eigenvalues or using "
                "sparse matrix methods for large proteins"
            )
        elif "SASA" in t.name and pct > 20:
            recommendations.append(
                f"- SASA ({pct:.1f}%): Already using Rust backend. Consider reducing "
                "n_sphere_points if accuracy permits"
            )
        elif "load_rotamer_library" in t.name and pct > 10:
            recommendations.append(
                f"- Rotamer library loading ({pct:.1f}%): Consider caching the loaded "
                "library if processing multiple structures"
            )
        elif "Structure loading" in t.name and pct > 15:
            recommendations.append(
                f"- Structure loading ({pct:.1f}%): MDTraj I/O overhead. Consider "
                "pre-loading structures for batch processing"
            )

    if recommendations:
        for r in recommendations:
            print(r)
    else:
        print("No major bottlenecks identified. The code is well-optimized!")


if __name__ == '__main__':
    main()
