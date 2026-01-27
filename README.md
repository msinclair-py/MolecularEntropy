# Molecular Entropy

Compute entropic contributions to protein binding free energy.

## Features

- **SASA-based solvent entropy**: Shrake-Rupley algorithm via Rust (fast)
- **ANM vibrational entropy**: Sparse eigensolvers for large proteins (20x faster than dense)
- **Side-chain rotamer entropy**: Dunbrack backbone-dependent library
- **Translational/rotational entropy**: Configurable penalty term

## Installation

### From PyPI (recommended)

```bash
pip install molecular-entropy
```

Pre-built wheels are available for:
- Linux (x86_64, aarch64)
- macOS (x86_64, Apple Silicon)
- Windows (x86_64)

### From source (requires Rust)

```bash
# Clone the repository
git clone https://github.com/msinclair-py/MolecularEntropy.git
cd MolecularEntropy

# Install with maturin (builds Rust extension)
pip install maturin
maturin develop --release

# Or install in editable mode
pip install -e .
```

### Development installation

```bash
pip install molecular-entropy[dev]
```

## Quick Start

### Python API

```python
from molecular_entropy import BindingEntropyCalculator

# Initialize calculator with rotamer library
calc = BindingEntropyCalculator(
    rotlib_path="rotamer_library/simple_library.parquet"
)

# Calculate binding entropy for a protein complex
result = calc.calculate("complex.pdb", chain_a="A", chain_b="B")

# View results
print(result.to_dataframe())
print(f"Total -T*dS = {result.total_negT_dS:.2f} kcal/mol")
```

### Command Line

```bash
molecular-entropy --pdb complex.pdb --chains A,B --rotlib rotamer_library/simple_library.parquet
```

### Individual Calculators

```python
from molecular_entropy import SASACalculator, ANMEntropyCalculator, RotamerEntropyCalculator
import mdtraj as md

# Load structure
traj = md.load("complex.pdb")

# SASA-based solvent entropy
sasa_calc = SASACalculator()
sasa_result = sasa_calc.calculate_binding_delta(traj, chain_a="A", chain_b="B")
print(f"Solvent: -T*dS = {sasa_result.negT_dS_solv:.2f} kcal/mol")

# ANM vibrational entropy (uses sparse solver by default - 20x faster)
anm_calc = ANMEntropyCalculator()
anm_result = anm_calc.calculate_binding_delta(traj, chain_a="A", chain_b="B")
print(f"Vibrational: -T*dS = {anm_result.negT_dS_kcal:.2f} kcal/mol")

# Rotamer entropy
rot_calc = RotamerEntropyCalculator(rotlib_path="rotamer_library/simple_library.parquet")
rot_result = rot_calc.calculate_binding_delta(traj, chain_a="A", chain_b="B")
print(f"Side-chain: -T*dS = {rot_result.total_negT_dS_kcal:.2f} kcal/mol")
```

## Performance

Benchmarked on a 1,448 CA atom protein complex:

| Component | Time | Notes |
|-----------|------|-------|
| ANM (sparse) | 270ms | ARPACK shift-invert eigensolver |
| SASA | 10ms | Rust KD-tree optimization |
| Rotamer | 20ms | Rust batch processing |
| **Total** | **~350ms** | 20x faster than naive implementation |

The ANM eigensolver uses scipy's ARPACK wrapper which is essentially optimal for sparse symmetric eigenvalue problems.

## Dependencies

- mdtraj - Structure loading
- numpy, scipy - Numerical computing (ARPACK eigensolver)
- polars - DataFrames
- rust-simulation-tools - SASA calculation

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Build wheels
maturin build --release
```

## License

MIT

## Citation

If you use this software, please cite:

```bibtex
@software{molecular_entropy,
  author = {Sinclair, Matt},
  title = {Molecular Entropy: Binding entropy calculations for protein complexes},
  url = {https://github.com/msinclair-py/MolecularEntropy}
}
```
