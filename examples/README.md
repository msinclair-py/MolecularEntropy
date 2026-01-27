# Molecular Entropy Examples

This directory contains examples demonstrating how to use the Molecular Entropy library.

## Examples

| File | Description |
|------|-------------|
| `01_basic_usage.py` | Simplest way to calculate binding entropy for a complex |
| `02_custom_parameters.py` | Customize temperature, cutoffs, and skip components |
| `03_individual_calculators.py` | Use SASA, ANM, and Rotamer calculators separately |
| `04_batch_processing.py` | Process multiple PDB files and generate summary |
| `05_trajectory_analysis.py` | Analyze MD trajectories for entropy fluctuations |
| `06_command_line.sh` | Command-line interface usage examples |

## Quick Start

```python
from molecular_entropy import BindingEntropyCalculator

calc = BindingEntropyCalculator(
    rotlib_path="rotamer_library/simple_library.parquet"
)
result = calc.calculate("complex.pdb", chain_a="A", chain_b="B")
print(f"Total -T*dS: {result.total_negT_dS:.2f} kcal/mol")
```

## Requirements

- Python 3.10+
- Install with: `pip install molecular-entropy`
- Or from source: `maturin develop --release`

## Input Format

- **PDB files**: Standard PDB format with chain identifiers
- **Trajectories**: DCD, XTC, TRR formats (requires topology PDB)
- **Rotamer library**: Parquet files in `rotamer_library/` directory

## Output

Results include four entropy contributions (in kcal/mol):

- **SASA**: Solvent entropy from buried surface area
- **ANM**: Vibrational entropy from normal modes
- **Rotamer**: Side-chain conformational entropy
- **Trans/Rot**: Penalty for loss of translational/rotational freedom
