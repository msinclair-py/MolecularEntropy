"""
Example of analyzing a molecular dynamics trajectory.

Calculates binding entropy for multiple frames to assess
conformational sampling and entropy fluctuations.
"""

from molecular_entropy import BindingEntropyCalculator
import polars as pl

# Initialize calculator
calc = BindingEntropyCalculator(
    rotlib_path="rotamer_library/simple_library.parquet"
)

# Analyze trajectory
# Supports common formats: DCD, XTC, TRR, etc.
results = calc.calculate_from_trajectory(
    traj_path="data/dynamic.dcd",
    topology_path="data/dynamic.pdb",
    chain_a="A",
    chain_b="B",
    # Optional: analyze specific frames (default: all frames)
    # Optional: stride for analyzing every Nth frame
    # stride=10,
)

# Results is a list of BindingEntropyResult objects
print(f"Analyzed {len(results)} frames\n")

# Collect data for analysis
frame_data = []
for i, result in enumerate(results):
    frame_data.append({
        "frame": i,
        "sasa": result.negT_dS_sasa,
        "anm": result.negT_dS_vib,
        "rotamer": result.negT_dS_rotamer,
        "total": result.total_negT_dS,
    })

# Create DataFrame for analysis
df = pl.DataFrame(frame_data)

# Print statistics
print("Entropy Statistics Across Trajectory")
print("=" * 50)
print(df.select([
    pl.col("sasa").mean().alias("SASA (mean)"),
    pl.col("sasa").std().alias("SASA (std)"),
    pl.col("anm").mean().alias("ANM (mean)"),
    pl.col("anm").std().alias("ANM (std)"),
    pl.col("rotamer").mean().alias("Rotamer (mean)"),
    pl.col("rotamer").std().alias("Rotamer (std)"),
    pl.col("total").mean().alias("Total (mean)"),
    pl.col("total").std().alias("Total (std)"),
]))

# Save to CSV
df.write_csv("trajectory_entropy.csv")
print("\nResults saved to trajectory_entropy.csv")
