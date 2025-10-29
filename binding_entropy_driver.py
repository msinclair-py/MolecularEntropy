import argparse
import subprocess
from pathlib import Path
import json
import pandas as pd

# Defaults
DEFAULT_TR_KCAL = 8.0 # -TΔS_trans+rot at 298 K (rule-of-thumb)
DEFAULT_ALPHA_NP = 0.0 # kcal/mol per Å^2 of nonpolar ΔSASA (set >0 to include hydrophobic water-release entropy)
DEFAULT_BETA_POL = 0.0 # kcal/mol per Å^2 of polar ΔSASA (often ~0 to slightly negative)

def run(cmd):
    print("[run]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)

def read_delta_sasa(outdir: Path):
    # totals
    totals = pd.read_csv(outdir / 'delta_sasa_summary.csv')
    row_delta = totals.loc[totals['context'] == 'delta (unbound - bound)']
    d_total = float(row_delta['SASA_total_A2'])
    d_polar = float(row_delta['SASA_polar_A2'])
    d_nonpolar = float(row_delta['SASA_nonpolar_A2'])
    
    # per-residue (strict file) if present
    perres = None
    perres_path = outdir / 'delta_sasa_per_residue.csv'
    if perres_path.exists():
        perres = pd.read_csv(perres_path)

    return d_total, d_polar, d_nonpolar, perres

def read_rotamer_entropy(outdir: Path):
    summ = pd.read_csv(outdir / 'rotamer_entropy_summary.csv')
    # We wrote two rows: sum_dS_kcal_per_mol and sum_-T_dS_kcal_per_mol_at_298K
    dS_sum = float(summ.loc[summ['metric']=='sum_dS_kcal_per_mol','value'])
    negT_dS = float(summ.loc[summ['metric']=='sum_-T_dS_kcal_per_mol_at_298K','value'])
    return dS_sum, negT_dS

def read_anm(outcsv: Path, T: float):
    df = pd.read_csv(outcsv)
    # last row has -TΔS_vib (kcal/mol)
    val = df.loc[df['term']=='-TΔS_vib (kcal/mol)','kcal_per_K'].values
    if len(val):
        return float(val[0])
    
    # fallback: compute from ΔS row
    dS_kcal_per_K = df.loc[df['term']=='ΔS_vib','kcal_per_K'].values
    if len(dS_kcal_per_K):
        return -T * float(dS_kcal_per_K[0])

    return 0.0

def main():
    ap = argparse.ArgumentParser(description='End-to-end binding entropy driver (SASA + Dunbrack + ANM)')
    ap.add_argument('--pdb', required=True)
    ap.add_argument('--chains', required=True, help='e.g., A,B')
    ap.add_argument('--rotlib', required=True, help='Path to Dunbrack bbdep JSON/TSV (JSON per line)')
    ap.add_argument('--out', default='results')
    ap.add_argument('--T', type=float, default=298.15)
    
    
    # SASA options
    ap.add_argument('--probe-radius', type=float, default=1.4)
    ap.add_argument('--interface-threshold', type=float, default=5.0)
    
    
    # ANM options
    ap.add_argument('--anm-cutoff', type=float, default=15.0)
    ap.add_argument('--anm-gamma', type=float, default=1.0)
    
    
    # Thermodynamic mixing options
    ap.add_argument('--tr_penalty', type=float, default=DEFAULT_TR_KCAL, help='-TΔS_trans+rot to add (kcal/mol), set 0 to omit')
    ap.add_argument('--alpha_np', type=float, default=DEFAULT_ALPHA_NP, help='kcal/mol per Å^2 for nonpolar ΔSASA')
    ap.add_argument('--beta_pol', type=float, default=DEFAULT_BETA_POL, help='kcal/mol per Å^2 for polar ΔSASA')
    
    
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # 1) SASA
    run([
        'python', 'compute_sasa.py',
        '--pdb', args.pdb,
        '--chains', args.chains,
        '--probe-radius', str(args.probe_radius),
        '--interface-threshold', str(args.interface_threshold),
        '--out', str(outdir)
    ])

    d_total, d_polar, d_nonpolar, _ = read_delta_sasa(outdir)
    
    
    # 2) Rotamer entropy (Dunbrack)
    run([
        'python', 'rotamer_entropy_dunbrack.py',
        '--pdb', args.pdb,
        '--chains', args.chains,
        '--rotlib', args.rotlib,
        '--T', str(args.T),
        '--out', str(outdir)
    ])

    dS_sc, negT_dS_sc = read_rotamer_entropy(outdir)
    
    # 3) ANM vibrational entropy
    anm_csv = outdir / 'anm_entropy.csv'
    run([
        'python', 'anm_entropy.py',
        '--pdb', args.pdb,
        '--chains', args.chains,
        '--cutoff', str(args.anm_cutoff),
        '--gamma', str(args.anm_gamma),
        '--T', str(args.T),
        '--out', str(anm_csv)
    ])

    negT_dS_vib = read_anm(anm_csv, args.T)
    
    # 4) Optional solvent entropy from SASA (empirical)
    negT_dS_solv = - (args.alpha_np * d_nonpolar + args.beta_pol * d_polar)
    
    # 5) Optional TR penalty
    negT_dS_TR = float(args.tr_penalty)
    
    # 6) Collate summary
    terms = [
        ('-TΔS_sidechain (Dunbrack)', negT_dS_sc),
        ('-TΔS_vibrational (ANM)', negT_dS_vib),
        ('-TΔS_solv (from ΔSASA)', negT_dS_solv),
        ('-TΔS_trans+rot (user)', negT_dS_TR),
    ]
    
    total = sum(v for _, v in terms)

    # Write summary CSV and markdown
    summary_rows = [{'term': name, 'value_kcal_per_mol': v} for name, v in terms]
    summary_rows.append({'term': 'TOTAL -TΔS (kcal/mol)', 'value_kcal_per_mol': total})
    
    # Also echo ΔSASA totals
    meta = {
        'ΔSASA_total_A2': d_total,
        'ΔSASA_polar_A2': d_polar,
        'ΔSASA_nonpolar_A2': d_nonpolar,
        'T_K': args.T,
        'alpha_np_kcal_per_A2': args.alpha_np,
        'beta_pol_kcal_per_A2': args.beta_pol,
        'tr_penalty_kcal': args.tr_penalty,
        'anm_cutoff_A': args.anm_cutoff,
        'anm_gamma': args.anm_gamma,
        'probe_radius_A': args.probe_radius,
    }
    
    pd.DataFrame(summary_rows).to_csv(outdir / 'binding_entropy_summary.csv', index=False)
    with open(outdir / 'binding_entropy_meta.json','w') as fh:
        json.dump(meta, fh, indent=2)

    md_report = [
        f"# Binding Entropy Report",
        f"- PDB: `{args.pdb}`",
        f"- Chains: `{args.chains}`",
        f"- Temperature: {args.T:.2f} K",
        f"## ΔSASA totals (Å²)",
        f"- ΔSASA_total: {d_total:.1f}",
        f"- ΔSASA_polar: {d_polar:.1f}",
        f"- ΔSASA_nonpolar: {d_nonpolar:.1f}",
        f"## Entropy contributions (-TΔS, kcal/mol)",
    ]

    for name, v in terms:
        md_report.append(f"- {name}: {v:.3f}")
    
    md_report.append(f"**TOTAL -TΔS:** {total:.3f} kcal/mol")
    
    (outdir / 'binding_entropy_report.md').write_text(''.join(md_report))
    print('Wrote:', outdir / 'binding_entropy_summary.csv')
    print('Wrote:', outdir / 'binding_entropy_meta.json')
    print('Wrote:', outdir / 'binding_entropy_report.md')

if __name__ == '__main__':
    main()
