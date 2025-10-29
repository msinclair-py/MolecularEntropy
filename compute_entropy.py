import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import mdtraj as md

# --- Atom typing for polar/nonpolar split ---
POLAR_ELEMENTS = {"O", "N", "S"} # crude but effective for SASA partition

def load_structure(pdb_path):
    traj = md.load(pdb_path)
    return traj

def select_chains(traj, chains_csv):
    chains = [c.strip() for c in chains_csv.split(',') if c.strip()]
    sel = []
    for c in chains:
        sel.append(traj.topology.select(f"chainid == {chain_id_from_label(traj, c)}"))
    
    sel_idx = np.concatenate(sel).astype(int)
    return traj.atom_slice(sel_idx)

def chain_id_from_label(traj, chain_label):
    # Map PDB chain IDs (A,B,...) to mdtraj chain indices
    for chain in traj.topology.chains:
        if chain.chain_id == chain_label:
            return chain.index
    
    # Fallback: try label via 'chain' property if available
    for chain in traj.topology.chains:
        if getattr(chain, 'id', None) == chain_label:
            return chain.index

    raise ValueError(f"Chain label '{chain_label}' not found in topology.")

def write_subpdb(traj, out_path):
    traj.save_pdb(out_path)

def compute_freesasa_sasa(traj, probe_radius=1.4):
    """Return per-atom SASA array (nm^2) computed with FreeSASA via MDTraj.
    MDTraj returns nm^2; convert to Å^2 for reporting.
    """
    sasa = md.shrake_rupley(traj, probe_radius=probe_radius)
    # Convert nm^2 -> Å^2 (1 nm^2 = 100 Å^2)
    return sasa[0] * 100.0

def per_residue_sasa(traj, per_atom_sasa):
    res_sasa = np.zeros(traj.topology.n_residues)
    for atom in traj.topology.atoms:
        res_sasa[atom.residue.index] += per_atom_sasa[atom.index]

    return res_sasa

def polar_nonpolar_split(traj, per_atom_sasa):
    polar = np.zeros_like(per_atom_sasa)
    nonpolar = np.zeros_like(per_atom_sasa)
    for atom in traj.topology.atoms:
        elem = (atom.element.symbol if atom.element is not None else '')
        if elem in POLAR_ELEMENTS:
            polar[atom.index] = per_atom_sasa[atom.index]
        else:
            nonpolar[atom.index] = per_atom_sasa[atom.index]

    return polar, nonpolar

def dataframe_per_atom(traj, sasa_total, sasa_polar, sasa_nonpolar, tag):
    rows = []
    for atom in traj.topology.atoms:
        rows.append({
            'context': tag,
            'atom_index': atom.index,
            'resSeq': atom.residue.resSeq,
            'resName': atom.residue.name,
            'chain_id': atom.residue.chain.chain_id,
            'atomName': atom.name,
            'element': (atom.element.symbol if atom.element else ''),
            'sasa_total_A2': float(sasa_total[atom.index]),
            'sasa_polar_A2': float(sasa_polar[atom.index]),
            'sasa_nonpolar_A2': float(sasa_nonpolar[atom.index]),
        })

    return pd.DataFrame(rows)

def dataframe_per_residue(traj, sasa_total, tag):
    rows = []
    res_total = per_residue_sasa(traj, sasa_total)
    for res in traj.topology.residues:
        rows.append({
            'context': tag,
            'resSeq': res.resSeq,
            'resName': res.name,
            'chain_id': res.chain.chain_id,
            'sasa_res_total_A2': float(res_total[res.index]),
        })

    return pd.DataFrame(rows)

def detect_interface(res_df_bound, res_df_unboundA, res_df_unboundB, delta_threshold=1.0):
    """ΔSASA per residue computed from matching (chain_id, resSeq).
    Positive ΔSASA (A+B -> AB) indicates burial upon binding. threshold in Å^2.
    """
    key = ['chain_id', 'resSeq', 'resName']
    # Build unbound combined table with zeros for non-present residues
    unbound = pd.concat([res_df_unboundA, res_df_unboundB], ignore_index=True)
    un = unbound.groupby(key, as_index=False)['sasa_res_total_A2'].sum()
    
    bd = res_df_bound.groupby(key, as_index=False)['sasa_res_total_A2'].sum()
    
    merged = pd.merge(un, bd, on=key, how='outer', suffixes=('_unbound', '_bound')).fillna(0.0)
    merged['delta_sasa_A2'] = merged['sasa_res_total_A2_unbound'] - merged['sasa_res_total_A2_bound']
    interface = merged.loc[merged['delta_sasa_A2'] >= delta_threshold].copy()
    interface.sort_values(['chain_id','resSeq'], inplace=True)

    return interface, merged

def summarize_totals(df_bound_atoms, df_unboundA_atoms, df_unboundB_atoms):
    def totals(df):
        return pd.Series({
            'SASA_total_A2': df['sasa_total_A2'].sum(),
            'SASA_polar_A2': df['sasa_polar_A2'].sum(),
            'SASA_nonpolar_A2': df['sasa_nonpolar_A2'].sum(),
        })

    tb = totals(df_bound_atoms)
    ta = totals(df_unboundA_atoms)
    tbm = totals(df_unboundB_atoms)
    
    unbound_tot = ta + tbm
    delta = unbound_tot - tb
    out = pd.DataFrame([
        pd.Series({'context': 'unbound_total'}) | unbound_tot,
        pd.Series({'context': 'bound'}) | tb,
        pd.Series({'context': 'delta (unbound - bound)'}) | delta,
    ])

    return out

def main():
    ap = argparse.ArgumentParser(description='Compute SASA & interface for complex vs monomers')
    ap.add_argument('--pdb', required=True, help='Complex PDB path')
    ap.add_argument('--chains', required=True, help='Comma-separated chain labels for the two partners, e.g., A,B')
    ap.add_argument('--probe-radius', type=float, default=1.4, help='Probe radius in Å (1.4 Å = water)')
    ap.add_argument('--interface-threshold', type=float, default=1.0, help='Residue ΔSASA threshold (Å^2)')
    ap.add_argument('--out', required=True, help='Output directory')
    args = ap.parse_args()
    
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    
    complex_traj = load_structure(args.pdb)
    chains = [x.strip() for x in args.chains.split(',')]
    if len(chains) != 2:
        raise SystemExit('Please pass exactly two chains, e.g. --chains A,B')
    
    # Build monomer A and B by deleting the partner
    A_traj = select_chains(complex_traj, chains[0])
    B_traj = select_chains(complex_traj, chains[1])
    
    # Compute SASA for complex and monomers
    sasa_bound = compute_freesasa_sasa(complex_traj, probe_radius=args.probe_radius)
    pb, nb = polar_nonpolar_split(complex_traj, sasa_bound)
    df_bound_atoms = dataframe_per_atom(complex_traj, sasa_bound, pb, nb, tag='bound')
    df_bound_atoms.to_csv(outdir / 'sasa_bound.csv', index=False)
    
    # Per-residue bound
    df_bound_res = dataframe_per_residue(complex_traj, sasa_bound, tag='bound')
    
    # A alone
    sasa_A = compute_freesasa_sasa(A_traj, probe_radius=args.probe_radius)
    pA, nA = polar_nonpolar_split(A_traj, sasa_A)
    df_A_atoms = dataframe_per_atom(A_traj, sasa_A, pA, nA, tag='A_unbound')
    df_A_atoms.to_csv(outdir / 'sasa_unbound_A.csv', index=False)
    df_A_res = dataframe_per_residue(A_traj, sasa_A, tag='A_unbound')
    
    # B alone
    sasa_B = compute_freesasa_sasa(B_traj, probe_radius=args.probe_radius)
    pB, nB = polar_nonpolar_split(B_traj, sasa_B)
    df_B_atoms = dataframe_per_atom(B_traj, sasa_B, pB, nB, tag='B_unbound')
    df_B_atoms.to_csv(outdir / 'sasa_unbound_B.csv', index=False)
    df_B_res = dataframe_per_residue(B_traj, sasa_B, tag='B_unbound')
    
    # Interface residues by ΔSASA
    interface_df, merged_res = detect_interface(df_bound_res, df_A_res, df_B_res,
    delta_threshold=args.interface_threshold)
    interface_df.to_csv(outdir / 'interface_residues.csv', index=False)
    merged_res.to_csv(outdir / 'residue_sasa_merged.csv', index=False)
    
    # Totals
    totals_df = summarize_totals(df_bound_atoms, df_A_atoms, df_B_atoms)
    totals_df.to_csv(outdir / 'delta_sasa_summary.csv', index=False)
    
    print('Wrote:',
          outdir / 'sasa_bound.csv',
          outdir / 'sasa_unbound_A.csv',
          outdir / 'sasa_unbound_B.csv',
          outdir / 'interface_residues.csv',
          outdir / 'delta_sasa_summary.csv')

if __name__ == '__main__':
    main()
