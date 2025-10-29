import argparse
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import mdtraj as md

R = 1.98720425864083e-3 # kcal/mol/K

# Residues with side-chain chi angles (skip GLY, ALA, etc.)
CHI_RES = {
    'ARG','LYS','GLU','GLN','MET','ILE','LEU','VAL','THR','SER','TYR',
    'PHE','TRP','ASN','ASP','HIS','CYS','PRO'
}

def load_traj(pdb_path):
    return md.load(pdb_path)

def select_chain(traj, label):
    for chain in traj.topology.chains:
        if chain.chain_id == label:
            atom_idx = [a.index for a in chain.atoms]
            return traj.atom_slice(atom_idx)

    raise ValueError(f"Chain {label} not found")

def load_dunbrack_bbdep(rotlib_path):
    """Expected format (tab/JSON flexible): rows keyed by (resname, phi_bin, psi_bin) with
    columns listing rotamers and probabilities. To keep this example general, we allow JSON too.
    Minimal schema example per key:
      {
        "resname": "LEU", "phi_bin": -60, "psi_bin": -45,
        "rotamers": [
            {"chi": [60, 180], "p": 0.55},
            {"chi": [300, 180], "p": 0.35},
            {"chi": [180, 60], "p": 0.10}
        ]
      }
    """
    p = Path(rotlib_path)
    if p.suffix.lower() == '.json':
        data = json.loads(Path(rotlib_path).read_text())
        return data
    else:
        # very light TSV reader that expects a JSON object per line (flexible and explicit)
        rows = []
        with open(rotlib_path, 'r') as fh:
            for line in fh:
                line=line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    raise SystemExit("Rotamer library TSV must have one JSON object per non-comment line, or provide a single JSON file.")
        
        return rows

def angle_bin(angle_deg, centers=(-180,-120,-60,0,60,120,180)):
    # assign to nearest 60° bin center by default
    diffs = [abs((angle_deg - c + 180) % 360 - 180) for c in centers]
    return centers[int(np.argmin(diffs))]

def backbone_bins(traj):
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)
    # Map residue index -> (phi_bin, psi_bin)
    bins = {}
    # mdtraj returns radians arrays aligned; we need by residue index
    # Build dicts of residue index to angle in degrees
    phi_map = {traj.topology.atom(i).residue.index: np.degrees(val[0]) for i, val in zip(phi_idx[:,1], phi)}
    psi_map = {traj.topology.atom(i).residue.index: np.degrees(val[0]) for i, val in zip(psi_idx[:,1], psi)}
    for res in traj.topology.residues:
        pb = angle_bin(phi_map.get(res.index, 360.0)) if res.index in phi_map else None
        qb = angle_bin(psi_map.get(res.index, 360.0)) if res.index in psi_map else None
        bins[res.index] = (pb, qb)

    return bins

def build_rotlib_index(rotlib_rows):
    idx = {}
    for row in rotlib_rows:
        key = (row['resname'].upper(), row['phi_bin'], row['psi_bin'])
        idx[key] = row['rotamers'] # list of {"chi": [...], "p": float}

    return idx

def heuristic_feasible_mask(traj_bound, traj_unbound, partner_traj, residue, rotamers):
    """Placeholder: returns a mask of feasible rotamers based on proximity of current side-chain
    to partner atoms. This is deliberately conservative; replace with explicit placement later.
    - If any partner atom within 3.5 Å of side-chain heavy atoms, we assume the most divergent
    chi states become infeasible (keep top-N by Dunbrack p).
    """
    # Find side-chain heavy atoms for residue
    sc_atoms = [a.index for a in residue.atoms if a.name not in ("N","CA","C","O","OXT","H","H1","H2","H3")]
    if not sc_atoms:
        return np.ones(len(rotamers), dtype=bool)
    
    # coordinates
    xyz = traj_bound.xyz[0]
    partner_xyz = partner_traj.xyz[0]
    sc_coords = xyz[sc_atoms]
    # Partner all heavy atoms
    partner_heavy = [a.index for a in partner_traj.topology.atoms if (a.element is not None and a.element.symbol != 'H')]
    if not partner_heavy:
        return np.ones(len(rotamers), dtype=bool)

    pcoords = partner_xyz[partner_heavy]
    
    # Distance check
    close = False
    for v in sc_coords:
        d2 = np.min(np.sum((pcoords - v)**2, axis=1))
        if d2 <= (3.5/10.0)**2: # nm in mdtraj (3.5 Å = 0.35 nm)
            close = True
            break
    
    
    if not close:
        return np.ones(len(rotamers), dtype=bool)

    # If close contact, keep the top 2 rotamers by baseline p as a conservative filter
    order = np.argsort([-r['p'] for r in rotamers])
    keep = np.zeros(len(rotamers), dtype=bool)
    keep[order[:min(2, len(rotamers))]] = True

    return keep

def residue_entropy_kcal(T, probs):
    # Normalize and compute -R T sum p ln p
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum()
    S = -R * T * float(np.sum(probs * np.log(probs)))

    return S

def compute_sidechain_entropy_change(pdb_path, chainA, chainB, rotlib_path, T=298.15, outdir='results'):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    
    traj = load_traj(pdb_path)
    A = select_chain(traj, chainA)
    B = select_chain(traj, chainB)
    
    bins = backbone_bins(traj) # for complex geometry
    rotrows = load_dunbrack_bbdep(rotlib_path)
    rot_index = build_rotlib_index(rotrows)
    
    rows = []
    for res in traj.topology.residues:
        if res.name not in CHI_RES:
            continue
        
        if res.chain.chain_id not in {chainA, chainB}:
            continue
        
        pb, qb = bins.get(res.index, (None, None))
        if pb is None or qb is None:
            continue
        
        key = (res.name, int(pb), int(qb))
        rotamers = rot_index.get(key)
        if not rotamers:
            continue

        # Baseline (unbound) = Dunbrack probabilities
        p_unbound = [r['p'] for r in rotamers]
    
        # Bound feasibility mask: partner is the opposite chain
        partner = B if res.chain.chain_id == chainA else A
        feasible = heuristic_feasible_mask(traj, traj, partner, res, rotamers)
        p_bound_raw = np.array([r['p'] for r in rotamers]) * feasible.astype(float)
        if p_bound_raw.sum() == 0.0:
            # if all filtered, fall back to keep the top-1
            k = int(np.argmax([r['p'] for r in rotamers]))
            p_bound_raw[k] = 1.0

        # Normalize
        p_bound = (p_bound_raw / p_bound_raw.sum()).tolist()
        
        S_un = residue_entropy_kcal(T, p_unbound)
        S_bd = residue_entropy_kcal(T, p_bound)
        rows.append({
            'chain_id': res.chain.chain_id,
            'resSeq': res.resSeq,
            'resName': res.name,
            'S_unbound_kcal_per_mol': S_un,
            'S_bound_kcal_per_mol': S_bd,
            'dS_kcal_per_mol': S_bd - S_un,
        })

    df = pd.DataFrame(rows)
    df.sort_values(['chain_id','resSeq'], inplace=True)
    df.to_csv(outdir / 'rotamer_entropy_dunbrack.csv', index=False)
    
    summary = pd.DataFrame([
        {'metric': 'sum_dS_kcal_per_mol', 'value': df['dS_kcal_per_mol'].sum()},
        {'metric': 'sum_-T_dS_kcal_per_mol_at_298K', 'value': -1.0 * df['dS_kcal_per_mol'].sum()},
    ])
    summary.to_csv(outdir / 'rotamer_entropy_summary.csv', index=False)
    
    return df

def main():
    ap = argparse.ArgumentParser(description='Side-chain rotamer entropy via Dunbrack (API-ready)')
    ap.add_argument('--pdb', required=True)
    ap.add_argument('--chains', required=True, help='e.g., A,B')
    ap.add_argument('--rotlib', required=True, help='Path to Dunbrack bbdep file (JSON or TSV with one JSON per line)')
    ap.add_argument('--T', type=float, default=298.15)
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    
    chainA, chainB = [x.strip() for x in args.chains.split(',')]
    compute_sidechain_entropy_change(args.pdb, chainA, chainB, args.rotlib, T=args.T, outdir=args.out)

if __name__ == '__main__':
    main()
