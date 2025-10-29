import math
from collections import defaultdict, deque
from typing import List, Dict, Tuple
import numpy as np
import mdtraj as md

# Simple Bondi vdW radii (Å)
VDW = {'H':1.20, 'C':1.70, 'N':1.55, 'O':1.52, 'S':1.80, 'P':1.80}
BACKBONE = {"N","CA","C","O","OXT"}

# χ definitions by residue as (a,b,c,d) atom-name tuples; rotatable bond is (b-c)
CHI_DEFS: Dict[str, List[Tuple[str,str,str,str]]] = {
    'ARG':[('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','NE'), ('CG','CD','NE','CZ')],
    'LYS':[('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','CE'), ('CG','CD','CE','NZ')],
    'GLU':[('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','OE1')],
    'GLN':[('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','NE2')],
    'MET':[('N','CA','CB','CG'), ('CA','CB','CG','SD'), ('CB','CG','SD','CE')],
    'ILE':[('N','CA','CB','CG1'), ('CA','CB','CG1','CD1')],
    'LEU':[('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'VAL':[('N','CA','CB','CG1')],
    'THR':[('N','CA','CB','OG1')],
    'SER':[('N','CA','CB','OG')],
    'TYR':[('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'PHE':[('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'TRP':[('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'ASN':[('N','CA','CB','CG')],
    'ASP':[('N','CA','CB','CG')],
    'HIS':[('N','CA','CB','CG'), ('CA','CB','CG','ND1')],
    'CYS':[('N','CA','CB','SG')],
    'PRO':[('N','CA','CB','CG')],
}

def build_graph(top: md.Topology):
    G = defaultdict(list)
    for b in top.bonds:
        i, j = b[0].index, b[1].index
        G[i].append(j); G[j].append(i)
    return G


def dihedral_deg(xyz, i,j,k,l):
    # returns angle in degrees
    p = xyz[[i,j,k,l]]
    b0 = p[1]-p[0]; b1 = p[2]-p[1]; b2 = p[3]-p[2]
    b1n = b1/np.linalg.norm(b1)
    v = b0 - np.dot(b0,b1n)*b1n
    w = b2 - np.dot(b2,b1n)*b1n
    x = np.dot(v,w)
    y = np.dot(np.cross(b1n,v), w)
    ang = math.degrees(math.atan2(y,x))
    return ang

def rotate_about_axis(coords, origin, axis, angle_deg):
    # Rodrigues rotation for an array of points
    theta = math.radians(angle_deg)
    k = axis/np.linalg.norm(axis)
    p = coords - origin
    p_rot = (p*np.cos(theta) + np.cross(k, p)*np.sin(theta)
             + k*np.dot(p, k)[:,None]*(1-np.cos(theta)))
    return p_rot + origin

def downstream_atoms(G, top, bond_b, bond_c, fixed_side_root):
    # Return atoms on the "far" side of bond_b—bond_c away from fixed_side_root
    far_root = bond_c if fixed_side_root == bond_b else bond_b
    stop = fixed_side_root
    seen = {stop}
    dq = deque([far_root])
    out = set([far_root])
    while dq:
        u = dq.popleft()
    for v in G[u]:
        if v in seen: continue
        seen.add(v)
        
        # do not cross the rotatable bond back to the fixed side
        if (u==bond_b and v==bond_c) or (u==bond_c and v==bond_b):
            continue
        dq.append(v)
        out.add(v)

    return sorted(list(out))

def residue_atom_indices(res):
    return [a.index for a in res.atoms]

def name_to_index_map(res):
    return {a.name:a.index for a in res.atoms}

def place_rotamer_once(traj: md.Trajectory, res, chi_targets: List[float]):
"""Return a copy of xyz with the residue rotated to chi_targets (degrees)."""
    resname = res.name
    if resname not in CHI_DEFS: return traj.xyz[0].copy()
    defs = CHI_DEFS[resname]
    xyz = traj.xyz[0].copy() # nm
    top = traj.topology
    G = build_graph(top)
    name2idx = name_to_index_map(res)
    
    for chi_i, chi_def in enumerate(defs[:len(chi_targets)], start=1):
        a,b,c,d = chi_def
        if a not in name2idx or b not in name2idx or c not in name2idx or d not in name2idx:
            continue
        i,j,k,l = name2idx[a], name2idx[b], name2idx[c], name2idx[d]
        current = dihedral_deg(xyz, i,j,k,l)
        delta = ((chi_targets[chi_i-1] - current + 180) % 360) - 180
        if abs(delta) < 1e-3: # nothing to do
            continue
        # rotate all atoms downstream of (j-k) that lie on the k side
        rot_axis = xyz[k] - xyz[j]
        origin = xyz[j]
        moving = downstream_atoms(G, top, j, k, fixed_side_root=j)
        xyz[moving] = rotate_about_axis(xyz[moving], origin, rot_axis, delta)

    return xyz

def clash_with_partner(xyz_res, res_atom_indices, partner_xyz, partner_heavy, scale=0.8):
    # xyz in nm; convert vdW Å to nm when comparing
    for ai in res_atom_indices:
        # skip hydrogens by element if available via partner arrays; we assume input indices already heavy
        pass
    # Build per-atom radii for residue (heavy atoms only)
    # Here we’ll obtain element from indices via closure; caller provides arrays filtered to heavy atoms
    return False

def feasible_mask_for_residue(traj: md.Trajectory, partner: md.Trajectory, res, rotamers: List[Dict], cutoff_scale=0.8):
"""Return boolean mask of feasible rotamers using explicit χ placement + vdW clash test.
rotamers: list of {"chi": [deg,...], "p": float}
"""
    # Precompute partner heavy atoms
    partner_heavy = [a.index for a in partner.topology.atoms if (a.element and a.element.symbol != 'H')]
    partner_xyz = partner.xyz[0]
    
    # Build residue heavy list with per-atom vdW radii (nm)
    res_heavy = []
    res_radii = []
    for a in res.atoms:
        if a.element and a.element.symbol != 'H':
            res_heavy.append(a.index)
            res_radii.append(VDW.get(a.element.symbol, 1.7)/10.0)

    res_heavy = np.array(res_heavy, dtype=int)
    res_radii = np.array(res_radii, dtype=float)
    
    # Partner radii
    partner_radii = np.array(
        [
            VDW.get(a.element.symbol if a.element else 'C', 1.7)/10.0 
            for a in partner.topology.atoms if (a.element and a.element.symbol != 'H')
        ], dtype=float
    )
    
    feasible = []
    for rot in rotamers:
        chi = rot.get('chi', [])
        xyz_new = place_rotamer_once(traj, res, chi)
        res_xyz = xyz_new[res_heavy]
        # Pairwise min distance to partner heavy atoms (vectorized)
        # res_xyz: (m,3), partner_xyz[partner_heavy]: (n,3)
        ph_xyz = partner_xyz[partner_heavy]
        # Broadcast distances
        diff = res_xyz[:,None,:] - ph_xyz[None,:,:]
        d2 = np.einsum('mnj,mnj->mn', diff, diff)
        d = np.sqrt(d2)
        # Sum radii
        sum_r = res_radii[:,None] + partner_radii[None,:]
        # Clash if any pair is closer than scale * (r_i + r_j)
        clash = np.any(d < (cutoff_scale*sum_r))
        feasible.append(not clash)

    return np.array(feasible, dtype=bool)
