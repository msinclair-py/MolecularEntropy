import argparse
import numpy as np
import pandas as pd
from prody import parsePDB, calcANM, AtomGroup, writePDB

kB_kcal = 1.98720425864083e-3 # kcal/mol/K

def vib_entropy_from_anm(anm, T):
    # anm.getEigvals() returns eigenvalues ~ ω^2 (arbitrary scale). Skip 6 zero modes.
    evals = anm.getEigvals()
    # defensively filter small/negative numerical noise
    evals = np.array([e for e in evals if e > 1e-12])
    # Classical S ≈ k_B * sum[ ln(k_BT / ħω) + 1 ] ; with ENM scale, use ln(1/ω) up to constant.
    # Constants cancel in ΔS; keep form: S* = sum[-0.5 ln(evals)] + const. We report in k_B units then convert.
    S_kB = np.sum(-0.5*np.log(evals))
    S_kcal_per_K = S_kB * kB_kcal
    return S_kB, S_kcal_per_K

def load_chain(pdb_path, chain_id):
    ag = parsePDB(pdb_path, chain=chain_id)
    return ag

def run(pdb, chains, cutoff, gamma, T, out):
    A_id, B_id = [c.strip() for c in chains.split(',')]
    AB = parsePDB(pdb)
    A = parsePDB(pdb, chain=A_id)
    B = parsePDB(pdb, chain=B_id)
    
    # Select CA atoms for ANM
    selAB = AB.select('name CA')
    selA = A.select('name CA')
    selB = B.select('name CA')
    
    anmAB = calcANM(selAB, cutoff=cutoff, gamma=gamma)
    anmA = calcANM(selA, cutoff=cutoff, gamma=gamma)
    anmB = calcANM(selB, cutoff=cutoff, gamma=gamma)
    
    SAB_kB, SAB_kcalK = vib_entropy_from_anm(anmAB, T)
    SA_kB, SA_kcalK = vib_entropy_from_anm(anmA, T)
    SB_kB, SB_kcalK = vib_entropy_from_anm(anmB, T)
    
    dS_kB = SAB_kB - SA_kB - SB_kB
    dS_kcalK = SAB_kcalK - SA_kcalK - SB_kcalK
    
    df = pd.DataFrame([
        {"term":"S_vib(AB)","kB":SAB_kB,"kcal_per_K":SAB_kcalK},
        {"term":"S_vib(A)","kB":SA_kB,"kcal_per_K":SA_kcalK},
        {"term":"S_vib(B)","kB":SB_kB,"kcal_per_K":SB_kcalK},
        {"term":"ΔS_vib","kB":dS_kB,"kcal_per_K":dS_kcalK},
        {"term":"-TΔS_vib (kcal/mol)","kB":np.nan,"kcal_per_K":-T*dS_kcalK}
    ])
    df.to_csv(out, index=False)
    print("Wrote:", out)

if __name__ == '__main__':
    run(args.pdb, args.chains, args.cutoff, args.gamma, args.T, args.out)
