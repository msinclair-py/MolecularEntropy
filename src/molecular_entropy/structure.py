"""
Structure loading and manipulation utilities using mdtraj.

This module provides a unified interface for loading PDB files and trajectories,
selecting chains, and extracting structural information.
"""

__all__ = [
    "StructureLoader",
    "StructureLoadError",
    "ChainNotFoundError",
]

import logging
from pathlib import Path

import mdtraj as md
import numpy as np

logger = logging.getLogger(__name__)


class StructureLoadError(Exception):
    """Raised when a structure cannot be loaded."""

    pass


class ChainNotFoundError(Exception):
    """Raised when a specified chain is not found in the structure."""

    pass


class StructureLoader:
    """Unified structure loading interface using mdtraj."""

    @staticmethod
    def load(path: str | Path, topology: str | Path | None = None) -> md.Trajectory:
        """
        Load a structure from PDB or trajectory file.

        Args:
            path: Path to structure file (PDB, DCD, XTC, etc.)
            topology: Optional topology file for trajectory formats

        Returns:
            MDTraj Trajectory object

        Raises:
            StructureLoadError: If file cannot be loaded
        """
        path = Path(path)
        if not path.exists():
            raise StructureLoadError(f"File not found: {path}")

        try:
            if topology:
                traj = md.load(str(path), top=str(topology))
            else:
                traj = md.load(str(path))
            logger.info(
                f"Loaded structure: {traj.n_atoms} atoms, "
                f"{traj.n_residues} residues, {traj.n_frames} frames"
            )
            return traj
        except Exception as e:
            raise StructureLoadError(f"Failed to load {path}: {e}") from e

    @staticmethod
    def get_chain_ids(traj: md.Trajectory) -> list[str]:
        """Get list of chain IDs in the trajectory."""
        return [chain.chain_id for chain in traj.topology.chains]

    @staticmethod
    def select_chain(traj: md.Trajectory, chain_id: str) -> md.Trajectory:
        """
        Select a single chain from the trajectory.

        Args:
            traj: MDTraj trajectory
            chain_id: Chain identifier (e.g., 'A', 'B')

        Returns:
            New trajectory containing only the specified chain

        Raises:
            ChainNotFoundError: If chain is not found
        """
        for chain in traj.topology.chains:
            if chain.chain_id == chain_id:
                atom_idx = [a.index for a in chain.atoms]
                return traj.atom_slice(atom_idx)

        available = StructureLoader.get_chain_ids(traj)
        raise ChainNotFoundError(f"Chain '{chain_id}' not found. Available chains: {available}")

    @staticmethod
    def select_chains(traj: md.Trajectory, chain_ids: list[str]) -> md.Trajectory:
        """
        Select multiple chains from the trajectory.

        Args:
            traj: MDTraj trajectory
            chain_ids: List of chain identifiers

        Returns:
            New trajectory containing only the specified chains

        Raises:
            ChainNotFoundError: If any chain is not found
        """
        atom_idx = []
        found_chains = set()

        for chain in traj.topology.chains:
            if chain.chain_id in chain_ids:
                atom_idx.extend([a.index for a in chain.atoms])
                found_chains.add(chain.chain_id)

        missing = set(chain_ids) - found_chains
        if missing:
            available = StructureLoader.get_chain_ids(traj)
            raise ChainNotFoundError(f"Chains not found: {missing}. Available: {available}")

        return traj.atom_slice(sorted(atom_idx))

    @staticmethod
    def compute_backbone_dihedrals(
        traj: md.Trajectory,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """
        Compute phi and psi backbone dihedrals for all residues.

        Args:
            traj: MDTraj trajectory (single frame expected)

        Returns:
            Tuple of (phi_map, psi_map) dictionaries mapping
            residue index to dihedral angle in degrees.
        """
        # Compute phi angles
        phi_idx, phi_vals = md.compute_phi(traj)
        psi_idx, psi_vals = md.compute_psi(traj)

        # phi_idx has shape (n_phi, 4) with atom indices
        # phi_vals has shape (n_frames, n_phi)
        phi_map = {}
        for i, atom_indices in enumerate(phi_idx):
            # The residue is identified by the CA atom (index 1 in the quartet)
            ca_atom = traj.topology.atom(atom_indices[1])
            res_idx = ca_atom.residue.index
            phi_map[res_idx] = np.degrees(phi_vals[0, i])

        psi_map = {}
        for i, atom_indices in enumerate(psi_idx):
            ca_atom = traj.topology.atom(atom_indices[1])
            res_idx = ca_atom.residue.index
            psi_map[res_idx] = np.degrees(psi_vals[0, i])

        return phi_map, psi_map

    @staticmethod
    def get_residue_info(traj: md.Trajectory) -> list[dict]:
        """
        Get information about all residues in the trajectory.

        Returns:
            List of dicts with keys: index, name, resSeq, chain_id
        """
        residues = []
        for res in traj.topology.residues:
            residues.append(
                {
                    "index": res.index,
                    "name": res.name,
                    "resSeq": res.resSeq,
                    "chain_id": res.chain.chain_id,
                }
            )
        return residues

    @staticmethod
    def select_residues(traj: md.Trajectory, residue_indices: list[int]) -> md.Trajectory:
        """
        Select specific residues by their indices.

        Args:
            traj: MDTraj trajectory
            residue_indices: List of residue indices (0-based)

        Returns:
            New trajectory containing only the specified residues
        """
        residue_set = set(residue_indices)
        atom_idx = []
        for res in traj.topology.residues:
            if res.index in residue_set:
                atom_idx.extend([a.index for a in res.atoms])
        return traj.atom_slice(sorted(atom_idx))

    @staticmethod
    def assign_chains(
        traj: md.Trajectory,
        chain_a_residues: list[int],
        chain_b_residues: list[int],
    ) -> md.Trajectory:
        """
        Reassign chain IDs based on residue index groupings.

        Useful for AMBER prmtop topologies that lack chain ID information.
        Rebuilds the topology so that residues in ``chain_a_residues`` belong
        to chain "A" and residues in ``chain_b_residues`` belong to chain "B".
        Solvent / ion residues not in either list are placed on a third chain.

        Args:
            traj: MDTraj trajectory
            chain_a_residues: Residue indices (0-based) for chain A
            chain_b_residues: Residue indices (0-based) for chain B

        Returns:
            New trajectory with reassigned chain IDs
        """
        set_a = set(chain_a_residues)
        set_b = set(chain_b_residues)

        overlap = set_a & set_b
        if overlap:
            raise ValueError(
                f"Residue indices appear in both chain_a and chain_b: {sorted(overlap)}"
            )

        old_top = traj.topology
        new_top = md.Topology()

        chain_obj_a = new_top.add_chain()  # chain "A" (index 0)
        chain_obj_b = new_top.add_chain()  # chain "B" (index 1)
        chain_obj_other = new_top.add_chain()  # everything else

        atom_mapping: list[int] = []  # old atom index -> position in new

        for res in old_top.residues:
            if res.index in set_a:
                target_chain = chain_obj_a
            elif res.index in set_b:
                target_chain = chain_obj_b
            else:
                target_chain = chain_obj_other

            new_res = new_top.add_residue(res.name, target_chain, res.resSeq)
            for atom in res.atoms:
                new_top.add_atom(atom.name, atom.element, new_res)
                atom_mapping.append(atom.index)

        # Rebuild bonds
        old_atom_to_new = {old: new for new, old in enumerate(atom_mapping)}
        for bond in old_top.bonds:
            i, j = bond[0].index, bond[1].index
            if i in old_atom_to_new and j in old_atom_to_new:
                new_top.add_bond(
                    new_top.atom(old_atom_to_new[i]),
                    new_top.atom(old_atom_to_new[j]),
                )

        # Reorder coordinates to match new atom order
        new_xyz = traj.xyz[:, atom_mapping, :]
        new_traj = md.Trajectory(new_xyz, new_top, time=traj.time, unitcell_lengths=traj.unitcell_lengths, unitcell_angles=traj.unitcell_angles)

        logger.info(
            f"Assigned chains: A={len(chain_a_residues)} residues, "
            f"B={len(chain_b_residues)} residues"
        )
        return new_traj

    @staticmethod
    def select_residues_except(traj: md.Trajectory, exclude_indices: list[int]) -> md.Trajectory:
        """
        Select all residues except those specified.

        Args:
            traj: MDTraj trajectory
            exclude_indices: List of residue indices to exclude (0-based)

        Returns:
            New trajectory with specified residues removed
        """
        exclude_set = set(exclude_indices)
        atom_idx = []
        for res in traj.topology.residues:
            if res.index not in exclude_set:
                atom_idx.extend([a.index for a in res.atoms])
        return traj.atom_slice(sorted(atom_idx))
