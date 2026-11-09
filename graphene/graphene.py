"""Tentative de roproduire la structure de bande de la monocouche MoS2 avec les paramètres TB connus."""

import numpy as np
import tbplas as tb
from ase import Atoms
np.set_printoptions(legacy="1.25")

__all__ = ["make_graphene_atoms", "make_graphene_orbitals", "make_graphene_cell"]


# Base minimale sur chaque atome
orbital_label = {"C": ("pz",)}

orbital_energy = {"C": {"pz": 0.0}}


def make_graphene_atoms(a=2.460, c=10.0):
    """
    Construit une monocouche de graphene avec ASE.

    paramètre supplémentaire:
    :size: 
    :vacuum:

    :return: ase.Atoms
    """
    formula='C2'
    thickness=0.0
    cell = [[a, 0, 0], [a / 2, a * 3**0.5 / 2, 0], [0, 0, c]]
    basis = [[0, 0, -0.5 * thickness], [1 / 3, 1 / 3, 0.5 * thickness]]
    atoms = Atoms(formula, cell=cell, pbc=(1, 1, 0))
    atoms.set_scaled_positions(basis)

    return atoms 

def make_graphene_orbitals(energy=orbital_energy):
    """
    Crée la PrimitiveCell TBPLaS avec toutes les orbitales, sans saut.

    :param onsite: énergies sur site par espèce et par type d'orbitale
    :return: tb.PrimitiveCell
    """
    atoms = make_graphene_atoms()
    cell = tb.PrimitiveCell(atoms.cell.array, unit=tb.ANG)
    for sym, pos in zip(atoms.get_chemical_symbols(),atoms.get_scaled_positions()):
        for orb in orbital_label[sym]:
            cell.add_orbital(tuple(pos), energy=energy[sym][orb],label=f"{sym}_{orb}")
    return cell


SK_PARAMS_GRAPHENE = {
    "source": "",
    "ppp_1": -2.7,
    "ppp_2":  0.0
}



def calc_hop_graphene(sk: tb.SK, rij: np.ndarray, distance: float,
                   label_i: str, label_j: str,
                   params: dict = SK_PARAMS_GRAPHENE) -> complex:
    """
    Évalue l'intégrale de saut <i,0|H|j,R>.

    Ref:

    :param sk: instance SK
    :param rij: vecteur de l'orbitale i vers l'orbitale j, en nm
    :param distance: norme de rij, en nm
    :param label_i: label de l'orbitale i
    :param label_j: label de l'orbitale j
    :param params: jeu de paramètres
    :return: intégrale de saut en eV
    """
    sym_i, lm_i = label_i.split("_")
    sym_j, lm_j = label_j.split("_")

    if abs(distance - 0.142) < 0.01: 
        v = params["ppp_1"]
    elif abs(distance - 0.246) < 0.01:
        v = params["ppp_2"]
    else:                 
        raise ValueError(f"Distance {distance:.4f} nm : aucune couche de voisins ne correspond")

    return sk.eval(r=rij, label_i=lm_i, label_j=lm_j, v_ppp=v)


def make_graphene_cell(energy=orbital_energy, sk_params=SK_PARAMS_GRAPHENE, cutoff=0.26):
    """Maille TBPLaS complète : orbitales + sauts. cutoff en nm."""
    cell = make_graphene_orbitals(energy)
    neighbors = tb.find_neighbors(cell, a_max=2, b_max=2, c_max=0,
                                  max_distance=cutoff)
    sk = tb.SK()
    for term in neighbors:
        if term.distance < 1e-5:              # orbitales d'un même atome
            continue
        i, j = term.pair
        hop = calc_hop_graphene(sk, term.rij, term.distance,
                             cell.get_orbital(i).label,
                             cell.get_orbital(j).label, sk_params)
        if abs(hop) > 1e-8:
            cell.add_hopping(term.rn, i, j, hop)
    return cell
