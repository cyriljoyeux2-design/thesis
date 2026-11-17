"""
Construction of the tight-binding model of In2O3 (bixbyite) with TBPLaS.

The model is built from a structure (ase.Atoms, ex. read from the DFT reference with
qe_xml.read_structure) and two parameter dictionaries (see params.py):
    - on-site energies, per species and orbital;
    - Slater-Koster parameters, per pair of species.
This module does not read any file: the structure and the parameters are given as arguments.

Orbital basis: defined by orbital_label (currently In 5s and O 2p).
Hopping terms: Slater-Koster two-centre integrals between all orbital pairs closer than a
cutoff distance, either given directly in eV ("eV" mode) or scaled as eta * hbar^2/(m d^2)
("harrison" mode, valid for s and p orbitals only).

Ref:
[1] https://www.frontiersin.org/journals/materials/articles/10.3389/fmats.2024.1380180/full

Functions:
    make_in2o3_orbitals : primitive cell with the orbitals and on-site energies, no hopping.
    calc_hop_in2o3      : hopping integral between two orbitals.
    make_in2o3_cell     : complete primitive cell (orbitals and hopping terms).
"""
import numpy as np
import tbplas as tb
from scipy.constants import hbar, m_e, e, angstrom


orbital_label = {"In": ("s",),
                  "O": ("px", "py", "pz")}

#def make_in2o3_atoms(a=10.06, primitive=True):
#    """
#    Construit In2O3 bixbyite avec ASE (Ia-3, Jpn. J. Appl. Phys. - 2000 - Odaka).
#
#    :param a: paramètre de maille cubique en Å
#    :param primitive: True -> maille primitive (40 atomes),
#                      False -> maille conventionnelle (80 atomes)
#    :return: ase.Atoms
#    """
#    symbols = ['In', 'In', 'O']
#    basis = [
#        (0.25,  0.25, 0.25),   # In — site 8b
#        (0.467, 0.0,  0.25),   # In — site 24d
#        (0.391, 0.153, 0.383), # O  — site 48e
#    ]
#    spacegroup = 206
#    cellpar = [a, a, a, 90, 90, 90]
#    return crystal(symbols=symbols,basis=basis,spacegroup=spacegroup,cellpar=cellpar,primitive_cell=primitive)


def make_in2o3_orbitals(energy, atoms):
    """
    Create the TBPLaS primitive cell with all the orbitals and no hopping term.

    The lattice vectors and the atomic positions are those of atoms. Each atom carries
    the orbitals of orbital_label, labelled "<species>_<orbital>" (ex. "O_px").

    :param energy: on-site energies, {species: {orbital type: energy in eV}}, where the
                   orbital type is the first letter of the orbital ("s", "p"...)
    :param atoms: ase.Atoms, structure in Angstrom
    :return: tb.PrimitiveCell
    """
    cell = tb.PrimitiveCell(atoms.cell.array, unit=tb.ANG)
    for sym, pos in zip(atoms.get_chemical_symbols(),atoms.get_scaled_positions()):
        for orb in orbital_label[sym]:
            cell.add_orbital(tuple(pos), energy=energy[sym][orb[0]],label=f"{sym}_{orb}")
    return cell

#SK_in2o3_eV = {
#    "source": "Réf. [1] évaluée à d = 2.18 Å",
#    "mode": "eV",
#    "sss": -0.962,
#    "sps_in_o": 0.433,
#    "sps_o_in": 0.433,
#    "pps": 1.267,
#    "ppp": -0.337,
#}

def calc_hop_in2o3(sk: tb.SK, rij: np.ndarray, distance: float,
                   label_i: str, label_j: str,
                   params: dict) -> complex:
    """
    Evaluate the hopping integral <i,0|H|j,R> between two orbitals (Slater-Koster).

    The parameters of the pair are looked up as "A-B" or "B-A" in params. In "harrison"
    mode, they are multiplied by hbar^2/(m d^2); in "eV" mode, they are used directly.

    :param sk: tb.SK instance
    :param rij: (3,) float array, vector from orbital i to orbital j, in nm
    :param distance: norm of rij, in nm
    :param label_i: label of orbital i (ex. "In_s")
    :param label_j: label of orbital j (ex. "O_px")
    :param params: SK parameters dictionary, {"A-B": {"mode": ..., "sss": ..., ...}}
    :return: hopping integral in eV
    :raises ValueError: if the mode is neither "harrison" nor "eV"
    """
    sym_i, lm_i = label_i.split("_")
    sym_j, lm_j = label_j.split("_")

    # Seules les paires In–O sont décrites pour l'instant
    # if {sym_i, sym_j} != {"In", "O"}:
    #     return 0.0

    key = f"{sym_i}-{sym_j}" if f"{sym_i}-{sym_j}" in params else f"{sym_j}-{sym_i}"
    params = params.get(key)
    d = 10 * distance                          # nm → Å

    # Préfacteur selon la notation
    if params["mode"] == "harrison":
        V0 = hbar**2 / m_e / e / angstrom**2 / (10 * distance)**2    # ħ²/(m d²) en eV, distance nm → Å
    elif params["mode"] == "eV":
        V0 = 1.0
    else:
        raise ValueError(f"Mode inconnu : {params['mode']!r} (attendu 'harrison' ou 'eV')")

    # spσ hétéronucléaire : choisir selon l'atome qui porte l'orbitale s
    #s_sur_in = (lm_i == "s" and sym_i == "In") or (lm_j == "s" and sym_j == "In")
    #v_sps = params["sps_AB"] if s_sur_in else params["sps_BA"]

    #A = key.split("-")[0]
    #s_sur_A = (lm_i == "s" and sym_i == A) or (lm_j == "s" and sym_j == A)
    #v_sps = params["sps_AB"] if s_sur_A else params["sps_BA"]

    return sk.eval(r=rij, label_i=lm_i, label_j=lm_j,
                   v_sss=params["sss"] * V0, v_sps=params["sps"] * V0,
                   v_pps=params["pps"] * V0, v_ppp=params["ppp"] * V0)


def make_in2o3_cell(energy, sk_params, atoms, cutoff=0.5):
    """
    Build the complete TBPLaS primitive cell: orbitals and hopping terms.

    Hopping terms are added for every orbital pair closer than cutoff (except orbitals
    of the same atom), and kept only if larger than 1e-8 eV. Each pair is added once:
    TBPLaS adds the conjugate term itself.

    :param energy: on-site energies dictionary (see make_in2o3_orbitals)
    :param sk_params: SK parameters dictionary (see calc_hop_in2o3)
    :param atoms: ase.Atoms, structure in Angstrom
    :param cutoff: maximum hopping distance, in nm
    :return: tb.PrimitiveCell
    """
    cell = make_in2o3_orbitals(energy, atoms)
    neighbors = tb.find_neighbors(cell, a_max=3, b_max=3, c_max=3,
                                  max_distance=cutoff)
    sk = tb.SK()
    for term in neighbors:
        if term.distance < 1e-5:              # orbitales d'un même atome
            continue
        i, j = term.pair
        hop = calc_hop_in2o3(sk, term.rij, term.distance,
                             cell.get_orbital(i).label,
                             cell.get_orbital(j).label, sk_params)
        if abs(hop) > 1e-8:
            cell.add_hopping(term.rn, i, j, hop)
    return cell
