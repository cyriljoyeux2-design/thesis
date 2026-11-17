"""
Band-structure tools, and band structure of the In2O3 tight-binding model for a given
parameter file.

Functions:
    cart_to_frac   : Cartesian -> fractional coordinates of the k-points.
    k_path_length  : cumulated length along the k-path (x axis of band plots).
    align_bands    : number of occupied bands, VBM, CBM, gap, energies aligned on the VBM.
    find_corners   : indices of the high-symmetry points of a k-path.
    tb_bands       : bands of a TBPLaS model, sorted and aligned on its VBM.
    find_manifolds : groups of bands separated by gaps.
    select_window  : contiguous window of bands.
    check_lattice  : check that the model and the reference share the same lattice.
    band_errors    : RMS and point-by-point difference between two band structures.

Usage (main section):
    Set the reference file and the parameter file, then run: python calc_bands.py
    The model bands are computed on the k-path of the reference and plotted on top of
    the reference window (bands_in2o3.png).

This module does not depend on the fit (fit.py).
"""
import numpy as np
import matplotlib.pyplot as plt
import tbplas as tb
from in2o3 import make_in2o3_orbitals, make_in2o3_cell
from params import read_params, write_params, params_to_dicts, dicts_to_params
from qe_xml import bands_read, read_reciprocal, read_structure, read_nelec

def cart_to_frac(k_cart, B):
    """
    Convert k-points from Cartesian to fractional coordinates: k_cart = k_frac @ B.

    :param k_cart: (n_k, 3) float array, Cartesian coordinates
    :param B: (3, 3) float array, reciprocal vectors as rows, same units as k_cart
    :return: (n_k, 3) float array, fractional coordinates
    :raises ValueError: if the back-conversion does not give k_cart again
    """
    k_frac = k_cart @ np.linalg.inv(B)

    if np.allclose(k_cart, k_frac @ B) == False:
        raise ValueError(f"Problème dans la convertion en coordonnées réduite des k points")
    return k_frac


def k_path_length(k_cart, alat):
    """
    Cumulated length along the k-path, used as x axis of band plots.

    :param k_cart: (n_k, 3) float array, Cartesian k-points in 2pi/alat units
    :param alat: QE length unit, in Angstrom
    :return: (n_k,) float array, in 1/Angstrom, starting at 0
    """
    k_len = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(k_cart, axis=0), axis=1)))) * 2*np.pi/alat
    return k_len


def align_bands(energies_ev, nelec):
    """
    Locate the band edges and set the zero of energy at the valence band maximum.

    Assumes a non spin-polarised calculation (2 electrons per band).

    :param energies_ev: (n_k, nbnd) float array, sorted band energies in eV
    :param nelec: number of electrons
    :return: (n_occ, VBM, CBM, gap, k_max, k_min, energies_fermi)
        n_occ: number of occupied bands
        VBM, CBM: band edges after alignment (VBM = 0), in eV
        gap: CBM - VBM, in eV
        k_max, k_min: k-point indices of the VBM and of the CBM
        energies_fermi: (n_k, nbnd) float array, energies with the VBM at 0
    """
    n_occ = int(nelec/2)
    k_max = np.argmax(energies_ev[:,n_occ-1])
    k_min = np.argmin(energies_ev[:,n_occ])
    energies_fermi = energies_ev - energies_ev[k_max,n_occ-1]
    VBM = energies_fermi[k_max,n_occ-1]
    CBM = energies_fermi[k_min,n_occ]
    gap = CBM - VBM
    return n_occ, VBM, CBM, gap, k_max, k_min, energies_fermi


def find_corners(k_cart):
    """
    Find the high-symmetry points of a k-path, i.e. where the path changes direction.

    The first and last points are always included.

    :param k_cart: (n_k, 3) float array, k-points along the path
    :return: (n_corners,) int array, indices of the corners in k_cart
    """
    d1 = np.diff(k_cart, axis=0) # * 2*np.pi/alat
    d = d1 / np.linalg.norm(d1, axis=1)[:, None]
    p = np.sum(np.concatenate(([[0.0, 0.0, 0.0]], d)) * np.concatenate((d, [[0.0, 0.0, 0.0]])), axis=1)
    k_symm = np.where(p < 0.99)[0]
    return k_symm


def tb_bands(TB_cell, k_frac, n_occ):
    """
    Compute the band structure of a TBPLaS model, sorted and aligned on its VBM.

    :param TB_cell: tb.PrimitiveCell
    :param k_frac: (n_k, 3) float array, k-points in fractional coordinates
    :param n_val: number of valence bands of the model
    :return: (n_k, n_orb) float array, energies in eV with the VBM at 0
    """
    k_TB, bands = TB_cell.calc_bands(k_frac, echo_details=False)
    bands = np.sort(bands, axis=1)

    vbm = bands[:, n_occ - 1].max()
    cbm = bands[:, n_occ].min()
    bands_shifted = bands - vbm
    return bands_shifted


def find_manifolds(energies_fermi, nbnd, tol=0.05):
    """
    Split the bands into groups (manifolds) separated by energy gaps.

    There is a gap between bands n and n+1 if the minimum of band n+1 over all k-points
    is above the maximum of band n by more than tol.

    :param energies_fermi: (n_k, nbnd) float array, sorted band energies in eV
    :param nbnd: number of bands
    :param tol: minimum gap width, in eV
    :return: (band_group, n_band_group, en_band_group)
        band_group: (n_groups, 2) int array, first and last band of each group (included)
        n_band_group: (n_groups,) int array, number of bands of each group
        en_band_group: (n_groups, 2) float array, minimum and maximum energy of each group
    """
    band_max = np.max(energies_fermi, axis=0)
    band_min = np.min(energies_fermi, axis=0)
    band_delta = band_min[1:] - band_max[:-1]
    band_gap = np.concatenate(([-1], np.where(band_delta > tol)[0], [nbnd-1]))

    band_group = []
    for i in range(len(band_gap)-1):
        band_group.append((band_gap[i]+1, band_gap[i+1]))
    band_group = np.array(band_group)

    n_band_group = band_group[:, 1] - band_group[:, 0] + 1
    en_band_group = np.array([np.min(energies_fermi[:,band_group[:,0]], axis=0), np.max(energies_fermi[:,band_group[:,1]], axis=0)]).T
    return band_group, n_band_group, en_band_group


def select_window(energies, first, n_band):
    """
    Select a contiguous window of bands.

    :param energies: (n_k, n_bands) float array, band energies in eV
    :param first: index of the first band of the window
    :param n_band: number of bands in the window
    :return: (n_k, n_band) float array
    :raises ValueError: if the window goes beyond the available bands
    """
    if first < 0:
        raise ValueError(f"L'indexe de la première bande est négatif {first}")

    if n_band + first > energies.shape[1]:
        raise ValueError(f"La fenêtre sélectionnée dépasse le nombre totale de bandes calculées ({n_band - first} pour un totale de {energies.shape[1]} bandes)")
    return energies[:, first:first+ n_band]


def check_lattice(TB_cell, ref_atoms, tol=1e-5):
    """
    Check that the model and the reference have the same lattice vectors.

    Same vectors, in the same order and orientation: this guarantees that fractional
    k-points designate the same physical points in both.

    :param TB_cell: tb.PrimitiveCell (lattice vectors stored in nm)
    :param ref_atoms: ase.Atoms of the reference (lattice in Angstrom)
    :param tol: maximum difference per component, in Angstrom
    :raises ValueError: if a component differs by more than tol
    """
    diff = np.abs(TB_cell.lat_vec*10 - ref_atoms.cell.array)
    if np.max(diff) > tol:
        raise ValueError(f"Au moins une coordonnée de la maille du modèle différe de la maille de référence au dela du seuil {diff}")


def band_errors(e_tb, e_ref, tol=1e-8):
    """
    Compare two band structures band by band.

    Both arrays must use the same k-points, the same bands and the same zero of energy.
    Comparing band by band is valid because both are sorted by energy at each k-point.

    :param e_tb: (n_k, n_b) float array, model bands in eV
    :param e_ref: (n_k, n_b) float array, reference bands in eV
    :param tol: tolerance of the sorting check, in eV
    :return: (rms, d)
        rms: root-mean-square difference over all k-points and bands, in eV
        d: (n_k, n_b) float array, e_tb - e_ref, in eV
    :raises ValueError: if the shapes differ or if the bands are not sorted
    """
    if e_tb.shape != e_ref.shape:
        raise ValueError(f"Formes différentes : TB={e_tb.shape}, référence={e_ref.shape}")
    if np.any(np.diff(e_tb, axis=1) < -tol) or np.any(np.diff(e_ref, axis=1) < -tol):
        raise ValueError("Les bandes ne sont pas triées par ordre croissant d'énergie")

    d = e_tb - e_ref
    rms = np.sqrt(np.mean(d**2))
    return rms, d

# Chemin k de la maille cubique centrée, dans le repère de la maille primitive
# [0.0 ,  0.0 , 0.0 ],  G
# [1./2, -1./2, 1./2],  H 
# [0.0 ,  0.0 , 1./2],  N
# [0.0 ,  0.0 , 0.0 ],  G
# [1./4,  1./4, 1./4],  P
# [1./2, -1./2, 1./2]   H

# ASE identifie le type de réseau de Bravais et retrouve les coordonnées fractionnaires des poits de hautes symétries de la ZB ainsi que le chemin recommandé
#special = make_in2o3_atoms(energy).cell.bandpath(npoints=0).special_points


if __name__ == "__main__":
    # ----- Input files names -----
    file_name ="data-file-schema.xml"
    params_file_name = "params/fit_2026-11-13.json"

    # ----- qe_xml.py varaiables -----
    ref_atoms = read_structure(file_name)
    k_cart, energies_ev, nbnd, n_k = bands_read(file_name)
    alat, B = read_reciprocal(file_name)
    nelec = read_nelec(file_name)

    # ----- in2o3.py variables -----
    energy, sk_params = read_params(params_file_name)
    TB_cell = make_in2o3_cell(energy, sk_params, ref_atoms)
    n_val = 3 * ref_atoms.get_chemical_symbols().count("O")

    # ----- calc_bands.py variables -----
    k_frac = cart_to_frac(k_cart, B)
    k_len = k_path_length(k_cart, alat)
    k_symm = find_corners(k_cart)
    *_, energies_fermi = align_bands(energies_ev, nelec) 
    band_group, n_band_group, en_band_group = find_manifolds(energies_fermi, nbnd)
    bands = tb_bands(TB_cell, k_frac, n_val)

    # ----- plot settings -----
    plt.figure()
    label = ["G", "N", "P", "G", "H", "N"]
    plt.plot(k_len, select_window(energies_fermi, band_group[2, 0], TB_cell.num_orb), "k-", lw=0.8)
    plt.plot(k_len, bands, "r--", lw=0.8)
    plt.xticks(k_len[k_symm], label)
    plt.xlabel("k (1/ang)")
    #plt.ylim(-2, 0)
    plt.ylabel("Energy (eV)")
    for i in range(len(k_symm)):
        plt.axvline(k_len[k_symm[i]], color="grey", lw=0.5)

    plt.axhline(0, color="grey", lw=0.5)
    plt.savefig("bands_in2o3.png", dpi=200)
