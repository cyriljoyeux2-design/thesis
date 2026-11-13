import numpy as np
import matplotlib.pyplot as plt
import tbplas as tb
from in2o3 import *
from qe_xml import *

def cart_to_frac(k_cart, B):
    k_frac = k_cart @ np.linalg.inv(B)

    if np.allclose(k_cart, k_frac @ B) == False:
        raise ValueError(f"Problème dans la convertion en coordonnées réduite des k points")

    return k_frac

def k_path_length(k_cart, alat):
    k_len = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(k_cart, axis=0), axis=1)))) * 2*np.pi/alat

    return k_len

def align_bands(energies_ev, nelec):
    n_occ = int(nelec/2)
    k_max = np.argmax(energies_ev[:,n_occ-1])
    k_min = np.argmin(energies_ev[:,n_occ])
    energies_fermi = energies_ev - energies_ev[k_max,n_occ-1]
    VBM = energies_fermi[k_max,n_occ-1]
    CBM = energies_fermi[k_min,n_occ]
    gap = CBM - VBM

    return n_occ, VBM, CBM, gap, k_max, k_min, energies_fermi

def find_corners(k_cart):
    d1 = np.diff(k_cart, axis=0) # * 2*np.pi/alat
    d = d1 / np.linalg.norm(d1, axis=1)[:, None]
    p = np.sum(np.concatenate(([[0.0, 0.0, 0.0]], d)) * np.concatenate((d, [[0.0, 0.0, 0.0]])), axis=1)
    k_symm = np.where(p < 0.99)[0]

    return k_symm

def tb_bands(TB_cell, k_frac, n_occ):
    k_TB, bands = TB_cell.calc_bands(k_frac)
    bands = np.sort(bands, axis=1)

    vbm = bands[:, n_occ - 1].max()
    cbm = bands[:, n_occ].min()
    bands_shifted = bands - vbm

    return bands_shifted

def find_manifolds(energies_fermi, nbnd, tol=0.05):
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

    if first < 0:
        raise ValueError(f"L'indexe de la première bande est négatif {first}")

    if n_band + first > energies.shape[1]:
        raise ValueError(f"La fenêtre sélectionnée dépasse le nombre totale de bandes calculées ({n_band - first} pour un totale de {energies.shape[1]} bandes)")

    return energies[:, first:first+ n_band]


def check_lattice(TB_cell, ref_atoms, tol=1e-5):
    diff = np.abs(TB_cell.lat_vec*10 - ref_atoms.cell.array)
    if np.max(diff) > tol:
        raise ValueError(f"Au moins une coordonnée de la maille du modèle différe de la maille de référence au dela du seuil {diff}")


def band_errors(bands_shifted, window_dft, first_band, last_band, tol=1e-8):
    if bands_shifted[:, first_band:last_band].shape != window_dft.shape:
        raise ValueError(f"Les deux modèles n'ont pas le même nombre de point k et de bandes: TB={bands_shifted.shape}, dft={window_dft.shape}")

    if np.any(np.diff(bands_shifted, axis=1) < - tol) == True:
        raise ValueError(f"Les bandes ne sont pas triées par ordre croissant d'énergie sur tous les points k pour le modèle TB")

    if np.any(np.diff(window_dft, axis=1) < - tol) == True:
        raise ValueError(f"Les bandes ne sont pas triées par ordre croissant d'énergie sur tous les points k pour le modèle dft")

    RMS = np.sqrt(np.mean((bands_shifted[:, first_band:last_band] - window_dft)**2))
    d = bands_shifted[:, first_band:last_band] - window_dft

    return RMS, d


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
    # Input files names
    file_name ="data-file-schema.xml"
    params_file_name = "params/fit_2026-11-13.json"

    # qe_xml.py varaiables
    ref_atoms = read_structure(file_name)
    k_cart, energies_ev, nbnd, n_k = bands_read(file_name)
    alat, B = read_reciprocal(file_name)
    nelec = read_nelec(file_name)

    # in2o3.py variables
    energy, sk_params = read_params(params_file_name)
    TB_cell = make_in2o3_cell(energy, sk_params, ref_atoms)
    n_val = 3 * ref_atoms.get_chemical_symbols().count("O")

    # calc_bands.py variables
    k_frac = cart_to_frac(k_cart, B)
    k_len = k_path_length(k_cart, alat)
    k_symm = find_corners(k_cart)
    n_occ, VBM, CBM, gap, k_max, k_min, energies_fermi = align_bands(energies_ev, nelec) 
    band_group, n_band_group, en_band_group = find_manifolds(energies_fermi, nbnd)
    bands = tb_bands(TB_cell, k_frac, n_val)

    plt.figure()
    label = ["G", "N", "P", "G", "H", "N"]
    plt.plot(k_len, select_window(energies_fermi, band_group[2, 0], TB_cell.num_orb), "k-", lw=0.8)
    plt.plot(k_len, bands, "r--", lw=0.8)
    plt.xticks(k_len[k_symm], label)
    plt.xlabel("k (1/ang)")
    #plt.ylim(-6, 9)
    plt.ylabel("Energy (eV)")
    for i in range(len(k_symm)):
        plt.axvline(k_len[k_symm[i]], color="grey", lw=0.5)

    plt.axhline(0, color="grey", lw=0.5)
    plt.savefig("bands_in2o3.png", dpi=200)
