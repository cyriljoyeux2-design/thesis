#==================================================================================================
# Imports
#==================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import tbplas as tb
from in2o3 import read_params, make_in2o3_orbitals, make_in2o3_cell
from qe_xml import bands_read, read_reciprocal, read_structure, read_nelec
import copy
np.set_printoptions(legacy="1.25")


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


def tb_bands(TB_cell, k_frac, n_occ):
    k_TB, bands = TB_cell.calc_bands(k_frac)
    bands = np.sort(bands, axis=1)

    vbm = bands[:, n_occ - 1].max()
    cbm = bands[:, n_occ].min()
    bands_shifted = bands - vbm

    return bands_shifted


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


def params_to_dicts(params, energy_start, sk_start, params_names):
    if len(params) != len(params_names):
        raise ValueError(f"La liste des paramètres ne contient pas le bon nombre de paramètres : {len(params_names)} paramètres listés pour {len(params)} paramètres au total")
    energy = copy.deepcopy(energy_start)
    sk_params = copy.deepcopy(sk_start)
    for i, name in enumerate(params_names):
        part = name.split("_")
        if part[0] == "E":
            if part[2] not in energy[part[1]]:
                raise ValueError(f"{name} n'est pas un paramètre valide")
            energy[part[1]][part[2]]= float(params[i])
        elif part[0] == "SK":
            if part[2] not in sk_params[part[1]]:
                raise ValueError(f"{name} n'est pas un paramètre valide")
            sk_params[part[1]][part[2]]= float(params[i])
        else:
            raise ValueError("La liste des parametres n'a pas le bon format (les onsites energies doivent commencer par 'E' et les hopping terms doivent commencer par 'SK')")

    return energy, sk_params


def dicts_to_params(energy, sk_params, params_names):
    params = []
    for i, name in enumerate(params_names):
        part = name.split("_")
        if part[0] == "E":
            param = energy[part[1]][part[2]]
        elif part[0] == "SK":
            param = sk_params[part[1]][part[2]]
        else:
            raise ValueError("La liste des parametres n'a pas le bon format (les onsites energies doivent commencer par 'E' et les hopping terms doivent commencer par 'SK')")
        params.append(param)
    params=np.array(params)

    return params

if __name__ == "__main__":
    file_name ="data-file-schema.xml"
    params_file_name = "params/test.json"
    energy_start, sk_start = read_params(params_file_name)
    PARAMS_NAME = ["E_In_s", "E_O_p", "SK_In-O_sps", "SK_O-O_pps", "SK_O-O_ppp", "SK_In-In_sss"]
    params = np.array([-4.7, -15.8, 0.55, 1.23, -0.03, -0.32])
    energy, sk_params = params_to_dicts(params, energy_start, sk_start, PARAMS_NAME)
    params_rt = dicts_to_params(energy_start, sk_start, PARAMS_NAME)
    e_zero, sk_zero = params_to_dicts(np.zeros(len(PARAMS_NAME)), energy_start, sk_start, PARAMS_NAME)
    assert (energy_start, sk_start) == params_to_dicts(params_rt, e_zero, sk_zero, PARAMS_NAME), \
           "Échec du test d'aller-retour depuis des dictionnaires à zéro"
    label = ["G", "N", "P", "G", "H", "N"]
    k_cart, energies_ev, nbnd, n_k = bands_read(file_name)
    alat, B = read_reciprocal(file_name)
    k_frac = cart_to_frac(k_cart, B)
    k_len = k_path_length(k_cart, alat)
    nelec = read_nelec(file_name)
    n_occ, VBM, CBM, gap, k_max, k_min, energies_fermi = align_bands(energies_ev, nelec)
    k_symm = find_corners(k_cart)
    band_group, n_band_group, en_band_group = find_manifolds(energies_fermi, nbnd)
    ref_atoms = read_structure(file_name)
    TB_cell = make_in2o3_cell(energy, sk_params, ref_atoms)
    check_lattice(TB_cell, ref_atoms)
    n_val = 3 * ref_atoms.get_chemical_symbols().count("O")
    assert n_val == n_occ - band_group[2,0], "Le dessus de la bande de valence n'est pas équivalente aux deux modeles"
    bands_shifted = tb_bands(TB_cell, k_frac, n_val)
    regions = {
            "valence":          (band_group[2, 0], n_band_group[2], 0, n_val), # 72 orbitales O2p
            "1re conduction":   (band_group[3, 0], 1, n_val, n_val + 1),
            "conduction haute": (band_group[3, 0] + 1, n_band_group[3] - 3, n_val + 1, None),
    }

    for nom, (first_dft, n_dft, first_band, last_band) in regions.items():
        window_dft = select_window(energies_fermi, first_dft, n_dft)
        RMS, d = band_errors(bands_shifted, window_dft, first_band, last_band)
        print("==============================================================================================")
        #print(first_band)
        idx_k, idx_b =  np.unravel_index(np.argmax(np.abs(d)), d.shape)
        print(f"Ecart quadratique des bandes d'énergies entre la référence dft et le modele TB de la bande de {nom}", RMS)
        print(f"Ecart max des bandes d'énergies entre la référence dft et le modele TB de la bande de {nom}", d[idx_k, idx_b])
        print(f"Position de l'ecart max des bandes d'énergies entre la référence dft et le modele TB de la bande de {nom}", k_frac[idx_k,:])


    plt.figure()
    plt.plot(k_len, select_window(energies_fermi, band_group[2, 0], TB_cell.num_orb), "k-", lw=0.8)
    #plt.plot(k_len, window, "r--", lw=1)
    plt.plot(k_len, bands_shifted, "r--", lw=0.8)
    plt.xticks(k_len[k_symm], label)
    plt.xlabel("k (1/ang)")
    plt.ylim(-6, 9)
    plt.ylabel("Energy (eV)")
    for i in range(len(k_symm)):
        plt.axvline(k_len[k_symm[i]], color="grey", lw=0.5)

    plt.axhline(0, color="grey", lw=0.5)
    plt.savefig("TB_dft.png", dpi=200)

    #==================================================================================================
    # Affichage
    #==================================================================================================
    #print(k_cart.shape)
    #print(energies_ev.shape)
    #print("Nombre de bandes:", nbnd)                     # 194
    #print("Nombre de k points:", n_k)                    # 151
    #print("Coordonnées du premier point k:", k_cart[0])  # [0 0 0]
    #print("Coordonnées du sernier point k:", k_cart[-1]) # [4.330127018922194E-01   4.330127018922194E-01  -1.306919376461998E-17]
    #print("Energie des 5 bandes les plus hautes au point gamma:", np.round(energies_ev[0,-5:],3))
    #print("Energie des 5 bandes les plus basses au point gamma:", np.round(energies_ev[0,:5],3))
    #print("==============================================================================================")
    #print("Paramètre de maille:", alat)
    #print("Vecteurs du réseau réciproque:", B)
    #print("Coordonnées réduite du premier point k (gamma):", np.round(k_frac[0],3))
    #print("Coordonnées réduite du dernier point k (N):", np.round(k_frac[-1],3))
    #print("Longueur totale du chemin k:", k_len[-1])
    #print("==============================================================================================")
    #print("Nombre de bandes occupées:", n_occ)
    #print("Energie du gap indirect:", gap)
    #print("Coordonnées réduites du haut de la bande de valence:", k_frac[k_max,:])
    #print("Coordonnées réduites du bas de la bande de conduction:", k_frac[k_min,:])
    #print("Coordonnées des points de hautes symétrie:")
    #print("G:", np.round(k_frac[k_symm[0],:],2))
    #print("N:", np.round(k_frac[k_symm[1],:],2))
    #print("P:", np.round(k_frac[k_symm[2],:],2))
    #print("G:", np.round(k_frac[k_symm[3],:],2))
    #print("H:", np.round(k_frac[k_symm[4],:],2))
    #print("N:", np.round(k_frac[k_symm[-1],:],2))
    #print("Indices des points de haute symmétrie:", k_symm)
    #print("Ecart entre le maximum de la bande de valence et l'énergie au point Gamma (eV):", VBM - energies_fermi[0, n_occ-1] )
    #print("==============================================================================================")
    #print(f"{'Groupe':>6} {'Première':>9} {'Dernière':>9} {'Nombre':>7} {'E_min (eV)':>11} {'E_max (eV)':>11}")
    #for i in range(len(band_group)):
    #    print(f"{i:6d} {band_group[i,0]:9d} {band_group[i,1]:9d} {n_band_group[i]:7d} {en_band_group[i,0]:11.2f} {en_band_group[i,1]:11.2f}")
    #print("==============================================================================================")
    #print(energy)
    #print(sk_params)
    print(energy)
    print(sk_params)
    print(params)
