#==================================================================================================
# Imports
#==================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import tbplas as tb
from in2o3 import read_params, write_params, make_in2o3_orbitals, make_in2o3_cell
from qe_xml import bands_read, read_reciprocal, read_structure, read_nelec
from calc_bands import cart_to_frac, k_path_length, align_bands, find_corners, tb_bands, find_manifolds, select_window, check_lattice, band_errors
import copy
np.set_printoptions(legacy="1.25")


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

class In2O3Fit(tb.ParamFit):
    def __init__(self, k_points, weights, bands_ref, n_val, atoms, energy_ref, sk_ref, params_names):
        if bands_ref.shape[0] != len(k_points):
            raise ValueError(f"bands_ref a {bands_ref.shape[0]} points k, "
                             f"mais {len(k_points)} points k ont été fournis")
        self.bands_ref = bands_ref
        self.n_val = n_val
        self.params_names = params_names
        self.energy_ref = energy_ref
        self.sk_ref = sk_ref
        self.atoms = atoms
        super().__init__(k_points, weights)

    def calc_bands_ref(self):

        return self.bands_ref

    def calc_bands_fit(self, params):
        energy, sk_params = params_to_dicts(params, self.energy_ref, self.sk_ref, self.params_names)

        cell = make_in2o3_cell(energy, sk_params, self.atoms)
        bands = tb_bands(cell, self._k_points, self.n_val)

        return bands

if __name__ == "__main__":
    file_name ="data-file-schema.xml"
    params_file_name = "params/test.json"
    energy_ref, sk_ref = read_params(params_file_name)
    PARAMS_NAME = ["E_In_s", "SK_In-O_sps", "SK_O-O_pps", "SK_O-O_ppp", "SK_In-In_sss"]
    #params = np.array([-5.0, 0.27, 0.79, -0.21, -0.6])
    params_ref = dicts_to_params(energy_ref, sk_ref, PARAMS_NAME)
    energy_start, sk_start = params_to_dicts(params_ref, energy_ref, sk_ref, PARAMS_NAME)
    e_zero, sk_zero = params_to_dicts(np.zeros(len(PARAMS_NAME)), energy_ref, sk_ref, PARAMS_NAME)
    assert (energy_ref, sk_ref) == params_to_dicts(params_ref, e_zero, sk_zero, PARAMS_NAME), \
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
    TB_cell = make_in2o3_cell(energy_ref, sk_ref, ref_atoms)
    check_lattice(TB_cell, ref_atoms)
    n_val = 3 * ref_atoms.get_chemical_symbols().count("O")
    assert n_val == n_occ - band_group[2,0], "Le dessus de la bande de valence n'est pas équivalente aux deux modeles"
    bands_shifted = tb_bands(TB_cell, k_frac, n_val)
    weights = np.ones(TB_cell.num_orb)
    regions = {
            "valence":          (band_group[2, 0], n_band_group[2] - 3, 0, n_val - 3, 0.1), # 72 orbitales O2p
            "1re conduction":   (band_group[3, 0] - 3, 6, n_val - 3, n_val + 3, 10.0),
            "conduction haute": (band_group[3, 0] + 3, n_band_group[3] - 5, n_val + 3, None, 3.0),
    }
    for nom, (first_dft, n_dft, first_band, last_band, w) in regions.items():
        weights[first_band:last_band] = w
        window_dft = select_window(energies_fermi, first_dft, n_dft)
        RMS, d = band_errors(bands_shifted, window_dft, first_band, last_band)
        print("==============================================================================================")
        #print(first_band)
        idx_k, idx_b =  np.unravel_index(np.argmax(np.abs(d)), d.shape)
        print(f"Ecart quadratique des bandes d'énergies entre la référence dft et le modele TB de la bande de {nom}", RMS)
        print(f"Ecart max des bandes d'énergies entre la référence dft et le modele TB de la bande de {nom}", d[idx_k, idx_b])
        print(f"Position de l'ecart max des bandes d'énergies entre la référence dft et le modele TB de la bande de {nom}", k_frac[idx_k,:])

    cell_ref = make_in2o3_cell(energy_ref, sk_ref, ref_atoms)
    #bands_ref = tb_bands(cell_ref, k_frac, n_val)
    bands_ref = select_window(energies_fermi, band_group[2, 0], TB_cell.num_orb)
    fit_dft = In2O3Fit(k_frac, weights, bands_ref, n_val, ref_atoms, energy_start, sk_start, PARAMS_NAME)
    res_fit = fit_dft.fit(params_ref)
    energy_fit, sk_fit = params_to_dicts(res_fit, energy_ref, sk_ref, PARAMS_NAME)
    for nom, ref, fit in zip(PARAMS_NAME, params_ref, res_fit):
        print(f"{nom:>15} {ref:10.4f} {fit:10.4f}")
    cell_fit = make_in2o3_cell(energy_fit, sk_fit, ref_atoms)
    bands_fit = tb_bands(cell_fit, k_frac, n_val)
    write_params("params/fit_2026-11-13.json", energy_fit, sk_fit)

    for nom, (first_dft, n_dft, first_band, last_band, w) in regions.items():
        window_dft = select_window(energies_fermi, first_dft, n_dft)
        RMS, d = band_errors(bands_fit, window_dft, first_band, last_band)
        print("==============================================================================================")
        #print(first_band)
        idx_k, idx_b =  np.unravel_index(np.argmax(np.abs(d)), d.shape)
        print(f"Ecart quadratique des bandes d'énergies entre la référence dft et le modele TB de la bande de {nom}", RMS)
        print(f"Ecart max des bandes d'énergies entre la référence dft et le modele TB de la bande de {nom}", d[idx_k, idx_b])
        print(f"Position de l'ecart max des bandes d'énergies entre la référence dft et le modele TB de la bande de {nom}", k_frac[idx_k,:])
    
    plt.figure()
    #plt.plot(k_len, select_window(energies_fermi, band_group[2, 0], TB_cell.num_orb), "k-", lw=0.8)
    plt.plot(k_len, bands_ref, "k-", lw=1)
    plt.plot(k_len, bands_fit, "r--", lw=0.8)
    plt.xticks(k_len[k_symm], label)
    plt.xlabel("k (1/ang)")
    #plt.ylim(-6, 9)
    plt.ylabel("Energy (eV)")
    for i in range(len(k_symm)):
        plt.axvline(k_len[k_symm[i]], color="grey", lw=0.5)

    plt.axhline(0, color="grey", lw=0.5)
    plt.savefig("TB_fit.png", dpi=200)

    #==================================================================================================
    # Affichage
    #==================================================================================================
