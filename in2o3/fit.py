"""
Fit of tight-binding parameters for In2O3 on a reference band structure, using TBPLaS.

The reference is currently a DFT band structure computed with Quantum ESPRESSO
(data-file-schema.xml). Another reference format can be used, as long as it provides
the primitive cell, the k-points and the band energies (see qe_xml.py).

Workflow:
    1. read the reference (structure, k-points, bands) and align it on its VBM;
    2. build the starting model from a parameter file (JSON);
    3. fit the parameters listed in PARAMS_NAME, the others being kept fixed;
    4. report the errors per band region and write the fitted parameters to a new JSON file.

Modules:
    qe_xml.py     : reads the QE XML file (structure, reciprocal lattice, number of
                    electrons, k-points, band energies).
    in2o3.py      : builds the TBPLaS primitive cell (orbitals and hopping terms) from
                    a structure and the parameter dictionaries.
    calc_bands.py : band-structure tools (k-point conversion, alignment, band groups,
                    windows, errors) and plot of the bands for a given parameter file.
    params_io.py  : reads, writes and converts the tight-binding parameters (JSON files,
                    dictionaries and parameter vectors).

Usage:
    Set the input files, PARAMS_NAME, OVERRIDES and the weights in the main section,
    then run: python fit.py
"""
import numpy as np
import matplotlib.pyplot as plt
import tbplas as tb
from in2o3 import make_in2o3_orbitals, make_in2o3_cell
from qe_xml import bands_read, read_reciprocal, read_structure, read_nelec
from calc_bands import cart_to_frac, k_path_length, align_bands, find_corners, tb_bands, find_manifolds, select_window, check_lattice, band_errors
from params import read_params, write_params, params_to_dicts, dicts_to_params
from pathlib import Path
np.set_printoptions(legacy="1.25")

class In2O3Fit(tb.ParamFit):
    """
    Fit of the In2O3 tight-binding parameters on a fixed reference band structure.

    The reference bands are given at creation and are not recomputed. The model bands
    are built from the parameter vector, sorted and aligned on their VBM, like the
    reference.
    """
    def __init__(self, k_points, weights, bands_ref, n_val, atoms, energy_ref, sk_ref, params_names):
        """
        :param k_points: (n_k, 3) float array, fractional k-points of the reference
        :param weights: (n_orb,) float array, fitting weight of each band
        :param bands_ref: (n_k, n_orb) float array, reference bands in eV, sorted and
                          aligned on the VBM, same k-points and band order as the model
        :param n_val: number of valence bands of the model
        :param atoms: ase.Atoms, structure of the model
        :param energy_ref: on-site energies dictionary; provides the values of the
                           parameters that are not fitted
        :param sk_ref: SK parameters dictionary; same role as energy_ref
        :param params_names: names of the fitted parameters (order of the parameter vector)
        :raises ValueError: if bands_ref and k_points do not have the same number of k-points
        """
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
        """
        Return the reference bands given at creation (no calculation).

        :return: (n_k, n_orb) float array, in eV
        """
        return self.bands_ref

    def calc_bands_fit(self, params):
        """
        Compute the model bands for a parameter vector (called by the optimiser).

        :param params: (n_params,) float array, in the order of params_names
        :return: (n_k, n_orb) float array, in eV, sorted and aligned on the model VBM
        """
        energy, sk_params = params_to_dicts(params, self.energy_ref, self.sk_ref, self.params_names)

        cell = make_in2o3_cell(energy, sk_params, self.atoms)
        bands = tb_bands(cell, self._k_points, self.n_val)
        return bands


def report_errors(bands_tb, bands_ref, regions, k_frac):
    """
    Print the RMS and the largest difference between model and reference, per band region.

    :param bands_tb: (n_k, n_orb) float array, model bands in eV
    :param bands_ref: (n_k, n_orb) float array, reference bands in eV
    :param regions: dict {name: slice}, band indices of each region (window indices)
    :param k_frac: (n_k, 3) float array, fractional k-points (to locate the largest difference)
    """
    for nom, sl in regions.items():
        rms, d = band_errors(bands_tb[:, sl], bands_ref[:, sl])
        idx_k, idx_b = np.unravel_index(np.argmax(np.abs(d)), d.shape)
        print(f"{nom:>18} : RMS = {rms:.3f} eV, écart max = {d[idx_k, idx_b]:+.3f} eV en k = {k_frac[idx_k]}")


if __name__ == "__main__":
    # ----- inputs -----
    file_name = Path.home() / "work" / "thesis-data" / "in2o3" / "dft_reference" / "data-file-schema.xml"
    params_file_name = "params/ref.json"
    fitted_params_file_name = Path.home() / "work" / "thesis-data" / "in2o3" / "results" / "fit_2026-11-13.json"
    PARAMS_NAME = ["E_In_s", "SK_In-O_sps", "SK_O-O_pps", "SK_O-O_ppp", "SK_In-In_sss"]
    OVERRIDES = {}

    # ----- dft variables -----
    k_cart, energies_ev, nbnd, n_k = bands_read(file_name)
    alat, B = read_reciprocal(file_name)
    k_frac = cart_to_frac(k_cart, B)
    k_len = k_path_length(k_cart, alat)
    nelec = read_nelec(file_name)
    n_occ, VBM, CBM, gap, k_max, k_min, energies_fermi = align_bands(energies_ev, nelec)
    k_symm = find_corners(k_cart)
    band_group, n_band_group, en_band_group = find_manifolds(energies_fermi, nbnd)
    ref_atoms = read_structure(file_name)

    # ----- starting point -----
    energy_ref, sk_ref = read_params(params_file_name, overrides=OVERRIDES)
    #energy_start, sk_start = params_to_dicts(params, energy_ref, sk_ref, PARAMS_NAME)
    params_ref = dicts_to_params(energy_ref, sk_ref, PARAMS_NAME)
    e_zero, sk_zero = params_to_dicts(np.zeros(len(PARAMS_NAME)), energy_ref, sk_ref, PARAMS_NAME)
    assert (energy_ref, sk_ref) == params_to_dicts(params_ref, e_zero, sk_zero, PARAMS_NAME), \
           "Échec du test d'aller-retour depuis des dictionnaires à zéro"
    cell_start = make_in2o3_cell(energy_ref, sk_ref, ref_atoms)
    #cell_ref = make_in2o3_cell(energy_ref, sk_ref, ref_atoms)
    check_lattice(cell_start, ref_atoms)
    n_orb = cell_start.num_orb
    n_val = 3 * ref_atoms.get_chemical_symbols().count("O")
    assert n_val == n_occ - band_group[2,0], "Le dessus de la bande de valence n'est pas équivalente aux deux modeles"
    bands_ref = select_window(energies_fermi, band_group[2, 0], n_orb)
    bands_start = tb_bands(cell_start, k_frac, n_val)

    # ----- cutting the bands in multiple region -----
    n_bords = 5
    weights = np.ones(n_orb)
    weights[:n_val - n_bords] = 0.5                    # valence profonde
    weights[n_val - n_bords:n_val + n_bords] = 20.0    # bord du gap
    weights[n_val + n_bords:] = 3.0                    # conduction haute

    regions = {
        "valence profonde": slice(0, n_val - n_bords),
        "bord du gap":      slice(n_val - n_bords, n_val + n_bords),
        "conduction haute": slice(n_val + n_bords, None),
    }

    # ----- RMS before fitting -----
    print("\n===== Before fitting =====")
    report_errors(bands_start, bands_ref, regions, k_frac)

    # ----- fitting -----
    fit_dft = In2O3Fit(k_frac, weights, bands_ref, n_val, ref_atoms, energy_ref, sk_ref, PARAMS_NAME)
    res_fit = fit_dft.fit(params_ref)

    # ----- results of the fit -----
    energy_fit, sk_fit = params_to_dicts(res_fit, energy_ref, sk_ref, PARAMS_NAME)
    write_params(fitted_params_file_name, energy_fit, sk_fit,
                 {"reference": str(file_name), "start": params_file_name,
                  "dft_window": [int(band_group[2, 0]), int(band_group[2, 0] + n_orb - 1)],
                  "weights": weights.tolist(), "fitted_parameters": PARAMS_NAME})
    print(f"\n{'parameters':>15} {'start':>10} {'fitted':>10}")
    for nom, ref, fit in zip(PARAMS_NAME, params_ref, res_fit):
        print(f"{nom:>15} {ref:10.4f} {fit:10.4f}")

    # ----- RMS after fitting -----
    cell_fit = make_in2o3_cell(energy_fit, sk_fit, ref_atoms)
    bands_fit = tb_bands(cell_fit, k_frac, n_val)
    print("\n===== After fitting =====")
    report_errors(bands_fit, bands_ref, regions, k_frac)
    gap_tb = bands_fit[:, n_val].min() - bands_fit[:, n_val - 1].max()
    print(f"\nGap : DFT = {gap:.3f} eV, TB = {gap_tb:.3f} eV")
    
    # ----- plot settings -----
    plt.figure()
    label = ["G", "N", "P", "G", "H", "N"]
    plt.plot(k_len, bands_ref, "k-", lw=1)
    plt.plot(k_len, bands_fit, "r--", lw=0.8)
    plt.xticks(k_len[k_symm], label)
    plt.xlabel("k (1/ang)")
    #plt.ylim(-6, 9)
    plt.ylabel("Energy (eV)")
    for i in range(len(k_symm)):
        plt.axvline(k_len[k_symm[i]], color="grey", lw=0.5)

    plt.axhline(0, color="grey", lw=0.5)
    plt.savefig(Path.home() / "work" / "thesis-data" / "in2o3" / "results" / "TB_fit.png", dpi=200)
