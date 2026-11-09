import numpy as np
import tbplas as tb
from graphene import *
import matplotlib.pyplot as plt

class GrapheneFit(tb.ParamFit):
    def __init__(self, k_points, weights, t2_ref):
        self.t2_ref = t2_ref
        super().__init__(k_points, weights)

    def calc_bands_ref(self):
        cell = make_graphene_cell(sk_params={"ppp_1": -2.7, "ppp_2": self.t2_ref})

        k_len, bands = cell.calc_bands(self._k_points)

        return np.sort(bands, axis=1)

    def calc_bands_fit(self, params):
        e0, t_1 = params

        energy = {"C": {"pz": e0}}
        sk = {"ppp_1": t_1, "ppp_2": 0.0}

        cell = make_graphene_cell(energy=energy, sk_params=sk)

        k_len, bands = cell.calc_bands(self._k_points) # le "_" de _k_points est une convention utilisé dans cette version de TBPLaS mais peut être source d'erreur pour d'autres versions (AttributeError)

        return np.sort(bands, axis=1)

if __name__ == "__main__":
    # ----- test preliminaire -----
    k_points = tb.gen_kmesh((24, 24, 1))
    fit_test = GrapheneFit(k_points, np.ones(2), t2_ref=0.0)

    params_0 = np.array([0.0, -2.7])
    ref = fit_test.calc_bands_ref()
    mod = fit_test.calc_bands_fit(params_0)
    print(ref.shape, mod.shape)                  # (576, 2) pour les deux
    print(np.max(np.abs(ref - mod)))             # 0 à la précision numérique près
    params_0 = np.array([0.5, -2.7])
    mod2 = fit_test.calc_bands_fit(params_0)
    print(np.max(np.abs(ref - mod2)))            # doit valoir 0,5 : e0 décale les deux bandes en bloc
    print("==============================================")

    # ----- test premiers voisins -----
    weight = np.array([1.0, 1.0])
    params_A = np.array([0.5, -2.0])
    fit_A = GrapheneFit(k_points, weight, t2_ref=0.0)
    res_A = fit_A.fit(params_A)
    print("e0 et t après fitting")
    print(res_A)

    # ----- test premiers et seconds voisins -----
    weight = np.array([1.0, 1.0])
    params_B = np.array([0.5, -2.0])
    fit_B = GrapheneFit(k_points, weight, t2_ref=0.3)
    res_B = fit_B.fit(params_B)
    e0_fit, t_fit = res_B
    print("e0 et t_1 après fitting")
    print(res_B)
    RMS = np.sqrt(np.mean((fit_B.calc_bands_ref() - fit_B.calc_bands_fit(res_B))**2))
    print("RMS=", RMS)

    cell_ref = make_graphene_cell(sk_params={"ppp_1": -2.7, "ppp_2": 0.3})
    cell_fit = make_graphene_cell(energy={"C": {"pz": e0_fit}},
                                  sk_params={"ppp_1": t_fit, "ppp_2": 0.0})

    special = make_graphene_atoms().cell.bandpath(npoints=0).special_points

    k_label = ["G", "M", "K", "G"]
    k_points = np.array([special[l] for l in k_label])
    k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
    k_len, bands_ref = cell_ref.calc_bands(k_path) 
    k_len, bands_fit = cell_fit.calc_bands(k_path) 


    plt.figure()
    plt.plot(k_len, bands_ref, "k-", lw=0.8)
    plt.plot(k_len, bands_fit, "r--", lw=0.8)
    plt.xticks(k_len[k_idx], k_label)
    plt.xlabel("k (1/ang)")
    #plt.ylim(-6, 9)
    plt.ylabel("Energy (eV)")
    plt.axhline(0, color="grey", lw=0.5)
    plt.savefig("plot_fit_graphene.png", dpi=200)

    # ----- test pour différent poids -----
    weight = np.array([0.1, 1.0])
    params_B = np.array([0.5, -2.0])
    fit_B = GrapheneFit(k_points, weight, t2_ref=0.3)
    res_B = fit_B.fit(params_B)
    e0_fit, t_fit = res_B
    print("e0 et t_1 après fitting")
    print(res_B)
    RMS = np.sqrt(np.mean((fit_B.calc_bands_ref() - fit_B.calc_bands_fit(res_B))**2))
    print("RMS=", RMS)

    cell_ref = make_graphene_cell(sk_params={"ppp_1": -2.7, "ppp_2": 0.3})
    cell_fit = make_graphene_cell(energy={"C": {"pz": e0_fit}},
                                  sk_params={"ppp_1": t_fit, "ppp_2": 0.0})

    k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
    k_len, bands_ref = cell_ref.calc_bands(k_path)
    k_len, bands_fit = cell_fit.calc_bands(k_path)

    plt.figure()
    plt.plot(k_len, bands_ref, "k-", lw=0.8)
    plt.plot(k_len, bands_fit, "r--", lw=0.8)
    plt.xticks(k_len[k_idx], k_label)
    plt.xlabel("k (1/ang)")
    #plt.ylim(-6, 9)
    plt.ylabel("Energy (eV)")
    plt.axhline(0, color="grey", lw=0.5)
    plt.savefig("plot_fit_graphene_weights.png", dpi=200)
