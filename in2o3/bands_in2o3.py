import numpy as np
import matplotlib.pyplot as plt
import tbplas as tb
from in2o3 import *
from qe_xml import *

params_file_name = "params/test.json"
energy, sk_params = read_params(params_file_name)

cell = make_in2o3_cell(energy, sk_params)

# Chemin k de la maille cubique centrée, dans le repère de la maille primitive
# [0.0 ,  0.0 , 0.0 ],  G
# [1./2, -1./2, 1./2],  H 
# [0.0 ,  0.0 , 1./2],  N
# [0.0 ,  0.0 , 0.0 ],  G
# [1./4,  1./4, 1./4],  P
# [1./2, -1./2, 1./2]   H

# ASE identifie le type de réseau de Bravais et retrouve les coordonnées fractionnaires des poits de hautes symétries de la ZB ainsi que le chemin recommandé
#special = make_in2o3_atoms(energy).cell.bandpath(npoints=0).special_points

#k_label = ["G", "H", "N", "G", "P", "H"]
#k_points = np.array([special[l] for l in k_label])
#k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40, 40, 40])
#k_len, bands = cell.calc_bands(k_frac)
#tb.Visualizer().plot_bands(k_len, bands, k_idx, k_label)


n_occ = 71

vbm = bands[:, n_occ - 1].max() 
cbm = bands[:, n_occ].min()          
bands_shifted = bands - vbm          
print(f"Gap = {cbm - vbm:.3f} eV")
print(bands_shifted[:, n_occ - 1].max())   # doit afficher 0.0
print(bands_shifted[:, n_occ].min())       # doit afficher le gap, 2.755

tb.Visualizer().plot_bands(k_len, bands_shifted, k_idx, k_label)
#vis.plot_bands(k_len, bands_shifted, k_idx, ["G", "H", "N", "G", "P"])


e_min, e_max = -2.0, 9.0   # eV, relatif au VBM

def beautifier(*args, **kwargs):
    plt.ylim(e_min, e_max)
    plt.axhline(0.0, color="gray", ls="--", lw=0.5)   # optionnel : repère du VBM

tb.Visualizer().plot_bands(
    k_len, bands_shifted, k_idx, k_label,
    fig_name="bands_In2O3.png",
    y_label=r"Energy (eV)",
    beautifier=beautifier,
)
