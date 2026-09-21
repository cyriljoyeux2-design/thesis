"""
plot_graphene_delta_sweep.py — Effet du paramètre delta sur le gap en K.

À venir : le même principe de balayage servira à explorer l'effet de
chaque paramètre TB de In2O3 (ex. t_InO) AVANT de lancer le fit complet
contre les données DFT du postdoc — utile pour choisir un point de
départ raisonnable (p0) pour scipy.optimize.least_squares.
"""
import numpy as np
import matplotlib.pyplot as plt
from models import build_graphene_model
from plotting import plot_bands

my_model = build_graphene_model()

k_nodes = [[0, 0], [2 / 3, 1 / 3], [1 / 2, 1 / 2], [0, 0]]
k_node_labels = (r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$")
nk = 121
k_path, k_dist, k_node_dist = my_model.k_path(k_nodes, nk)

delta_values = np.array([0.0, 0.2, 0.5])
evals_sweep = my_model.solve_ham(k_path, delta=delta_values, t=-1.0)
# evals_sweep a la forme (Nk, len(delta_values), nbandes)

fig, ax = plt.subplots()
colors = ["k", "b", "r"]
for i, d in enumerate(delta_values):
    plot_bands(k_dist, k_node_dist, k_node_labels, evals_sweep[:, i, :],
               ax=ax, c=colors[i], label=f"δ={d}")

ax.legend()
ax.set_title("Effet de δ sur le cône de Dirac en K")
plt.show()
