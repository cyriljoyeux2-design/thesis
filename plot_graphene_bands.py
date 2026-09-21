"""
plot_graphene_bands.py — Structure de bande du graphène pur (sans gap).

À venir : plot_in2o3_bands.py suivra exactement ce même schéma —
importer build_in2o3_model() au lieu de build_graphene_model(),
et un chemin k propre à la zone de Brillouin 3D de In2O3
(via atoms.cell.bandpath() plutôt que les points Gamma/K/M codés en dur ici,
qui sont spécifiques au réseau hexagonal 2D du graphène).
"""
import matplotlib.pyplot as plt
from models import build_graphene_model
from plotting import plot_bands

my_model = build_graphene_model()

# Chemin dans la zone de Brillouin (spécifique au graphène)
k_nodes = [[0, 0], [2 / 3, 1 / 3], [1 / 2, 1 / 2], [0, 0]]
k_node_labels = (r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$")
nk = 121
k_path, k_dist, k_node_dist = my_model.k_path(k_nodes, nk)

# Évaluation avec des valeurs numériques concrètes
evals = my_model.solve_ham(k_path, delta=0.0, t=-1.0)

plot_bands(k_dist, k_node_dist, k_node_labels, evals,
           title="Graphène — δ=0 (cône de Dirac intact)", c="b")
plt.show()
