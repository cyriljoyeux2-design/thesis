"""
plotting.py — Fonctions de visualisation communes à tous les modèles TB.

Ce module ne dépend d'aucun matériau précis — il prend en entrée des
tableaux numériques (k_dist, evals, ...) déjà calculés par PythTB.
Servira tel quel pour In2O3, aucune modification nécessaire ici.
"""
import matplotlib.pyplot as plt


def plot_bands(k_dist, k_node_dist, k_node_labels, evals, ax=None, title="", **plot_kwargs):
    """
    Trace une structure de bande.

    evals peut être 2D (Nk, nbandes) pour un seul jeu de paramètres,
    ou on peut appeler cette fonction plusieurs fois sur le même `ax`
    pour superposer plusieurs jeux de paramètres (voir le script de sweep).
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xlim(k_node_dist[0], k_node_dist[-1])
    ax.set_xticks(k_node_dist)
    ax.set_xticklabels(k_node_labels)
    for n in k_node_dist:
        ax.axvline(x=n, linewidth=0.5, color="k")

    ax.plot(k_dist, evals, **plot_kwargs)
    ax.set_title(title)
    ax.set_xlabel("Chemin dans l'espace k")
    ax.set_ylabel("Énergie")
    return ax
