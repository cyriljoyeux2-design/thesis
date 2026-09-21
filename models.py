"""
models.py — Constructeurs de modèles tight-binding.

Chaque fonction construit et retourne un TBModel prêt à l'emploi.
Ce fichier ne contient QUE des définitions (def) — rien ne s'exécute
si tu l'importes ailleurs.

À venir : build_in2o3_model() sera ajoutée ici, à côté de build_graphene_model().
Elle utilisera structures.in2o3_primitive() pour les vecteurs de maille et les
positions atomiques (au lieu des lat_vecs/orb_vecs codés en dur ci-dessous,
qui n'existent que parce que le graphène est un cas jouet à 2 atomes sans
vraie structure cristalline 3D à importer).
"""
from pythtb import TBModel, Lattice
import numpy as np


def build_graphene_model():
    """
    Modèle tight-binding du graphène, paramétrique.

    Paramètres du modèle (à fournir lors de l'évaluation, ex. solve_ham(..., delta=0.2, t=-1.0)) :
        delta : décalage d'énergie entre les deux sous-réseaux (site A à -delta, site B à +delta).
                delta=0 -> graphène pur, cône de Dirac intact en K.
                delta!=0 -> ouvre un gap en K (brise la symétrie des deux sous-réseaux).
        t     : intégrale de saut entre premiers voisins (même valeur pour les 3 liaisons,
                le graphène n'a qu'un seul type de liaison C-C).
    """
    # Vecteurs de maille du réseau hexagonal (2D)
    lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
    # Position des deux atomes de la maille (coordonnées fractionnaires)
    orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]

    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=...)  # ... = toutes les directions périodiques
    model = TBModel(lat)

    # Énergies sur site symboliques : +delta et -delta sur les deux sous-réseaux
    model.set_onsite(lambda delta: -delta, ind_i=0)
    model.set_onsite(lambda delta: delta, ind_i=1)

    # Saut symbolique — même paramètre "t" pour les 3 liaisons vers les 3 premiers voisins
    model.set_hop("t", 0, 1, [0, 0])
    model.set_hop("t", 1, 0, [1, 0])
    model.set_hop("t", 1, 0, [0, 1])

    return model


# ------------------------------------------------------------------
# EMPLACEMENT FUTUR : build_in2o3_model()
#
# def build_in2o3_model():
#     from structures import in2o3_primitive
#     atoms = in2o3_primitive()
#     lat_vecs = atoms.cell.array
#     orb_vecs = atoms.get_scaled_positions()   # + gestion multi-orbitales par atome
#     ...
#     # onsite symboliques par élément/site (e_In_8b, e_In_24d, e_O, ...)
#     # hoppings symboliques par type de liaison (t_InO, ...), générés via ase.neighborlist
#     return model
# ------------------------------------------------------------------
