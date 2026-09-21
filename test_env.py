"""
test_env.py — Sanity check de l'environnement tb-env.
À lancer une fois après toute modification de l'environnement conda.
"""
import ase
import pythtb

print("ASE:", ase.__version__)
print("PythTB:", pythtb.__version__)
