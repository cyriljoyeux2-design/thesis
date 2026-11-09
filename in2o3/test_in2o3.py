#! /usr/bin/env python

import numpy as np
from ase.io import write
from in2o3 import make_in2o3_atoms, make_in2o3_orbitals

atoms = make_in2o3_atoms()
print(atoms.get_chemical_formula(), len(atoms))

# Distances In–O
d = atoms.get_all_distances(mic=True)
sym = np.array(atoms.get_chemical_symbols())
d_in_o = d[np.ix_(sym == "In", sym == "O")]
print("Distances In–O < 2.5 Å :", np.unique(np.round(d_in_o[d_in_o < 2.5], 3)))
print("Nombre de O autour de chaque In :", (d_in_o < 2.5).sum(axis=1))

# Export pour VESTA
write("In2O3_prim.cif", atoms)
write("In2O3_conv.cif", make_in2o3_atoms(primitive=False))

cell = make_in2o3_orbitals()
print("Nombre d'orbitales :", cell.num_orb)

cell.plot()
