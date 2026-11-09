import xml.etree.ElementTree as ET         # lecture des fichiers XML
from ase.units import Hartree, Bohr
import numpy as np
from ase import Atoms

#==================================================================================================
# Lecture du fichier XLM et séparation en plusieurs bloc
#==================================================================================================

def bands_read(file_name):
    # ET.parse() extrait la structure du fichier sous forme d'arbre (dossier/sous-dosier/...) et
    # getroot() extrait la racine (<qes:espresso>)
    root = ET.parse(file_name).getroot()

    # extraction du bloc contennant les résultats des calculs
    bloc_output = root.find("output")
    bloc_band = bloc_output.find("band_structure")
    bloc_k = bloc_band.findall("ks_energies/k_point")
    bloc_eigenvalues = bloc_band.findall("ks_energies/eigenvalues")

    #==================================================================================================
    # Info du calcul
    #==================================================================================================

    nbnd = int(bloc_band.find("nbnd").text) # nbr de bandes
    n_k = int(bloc_band.find("nks").text) # nbr de k points

    k_cart = []
    for k in bloc_k:
        comps = k.text.split()
        position = []
        for comp in comps:
            position.append(float(comp))
        k_cart.append(position)
    k_cart = np.array(k_cart)

    if k_cart.shape != (n_k, 3):
        raise ValueError(f"Forme innatendue pour la liste des k points : {k_cart.shape} au lieu de {(n_k, 3)}")

    energies = []
    for e in bloc_eigenvalues:
        eigenvalues = e.text.split()
        energie = []
        for eigenvalue in eigenvalues:
            energie.append(float(eigenvalue))
        energies.append(energie)
    energies_hartree = np.array(energies)
    energies_ev = energies_hartree * Hartree

    if energies_ev.shape != (n_k, nbnd):
        raise ValueError(f"Forme innatendue pour la liste des bandes : {energies_ev.shape} au lieu de {(n_k, nbnd)}")

    return k_cart, energies_ev, nbnd, n_k



def read_reciprocal(file_name):
    root = ET.parse(file_name).getroot()
    bloc_output = root.find("output")
    bloc_b = bloc_output.find("basis_set/reciprocal_lattice")
    alat = float(bloc_output.find("atomic_structure").get("alat")) * Bohr

    B = []
    for b in bloc_b:
        vects = b.text.split()
        vec_reciproc = []
        for vect in vects:
            vec_reciproc.append(float(vect))
        B.append(vec_reciproc)
    B = np.array(B)

    return alat, B

def read_nelec(file_name):
    root = ET.parse(file_name).getroot()
    bloc_output = root.find("output")
    nelec = int(float(bloc_output.find("band_structure/nelec").text))

    if nelec % 2 != 0:
        raise ValueError(f"Nombre impaire d'électron détecté (nelec).")

    return nelec

def read_structure(file_name):
    root = ET.parse(file_name).getroot()
    bloc_struct = root.find("output/atomic_structure")

    # Sécurité si le bloc n'est pas trouvé
    if bloc_struct is None:
        raise ValueError("Balise <atomic_structure> introuvable dans le fichier")

    bloc_maille = bloc_struct.find("cell")
    bloc_atomes = bloc_struct.findall("atomic_positions/atom")

    # -------------------------------------------------------------------------------------------------
    # Vecteurs de maille
    # -------------------------------------------------------------------------------------------------
    vecteurs_maille = []                 # contiendra 3 listes de 3 nombres

    for a in ["a1", "a2", "a3"]:
        composantes = bloc_maille.find(a).text.split()

        vecteur = []
        for composante in composantes:
            vecteur.append(float(composante))

        vecteurs_maille.append(vecteur)

    # Liste de listes -> tableau numpy 3x3, puis conversion Bohr -> Å
    cell_bohr = np.array(vecteurs_maille)
    cell_angstrom = cell_bohr * Bohr

    # -------------------------------------------------------------------------------------------------
    # Elément et position des atomes
    # -------------------------------------------------------------------------------------------------
    symbols = []                        # ['In', 'In', ..., 'O', ...]
    positions_bohr = []                  # [[x, y, z], [x, y, z], ...]

    for atome in bloc_atomes:
        # L'espèce est un ATTRIBUT de la balise : name="In"
        symbole = atome.get("name")
        symbols.append(symbole)

        # Les coordonnées sont le TEXTE entre les balises : "x y z"
        composantes = atome.text.split()
        position = []
        for composante in composantes:
            position.append(float(composante))

        positions_bohr.append(position)

    # Liste de listes -> tableau numpy (N, 3), puis conversion Bohr -> Å
    positions_bohr = np.array(positions_bohr)
    positions_angstrom = positions_bohr * Bohr

    return Atoms(symbols=symbols, cell=cell_angstrom,
                 positions=positions_angstrom, pbc=True)
