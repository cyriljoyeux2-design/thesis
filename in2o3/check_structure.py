"""
Lecture de la structure (maille + atomes) depuis le fichier XML
de Quantum ESPRESSO (data-file-schema.xml) et conversion en objet ASE.

Version volontairement détaillée : une opération par ligne,
boucles explicites, variables intermédiaires.
"""

#==================================================================================================
# Imports
#==================================================================================================
import xml.etree.ElementTree as ET         # lecture des fichiers XML
import numpy as np
from ase import Atoms              
from ase.units import Bohr                 # 1 Bohr = 0.529177 Å
from ase.neighborlist import neighbor_list
from in2o3 import make_in2o3_atoms


#==================================================================================================
# Constantes
#==================================================================================================
# Tolérances
TOL_LEN = 1e-3    # Å
TOL_ANG = 1e-2    # degrés
TOL_FRAC = 1e-3   # coordonnées réduites

# Couches comparées : (centre, voisin) -> nombre de couches à vérifier
NB_COUCHES = {
    ("In(8b)",  "O"):  1,   # 6 O à ~2.18 Å
    ("In(24d)", "O"):  3,   # 2 O à ~2.12, ~2.19 et ~2.22 Å
}

#==================================================================================================
# Lecture du fichier XLM et séparation en plusieurs bloc
#==================================================================================================
file_name="data-file-schema.xml"

# ET.parse() extrait la structure du fichier sous forme d'arbre (dossier/sous-dosier/...) et
# getroot() extrait la racine (<qes:espresso>)
root = ET.parse("data-file-schema.xml").getroot()

# extraction du bloc contennant toute la structure du système (maille, atome)
bloc_struct = root.find("output/atomic_structure")

# Sécurité si le bloc n'est pas trouvé
if bloc_struct is None:
    raise ValueError("Balise <atomic_structure> introuvable dans le fichier")

# extraction du bloc contennant les vecteurs de maille
bloc_maille = bloc_struct.find("cell")

# extraction du bloc contennant la liste des éléments et des coordonnées des atomes
bloc_atome = bloc_struct.findall("atomic_positions/atom")


#==================================================================================================
# Lecture de la structure de référence
#==================================================================================================
# -------------------------------------------------------------------------------------------------
# Vecteurs de maille
# -------------------------------------------------------------------------------------------------
vecteurs_maille = []                 # contiendra 3 listes de 3 nombres

for a in ["a1", "a2", "a3"]:
    composantes = bloc_maille.find(a).text.split()     # ex. <a1>19.12 0.0 0.0</a1>

    vecteur = []
    for composante in composantes:
        vecteur.append(float(composante))

    vecteurs_maille.append(vecteur)

# Liste de listes -> tableau numpy 3x3, puis conversion Bohr -> Å
cell_bohr = np.array(vecteurs_maille)
cell_angstrom = cell_bohr * Bohr

print("Vecteurs de maille (Å) :")
print(cell_angstrom)


# -------------------------------------------------------------------------------------------------
# Elément et position des atomes
# -------------------------------------------------------------------------------------------------
symboles = []                        # ['In', 'In', ..., 'O', ...]
positions_bohr = []                  # [[x, y, z], [x, y, z], ...]

for atome in bloc_atome:
    # L'espèce est un ATTRIBUT de la balise : name="In"
    symbole = atome.get("name")
    symboles.append(symbole)

    # Les coordonnées sont le TEXTE entre les balises : "x y z"
    composantes = atome.text.split()
    position = []
    for composante in composantes:
        position.append(float(composante))

    positions_bohr.append(position)

# Liste de listes -> tableau numpy (N, 3), puis conversion Bohr -> Å
positions_bohr = np.array(positions_bohr)
positions_angstrom = positions_bohr * Bohr

print("Nombre d'atomes lus :", len(symboles))


#==================================================================================================
# Maille de référence construite avec ASE
#==================================================================================================
ref = Atoms(
    symbols=symboles,                # espèces chimiques
    positions=positions_angstrom,    # positions cartésiennes en Å
    cell=cell_angstrom,              # vecteurs de maille en Å
    pbc=True,                        # cristal : périodique dans les 3 directions
)


#==================================================================================================
# Grandeurs comparées
#==================================================================================================
def lattice_3d(atoms):
    a1, a2, a3 = atoms.cell[0], atoms.cell[1], atoms.cell[2]
    la1, la2, la3 = np.linalg.norm(a1), np.linalg.norm(a2),  np.linalg.norm(a3)
    gamma1, gamma2, gamma3 = np.degrees(np.arccos(np.dot(a1, a2) / (la1 * la2))), np.degrees(np.arccos(np.dot(a2, a3) / (la2 * la3))), np.degrees(np.arccos(np.dot(a1, a3) / (la1 * la3)))
    return la1, la2, la3, gamma1, gamma2, gamma3

def neighbor_shells(atoms, cutoff=4.2, tol=5e-3, tol_site=0.02):
    """
    Couches de voisins par paire (centre → voisin) pour In2O3.
    Les In sont séparés en In(8b) (6 O équidistants) et In(24d) (3 distances In-O).
    Renvoie {("In(8b)", "O"): [(d, nb de voisins par centre), ...], ...}
    """
    sym = np.array(atoms.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", atoms, cutoff)

    # Étiquette de site : In(8b) si les 6 premiers In-O sont égaux, sinon In(24d)
    site = sym.astype(object)
    for c in np.where(sym == "In")[0]:
        d_in_o = np.sort(d[(i == c) & (sym[j] == "O")])[:6]
        site[c] = "In(8b)" if d_in_o[-1] - d_in_o[0] < tol_site else "In(24d)"

    shells = {}
    for (a, b), n_couches in NB_COUCHES.items():
        n_centers = np.sum(site == a)
        if n_centers == 0:
            continue
        mask = (site[i] == a) & (sym[j] == b)
        ds = np.sort(d[mask])
        groups = []                                  # [somme des d, nombre]
        for x in ds:
            if groups and abs(x - groups[-1][0] / groups[-1][1]) < tol:
                groups[-1][0] += x
                groups[-1][1] += 1
            else:
                groups.append([x, 1])
        shells[(a, b)] = [(g[0] / g[1], g[1] / n_centers) for g in groups[:n_couches]]
    return shells

#==================================================================================================
# Affichage
#==================================================================================================
results = []


def compare(name, ref, val, tol, unit=""):
    ok = abs(ref - val) < tol
    results.append(ok)
    flag = "OK   " if ok else "ÉCART"
    print(f"  [{flag}] {name:<28} ref = {ref:9.4f}   TB = {val:9.4f} {unit}"
          + ("" if ok else f"   (Δ = {val - ref:+.4f})"))


def main():
    mine = make_in2o3_atoms()

    print("\n== Réseau dans le plan ==")
    r1, r2, r3, rg1, rg2, rg3 = lattice_3d(ref)
    m1, m2, m3, mg1, mg2, mg3 = lattice_3d(mine)
    compare("|a1|", r1, m1, TOL_LEN, "Å")
    compare("|a2|", r2, m2, TOL_LEN, "Å")
    compare("|a3|", r3, m3, TOL_LEN, "Å")
    compare("angle (a1, a2)", rg1, mg1, TOL_ANG, "°")
    compare("angle (a2, a3)", rg2, mg2, TOL_ANG, "°")
    compare("angle (a1, a3)", rg3, mg3, TOL_ANG, "°")

    print("\n== Premiers voisins (distance, nombre par centre) ==")
    rs, ms = neighbor_shells(ref), neighbor_shells(mine)
    for (a, b), n in NB_COUCHES.items():
        print(f"  {b} autour de {a} :")
        if (a, b) not in rs or (a, b) not in ms or len(ms[(a, b)]) < n:
            print(f"          couches manquantes (sites In mal identifiés ?)")
            results.append(False)
            continue
        for k in range(n):
            (rd, rn), (md, mn) = rs[(a, b)][k], ms[(a, b)][k]
            compare(f"  couche {k + 1} : distance", rd, md, TOL_LEN, "Å")
            if abs(rn - mn) > 1e-6:
                print(f"          nombre de voisins : ref = {rn:g}, TB = {mn:g}")
                results.append(False)

    n_bad = results.count(False)
    print("\n" + ("Maille conforme à la référence."
                  if n_bad == 0 else f"{n_bad} écart(s) détecté(s)."))


if __name__ == "__main__":
    main()

