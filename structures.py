"""
structures.py — Géométries cristallines construites via ASE.

Contrairement au graphène (réseau 2D construit à la main dans models.py),
les matériaux 3D réels comme In2O3 passent par ase.spacegroup.crystal(),
qui a besoin d'un groupe d'espace et de positions de Wyckoff.
"""
from ase.spacegroup import crystal


def in2o3_primitive(a=10.117):
    """Maille primitive de In2O3 bixbyite (Ia-3, Marezio 1966) — 40 atomes."""
    return crystal(
        symbols=['In', 'In', 'O'],
        basis=[
            (0.25,  0.25, 0.25),
            (0.467, 0.0,  0.25),
            (0.391, 0.153, 0.383),
        ],
        spacegroup=206,
        cellpar=[a, a, a, 90, 90, 90],
        primitive_cell=True,
    )


def in2o3_conventional(a=10.117):
    """Maille conventionnelle de In2O3 bixbyite — 80 atomes."""
    return crystal(
        symbols=['In', 'In', 'O'],
        basis=[
            (0.25,  0.25, 0.25),
            (0.467, 0.0,  0.25),
            (0.391, 0.153, 0.383),
        ],
        spacegroup=206,
        cellpar=[a, a, a, 90, 90, 90],
    )
