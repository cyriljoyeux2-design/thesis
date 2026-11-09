import json
from in2o3 import orbital_energy, SK_PARAMS

donnees = {"energy": orbital_energy, "sk_params": SK_PARAMS}

with open("params/test.json", "w") as f:
    json.dump(donnees, f, indent=2)
