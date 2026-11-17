"""
Input/output and conversion of the tight-binding parameters.

The parameters are stored in JSON files with two sections:
    "orbital_energy": {species: {orbital: on-site energy in eV}}
    "sk_params":      {"A-B": {"mode": "harrison" or "eV", "sss": ..., "sps": ..., ...}}
and an optional "fit" section describing how a fitted set was obtained.

A parameter is designated by a name:
    "E_<species>_<orbital>"  -> orbital_energy[species][orbital]   (ex. "E_In_s")
    "SK_<pair>_<parameter>"  -> sk_params[pair][parameter]         (ex. "SK_In-O_sps")
A list of such names (PARAMS_NAME) defines the order of the parameter vector used by
the optimiser.

Functions:
    read_params     : read a JSON file, with optional overridden values.
    write_params    : write a JSON file.
    params_to_dicts : parameter vector -> dictionaries.
    dicts_to_params : dictionaries -> parameter vector.
"""
import numpy as np
import json
import copy

def read_params(params_file_name, overrides=None):
    """
    Read a JSON parameter file.

    :param params_file_name: path to the .json file
    :param overrides: dict {name: value} of parameters to overwrite, with the names of
                      PARAMS_NAME (ex. {"SK_In-O_sps": 0.34}).
                      None: values of the file, unmodified.
    :return: (energy, sk_params), on-site energies and SK parameters dictionaries
    :raises ValueError: if an overridden name is not a valid parameter
    """
    with open(params_file_name) as f:
        data = json.load(f)

    energy = data["orbital_energy"]
    sk_params = data["sk_params"]

    if overrides:
        energy, sk_params = params_to_dicts(np.array(list(overrides.values())),
                                            energy, sk_params,
                                            list(overrides.keys()))
    return energy, sk_params


def write_params(params_file_name, energy, sk_params, fit_info=None):
    """
    Write a JSON parameter file, readable by read_params.

    :param params_file_name: path to the .json file (overwritten if it exists)
    :param energy: on-site energies dictionary
    :param sk_params: SK parameters dictionary
    :param fit_info: optional dict describing the fit (reference, weights, fitted
                     parameters...), stored in the "fit" section. Its values must be
                     JSON-serialisable (use .tolist() for numpy arrays, int() for numpy integers).
    """
    data = {"orbital_energy": energy, "sk_params": sk_params}
    if fit_info is not None:
        data["fit"] = fit_info
    with open(params_file_name, "w") as f:
        json.dump(data, f, indent=2)


def params_to_dicts(params, energy_start, sk_start, params_names):
    """
    Build the parameter dictionaries from a parameter vector.

    The starting dictionaries are deep-copied and left unchanged: only the parameters
    listed in params_names are replaced, all the others keep their starting value.

    :param params: (n_params,) float array, values in the order of params_names
    :param energy_start: on-site energies dictionary (complete)
    :param sk_start: SK parameters dictionary (complete)
    :param params_names: list of n_params parameter names ("E_..." or "SK_...")
    :return: (energy, sk_params), new dictionaries
    :raises ValueError: if params and params_names have different lengths, or if a
                        name does not match an existing entry of the dictionaries
    """
    if len(params) != len(params_names):
        raise ValueError(f"La liste des paramètres ne contient pas le bon nombre de paramètres : {len(params_names)} paramètres listés pour {len(params)} paramètres au total")
    energy = copy.deepcopy(energy_start)
    sk_params = copy.deepcopy(sk_start)
    for i, name in enumerate(params_names):
        part = name.split("_")
        if part[0] == "E":
            if part[2] not in energy[part[1]]:
                raise ValueError(f"{name} n'est pas un paramètre valide")
            energy[part[1]][part[2]]= float(params[i])
        elif part[0] == "SK":
            if part[2] not in sk_params[part[1]]:
                raise ValueError(f"{name} n'est pas un paramètre valide")
            sk_params[part[1]][part[2]]= float(params[i])
        else:
            raise ValueError("La liste des parametres n'a pas le bon format (les onsites energies doivent commencer par 'E' et les hopping terms doivent commencer par 'SK')")
    return energy, sk_params


def dicts_to_params(energy, sk_params, params_names):
    """
    Extract a parameter vector from the dictionaries (inverse of params_to_dicts).

    :param energy: on-site energies dictionary
    :param sk_params: SK parameters dictionary
    :param params_names: list of parameter names ("E_..." or "SK_...")
    :return: (n_params,) float array, values in the order of params_names
    :raises ValueError: if a name has neither the "E" nor the "SK" prefix
    :raises KeyError: if a name does not match an entry of the dictionaries
    """
    params = []
    for i, name in enumerate(params_names):
        part = name.split("_")
        if part[0] == "E":
            param = energy[part[1]][part[2]]
        elif part[0] == "SK":
            param = sk_params[part[1]][part[2]]
        else:
            raise ValueError("La liste des parametres n'a pas le bon format (les onsites energies doivent commencer par 'E' et les hopping terms doivent commencer par 'SK')")
        params.append(param)
    params=np.array(params)
    return params
