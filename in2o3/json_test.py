import json

def read_params(params_file_name):
    with open(params_file_name) as f:
        data = json.load(f)

        orbital_energy = data["orbital_energy"]
        sk_params = data["sk_params"]

    return orbital_energy, sk_params

if __name__ == "__main__":
    params_file_name = "params/test.json"
    orbital_energy, sk_params = read_params(params_file_name)

    print(orbital_energy)
    print(sk_params)
