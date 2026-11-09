from graphene import make_graphene_cell

# Test 1 : paramètres par défaut (ppp_2 = 0)
cell = make_graphene_cell()
cell.print()

# Test 2 : avec seconds voisins
cell = make_graphene_cell(sk_params={"ppp_1": -2.7, "ppp_2": 0.3})
cell.print()
