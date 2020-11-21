import pandas as pd

city_mod = pd.read_csv("cities_mod.csv")
city_mod["X"] = city_mod["X"] *1000
city_mod["Y"] = city_mod["Y"] * 1000
city_mod.to_csv("cities_mod.csv", index=False)
