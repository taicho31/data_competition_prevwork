import pandas as pd
from basic import total_distance
from basic import sieve_of_eratosthenes

city = pd.read_csv("cities.csv")
LKH = pd.read_csv("santa_output1227.csv")
prime_cities = sieve_of_eratosthenes(max(city.CityId))

LKH2 = pd.DataFrame(LKH[5:-2]).reset_index(drop=True)
LKH2.columns = ["Path"]
LKH2["Path"] = LKH2["Path"].astype("int64")
LKH2["Path"] -=1
lkh_path = list(LKH2["Path"])+ [0]
dist = total_distance(city, lkh_path, prime_cities)
print(dist)

pd.DataFrame({'Path':lkh_path}).to_csv('lkh_path_'+str(dist)+'.csv',index  = False)
