import pandas as pd
import numpy as np
from basic import sieve_of_eratosthenes
from basic import total_distance
from basic import sub_distance
from intersect_check_and_swap import city_swap3
from intersect_check_and_swap import intersect_mod
import greedy_algorithm as ga
import time
import opt
from swap import prime_swap
from swap import shuffle

city = pd.read_csv("../input/cities_mod.csv") 

prime_cities = sieve_of_eratosthenes(max(city.CityId))
city[prime_cities].to_csv("../input/prime_city.csv", index=False)

not_prime_cities = list(~np.array(prime_cities))
city[not_prime_cities].to_csv("../input/not_prime_city.csv", index=False)

