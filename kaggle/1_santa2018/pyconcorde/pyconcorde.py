from concorde.tsp import TSPSolver

import numpy as np
import pandas as pd
import time
import basic as bs
from basic import total_distance

total_dist =1555685 
city = pd.read_csv("../input/not_prime_city.csv")
prime_cities = bs.sieve_of_eratosthenes(max(city.CityId))

for i in range(10):
    solver = TSPSolver.from_data(
    city.X,
    city.Y,
    norm="EUC_2D"
)

    t = time.time()
    tour_data = solver.solve(time_bound = 3.0, verbose = True, random_seed = 43) # solve() doesn't seem to respect time_bound for certain values?

    print(time.time() - t)
    print(tour_data.found_tour)
    tmp_distance = total_distance(city, np.append(tour_data.tour,[0]),prime_cities)/1000
    print('Total distance with the Nearest Neighbor path '+  "is {:,}".format(tmp_distance))
    if tmp_distance < total_dist:
        total_dist = tmp_distance
        pd.DataFrame({'Path': np.append(tour_data.tour,[0])}).to_csv('../input/s_not_prime_city_path_'+str(total_dist)+'.csv', index=False)
