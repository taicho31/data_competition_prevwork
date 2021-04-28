import pandas as pd
import numpy as np
import time 
import basic as bs

def nearest_neighbour(city):
    ids = city.CityId.values[1:]
    xy = np.array([city.X.values, city.Y.values]).T[1:]
    path = [0,]
    while len(ids) > 0:
        last_x, last_y = city.X[path[-1]], city.Y[path[-1]]
        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
    path.append(0)
    return path


