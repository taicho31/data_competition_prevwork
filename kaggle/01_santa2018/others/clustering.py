import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time

city = pd.read_csv("../input/cities.csv")

XY = pd.concat([city["X"], city["Y"]],axis=1)
XY = np.array(XY)
kmeans = KMeans(n_clusters=100, random_state=0).fit(XY)
label = pd.DataFrame(kmeans.labels_)
city["label"]=label

city.to_csv("../input/city_with_label.csv", index=False)
