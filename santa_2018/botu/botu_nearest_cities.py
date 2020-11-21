import pandas as pd
import numpy as np

NEAREST_CITIES = 30

city = pd.read_csv("../input/cities.csv") 

for i in range(NEAREST_CITIES):
    city[""+str(i)+""] = 0

city["passed"] = 0

# やはり非常に時間がかかる
for i in range(city.shape[0]):
    print(i)
    x_cor = city.loc[i,"X"]
    y_cor = city.loc[i,"Y"]
    x_diff = city["X"] - x_cor
    y_diff = city["Y"] - y_cor
    dist = np.sqrt(x_diff * x_diff + y_diff * y_diff)
    dist = dist.sort_values().head(NEAREST_CITIES+1)
    dist = dist.drop(i) # 先頭が０なのは当たり前なので削除
    for j in range(NEAREST_CITIES):
        city.iloc[i, 3 +j] = dist.head(NEAREST_CITIES).index[j]
