import pandas as pd
import numpy as np

city = pd.read_csv("../input/cities_mod.csv")

NEAREST_CITIES = 30


path = [0]
remain_city = list(range(1,city.shape[0]))# 訪れていない都市リスト
for k in range(city.shape[0]):
    # print("k:%d" %k)
    if k == len(path): # 下のfor文の中で新たに都市が追加されなかったら訪問されなかった都市を付け足して終了
        for j in range(len(remain_city)):
            path.append(remain_city[j])
        break
    last = path[k]
    for i in range(NEAREST_CITIES):
        # print("i:%d" %i)
        cand = city.loc[last, ""+str(i)+""] 
        city.loc[last,"passed"] = 1 # 今いる都市の通過フラグを1にする
        if city.loc[cand, "passed"]==0: # まだ通過していないなら
            path.append(cand)               # pathに追加
            remain_city.remove(cand) # 訪れていない都市リストを更新
            last = cand
            # print(path)
            # print("pass")
            break

path = pd.DataFrame(path)
path.to_csv("../input/path.csv", index=False)

