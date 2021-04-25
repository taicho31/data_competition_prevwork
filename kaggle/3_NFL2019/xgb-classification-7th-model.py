# bronze medal winning solution
#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold,TimeSeriesSplit,KFold,GroupKFold
import xgboost as xgb
from datetime import datetime
from bayes_opt import BayesianOptimization
from kaggle.competitions import nflrush
import math
import tqdm
from scipy.special import expit
from scipy.spatial import Delaunay, delaunay_plot_2d, Voronoi, voronoi_plot_2d, ConvexHull
pd.set_option("display.max_rows",1000)
env = nflrush.make_env()




def crps_score(y_pred, data):
    y_pred = y_pred.reshape(-1,199)
    y_true = np.array(data.get_label()).astype("int")
    y_true_cdf = np.zeros([y_true.shape[0], 199])
    for i, y in enumerate(y_true):
        y_true_cdf[i, y] = 1
    y_true_cdf = np.clip(np.cumsum(y_true_cdf, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return "CRPS", ((y_true_cdf - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0]), False

def simple_crps(y_true, y_pred):
    y_true_cdf = np.zeros([y_true.shape[0], 199])
    for i, y in enumerate(y_true):
        y_true_cdf[i, y] = 1
    y_true_cdf = np.clip(np.cumsum(y_true_cdf, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true_cdf - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])




train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv',low_memory=False)


# # preprocess and feature engineering 



def transform_time_quarter(str1):
    return int(str1[:2])*60 + int(str1[3:5])
  
def transform_time_all(str1,quarter):
    if quarter<=4:
        return 15*60 - (int(str1[:2])*60 + int(str1[3:5])) + (quarter-1)*15*60
    if quarter ==5:
        return 10*60 - (int(str1[:2])*60 + int(str1[3:5])) + (quarter-1)*15*60
      
def back_direction(orientation):
    if orientation > 180.0:
        return 1
    else:
        return 0
    
def transform_height(te):
    return (int(te.split('-')[0])*12 + int(te.split('-')[1]))*2.54/100

def voronoi_volumes(points, selected_index):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
      
    for i, reg_num in enumerate(v.point_region):
        if reg_num == v.point_region[selected_index]:
            indices = v.regions[reg_num]
            if -1 in indices: # some regions can be opened
                vol = -999 ## insert missing value when the area is open
            else:
                vol = ConvexHull(v.vertices[indices]).volume      
            break
    return vol

def radius_calc(dist_to_ball):
    return 4 + 6 * (dist_to_ball >= 15) + (dist_to_ball ** 3) / 560 * (dist_to_ball < 15)

def compute_influence_to_ball(row):
    point = np.array([row["RushX"], row["RushY"]])
    theta = math.radians(row['Orientation'])
    speed = row['S']
    player_coords = row[['X', 'Y']].values
    dist_to_ball = row["DisToRusher"]    

    S_ratio = (speed / 13) ** 2    # we set max_speed to 13 m/s ((dominance_df["S"]*0.9144).max() #8.64m/s)
    RADIUS = radius_calc(dist_to_ball)  # updated

    S_matrix = np.matrix([[RADIUS * (1 + S_ratio), 0], [0, RADIUS * (1 - S_ratio)]])
    R_matrix = np.matrix([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    COV_matrix = np.dot(np.dot(np.dot(R_matrix, S_matrix), S_matrix), np.linalg.inv(R_matrix))
    
    norm_fact = (1 / 2 * np.pi) * (1 / np.sqrt(np.linalg.det(COV_matrix)))    
    mu_play = player_coords + speed * np.array([np.cos(theta), np.sin(theta)]) / 2
    
    intermed_scalar_player = np.dot(np.dot((player_coords - mu_play),
                                    np.linalg.inv(COV_matrix)),
                             np.transpose((player_coords - mu_play)))
    player_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_player[0, 0])
    
    intermed_scalar_point = np.dot(np.dot((point - mu_play), 
                                    np.linalg.inv(COV_matrix)), 
                             np.transpose((point - mu_play)))
    point_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_point[0, 0])
    
    return point_influence / player_influence




remove_features = ['GameId','PlayId','DisplayName','GameClock','TimeHandoff','TimeSnap', 'PlayDirection', 'TeamOnOffense', 
                    'PlayerBirthDate', 'is_run', 'NflIdRusher', 'date_game', 'RushX', 'RushY', 'PossessionTeam', 
                   'FieldPosition', 'Position', 'PlayerHeight', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Turf', 'Quarter']
top20_weather = list(train.GameWeather.value_counts(normalize=True, dropna=False).cumsum().head(20).index)




def transform_data(df):
    df.loc[df.VisitorTeamAbbr == "ARI",'VisitorTeamAbbr'] = "ARZ"
    df.loc[df.HomeTeamAbbr == "ARI",'HomeTeamAbbr'] = "ARZ"

    df.loc[df.VisitorTeamAbbr == "BAL",'VisitorTeamAbbr'] = "BLT"
    df.loc[df.HomeTeamAbbr == "BAL",'HomeTeamAbbr'] = "BLT"

    df.loc[df.VisitorTeamAbbr == "CLE",'VisitorTeamAbbr'] = "CLV"
    df.loc[df.HomeTeamAbbr == "CLE",'HomeTeamAbbr'] = "CLV"

    df.loc[df.VisitorTeamAbbr == "HOU",'VisitorTeamAbbr'] = "HST"
    df.loc[df.HomeTeamAbbr == "HOU",'HomeTeamAbbr'] = "HST"

    df['is_run'] = df.NflId == df.NflIdRusher

    if 2017 in list(df["Season"].unique()):
        df.loc[df['Season'] == 2017, 'S'] = (df['S'][df['Season'] == 2017] - 2.4355) / 1.2930 * 1.4551 + 2.7570

    df['ToLeft'] = df.PlayDirection == "left"
    df['TeamOnOffense'] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    df['OnOffense'] = df.Team == df.TeamOnOffense 
    df['YardLine_std'] = 100 - df.YardLine.copy()
    df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,  
            'YardLine_std'
             ] = df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,  
              'YardLine']
    df['X_std'] = df.X.copy()
    df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X'] 
    df['Y_std'] = df.Y.copy()
    df.loc[df.ToLeft, 'Y_std'] = 53.3 - df.loc[df.ToLeft, 'Y'] 
    df['Orientation_std'] = df.Orientation.copy()
    df.loc[df.ToLeft, 'Orientation_std'] = np.mod(180 + df.loc[df.ToLeft, 'Orientation_std'], 360)
    df['Dir_std'] = df.Dir.copy()
    df.loc[df.ToLeft, 'Dir_std'] = np.mod(180 + df.loc[df.ToLeft, 'Dir_std'], 360)
    df.loc[df['Season'] == 2017, 'Orientation_std'] = np.mod(90 + df.loc[df['Season'] == 2017, 'Orientation_std'], 360) 
    df.drop(["X", "Y", "Orientation", "YardLine", "Dir", "ToLeft"], axis=1, inplace=True)
    df.rename(columns={'X_std': 'X', 'Y_std': 'Y', 'Orientation_std': 'Orientation', 'Dir_std': 'Dir', "YardLine_std": "YardLine"}, inplace=True)
    
    df['date_game'] = df.GameId.map(lambda x:pd.to_datetime(str(x)[:8]))
    df['age'] = (df.date_game.map(pd.to_datetime) - df.PlayerBirthDate.map(pd.to_datetime)).map(lambda x:x.days)/365

    df["Momentum"] = df["S"] * df["PlayerWeight"]
    
    #df["Jersey_cat"] = df["JerseyNumber"].apply(lambda x: 0 if x>=1 and x <=9 else #: quarterbacks, kickers, and punters
    #                                            (1 if x>=10 and x <= 19 else
    ##                                            (2 if x>=20 and x <= 39 else
    #                                            (3 if x>=40 and x <= 49 else
    #                                            (4 if x>=50 and x <= 59 else
    #                                            (5 if x>=60 and x <= 79 else
    #                                            (6 if x>=80 and x <= 89 else
    #                                            (7 if x>=90 and x <= 99 else 8))))))))

    #df["F"] = df["A"] * df["PlayerWeight"]

    rusher_x = np.array(df.groupby(["PlayId", "is_run"])["X"].agg(np.mean)[1::2])
    df["RushX"] = np.repeat(rusher_x, 22) # repeat each elemnt 22 times
    rusher_y = np.array(df.groupby(["PlayId", "is_run"])["Y"].agg(np.mean)[1::2])
    df["RushY"] = np.repeat(rusher_y, 22) 
    df["DisToRusher"] = np.sqrt((df["X"] - df["RushX"]) ** 2 + (df["Y"] - df["RushY"]) ** 2)
    df["TackleTimeToRusher"] = df["DisToRusher"] / df["S"] # includes nan when the speed of rusher is 0
    #df.loc[df.is_run==True, "TackleTimeToRusher"] = 0
    
    #df["InfluToBall"] = df.apply(compute_influence_to_ball, axis=1)

    #df["Dir_sin"] = df["Dir"].apply(lambda x : np.sin((450-x) * np.pi/ 180))
    df["Dir_cos"] = df["Dir"].apply(lambda x : np.cos((450-x) * np.pi/ 180))
    #df["Orientation_sin"] = df["Orientation"].apply(lambda x : np.cos((450-x) * np.pi/ 180))
    #df["Orientation_cos"] = df["Orientation"].apply(lambda x : np.sin((450-x) * np.pi/ 180))
    
    df["Momentum_cos"] = df["Momentum"] * df["Dir_cos"]
    #df["Momentum_sin"] = df["Momentum"] * df["Dir_sin"]
    
    df["AttackAngle"] =  np.arctan((df["Y"] - df["RushY"]) / (df["X"] - df["RushX"])) * 180 / np.pi

    #rusher_s = np.array(df.groupby(["PlayId", "is_run"]).agg(np.mean)["S"][1::2])
    #rusher_s[rusher_s == 0] = 1e-15 # replace velocity 0 with very small values
    #rusher_s = np.repeat(rusher_s, 22)
    #df["RatioSToRusher"] = df["S"] / rusher_s
    #df.loc[df.is_run==True, "RatioSToRusher"] = 1
    
    df_single = df[df.is_run==True].copy()
        
    #df_single["NecDisPerDown"] = df_single["Distance"] / (5 - df_single["Down"])
        
    #df_single['time_quarter'] = df_single.GameClock.map(lambda x:transform_time_quarter(x))
    df_single['time_end'] = df_single.apply(lambda x:transform_time_all(x.loc['GameClock'],x.loc['Quarter']),axis=1)

    df_single["Stadium"] = df_single["Stadium"].map(lambda x: "Broncos Stadium at Mile High" if x=="Broncos Stadium At Mile High" 
                                             else ("CenturyLink Field" if x == "CenturyField" or x == x=="CenturyLink"
                                             else ("Everbank Field" if x == "EverBank Field"
                                             else ("FirstEnergy Stadium" if x =="First Energy Stadium" or x=="FirstEnergy" or x == "FirstEnergyStadium"
                                             else ("Lambeau Field" if x == "Lambeau field"
                                             else ("Los Angeles Memorial Coliseum" if x == "Los Angeles Memorial Coliesum"
                                             else ("M&T Bank Stadium" if x == "M & T Bank Stadium" or x == "M&T Stadium"
                                             else ("Mercedes-Benz Superdome" if x == "Mercedes-Benz Dome"
                                             else ("MetLife Stadium" if x == "MetLife" or x == "Metlife Stadium"
                                             else ("NRG Stadium" if x == "NRG"
                                             else ("Oakland-Alameda County Coliseum" if x == "Oakland Alameda-County Coliseum"
                                             else ("Paul Brown Stadium" if x == "Paul Brown Stdium"
                                             else ("Twickenham Stadium" if x == "Twickenham" else x)))))))))))))
                                             #"State Farm Stadium" if x == "University of Phoenix Stadium"
                                             #"Empower Field at Mile High" if x == "Sports Authority Field at Mile High" or x == "Broncos Stadium at Mile High" 

    df_single["Location"] = df_single["Location"].map(lambda x: "Arlington, TX" if x == "Arlington, Texas"
                                            else ("Baltimore, MD" if x == "Baltimore, Maryland" or x == "Baltimore, Md."
                                            else ("Charlotte, NC" if x == "Charlotte, North Carolina"
                                            else ("Chicago, IL" if x == "Chicago. IL"
                                            else ("Cincinnati, OH" if x == "Cincinnati, Ohio"
                                            else ("Cleveland, OH" if x == "Cleveland" or x == "Cleveland Ohio" or x == "Cleveland, Ohio" or x == "Cleveland,Ohio"
                                            else ("Detroit, MI" if x == "Detroit"
                                            else ("East Rutherford, NJ" if x == "E. Rutherford, NJ" or x == "East Rutherford, N.J."
                                            else ("Foxborough, MA" if x == "Foxborough, Ma"
                                            else ("Houston, TX" if x == "Houston, Texas"
                                            else ("Jacksonville, FL" if x == "Jacksonville Florida" or x == "Jacksonville, Fl" or x == "Jacksonville, Florida"
                                            else ("London" if x == "London, England"
                                            else ("Los Angeles, CA" if x == "Los Angeles, Calif."
                                            else ("Miami Gardens, FLA" if x == "Miami Gardens, Fla."
                                            else ("New Orleans, LA" if x == "New Orleans" or x == "New Orleans, La."
                                            else ("Orchard Park, NY" if x == "Orchard Park NY"
                                            else ("Philadelphia, PA" if x == "Philadelphia, Pa."
                                            else ("Pittsburgh, PA" if x == "Pittsburgh"
                                            else ("Seattle, WA" if x == "Seattle" else x)))))))))))))))))))

    grass_labels = ['grass', 'natural grass', 'natural', 'naturall grass']
    df_single['Grass'] = np.where(df_single.Turf.str.lower().isin(grass_labels), "Natural", "Artificial")
                                                                 
    #df_single["GameWeather"] = df_single["GameWeather"].apply(lambda x: "Others" if x not in top20_weather else x)
                                                                 
    df_single["OffenseFormation"] = df_single["OffenseFormation"].fillna("Unknown") 
    df_single['DefendersInTheBox_vs_Distance'] = df_single['DefendersInTheBox'] / df_single['Distance']
                                                                 
    #df_single['back_oriented_down_field'] = df_single['Orientation'].apply(lambda x: back_direction(x))
    #df_single['back_moving_down_field'] = df_single['Dir'].apply(lambda x: back_direction(x))

    #arr = [[int(s[0]) for s in t.split(", ")] for t in df_single["DefensePersonnel"]]
    #df_single["DefenseDL"] = np.array([a[0] for a in arr])
    #df_single["DefenseLB"] = np.array([a[1] for a in arr])
    #df_single["DefenseDB"] = np.array([a[2] for a in arr])
    #df_single["DefenseOL"] = np.array([a[3] if len(a) == 4 else 0 for a in arr])
  
    #df_single["OffenseRB"] = df_single["OffensePersonnel"].apply(lambda x: 
    #                        int(x.replace(",", "").split(" RB")[0][-1]) if "RB" in x else 0)
    #df_single["OffenseTE"] = df_single["OffensePersonnel"].apply(lambda x: 
    #                        int(x.replace(",", "").split(" TE")[0][-1]) if "TE" in x else 0)
    #df_single["OffenseWR"] = df_single["OffensePersonnel"].apply(lambda x: 
    #                        int(x.replace(",", "").split(" WR")[0][-1]) if "WR" in x else 0)
    #df_single["OffenseOL"] = df_single["OffensePersonnel"].apply(lambda x: 
    #                        int(x.replace(",", "").split(" OL")[0][-1]) if "OL" in x else 0)
    #df_single["OffenseDL"] = df_single["OffensePersonnel"].apply(lambda x: 
    #                        int(x.replace(",", "").split(" DL")[0][-1]) if "DL" in x else 0)
    #df_single["OffenseQB"] = df_single["OffensePersonnel"].apply(lambda x: 
    #                        int(x.replace(",", "").split(" QB")[0][-1]) if "QB" in x else 0)
  
    df_single["DisToQB"] = np.array(df[(df.Position=="QB") | (df.Position=="C")].groupby(["PlayId"]).agg(np.mean)["DisToRusher"])

    #df_single["Margin"] = df_single["HomeScoreBeforePlay"] - df_single["VisitorScoreBeforePlay"]
    #df_single.loc[df_single['Team'] == "away", 'Margin'] = (df_single['VisitorScoreBeforePlay'][df_single['Team'] == "away"] - df_single['HomeScoreBeforePlay'][df_single['Team'] == "away"])

    df_single['runner_height'] = df_single.PlayerHeight.map(transform_height)
    df_single.drop(remove_features,axis=1,inplace=True) 

    tmp = df.groupby(["PlayId", "OnOffense"]).agg(np.mean)[["X", "Y", "Momentum", "Momentum_cos"]]
    df_single["DefenseAveX"] = np.array(tmp[0::2]["X"])
    df_single["OffenseAveX"] = np.array(tmp[1::2]["X"])
    df_single["DefenseAveY"] = np.array(tmp[0::2]["Y"]) 
    df_single["OffenseAveY"] = np.array(tmp[1::2]["Y"]) 
    df_single["DefenseAveMomentum"] = np.array(tmp[0::2]["Momentum"])
    df_single["OffenseAveMomentum"] = np.array(tmp[1::2]["Momentum"])
    df_single["DefenseAveMomentum_cos"] = np.array(tmp[0::2]["Momentum_cos"])
    df_single["OffenseAveMomentum_cos"] = np.array(tmp[1::2]["Momentum_cos"])
    
    #df_single["DefenseAveAge"] = np.array(tmp[0::2]["age"])
    #df_single["OffenseAveAge"] = np.array(tmp[1::2]["age"])

    tmp = df.groupby(["PlayId", "OnOffense"]).agg(["std"])[["X", "Y", "Momentum", "Momentum_cos"]]
    df_single["DefenseStdX"] = np.array(tmp[0::2]["X"])
    df_single["OffenseStdX"] = np.array(tmp[1::2]["X"])
    df_single["DefenseStdY"] = np.array(tmp[0::2]["Y"])
    df_single["OffenseStdY"] = np.array(tmp[1::2]["Y"])
    df_single["DefenseStdMomentum"] = np.array(tmp[0::2]["Momentum"])
    df_single["OffenseStdMomentum"] = np.array(tmp[1::2]["Momentum"])
    df_single["DefenseStdMomentum_cos"] = np.array(tmp[0::2]["Momentum_cos"])
    df_single["OffenseStdMomentum_cos"] = np.array(tmp[1::2]["Momentum_cos"])
    
    df_single["RunnerToDefenseCentoid"] = np.sqrt((df_single["X"] - df_single["DefenseAveX"]) ** 2 + (df_single["Y"] - df_single["DefenseAveY"]) ** 2)
    df_single["RunnerToOffenseCentoid"] = np.sqrt((df_single["X"] - df_single["OffenseAveX"]) ** 2 + (df_single["Y"] - df_single["OffenseAveY"]) ** 2)

    tmp_max = df.groupby(["PlayId", "OnOffense"])["X"].max()
    tmp_min = df.groupby(["PlayId", "OnOffense"])["X"].min()
    df_single["DefenseSpreadX"] = np.array(tmp_max[0::2]- tmp_min[0::2])
    df_single["OffenseSpreadX"] = np.array(tmp_max[1::2]- tmp_min[1::2])

    df_single["RunnerToScrimmage"] = df_single["X"] - (df_single["YardLine"] + 10)

    df_single["MinTackleTime"] = np.array(df.groupby(["PlayId", "OnOffense"])["TackleTimeToRusher"].min()[0::2])
    df_single["1stDefender_Momentum_cos"] = np.array(df.loc[df.groupby(["PlayId", "OnOffense"])["DisToRusher"].idxmin()[0::2]]["Momentum_cos"])
    first_offenser_index =df.groupby(["PlayId", "OnOffense"])["DisToRusher"].nsmallest(2)[3::4].reset_index()["level_2"]
    df_single["1stOffenser_Momentum_cos"] = np.array(df.loc[first_offenser_index]["Momentum_cos"])
    #df_single["1stDefender_Momentum_sin"] = np.array(df.loc[df.groupby(["PlayId", "OnOffense"])["DisToRusher"].idxmin()[0::2]]["Momentum_sin"])
    #df_single["1stDefender_A"] = np.array(df.loc[df.groupby(["PlayId", "OnOffense"])["DisToRusher"].idxmin()[0::2]]["A"])
    df_single["1stDefender_AttackAngle"] = np.array(df.loc[df.groupby(["PlayId", "OnOffense"])["DisToRusher"].idxmin()[0::2]]["AttackAngle"])
    #df_single["1stDefenderID"] = np.array(df.loc[df.groupby(["PlayId", "OnOffense"])["DisToRusher"].idxmin()[0::2]]["NflId"])

    #df_single["Rusher1stDefSpeedRatio"] = df.loc[df.groupby(["PlayId", "OnOffense"])["DisToRusher"].idxmin()[0::2]]["RatioSToRusher"]

    pts = np.array(df[["X", "Y"]]).reshape(df.shape[0]//22, 22, 2) # plays * players * (X, Y, rusher)
    rusher_index = list(df[df.is_run==True].index % 22) 
    closest_def_index = list(df.loc[df.groupby(["PlayId", "OnOffense"])["DisToRusher"].idxmin()[0::2]].index % 22)
    #closest_off_index = list(first_offenser_index % 22)
    rusher_voronoi = []
    closest_def_voronoi = []
    #closest_off_voronoi = []

    for i in range(0, df.shape[0] //22):
        rusher_voronoi.append(voronoi_volumes(pts[i], rusher_index[i]))
        closest_def_voronoi.append(voronoi_volumes(pts[i], closest_def_index[i]))
        #closest_off_voronoi.append(voronoi_volumes(pts[i], closest_off_index[i]))
    df_single["RusherVoronoi"] = rusher_voronoi    
    df_single["FirstDefenderVoronoi"] = closest_def_voronoi 
    #df_single["FirstOffenserVoronoi"] = closest_off_voronoi
    
    #def_influ = np.array(df.groupby(["PlayId", "OnOffense"]).agg("sum")["InfluToBall"][0::2])
    #off_influ = np.array(df.groupby(["PlayId", "OnOffense"]).agg("sum")["InfluToBall"][1::2])
    #df_single["PitchControl"] = expit(off_influ - def_influ)
    
    df_single.fillna(-999,inplace=True) 
    remove_features2 = ["OnOffense", "DisToRusher", "TackleTimeToRusher", "OffenseAveX", "OffenseAveY"]#"InfluToBall",, "RatioSToRusher"]
    df_single.drop(remove_features2, axis=1, inplace=True)

    return df_single

train_single = transform_data(train)
y_train = train_single.Yards + 99 # to categorize
X_train = train_single.drop(['Yards'],axis=1)
for f in X_train.columns:
    if X_train[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f])+[-999])
        X_train[f] = lbl.transform(list(X_train[f]))


# # modelling
n_folds=5
kf=KFold(n_splits = n_folds, random_state=1125)
average_crps = 0
models = []
crps_scores = []
xgb_params = {
    "objective" : "multi:softprob",
    "eval_metric" : "mlogloss", 
    "max_depth" : 4,
    "boosting": 'gbdt',
    "num_class": 199,
    "num_leaves" : 13,
    "learning_rate" : 0.05,
    #"alpha": 0.1
}
#evals_result = {}
num_boost_round=100000
for i , (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
    X_train2= X_train.iloc[train_index,:]
    y_train2= y_train.iloc[train_index]
    X_test2= X_train.iloc[test_index,:]
    y_test2= y_train.iloc[test_index]
    xgb_train = xgb.DMatrix(X_train2, label = y_train2)
    xgb_eval = xgb.DMatrix(X_test2, label = y_test2)
    watchlist = [(xgb_train, "train"), (xgb_eval, "eval")]
    
    clf = xgb.train(
        xgb_params, xgb_train, num_boost_round, watchlist,
        early_stopping_rounds=10,
        #evals_result=evals_result,
        #feval=crps_score,
    )
    
    models.append(clf)
    temp_predict = clf.predict(xgb_eval, ntree_limit=clf.best_ntree_limit)
    score = simple_crps(y_test2, temp_predict)
    print(score)
    crps_scores.append(score)
    average_crps += score / n_folds
    if i == 0:
        feature_importance_df = pd.DataFrame(clf.get_score(importance_type="total_gain").items(), columns =["Features", "Fold_"+str(i+1)])
    else:
        feature_importance_df["Fold_"+str(i+1)] = list(clf.get_score(importance_type="total_gain").values())
    
feature_importance_df["Average"] = np.mean(feature_importance_df.iloc[:,1:n_folds+1], axis=1)
feature_importance_df["Std"] = np.std(feature_importance_df.iloc[:,1:n_folds+1], axis=1)
feature_importance_df["Cv"] = feature_importance_df["Std"] / feature_importance_df["Average"]
print('crps list:', crps_scores)
print('average crps:',average_crps)


# - DIrから最後30個まで
# - crps list: [0.01306208230030206, 0.012845522370741002, 0.012312934091543165, 0.014272256152330345, 0.01411467909845058]
# - average crps: 0.013321494802673431
# - 
# - 全変数での確認
# - xgb_params = { "max_depth" : 8,"boosting": 'gbdt',"num_class": 199,"num_leaves" : 13,"learning_rate" : 0.05}
# - crps list: [0.012463530769226253, 0.012254392817461731, 0.011967620183047138, 0.01439946003900104, 0.014263435465008889]
# - average crps: 0.01306968785474901 public 0.01372　過学習
# - 
# - xgb_params = { "max_depth" : 7,"boosting": 'gbdt',"num_class": 199,"num_leaves" : 13,"learning_rate" : 0.05}
# - crps list: [0.01232583358804592, 0.012123005183270549, 0.011848985424717825, 0.014264628496662391, 0.014080748133163139]
# - average crps: 0.012928640165171965
# - 
# - xgb_params = { "max_depth" : 6,"boosting": 'gbdt',"num_class": 199,"num_leaves" : 13,"learning_rate" : 0.05}
# - crps list: [0.012277130968882225, 0.012068885358419669, 0.011805369240065696, 0.01421829961531137, 0.014034948755887455]
# - average crps: 0.012880926787713283
# - 
# - xgb_params = { "max_depth" : 5,"boosting": 'gbdt',"num_class": 199,"num_leaves" : 13,"learning_rate" : 0.05}
# - crps list: [0.012266423714509405, 0.012042828801204651, 0.011763040846812843, 0.014174927089895447, 0.01401563346066939]
# - average crps: 0.012852570782618346
# - 
# - xgb_params = { "max_depth" : 4,"boosting": 'gbdt',"num_class": 199,"num_leaves" : 13,"learning_rate" : 0.05}
# - crps list: [0.012270968531269785, 0.012060295219752368, 0.011752867464111713, 0.014142862399415956, 0.013976765969843291]
# - average crps: 0.012840751916878623



X_train.columns




feature_importance_df.sort_values("Average").head(70).reset_index(drop=True)


# # prediction



for (test_df, sample_prediction_df) in env.iter_test():
    X_test = transform_data(test_df)
    for f in X_test.columns:
        if X_test[f].dtype=='object':
            X_test[f] = X_test[f].map(lambda x:x if x in set(X_train[f]) else -999)
    for f in X_test.columns:
        if X_test[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(X_train[f])+[-999])
            X_test[f] = lbl.transform(list(X_test[f])) 
    pred_value = 0
    dtest = xgb.DMatrix(X_test)
    for model in models:
        pred_value += model.predict(dtest, ntree_limit = model.best_ntree_limit)[0]/5
    pred_data = np.clip(np.cumsum(pred_value), 0, 1)
    pred_data = np.array(pred_data).reshape(1,199)
    pred_target = pd.DataFrame(index = sample_prediction_df.index,                                columns = sample_prediction_df.columns,                                data = pred_data)
    env.predict(pred_target)
env.write_submission_file()






