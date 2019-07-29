import pandas as pd
import numpy as np
import feather
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from catboost import CatBoostClassifier,Pool
import os
import pickle
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import warnings
import sys
import datetime
warnings.filterwarnings('ignore')

logger = getLogger(__name__)

TRAIN = '../input/train.feather'
TEST = '../input/test.feather'

DIR = '../result/logfile'

start_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(DIR + '_cat_train.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)

logger.info('start')
logger.info('CatBoost')

args = sys.argv
id_feature = args[1]
target_feature = args[2]
print("id_feature", id_feature)
print("target_feature", target_feature)

logger.info('install data')
train = feather.read_dataframe(TRAIN)
test = feather.read_dataframe(TEST)

features = [c for c in train.columns if c not in [id_feature, target_feature]]
target= train[target_feature]
logger.info('data install complete')

params = {
    "eval_metric": "AUC",
    "objective": "Logloss",
    "early_stopping_rounds": 1000,
    "learning_rate": 0.01,
    "iterations": 10000,
    #"subsample": 0.6,
    "colsample_bylevel": 0.01,
    "max_depth": 10,
    #"rsm": 0.1,
    "scale_pos_weight": 5,
#"boosting": "Ordered"
}

#logger.info('Paramter tuning by BayesSearch')
#params = {'eval_metric':"AUC", 'objective':"Logloss", 'early_stopping_rounds':1000}
#bayes_cv_tuner = BayesSearchCV(
#            estimator = CatBoostClassifier(eval_metric="AUC", objective="Logloss", early_stopping_rounds=1000),
#            search_spaces = {
#                            'learning_rate': (0.01, 0.1),
#                            'iterations': (10000, 50000),
#                            'subsample': (0.6, 1.0),
#                            'colsample_bylevel': (0.01,0.1),
#                            'max_depth': (30, 50),
#                            'depth':(6, 16),
#                            'rsm':(0.1, 0.99),
#                            'scale_pos_weight': (1,10),
#                            'boosting': ["Ordered", "Plain"],
#                            },
#            scoring = "roc_auc",
#            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
#            n_jobs = -3,
#            n_iter = 10,
#            verbose = 0,
#            refit = True,
#            random_state = 42
#            )

#result = bayes_cv_tuner.fit(train[selected_features].values, target.values, callback=status_print)
#logger.info('found parameters by bayes searchCV: {}'.format(bayes_cv_tuner.best_params_))
#logger.info('best scores by bayes searchCV: {}'.format(bayes_cv_tuner.best_score_))

#params.update(bayes_cv_tuner.best_params_)

#path = "../result/parameter_catboost.csv"
#keys = pd.DataFrame(list(my_dict.keys()))
#values = pd.DataFrame(list(my_dict.values()))
#current = pd.concat([keys, values], axis=1)
#current.columns = [str(start_time)+"keys", str(start_time)+"values"]
#if os.path.isfile(path):
#    data = pd.read_csv(path)
#   data = pd.concat([data, current], axis=1)
#    data.to_csv(path)
#else:
#    current.to_csv(path)

logger.info('Learning start')
folds = StratifiedKFold(n_splits=2, shuffle=False, random_state=44000)
oof = train[[id_feature, target_feature]]
oof['predict'] = 0
predictions = pd.DataFrame(test[id_feature])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    logger.info('Fold {}'.format(fold_+1))
    cat = CatBoostClassifier(**params)
    cat.fit(train.iloc[trn_idx][features], target.iloc[trn_idx])
    oof['predict'][val_idx] = cat.predict_proba(train.iloc[val_idx][features])[:,1]

    predictions["Fold_"+str(fold_+1)] = cat.predict_proba(test[features])[:,1]
    logger.debug('CV score: {:<8.5f}'.format(roc_auc_score(target.iloc[val_idx], oof["predict"][val_idx])))

logger.info('Learning end')
score = roc_auc_score(target, oof["predict"])

predictions["Result"] = np.mean(predictions.iloc[:,2:], axis=1)

logger.info('record oof')
path = "../result/catboost_oof.csv"
if os.path.isfile(path):
    data = pd.read_csv(path)
else:
    data = pd.DataFrame()
data[str(start_time)+str(target_feature)] = oof["predict"]
data.to_csv(path, index=None)

logger.info('make submission file')
sub_df = pd.DataFrame({str(id_feature):test[id_feature].values})
sub_df[target_feature] = predictions["Result"]
sub_df.to_csv("../result/submission_cat_"+str(score)+".csv", index=False)

logger.info('record submission contents')
path = "../result/catboost_submission_sofar.csv"
if os.path.isfile(path):
    data = pd.read_csv(path)
else:
    data = pd.DataFrame()
    data[id_feature] = sub_df[id_feature]
data = pd.concat([data, sub_df[target_feature]], axis=1)
data= data.rename(columns={str(target_feature):str(start_time[:4])+"/"+str(start_time[5:7])+"/"+str(start_time[8:10])+"/"+str(start_time[11:13])+":"+str(start_time[14:16])+"/"+str(score)[:7]})
data.to_csv(path, index=None)

logger.info('end')
