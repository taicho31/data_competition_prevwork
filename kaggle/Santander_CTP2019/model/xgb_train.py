import pandas as pd
import numpy as np
import feather
from sklearn.metrics import confusion_matrix, log_loss, classification_report, accuracy_score, roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
import warnings
import xgboost as xgb
import pickle
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import os
import sys
import datetime
warnings.filterwarnings('ignore')

logger = getLogger(__name__)

TRAIN = '../input/train_mod.feather'
TEST = '../input/test_mod.feather'

DIR = '../result/logfile'

start_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)
    
handler = FileHandler(DIR + '_xgb_train.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)
    
logger.info('start')
logger.info('XGBoostClassifier')

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

logger.info('---------------Learning start-------------')
CLASS = 9
n_folds = 5

params = {
    "booster": "gbtree",
    "nthread": 8,
    "objective": "multi:softprob", # or "regression"
    "eval_metric" : "auc", # or rmse
    "boosting": 'gbdt',
    "max_depth" : 8,
    "num_class": CLASS,
    "learning_rate" : 0.01,
    "verbosity" : 1,
}

folds = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=44000)
oof = np.zeros((len(train_df),CLASS))
predictions = pd.DataFrame(test[id_feature])
val_scores = []
feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = features
yp = np.zeros((test.shape[0] ,CLASS))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_+1))
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(train.iloc[trn_idx][features], target.iloc[trn_idx])
    
    valid_result = xgb_model.predict_proba(train.iloc[val_idx][features])
    yp += xgb_model.predict_proba(test[features]) / n_folds
    
    oof[val_idx] = valid_result
    val_score = roc_auc_score(target[val_idx], np.argmax(valid_result, axis=1))
    val_scores.append(val_score)
    feature_importance_df["importance_fold"+str(i)] = xgb_model.feature_importances_

logger.info('Learning end')

logger.info('-------Performance check and prediction-------')
mean_score = np.mean(val_scores)
std_score = np.std(val_scores)

oof_prediction = np.argmax(oof, axis=1)
all_score = roc_auc_score(target, oof_prediction)
logger.debug("Mean score: %.9f, std: %.9f. All score: %.9f." % (mean_score, std_score, all_score))
print(confusion_matrix(target, oof_prediction))
print(classification_report(target, oof_prediction))

predictions[target_feature] = np.argmax(yp, axis=1)

#logger.info('----------record oof contents-------------')
#path = "../result/xgboost_oof.csv"
#if os.path.isfile(path):
#    data = pd.read_csv(path)
#else:
#    data = pd.DataFrame()
#data[[str(start_time)+str(i) for i in target_feature]] = oof
#data.to_csv(path, index=None)

logger.info('-----------make submission file------------')
sub_df = pd.DataFrame({str(id_feature):test[id_feature].values})
sub_df[target_feature] = predictions["Result"]
sub_df.to_csv("../result/submission_xgb_"+str(score)+".csv", index=False)

logger.info('---------record submission contents---------')
path = "../result/xgboost_submission_sofar.csv"
if os.path.isfile(path):
    data = pd.read_csv(path)
else:
    data = pd.DataFrame()
    data[id_feature] = sub_df[id_feature]
data = pd.concat([data, sub_df[target_feature]], axis=1)
data = data.rename(columns={str(target_feature): str(start_time[:4])+"/"+str(start_time[5:7])+"/"+str(start_time[8:10])+"/"+str(start_time[11:13])+":"+str(start_time[14:16])+"/"+str(score)[:7]})
data.to_csv(path, index=None)

logger.info('end')
