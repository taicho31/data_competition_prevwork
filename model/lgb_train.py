import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import confusion_matrix, log_loss, classification_report, accuracy_score, roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
import feather
import warnings
import pickle
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import os
import sys
import datetime
from functools import partial
import optuna
from sklearn.feature_selection import RFE
from hyperopt import hp, tpe, Trials, fmin
from hyperopt import space_eval
warnings.filterwarnings('ignore')

logger = getLogger(__name__)

TRAIN = '../input/train_mod.feather'
TEST = '../input/test_mod.feather'

DIR = '../result/logfile'

random_state = 42
np.random.seed(random_state)

# Data augmentation for model development and leaning ---------------------------
def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y

def objective(X, y, trial):
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        
    # optimize the number of features to use by RFE
    n_features_to_select = trial.suggest_int('n_features_to_select', 1, 20),
    rfe = RFE(estimator=clf, n_features_to_select=n_features_to_select)
                                 
    X_train, X_eval, y_train, y_eval = train_test_split(X, y,shuffle=True,random_state=42)
    rfe.fit(X_eval, y_eval)
                                 
    X_train_selected = X_train.iloc[:, rfe.support_]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_train_selected, y_train,cv=skf, method='predict_proba')
                                 
    metric = roc_auc_score(y_train, y_pred[:, 1])
    return metric

opt_params = {
    "objective" : "binary", # or "regression"
    "metric" : "auc", # or rmse
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state,
    "tree_learner": "serial",
    "max_depth" : -1,
    "boost_from_average": "false",
    "boosting": 'gbdt',
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
}

check_params = {
    "objective" : "binary", # or "regression"
    "metric" : "auc", # or rmse
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state,
    "tree_learner": "serial",
    "max_depth" : -1,
    "boost_from_average": "false"
}

start_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(DIR + '_lgb_train.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)

logger.info('start')
logger.info('LightGBM')

args = sys.argv
id_feature = args[1]
target_feature = args[2]
print("id_feature", id_feature)
print("target_feature", target_feature)

logger.info('install data')
train_df = feather.read_dataframe(TRAIN)
test_df = feather.read_dataframe(TEST)

features = [c for c in train_df.columns if c not in [id_feature, target_feature]]
train = train_df[features]
target = train_df[target_feature]
X_test = test_df.values
logger.info('data install complete')

#logger.info('---------Determine Feature Selection Num by Optuna and RFE----------')
#tmp_features = [c for c in features if "int" in str(train[c].dtype) or "float" in str(train[c].dtype)][:20]
#f = partial(objective, train[tmp_features], target)
#study = optuna.create_study(direction='maximize') # optimize by optuna
#study.optimize(f, n_trials=3)
#print('no of params:', study.best_params["n_features_to_select"]) # output discovered parameters
#feature_num = study.best_params["n_features_to_select"]
#logger.info('Determine Feature Selection Num End')

#logger.info('---------Feature Selection by RFE----------')
#clf = lgb.LGBMClassifier(n_estimators=100, random_state=42)
#rfe = RFE(estimator=clf, n_features_to_select=feature_num,verbose=1)
#X_train, X_eval, y_train, y_eval = train_test_split(train[tmp_features], target, shuffle=True, random_state=42)
#rfe.fit(X_eval, y_eval) # Learning by RFE

#train_selected = train[tmp_features].iloc[:, rfe.support_]
#selected_features = list(train_selected.columns)
#logger.info("Selected features:", selected_features)
#logger.info('Feature Selection End')

logger.info('-----------Parameter tuning------------')

X = train[features]
Y = target

def para_tuning_obj(params):
    params = {
        'bagging_freq': int(params['bagging_freq']),
        'bagging_fraction': float(params['bagging_fraction']),
        'num_leaves': int(params['num_leaves']),
        'feature_fraction': float(params['feature_fraction']),
        'learning_rate': float(params['learning_rate']),
        'min_data_in_leaf': int(params['min_data_in_leaf']),
        'min_sum_hessian_in_leaf': int(params['min_sum_hessian_in_leaf']),
        'boosting': params['boosting'],
    #'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
}
    
    score = []
    clf = lgb.LGBMClassifier(objective="binary", metric="auc", seed= random_state,
                             verbose=1, bagging_seed = random_state, tree_learner= "serial",
                             max_depth = -1, boost_from_average= "false", **params)
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for trn_idx, val_idx in skf.split(X, Y):
        clf.fit(X.iloc[trn_idx], Y.iloc[trn_idx])
        predicts = clf.predict(X.iloc[val_idx])
        score.append(roc_auc_score(Y.iloc[val_idx], predicts))
    
    return -1 * np.mean(score)

trials = Trials()

space ={
    'bagging_freq': hp.quniform('bagging_freq', 1, 10, 1),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.2, 1.0),
    'num_leaves': hp.quniform('num_leaves', 8, 128, 1),
    'feature_fraction': hp.uniform('feature_fraction', 0.2, 1.0),
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 8, 128, 1),
    'min_sum_hessian_in_leaf': hp.quniform('min_sum_hessian_in_leaf', 5, 30, 1),
    'boosting': hp.choice('boosting', ['gbdt', 'dart', 'rf']),
    #'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0)
}

best = fmin(para_tuning_obj, space = space, algo=tpe.suggest, max_evals=2, trials=trials, verbose=1)

logger.info(best) # check the optimal parameters
best = space_eval(space, best)
check_params.update(best)
check_params['num_leaves'] = int(check_params['num_leaves'])
check_params['min_data_in_leaf'] = int(check_params['min_data_in_leaf'])
check_params['bagging_freq'] = int(check_params['bagging_freq'])

logger.info('-----------Learning start------------')
CLASS = 9
n_folds = 5
folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
oof = np.zeros((len(train_df),CLASS))
predictions = pd.DataFrame(test_df[id_feature])
val_aucs = []
feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = features
yp = np.zeros((test.shape[0] ,CLASS))

for fold, (trn_idx, val_idx) in enumerate(folds.split(train, target)):
    print("fold: {}" .format(fold+1))
    X_train, y_train = train.iloc[trn_idx][features], target.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], target.iloc[val_idx]

    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    lgb_clf = lgb.train(check_params, #opt_params
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000
                       )
    valid_result = lgb_clf.predict(X_valid, num_iteration = lgb_clf.best_iteration)
    yp += lgb_clf.predict(test[features], num_iteration = lgb_clf.best_iteration) / n_folds
    oof[val_idx] = valid_result
    val_score = roc_auc_score(y_valid, np.argmax(valid_result, axis=1))
    val_aucs.append(val_score)
    feature_importance_df["importance_fold"+str(i)] = lgb_clf.feature_importance()

logger.info('Learning End')


logger.info('-------Performance check and prediction-------')
mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)

oof_prediction = np.argmax(oof, axis=1)
all_auc = roc_auc_score(target, oof_prediction)
logger.debug("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))
print(confusion_matrix(target, oof_prediction))
print(classification_report(target, oof_prediction))

predictions[target_feature] = np.argmax(yp, axis=1)

# check accuracy considering the highest K classes in prediction
#K = 3
#unsorted_max_indices = np.argpartition(-oof, K)[:,:K]
#y = []
#for i in range(unsorted_max_indices.shape[0]):
#    y.append(oof[i, unsorted_max_indices[i]])
#y = np.array(y)
#indices = np.argsort(-y, axis=1)
#max_k_indices = []
#for i in range(indices.shape[0]):
#    max_k_indices.append(list(unsorted_max_indices[i][indices[i]]))
#max_k_indices = np.array(max_k_indices)
#count = 0
#for i in range(unsorted_max_indices.shape[0]):
#    if target[i] in max_k_indices[i]:
#        count += 1
#print(count * 100 / unsorted_max_indices.shape[0])

#logger.info('-------------record oof contents-------------')
#path = "../result/lgb_oof.csv"
#if os.path.isfile(path):
#    data = pd.read_csv(path)
#else:
#    data = oof[[id_feature, target_feature]]
#data = pd.concat([data, oof['predict']], axis=1)
#data = data.rename(columns={'predict': start_time})
#data.to_csv(path, index=None)

logger.info('-------------make submission file-------------')
sub_df = pd.DataFrame({str(id_feature):test[id_feature].values})
sub_df[target_feature] = predictions[target_feature]
sub_df.to_csv("../result/submission_lgb_"+str(mean_auc)+".csv", index=False)

logger.info('-------------record submission contents-------------')
path = "../result/lgb_submission_sofar.csv"
if os.path.isfile(path):
    data = pd.read_csv(path)
else:
    data = pd.DataFrame()
    data[id_feature] = sub_df[id_feature]
data = pd.concat([data, sub_df[target_feature]], axis=1)
data = data.rename(columns={str(target_feature):str(start_time[:4])+"/"+str(start_time[5:7])+"/"+str(start_time[8:10])+"/"+str(start_time[11:13])+":"+str(start_time[14:16])+"/"+str(mean_auc)[:7]})
data.to_csv(path, index=None)

logger.info('end')
