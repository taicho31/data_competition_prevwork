import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import xgboost
import warnings
warnings.filterwarnings('ignore')

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

train = pd.read_csv("../input/santander-customer-transaction-prediction/train_mod.csv")
test = pd.read_csv("../input/santander-customer-transaction-prediction/test_mod.csv")

print("train shape: {}". format(train.shape))
print("test shape: {}". format(test.shape))

features = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target']

# hyperparameter setting
param = {
      'bagging_freq': 5,
      'bagging_fraction': 0.4,
      'boost_from_average':'false',
      'boost': 'gbdt',
      'feature_fraction': 0.05,
      'learning_rate': 0.01,
      'max_depth': -1,  
      'metric':'auc',
      'min_data_in_leaf': 80,
      'min_sum_hessian_in_leaf': 10.0,
      'num_leaves': 13,
      'num_threads': 8,
      'tree_learner': 'serial',
      'objective': 'binary', 
      'verbosity': 1
    }

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
                
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
                
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))


sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("../result/submission_lgb_2nd.csv", index=False)
