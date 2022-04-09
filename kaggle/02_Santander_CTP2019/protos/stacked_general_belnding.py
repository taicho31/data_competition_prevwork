from catboost import CatBoostClassifier
import pandas as pd
import lightgbm as lgb
import numpy as np
import feather
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import warnings
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
warnings.filterwarnings('ignore')

path_train = '../input/train.feather'
path_test = '../input/test.feather'

path_train_mod = '../input/train_mod.feather'
path_test_mod = '../input/test_mod.feather'

path_train_std = '../input/train_std.feather'
path_test_std = '../input/test_std.feather'

path_train_mod_std = '../input/train_mod_std.feather'
path_test_mod_std = '../input/test_mod_std.feather'

def Stacking(model,train,target,test,n_fold):
   folds=StratifiedKFold(n_splits=n_fold,random_state=0)
   train_pred = np.zeros(len(train))
   predictions_xgb = np.zeros(len(test))
   for train_indices,val_indices in folds.split(train.values, target.values):
      model.fit(train.iloc[trn_idx], target.iloc[trn_idx])
      train_pred[val_idx] = model.predict_proba(train.iloc[val_idx])[:,1]

   test_pred= model.predict_proba(test)[:,1]
   return test_pred ,train_pred

# data installation -----------------------------------------------------------------
print("install data")
#train = pd.read_csv("../input/santander-customer-transaction-prediction/train_mod_std.csv")
#test = pd.read_csv("../input/santander-customer-transaction-prediction/test_mod_std.csv")

train_mod_std = feather.read_dataframe(path_train_mod_std)
test_mod_std = feather.read_dataframe(path_test_mod_std)

train_mod = feather.read_dataframe(path_train_mod)
test_mod = feather.read_dataframe(path_test_mod)

features = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target']

print("data install complete")

print("stacking start")
model_lr = LogisticRegression(warm_start=True, random_state=0, n_jobs=-1, solver='lbfgs', max_iter=1500)
test_pred_lr ,train_pred_lr = Stacking(model=model_lr,n_fold=10, train=train_mod_std[features],test=test_mod_std[features],y=target)
train_pred_lr =pd.DataFrame(train_pred_lr)
test_pred_lr = pd.DataFrame(test_pred_lr)

model_xgb = xgb.XGBClassifier(n_estimators=300,silent=True,nthread=1,eval_metric="auc",objective="binary:logistic",   min_child_weight= 1, gamma=2,subsample=0.8, colsample_bytree=1.0)
test_pred_xgb ,train_pred_xgb = Stacking(model=model_xgb,n_fold=10,train=train_mod[features],test=test_mod[features],y=target)
train_pred_xgb = pd.DataFrame(train_pred_xgb)
test_pred_xgb = pd.DataFrame(test_pred_xgb)

model_cat = CatBoostClassifier(learning_rate=0.03, objective="Logloss",eval_metric='AUC',iterations=50000,
                              colsample_bylevel=0.03)
test_pred_cat ,train_pred_cat = Stacking(model=model_cat, n_fold=10,train=train_mod[features],test=test_mod[features],y=target)
train_pred_cat = pd.DataFrame(train_pred_cat)
test_pred_cat = pd.DataFrame(test_pred_cat)

model_etc = ExtraTreesClassifier(n_estimators=300,max_depth=20, random_state=0, class_weight="balanced")
test_pred_etc ,train_pred_etc = Stacking(model=model_etc, n_fold=10,train=train_mod[features],test=test_mod[features],y=target)
train_pred_etc = pd.DataFrame(train_pred_etc)
test_pred_etc = pd.DataFrame(test_pred_etc)

model_gb = GradientBoostingClassifier(loss="exponential", n_estimators=500,random_state=0, warm_start=True,verbose=1, validation_fraction=0.1, n_iter_no_change=100)
test_pred_etc ,train_pred_etc = Stacking(model=model_gb, n_fold=10,train=train_mod[features],test=test_mod[features],y=target)
train_pred_gb = pd.DataFrame(train_pred_gb)
test_pred_gb = pd.DataFrame(test_pred_gb)

train_stack = pd.concat([train_mod[features], train_pred_lr, train_pred_rfc, train_pred_xgb, train_pred_cat, train_pred_etc, train_pred_gb], axis=1)
test_stack = pd.concat([test_mod[features], test_pred_lr, test_pred_rfc, test_pred_xgb, test_pred_cat, test_pred_etc, test_pred_gb], axis=1)

# final answer --------------------------------------------------
model = xgb.XGBClassifier(random_state=1)
model.fit(train_stack , target)
predictions = model.predict_proba(test_stack)

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("../result/submission_stacking.csv", index=False)
