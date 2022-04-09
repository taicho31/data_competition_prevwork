from catboost import CatBoostClassifier,Pool
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.svm import NuSVR
from scipy.stats import norm
import xgboost as xgb
import warnings
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
warnings.filterwarnings('ignore')

# data installation -----------------------------------------------------------------
print("install data")
train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train_mod = pd.read_csv("../input/santander-customer-transaction-prediction/train_mod.csv")
test_mod = pd.read_csv("../input/santander-customer-transaction-prediction/test_mod.csv")

train_std = pd.read_csv("../input/santander-customer-transaction-prediction/train_std.csv")
test_std = pd.read_csv("../input/santander-customer-transaction-prediction/test_std.csv")

train_mod_std = pd.read_csv("../input/santander-customer-transaction-prediction/train_mod_std.csv")
test_mod_std = pd.read_csv("../input/santander-customer-transaction-prediction/test_mod_std.csv")

print("train shape: {}". format(train.shape))
print("test shape: {}". format(test.shape))

features = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target']

X_train = train[features]
X_test = test[features]

X_train_mod = train_mod[features]
X_test_mod = test_mod[features]

X_train_std = train_std[features]
X_test_std = test_std[features]

X_train_mod_std = train_mod_std[features]
X_test_mod_std = test_mod_std[features]
print("data install complete")

# feature importances -----------------------------------------------------------------
print("feature importances -------------")
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train_mod, target)
importances = list(rf.feature_importances_)
columns = list(X_train.columns)

importances = pd.DataFrame(rf.feature_importances_, columns=["importances"])
columns = pd.DataFrame(X_train_mod.columns, columns=["variable"])

data = pd.concat([columns, importances], axis=1)
sort_data = data.sort_values(by="importances", ascending = False).reset_index(drop=True)

print(data.sort_values(by="importances", ascending = False).reset_index(drop=True).head(15))
for i in np.arange(50, train.shape[1], 50):
    print("sum of importances by highest {} features: {}".format(i, sort_data[:i].importances.sum()))

# learning and prediction -------------------------------------------------------------
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

# logistic regression ------------------------------------------
print("logistic regression-------------")
print("GridSearchCV-------------")
lr_tmp = LogisticRegression(warm_start=True, random_state=0, n_jobs=-1, solver='lbfgs', max_iter=1500)
param_lr ={
        'tol':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }

clf = GridSearchCV(lr_tmp, param_lr, scoring = "roc_auc", cv=10, n_jobs=-1)
clf = clf.fit(train[features], target)
print("param list: {}".format(clf.best_params_))
lr = clf.best_estimator_

print("model development--------")
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_+1))
    lr.fit(train.iloc[trn_idx][features], target.iloc[trn_idx])
    oof[val_idx] = lr.predict_proba(train.iloc[val_idx][features])[:,1]

    predictions += lr.predict_proba(test[features])[:,1] / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions
#sub_df.to_csv("../result/submission_lr_1st_mod.csv", index=False)
#sub_df.to_csv("../result/submission_lr_std.csv", index=False)
sub_df.to_csv("../result/submission_lr_2nd_mod.csv", index=False)


# random forest classifier ---------------------------------------
print("random forest classifier-------")
print("RandomizedSearchCV--------")
rfc_tmp = RandomForestClassifier(n_estimators=300,criterion='entropy', random_state=0, n_jobs=1,bootstrap=True)

param_rfc = {
        'min_samples_leaf':[2,3,4,5],
        'max_features':[0.2, 0.3, 0.5, 0.8, 1],
        'max_depth':[10, 20, 30],
        'min_samples_split':[50,100,150],
        }

clf = RandomizedSearchCV(rfc_tmp, param_rfc, scoring = "roc_auc", cv=10, n_jobs=-1)
clf = clf.fit(train[features], target)
print("param list: {}".format(clf.best_params_))
print(clf.cv_results_)
rfc = clf.best_estimator_

print("model development---------")
oof_rfc = np.zeros(len(train))
predictions_rfc = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_+1))
    rfc.fit(train.iloc[trn_idx][features], target.iloc[trn_idx])
    oof_rfc[val_idx] = rfc.predict_proba(train.iloc[val_idx][features])[:,1]

    predictions_rfc += rfc.predict_proba(test[features])[:,1] / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_rfc)))

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions_rfc
sub_df.to_csv("../result/submission_rfc.csv", index=False)


# xgboost ------------------------------------------------------
print("XGBoost------------------------")
print("randomizedSearchCV-------")
xgb_tmp = xgb.XGBClassifier(n_estimators=300,silent=True,nthread=1,eval_metric="auc",objective="binary:logistic",   min_child_weight= 1, gamma=2,subsample=0.8, colsample_bytree=1.0)

param_xgb = {
        'learning_rate': [0.01, 0.05, 0.1],
        'gamma': [1.4, 1.5,  2],
        'subsample': [0.6, 0.8,  1.0],
        'colsample_bytree': [0.8, 1.0],
        'max_depth': [30, 40, 50]
        }

clf = RandomizedSearchCV(xgb_tmp, param_xgb, scoring = "roc_auc", cv=10, n_jobs=-1)
clf = clf.fit(train[features], target)
print("param list: {}".format(clf.best_params_))
print(clf.cv_results_)
xgb = clf.best_estimator_

print("model development--------")
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_+1))
    xgb.fit(train.iloc[trn_idx][features], target.iloc[trn_idx])
    oof_xgb[val_idx] = xgb.predict(train.iloc[val_idx][features])

    predictions_xgb += xgb.predict(test[features]) / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_xgb)))

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions_xgb
sub_df.to_csv("../result/submission_xgb.csv", index=False)

# catboostclassifier -------------------------------------------
print("Catboosting------------------")
print("randomizedSearchCV-------")
cat_tmp = CatBoostClassifier( learning_rate=0.03, objective="Logloss",eval_metric='AUC',iterations=50000,
                              colsample_bylevel=0.03)

param_cat = {
        'learning_rate': [0.01, 0.03, 0.1],
        'subsample': [0.6, 0.8,  1.0],
        'colsample_bylevel': [0.01, 0.03, 0.1],
        'max_depth': [30, 40, 50]
        }

clf = RandomizedSearchCV(cat_tmp, param_cat, scoring = "roc_auc", cv=10, n_jobs=-1)
clf = clf.fit(train[features], target)
print("param list: {}".format(clf.best_params_))
print(clf.cv_results_)
cat = clf.best_estimator_

print("model development--------")
oof_cat = np.zeros(len(train))
predictions_cat = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_+1))
    cat.fit(train.iloc[trn_idx][features], target.iloc[trn_idx])
    oof_cat[val_idx] = cat.predict_proba(train.iloc[val_idx][features])[:,1]

    predictions_cat += cat.predict_proba(test[features])[:,1] / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_cat)))

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions_cat
sub_df.to_csv("../result/submission_cat_2nd_mod.csv", index=False)


print("ExtratreeClassifier--------------")
print("randomizedSearchCV---------")
etc_tmp = ExtraTreesClassifier(n_estimators=300,max_depth=20, random_state=0, class_weight="balanced")

param_etc = {
        'learning_rate': [0.01, 0.03, 0.1],
        'min_samples_split': [50, 80, 100],
        'max_depth': [30, 40, 50]
        }

clf = RandomizedSearchCV(etc_tmp, param_etc, scoring = "roc_auc", cv=10, n_jobs=-1)
clf = clf.fit(train[features], target)
print("param list: {}".format(clf.best_params_))
print(clf.cv_results_)
etc = clf.best_estimator_

print("model development----------")
oof_etc = np.zeros(len(train))
predictions_etc = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_+1))
    etc.fit(train.iloc[trn_idx][features], target.iloc[trn_idx])
    oof_etc[val_idx] = etc.predict_proba(train.iloc[val_idx][features])[:,1]

    predictions_etc += etc.predict_proba(test[features])[:,1] / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_etc)))

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions_etc
sub_df.to_csv("../result/submission_etc_2nd_mod.csv", index=False)

# gradient boosting ---------------------------------------------
print("Gradient Boosting----------------")
print("randomizedSearchCV---------")
gb_tmp = GradientBoostingClassifier(loss="exponential", n_estimators=500,random_state=0, warm_start=True,verbose=1, validation_fraction=0.1, n_iter_no_change=100)

param_gb = {
        'learning_rate': [0.01, 0.03, 0.1],
        'min_samples_split': [50, 80, 100],
        'max_depth': [30, 40, 50],
        "min_samples_leaf": [10, 20, 30]
        }

clf = RandomizedSearchCV(gb_tmp, param_gb, scoring = "roc_auc", cv=10, n_jobs=-1)
clf = clf.fit(train[features], target)
print("param list: {}".format(clf.best_params_))
print(clf.cv_results_)
etc = clf.best_estimator_

print("Model development----------")
oof_gb = np.zeros(len(train))
predictions_gb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_+1))
    gb.fit(train.iloc[trn_idx][features], target.iloc[trn_idx])
    oof_gb[val_idx] = gb.predict_proba(train.iloc[val_idx][features])[:,1]

    predictions_gb += gb.predict_proba(test[features])[:,1] / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_gb)))

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions_gb
sub_df.to_csv("../result/submission_gb_2nd_mod.csv", index=False)


# support vector machine ---------------------------------------
#print("SVM-----------------------------")
#print("RandomizedSearchCV------------")
#svm = SVC(kernel='rbf',degree=1, class_weight="balanced", gamma=0.4)

# param = {
#        'degree':[1,2,3,4,5,6],
#        'gamma':[0.1,0.2,0.3, 0.4, 0.5, 1],
#        'C':[0.1,1, 10,100]
#        }

#print("model development")
#oof_svm = np.zeros(len(train))
#predictions_svm = np.zeros(len(test))

#for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
#    print("Fold {}".format(fold_+1))
#    svm.fit(train.iloc[trn_idx][features], target.iloc[trn_idx])
#    oof_svm[val_idx] = svm.predict(train.iloc[val_idx][features])

#    predictions_svm += svm.predict(test[features]) / folds.n_splits

#    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_svm)))

#sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
#sub_df["target"] = predictions_svm
#sub_df.to_csv("../result/submission_svm_std.csv", index=False)
