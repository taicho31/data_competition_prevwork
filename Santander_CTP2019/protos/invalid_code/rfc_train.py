import pandas as pd
import numpy as np
import feather
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from scipy.stats import norm
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

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv("../result/bayesiantuning/"+clf_name+"_cv_results.csv")

# data installation ---------------------------------------------
print("install data")

train = feather.read_dataframe(path_train)
test = feather.read_dataframe(path_test)

print("train shape: {}". format(train.shape))

train_mod = feather.read_dataframe(path_train_mod)
test_mod = feather.read_dataframe(path_test_mod)

print("train_mod shape: {}". format(train_mod.shape))

train_std = feather.read_dataframe(path_train_std)
test_std = feather.read_dataframe(path_test_std)

print("train_std shape: {}". format(train_std.shape))

train_mod_std = feather.read_dataframe(path_train_mod_std)
test_mod_std = feather.read_dataframe(path_test_mod_std)

print("train_mod_std shape: {}". format(train_mod_std.shape))

features = [c for c in train_mod.columns if c not in ['ID_code', 'target']]
target = train['target']
target_mod = train_mod["target"]
target_std = train_std["target"]
target_mod_std = train_mod_std["target"]

print("data install complete")

# feature importances -----------------------------------------------------------------
#print("feature importances -------------")
#rf = RandomForestClassifier(random_state=0)
#rf.fit(train_mod[features], target_mod)
#importances = list(rf.feature_importances_)
#columns = list(train_mod[features].columns)

#importances = pd.DataFrame(rf.feature_importances_, columns=["importances"])
#columns = pd.DataFrame(train_mod[features].columns, columns=["variable"])

#data = pd.concat([columns, importances], axis=1)
#sort_data = data.sort_values(by="importances", ascending = False).reset_index(drop=True)

#print(data.sort_values(by="importances", ascending = False).reset_index(drop=True).head(15))
#for i in np.arange(50, train_mod_std.shape[1], 50):
#    print("sum of importances by highest {} features: {}".format(i, sort_data[:i].importances.sum()))

# learning and prediction -------------------------------------------------------------
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

# random forest classifier ---------------------------------------
print("random forest classifier")
#print("1. original data---------------------")
#rfc_tmp = RandomForestClassifier(n_estimators=300,criterion='entropy', random_state=0, n_jobs=1,bootstrap=True)

#clf = RandomizedSearchCV(rfc_tmp, param_rfc, scoring = "roc_auc", cv=10, n_jobs=-1)
#clf = clf.fit(train[features], target)
#print("param list: {}".format(clf.best_params_))
#print(clf.cv_results_)
#rfc = clf.best_estimator_

#print("1.1 model development")
#oof_rfc = np.zeros(len(train))
#predictions_rfc = np.zeros(len(test))

#for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
#    print("Fold {}".format(fold_+1))
#    rfc.fit(train.iloc[trn_idx][features], target.iloc[trn_idx])
#    oof_rfc[val_idx] = rfc.predict_proba(train.iloc[val_idx][features])[:,1]

#    predictions_rfc += rfc.predict_proba(test[features])[:,1] / folds.n_splits

#    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_rfc)))

#sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
#sub_df["target"] = predictions_rfc
#sub_df.to_csv("../result/submission_rfc_orig.csv", index=False)

#print("2. Std data--------------------------------------")
#print("2.1 model development")
#oof_rfc = np.zeros(len(train))
#predictions_rfc = np.zeros(len(test))

#for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_std.values, target_std.values)):
#    print("Fold {}".format(fold_+1))
#    rfc.fit(train_std.iloc[trn_idx][features], target_std.iloc[trn_idx])
#    oof_rfc[val_idx] = rfc.predict_proba(train_std.iloc[val_idx][features])[:,1]

#    predictions_rfc += rfc.predict_proba(test_std[features])[:,1] / folds.n_splits

#    print("CV score: {:<8.5f}".format(roc_auc_score(target_std, oof_rfc)))

#sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
#sub_df["target"] = predictions_rfc
#sub_df.to_csv("../result/submission_rfc_std.csv", index=False)

print("3. Mod data---------------------------------------")
bayes_cv_tuner = BayesSearchCV(
    estimator = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight = "balanced"),
    search_spaces = {
      "n_estimators" : (100, 500),
      "criterion" : ["gini", "entropy"],
      "bootstrap" : [True, False],
      "min_samples_leaf" : (1, 500),
      "max_depth" : (1,30),
      "max_leaf_nodes" : (1, 100),
      "oob_score": [True, False],
      "min_samples_split" : (10,100),
      "min_impurity_decrease" : (0, 0.5)
    },
    scoring = "roc_auc",
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs = -3,
    n_iter = 10,
    verbose = 0,
    refit = True,
    random_state = 42
)

result = bayes_cv_tuner.fit(train_mod[features].values, target_mod.values, callback=status_print)

print("3.1 model development")
oof_rfc = np.zeros(len(train))
predictions_rfc = np.zeros(len(test))

best_params = {
}

rfc = RandomForestClassifier(n_estimators=300,criterion='entropy', random_state=0, n_jobs=1,bootstrap=True, 
                            min_samples_leaf=2, max_features=0.8, max_depth=20, min_samples_split=40 )

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_mod.values, target_mod.values)):
    print("Fold {}".format(fold_+1))
    rfc.fit(train_mod.iloc[trn_idx][features], target_mod.iloc[trn_idx])
    oof_rfc[val_idx] = rfc.predict_proba(train_mod.iloc[val_idx][features])[:,1]

    predictions_rfc += rfc.predict_proba(test_mod[features])[:,1] / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target_mod, oof_rfc)))

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions_rfc
sub_df.to_csv("../result/submission_rfc_mod.csv", index=False)

#print("4. RandomizedSearchCV mod std data---------------------------------------")
#for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_mod_std.values, target_mod_std.values)):
#    print("Fold {}".format(fold_+1))
#    rfc.fit(train_mod_std.iloc[trn_idx][features], target_mod_std.iloc[trn_idx])
#    oof_rfc[val_idx] = rfc.predict_proba(train_mod_std.iloc[val_idx][features])[:,1]

#    predictions_rfc += rfc.predict_proba(test_mod_std[features])[:,1] / folds.n_splits

#    print("CV score: {:<8.5f}".format(roc_auc_score(target_mod_std, oof_rfc)))

#sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
#sub_df["target"] = predictions_rfc
#sub_df.to_csv("../result/submission_rfc_mod_std.csv", index=False)
