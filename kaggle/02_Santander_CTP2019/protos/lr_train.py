import pandas as pd
import lightgbm as lgb
import numpy as np
import feather
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
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

# data installation -----------------------------------------------------------------
print("install data")
train = feather.read_dataframe(path_train)
test = feather.read_dataframe(path_test)

train_mod = feather.read_dataframe(path_train_mod)
test_mod = feather.read_dataframe(path_test_mod)

train_std = feather.read_dataframe(path_train_std)
test_std = feather.read_dataframe(path_test_std)

train_mod_std = feather.read_dataframe(path_train_mod_std)
test_mod_std = feather.read_dataframe(path_test_mod_std)

features = [c for c in train_mod_std.columns if c not in ['ID_code', 'target']]
target = train['target']
target_mod = train_mod["target"]
target_std = train_std["target"]
target_mod_std = train_mod_std["target"]

print("data installation complete")

# feature importances -----------------------------------------------------------------
#print("feature importances -------------")
#rf = RandomForestClassifier(random_state=0)
#rf.fit(X_train_mod, target)
#importances = list(rf.feature_importances_)
#columns = list(X_train.columns)

#importances = pd.DataFrame(rf.feature_importances_, columns=["importances"])
#columns = pd.DataFrame(X_train_mod.columns, columns=["variable"])

#data = pd.concat([columns, importances], axis=1)
#sort_data = data.sort_values(by="importances", ascending = False).reset_index(drop=True)

#print(data.sort_values(by="importances", ascending = False).reset_index(drop=True).head(15))
#for i in np.arange(50, train.shape[1], 50):
#    print("sum of importances by highest {} features: {}".format(i, sort_data[:i].importances.sum()))

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

# learning and prediction -------------------------------------------------------------
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

# logistic regression ------------------------------------------
print("logistic regression-------------")
print("modified std data --------")
bayes_cv_tuner = BayesSearchCV(
    estimator = LogisticRegression(warm_start=True, random_state=0, n_jobs=-1, solver='lbfgs', max_iter=5000),
    search_spaces = {
      'tol': (0.1, 1000),
      'C': (0.1, 1000)
    },
    scoring = "roc_auc",
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs = -3,
    n_iter = 10,
    verbose = 0,
    refit = True,
    random_state = 42
)

result = bayes_cv_tuner.fit(train_mod_std[features].values, target_mod_std.values, callback=status_print)

#Model #4
#Best ROC-AUC: 0.8852
#Best params: {'C': 812.4147487585276, 'tol': 171.95437424042103}

best_params_mod_std = {
    warm_start=True,
    random_state=0,
    n_jobs=-1,
    solver='lbfgs',
    max_iter=5000,
    tol = 171.95437424042103
    C = 812.4147487585276,
}

lr = LogisticRegression(best_params_mod_std)

print("model developement------------")
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_mod_std.values, target_mod_std.values)):
    print("Fold {}".format(fold_+1))
    lr.fit(train_mod_std.iloc[trn_idx][features], target_mod_std.iloc[trn_idx])
    oof[val_idx] = lr.predict_proba(train_mod_std.iloc[val_idx][features])[:,1]

    predictions += lr.predict_proba(test_mod_std[features])[:, 1] / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target_mod_std, oof)))

sub_df = pd.DataFrame({"ID_code": test_mod_std["ID_code"].values})
sub_df["target"] = predictions 
sub_df.to_csv("../result/submission_lr_mod_std.csv", index=False)



