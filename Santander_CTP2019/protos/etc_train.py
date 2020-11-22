import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from scipy.stats import norm
import feather
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

# data installation -----------------------------------------------------------------
print("install data")
#train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
#test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

#train_mod = pd.read_csv("../input/santander-customer-transaction-prediction/train_mod.csv")
#test_mod = pd.read_csv("../input/santander-customer-transaction-prediction/test_mod.csv")

#train_std = pd.read_csv("../input/santander-customer-transaction-prediction/train_std.csv")
#test_std = pd.read_csv("../input/santander-customer-transaction-prediction/test_std.csv")

#train_mod_std = pd.read_csv("../input/santander-customer-transaction-prediction/train_mod_std.csv")
#test_mod_std = pd.read_csv("../input/santander-customer-transaction-prediction/test_mod_std.csv")

train = feather.read_dataframe(path_train)
test = feather.read_dataframe(path_test)

train_mod = feather.read_dataframe(path_train_mod)
test_mod = feather.read_dataframe(path_test_mod)

train_std = feather.read_dataframe(path_train_std)
test_std = feather.read_dataframe(path_test_std)

train_mod_std = feather.read_dataframe(path_train_mod_std)
test_mod_std = feather.read_dataframe(path_test_mod_std)

features = [c for c in train_mod.columns if c not in ['ID_code', 'target']]
target = train['target']
target_std = train_std['target']
target_mod = train_mod['target']
target_mod_std = train_mod_std['target']

print("data install complete")

# feature importances -----------------------------------------------------------------
print("feature importances -------------")
model = ExtraTreesClassifier(random_state=0)
model.fit(train_mod[features], target)
importances = list(model.feature_importances_)
columns = list(train_mod[features].columns)

importances = pd.DataFrame(model.feature_importances_, columns=["importances"])
columns = pd.DataFrame(train_mod[features].columns, columns=["variable"])

data = pd.conetc([columns, importances], axis=1)
sort_data = data.sort_values(by="importances", ascending = False).reset_index(drop=True)

print(data.sort_values(by="importances", ascending = False).reset_index(drop=True).head(15))
for i in np.arange(50, train_mod[features].shape[1], 50):
    print("sum of importances by highest {} features: {}".format(i, sort_data[:i].importances.sum()))

for i in range(sort_data.shape[0]):
    if sort_data.loc[:i,"importances"].sum() >= 0.95:
        selected_features = list(sort_data.loc[:i,"variable"])
        break

# learning and prediction -------------------------------------------------------------
print("BayesSearch")
bayes_cv_tuner = BayesSearchCV(
    estimator = ExtraTreesClassifier(n_estimators=300, random_state=0, class_weight="balanced"),
    search_spaces = {
      'criterion':["gini", "entropy"],
      'splitter': ["random", "best"],
      'min_samples_split': (2, 100),
      'min_samples_leaf': (1,100),
      'min_weight_fraction': (0, 1),
      'max_depth': (1,50),
    },
    scoring = "roc_auc",
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs = -3,
    n_iter = 10,
    verbose = 0,
    refit = True,
    random_state = 42
)

result = bayes_cv_tuner.fit(train_mod[selected_features].values, target_mod.values, callback=status_print)

#Model 
#Best ROC-AUC: 
#Best params: 

best_params= {
}

# learning and prediction -------------------------------------------------------------
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

print("3. RandomizedSearchCV mod data---------------------------------------")
etc = ExtraTreesClassifier(best_params)

print("3.1 model development")
oof_etc = np.zeros(len(train))
predictions_etc = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_mod.values, target_mod.values)):
    print("Fold {}".format(fold_+1))
    etc.fit(train_mod.iloc[trn_idx][selected_features], target_mod.iloc[trn_idx])
    oof_etc[val_idx] = etc.predict_proba(train_mod.iloc[val_idx][selected_features])[:,1]

    predictions_etc += etc.predict_proba(test_mod[selected_features])[:,1] / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target_mod, oof_etc)))

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions_etc
sub_df.to_csv("../result/submission_etc_mod.csv", index=False)

