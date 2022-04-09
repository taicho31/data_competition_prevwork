import pandas as pd
import numpy as np
import feather
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import norm
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
#train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
#test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

#train_mod = pd.read_csv("../input/santander-customer-transaction-prediction/train_mod.csv")
#test_mod = pd.read_csv("../input/santander-customer-transaction-prediction/test_mod.csv")

#train_std = pd.read_csv("../input/santander-customer-transaction-prediction/train_std.csv")
#test_std = pd.read_csv("../input/santander-customer-transaction-prediction/test_std.csv")

#train_mod_std = pd.read_csv("../input/santander-customer-transaction-prediction/train_mod_std.csv")
#test_mod_std = pd.read_csv("../input/santander-customer-transaction-prediction/test_mod_std.csv")

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
model = GradientBoostingClassifier(random_state=0)
model.fit(train_mod[features], target)
importances = list(model.feature_importances_)
columns = list(train_mod[features].columns)

importances = pd.DataFrame(model.feature_importances_, columns=["importances"])
columns = pd.DataFrame(train_mod[features].columns, columns=["variable"])

data = pd.congb([columns, importances], axis=1)
sort_data = data.sort_values(by="importances", ascending = False).reset_index(drop=True)

print(data.sort_values(by="importances", ascending = False).reset_index(drop=True).head(15))
for i in np.arange(50, train_mod[features].shape[1], 50):
    print("sum of importances by highest {} features: {}".format(i, sort_data[:i].importances.sum()))

for i in range(sort_data.shape[0]):
    if sort_data.loc[:i,"importances"].sum() >= 0.95:
        selected_features = list(sort_data.loc[:i,"variable"])
        break

# learning and prediction -------------------------------------------------------------
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

print("GradientBoostingClassifier")
bayes_cv_tuner = BayesSearchCV(
    estimator = GradientBoostingClassifier(loss="exponential", random_state=0,criterion=roc_auc_score, warm_start=True,verbose=1, validation_fraction=0.1, n_iter_no_change=100),
    search_spaces = {
        'learning_rate': (0.01, 0.1),
        'N_estimators': (10,1000),
        'min_weight_fraction_leaf': (0,1),
        'min_samples_split': (50, 100),
        'max_depth': (30, 50),
        "min_samples_leaf": (10, 20, 30)
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

#Model #4
#Best ROC-AUC: 0.8852
#Best params: {'C': 812.4147487585276, 'tol': 171.95437424042103}

best_params = {

}

gb = GradientBoostingClassifier((best_params)

print("3. Modified data---------------------------------------")
print("3.1 model development")
oof_gb = np.zeros(len(train))
predictions_gb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_mod.values, target_mod.values)):
    print("Fold {}".format(fold_+1))
    gb.fit(train_mod.iloc[trn_idx][selected_features], target_mod.iloc[trn_idx])
    oof_gb[val_idx] = gb.predict_proba(train_mod.iloc[val_idx][selected_features])[:,1]

    predictions_gb += gb.predict_proba(test_mod[selected_features])[:,1] / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target_mod, oof_gb)))

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions_gb
sub_df.to_csv("../result/submission_gb_mod.csv", index=False)

