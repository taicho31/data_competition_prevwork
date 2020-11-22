import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import feather
import warnings
warnings.filterwarnings('ignore')

random_state = 42
np.random.seed(random_state)

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

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
                        100000,
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

lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}

#train = pd.read_csv("../input/santander-customer-transaction-prediction/train_mod.csv")
#test = pd.read_csv("../input/santander-customer-transaction-prediction/test_mod.csv")
train = feather.read_dataframe(path_train_mod)
test = feather.read_dataframe(path_test_mod)

features = [c for c in train.columns if c not in ['ID_code', 'target']]

print("train shape: {}". format(train.shape))
print("test shape: {}". format(test.shape))

print("--------------------------Feature importances----------------------------------")
model = lgb.LGBMClassifier(class_weight='balanced')
model.fit(train[features], train["target"])

importances = list(lgb_clf.feature_importances_)
columns = list(train_mod[features].columns)

importances = pd.DataFrame(model.feature_importances_, columns=["importances"])
columns = pd.DataFrame(train_mod[features].columns, columns=["variable"])

data = pd.concat([columns, importances], axis=1)
sort_data = data.sort_values(by="importances", ascending = False).reset_index(drop=True)

print(data.sort_values(by="importances", ascending = False).reset_index(drop=True).head(15))
for i in np.arange(50, train_mod.shape[1], 50):
    print("sum of importances by highest {} features: {}".format(i, sort_data[:i].importances.sum()))

for i in range(sort_data.shape[0]):
    if sort_data.loc[:i,"importances"].sum() >= 0.95:
        selected_features = list(sort_data.loc[:i,"variable"])
        break
print("--------------------------Feature importances end-------------------------------")


features = [c for c in train.columns if c not in ['ID_code', 'target']]
X_test = test[features].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
oof = train[['ID_code', 'target']]
oof['predict'] = 0
predictions = test[['ID_code']]
val_aucs = []
feature_importance_df = pd.DataFrame()

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):
    X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']
    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']
    
    N = 5
    p_valid,yp = 0,0
    for i in range(N):
        X_t, y_t = augment(X_train.values, y_train.values)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')
    
        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(X_test)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    oof['predict'][val_idx] = p_valid/N
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
                   
    predictions['fold{}'.format(fold+1)] = yp/N

mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))

# submission
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('../result/lgb_augment_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("../result/lgb_augment_submission_2nd.csv", index=False)
oof.to_csv('../result/lgb_augment_oof.csv', index=False)
