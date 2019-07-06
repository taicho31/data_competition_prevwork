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
import pickle
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import os
import datetime
warnings.filterwarnings('ignore')

logger = getLogger(__name__)

TRAIN_MOD = '../input/train_mod.feather'
TEST_MOD = '../input/test_mod.feather'

TRAIN_MOD_STD = '../input/train_mod_std.feather'
TEST_MOD_STD = '../input/test_mod_std.feather'

DIR = '../result/logfile'

random_state = 42
np.random.seed(random_state)

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)
    
    # Get current parameters and the best parameters
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    logger.debug('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_))
        
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv("../result/logfile/bayesiantuning/"+clf_name+"_cv_results.csv")

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

if __name__ == "__main__":
    
    start_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)
    
    handler = FileHandler(DIR + 'lgb_train.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    
    logger.info('start')
    
    logger.info('LightGBMBoostClassifier')
    logger.info('install data')
    train = feather.read_dataframe(TRAIN_MOD)
    test = feather.read_dataframe(TEST_MOD)
    
    #train_mod_std = feather.read_dataframe(path_train_mod_std)
    #test_mod_std = feather.read_dataframe(path_test_mod_std)

    features = [c for c in train.columns if c not in ['ID_code', 'target']]
    target= train["target"]
    logger.info('data install complete')

    logger.info('feature importances')
    model = lgb.LGBMClassifier(class_weight='balanced')
    model.fit(train[features], target)

    importances = list(model.feature_importances_)
    columns = list(train[features].columns)

    importances = pd.DataFrame(model.feature_importances_, columns=["importances"])
    columns = pd.DataFrame(train[features].columns, columns=["variable"])

    data = pd.concat([columns, importances], axis=1)
    sort_data = data.sort_values(by="importances", ascending = False).reset_index(drop=True)

    logger.debug(data.sort_values(by="importances", ascending = False).reset_index(drop=True).head(15))
    for i in np.arange(50, train.shape[1], 50):
        logger.info("sum of importances by highest {} features: {}".format(i, sort_data[:i].importances.sum()))

    for i in range(sort_data.shape[0]):
        if sort_data.loc[:i,"importances"].sum() >= 0.95 * sort_data.importances.sum():
            selected_features = list(sort_data.loc[:i,"variable"])
            break

    use_cols = train[selected_features].columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end {}'.format(train[selected_features].shape))

    X_test = test[selected_features].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof = train[['ID_code', 'target']]
    oof['predict'] = 0
    predictions = test[['ID_code']]
    val_aucs = []
    feature_importance_df = pd.DataFrame()

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):
        X_train, y_train = train.iloc[trn_idx][selected_features], train.iloc[trn_idx]['target']
        X_valid, y_valid = train.iloc[val_idx][selected_features], train.iloc[val_idx]['target']

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
        fold_importance_df["feature"] = selected_features
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
    logger.debug("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))

    logger.info('record oof')
    path = "../result/lgb_oof.csv"
    if os.path.isfile(path):
        data = pd.read_csv(path)
    else:
        data = oof[['ID_code', 'target']]
    data = pd.concat([data, oof['predict']], axis=1)
    data = data.rename(columns={'predict': start_time})
    data.to_csv(path, index=None)

    predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
    #predictions.to_csv('../result/lgb_all_predictions.csv', index=None)

    # submission
    sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
    sub_df["target"] = predictions['target']
    sub_df.to_csv("../result/submission_lgb_"+str(mean_auc)+".csv", index=False)

    # record
    logger.info('record submission contents')
    path = "../result/lgb_submission_sofar.csv"
    if os.path.isfile(path):
        data = pd.read_csv(path)
    else:
        data = pd.DataFrame()
        data["ID_code"] = sub_df["ID_code"]
    data = pd.concat([data, sub_df["target"]], axis=1)
    data = data.rename(columns={'target': start_time})
    data.to_csv(path, index=None)
    
    logger.info('end')
