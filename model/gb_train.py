import pandas as pd
import numpy as np
import feather
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
import warnings
import pickle
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from datetime import datetime
import os
import datetime
warnings.filterwarnings('ignore')

logger = getLogger(__name__)

TRAIN = '../input/train_mod.feather'
TEST = '../input/test_mod.feather'

DIR = '../result/logfile'
CLASS = 2

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

if __name__ == "__main__":

    start_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'gb_train.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    logger.info('GradientBoostingClassifier')
    logger.info('install data')

    train = feather.read_dataframe(TRAIN)
    test = feather.read_dataframe(TEST)

    id_feature = ""
    target_feature = ""

    features = [c for c in train.columns if c not in [id_feature, target_feature]]
    target= train[target_feature]
    logger.info('data install complete')

    logger.info('feature importances')
    model = GradientBoostingClassifier(random_state=0)
    model.fit(train[features], target)
    importances = list(model.feature_importances_)
    columns = list(train[features].columns)

    importances = pd.DataFrame(model.feature_importances_, columns=["importances"])
    columns = pd.DataFrame(train[features].columns, columns=["variable"])

    data = pd.concat([columns, importances], axis=1)
    sort_data = data.sort_values(by="importances", ascending = False).reset_index(drop=True)

    logger.info(data.sort_values(by="importances", ascending = False).reset_index(drop=True).head(15))
    for i in np.arange(50, train[features].shape[1], 50):
        logger.debug("sum of importances by highest {} features: {}".format(i, sort_data[:i].importances.sum()))

    for i in range(sort_data.shape[0]):
        if sort_data.loc[:i,"importances"].sum() >= 0.95 * sort_data.importances.sum():
            selected_features = list(sort_data.loc[:i,"variable"])
            break

    use_cols = train[selected_features].columns.values
    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))
    logger.info('data preparation end {}'.format(train[selected_features].shape))

    logger.info('Paramter tuning by BayesSearch')
    params = {'loss':"exponential", 'random_state':0,'warm_start':"True",
                'verbose':1, 'validation_fraction':0.1, 'n_iter_no_change':100}

    bayes_cv_tuner = BayesSearchCV(
        estimator = GradientBoostingClassifier(loss="exponential", random_state=0, warm_start=True,verbose=1, validation_fraction=0.1, n_iter_no_change=100),
        search_spaces = {
            'learning_rate': (0.01, 0.1),
            'n_estimators': (10,1000),
            'min_weight_fraction_leaf': (0,0.5),
            'min_samples_split': (50, 100),
            'max_depth': (30, 50),
            'min_samples_leaf': (10, 30)
        },
        scoring = "roc_auc",
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs = -3,
        n_iter = 10,
        verbose = 0,
        refit = True,
        random_state = 42
        )

    result = bayes_cv_tuner.fit(train[selected_features].values, target.values, callback=status_print)

    logger.info('found parameters by bayes searchCV: {}'.format(bayes_cv_tuner.best_params_))
    logger.info('best scores by bayes searchCV: {}'.format(bayes_cv_tuner.best_score_))

    params.update(bayes_cv_tuner.best_params_)

    path = "../result/parameter_catboost.csv"
    keys = pd.DataFrame(list(my_dict.keys()))
    values = pd.DataFrame(list(my_dict.values()))
    current = pd.concat([keys, values], axis=1)
    current.columns = [str(start_time)+"keys", str(start_time)+"values"]
    if os.path.isfile(path):
        data = pd.read_csv(path)
        data = pd.concat([data, current], axis=1)
        data.to_csv(path)
    else:
        current.to_csv(path)

    logger.info('Learning start')
    folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
    oof = np.zeros((len(train), CLASS))
    predictions = np.zeros((len(test), CLASS))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        logger.info("Fold {}".format(fold_+1))
        gb = GradientBoostingClassifier(**params)
        gb.fit(train.iloc[trn_idx][selected_features], target.iloc[trn_idx])
        oof[val_idx] = gb.predict_proba(train.iloc[val_idx][selected_features])[:,1].reshape(CLASS, 1)

        predictions += gb.predict_proba(test[selected_features])[:,1].reshape(CLASS, 1) / folds.n_splits

        logger.debug("CV score: {:<8.5f}".format(roc_auc_score(target.iloc[val_idx], oof[val_idx])))

    logger.info('Learning end')
    score = roc_auc_score(target, oof)
    oof = pd.DataFrame(oof, columns=[str(start_time)+str(i) for i in target_feature])

    logger.info('record oof')
    path = "../result/gradientboost_oof.csv"
    if os.path.isfile(path):
        data = pd.read_csv(path)
    else:
        data = pd.DataFrame()
    data[[str(start_time)+str(i) for i in target_feature]] = oof
    data.to_csv(path, index=None)

    sub_df = pd.DataFrame({str(id_feature):test[id_feature].values})
    sub_df[target_feature] = predictions
    sub_df.to_csv("../result/submission_gb_"+str(score)+".csv", index=False)

    logger.info('record submission contents')
    path = "../result/gradientboost_submission_sofar.csv"
    if os.path.isfile(path):
        data = pd.read_csv(path)
    else:
        data = pd.DataFrame()
        data[id_feature] = sub_df[id_feature]
    data = pd.concat([data, sub_df[target_feature]], axis=1)
    data = data.rename(columns={str(target_feature): str(start_time.year)+"/"+str(start_time.month)+"/"+str(start_time.day)+
                                "/"+str(start_time.hour)+":"+str(start_time.minute)+"/"+str(mean_auc)[:7]})
    data.to_csv(path, index=None)

    logger.info('end')
