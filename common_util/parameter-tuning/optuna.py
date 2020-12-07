# 学習データの特徴量X, 学習データのtarget Yを準備
# parametersで最適化したいparameterを設定
# CVもしくはtime series splitなどでvalidation schemeを設定
# 最適化したい関数を設定
# maximize or minimizeを設定

import gc
import optuna
from optuna import Trial

def pr_auc_metric(y_predicted, y_true):
    return 'pr_auc', average_precision_score(y_true.get_label(), y_predicted), True

def objective(trial: Trial, fast_check=True, target_meter=0, return_info=False):  
    # data #################
    y = train.target   
    X = train.drop(['target'],axis=1) 
    ########################
    
    # validation scheme ####
    skf=StratifiedKFold(n_splits = 5, shuffle=True, random_state=0)
    ########################
    
    # parameters ###################################
    metric = "None"
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 2, 128),
        'objective': 'binary',
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.1, 1.0),
        "boosting": "gbdt",
        'random_seed': 0,
        'tree_learner': 'serial',
        "metric": metric,
        #'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        #'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        #"bagging_freq": 5,
        #"bagging_fraction": trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        #"feature_fraction": trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        #"verbosity": -1,
    }
    # parameters ###################################
    
    valid = np.zeros([X.shape[0]])
    
    # scoring ###################################
    for i , (train_index, val_index) in enumerate(skf.split(X, y)):
        print("fold: ", i)
        
        X_train = X.iloc[train_index,:]
        y_train = y.iloc[train_index]
        
        X_valid = X.iloc[val_index,:]
        y_valid = y.iloc[val_index]
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        
        model = lgb.train(params, lgb_train,valid_sets=[lgb_train, lgb_eval],
           num_boost_round=10000,early_stopping_rounds=20,verbose_eval = 30,
                          feval=pr_auc_metric, categorical_feature = object_feats)
        
        valid_predict = model.predict(X_valid, num_iteration = model.best_iteration)
        valid[val_index] = valid_predict
        
    valid_score = average_precision_score(y, valid)
    ####################################
    
    return valid_score

study = optuna.create_study(direction='maximize') #maximize or minimize
study.optimize(objective, n_trials=3)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
