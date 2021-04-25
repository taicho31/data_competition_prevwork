def my_hyperopt(X, Y):
    def para_tuning_obj(params):
        params = {
        'boosting_type': 'gbdt', 
        'metric': 'rmse', 
        'objective': 'regression', 
        'eval_metric': 'cappa', 
        "tree_learner": "serial",
        'max_depth': int(params['max_depth']),
        'bagging_freq': int(params['bagging_freq']),
        'bagging_fraction': float(params['bagging_fraction']),
        'num_leaves': int(params['num_leaves']),
        'feature_fraction': float(params['feature_fraction']),
        'learning_rate': float(params['learning_rate']),
        'min_data_in_leaf': int(params['min_data_in_leaf']),
        'min_sum_hessian_in_leaf': int(params['min_sum_hessian_in_leaf']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
}
    
        real = np.array([])
        pred = np.array([])
        skf = GroupKFold(n_splits=5)
        for trn_idx, val_idx in skf.split(X, Y, X["installation_id"]):
            x_train, x_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]
            y_train, y_val = Y.iloc[trn_idx], Y.iloc[val_idx]
            x_train.drop('installation_id', inplace = True, axis = 1)
            x_val.drop('installation_id', inplace = True, axis = 1)
            train_set = lgb.Dataset(x_train, y_train, categorical_feature = ['session_title'])
            val_set = lgb.Dataset(x_val, y_val, categorical_feature = ['session_title'])
        
            clf = lgb.train(params, train_set, num_boost_round = 100000, early_stopping_rounds = 100, 
                         valid_sets = [train_set, val_set], verbose_eval = 300)
            pred = np.concatenate((pred, np.array(clf.predict(x_val, num_iteration = clf.best_iteration))), axis=0) 
            real = np.concatenate((real, np.array(y_val)), axis=0) 
        score = np.sqrt(mean_squared_error(real, pred))
    
        return score

    trials = Trials()

    space ={
        'max_depth': hp.quniform('max_depth', 1, 30, 1),
        'bagging_freq': hp.quniform('bagging_freq', 1, 10, 1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.2, 1.0),
        'num_leaves': hp.quniform('num_leaves', 8, 128, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.2, 1.0),
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 8, 128, 1),
        'min_sum_hessian_in_leaf': hp.quniform('min_sum_hessian_in_leaf', 5, 30, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0)
    }

    best = fmin(para_tuning_obj, space = space, algo=tpe.suggest, max_evals=10, trials=trials, verbose=1)

    best_params = space_eval(space, best)
    return best_params

X = new_train.drop(["accuracy_group"], axis=1).copy()
X = pd.merge(X, train_labels[["game_session", "num_correct", "num_incorrect"]], on ="game_session")
Y = X.num_incorrect.copy()
Y.loc[Y >=2] = 2
random_state = 42
X = X.drop(['game_session', "num_correct", "num_incorrect"],axis=1) 
lbl = preprocessing.LabelEncoder()
lbl.fit(list(X["installation_id"]))
X["installation_id"] = lbl.transform(list(X["installation_id"]))
remove_features = [i for i in X.columns if "_4235" in i or i == "world_"+str(activities_world["NONE"]) 
                       or i in to_exclude]
for i in X.columns:
    if X[i].std() == 0 and i not in remove_features:
        remove_features.append(i)
X = X.drop(remove_features, axis=1)
X = X[sorted(X.columns.tolist())]
my_hyperopt(X, Y)
