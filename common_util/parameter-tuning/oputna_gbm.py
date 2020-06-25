# https://gist.github.com/smly/367c53e855cdaeea35736f32876b7416

import optuna.integration.lightgbm as lgb ###### optuna ######
import json

categoricals = []
def modelling_optuna(new_train, new_test):
    X_train = new_train.drop(['target'],axis=1).copy()
    y_train = new_train.target.copy()
    
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(X_train["chip_id"]))
    X_train["chip_id"] = lbl.transform(list(X_train["chip_id"]))
    
    remove_features = []
    for i in X_train.columns:
        if (X_train[i].std() == 0) and i not in remove_features:
            remove_features.append(i)
    X_train = X_train.drop(remove_features, axis=1)
    X_test = new_test.copy()
    X_test = X_test.drop(remove_features, axis=1)
    
    X_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]
    X_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]

    n_folds=5
    skf=StratifiedKFold(n_splits = n_folds, shuffle=True, random_state=0)
    models = []

    valid = np.array([])
    valid_lgb = pd.DataFrame(np.zeros([X_train.shape[0]]))
    real = np.array([])
    features_list = [i for i in X_train.columns if i != "chip_id"]
    feature_importance_df = pd.DataFrame(features_list, columns=["Feature"])
    mean_score = 0
    best_params_list = []
    for i , (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        print("Fold "+str(i+1))
        X_train2 = X_train.iloc[train_index,:]
        y_train2 = y_train.iloc[train_index]

        X_test2 = X_train.iloc[test_index,:]
        y_test2 = y_train.iloc[test_index]
        
        X_train2.drop(["chip_id"], axis=1, inplace=True)
        X_test2.drop(["chip_id"], axis=1, inplace=True)
        lgb_train = lgb.Dataset(X_train2, y_train2)
        lgb_eval = lgb.Dataset(X_test2, y_test2, reference=lgb_train)
        
        best_params, tuning_history = dict(), list()
        lgbm_params = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'tree_learner': 'serial'}
        
        clf = lgb.train(lgbm_params, lgb_train,valid_sets=[lgb_train, lgb_eval],
           num_boost_round=10000,early_stopping_rounds=100,verbose_eval = 0, feval=pr_auc_metric, categorical_feature = categoricals,
                    best_params=best_params, tuning_history=tuning_history) 
            
        valid_predict = clf.predict(X_test2, num_iteration = clf.best_iteration)
        mean_score += average_precision_score(y_test2,valid_predict) / n_folds
        valid = np.concatenate([valid, valid_predict])
        valid_lgb.iloc[test_index]  = clf.predict(X_test2, num_iteration = clf.best_iteration).reshape(X_test2.shape[0], 1)
        real = np.concatenate([real, y_test2])
        #feature_importance_df["Fold_"+str(i+1)] = clf.feature_importance()
        #models.append(clf)
        
        #pd.DataFrame(tuning_history).to_csv('./tuning_history.csv')
        best_params_list.append(best_params)
    
    for j in range(n_folds):
        print('Fold: ' + str(j+1) + ' Best parameters: ' + json.dumps(best_params_list[j], indent=4))

    #print('Best parameters: ' + json.dumps(best_params, indent=4))
    #feature_importance_df["Average"] = np.mean(feature_importance_df.iloc[:,1:n_folds+1], axis=1)
    #feature_importance_df["Std"] = np.std(feature_importance_df.iloc[:,1:n_folds+1], axis=1)
    #feature_importance_df["Cv"] = feature_importance_df["Std"] / feature_importance_df["Average"]

    score = average_precision_score(real, valid)
    print("mean score = {}".format(mean_score))
    print("average precision score = {}".format(score))
    #pred_value = np.zeros(X_test.shape[0])
    #for model in models:
    #    pred_value += model.predict(X_test, num_iteration = model.best_iteration) / len(models)
    return score, feature_importance_df #, pred_value

metric_skf, feature_importance_df_skf = modelling_optuna(new_train, new_test)
