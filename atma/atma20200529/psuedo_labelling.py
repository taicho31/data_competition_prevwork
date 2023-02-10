# pseudo labeling
test2 = new_test.copy()
test2["target"] = pred_value_skf
test2 = test2.sort_values("target", ascending=False).reset_index(drop=True)
#test2p = test2[ (test2['target']<=0.0001) | (test2['target']>=0.9999) ].copy() # select by score
top = list(range(20))
last = list(range(test2.shape[0]-50, test2.shape[0]))
test2p = test2[test2.index.isin(top) | test2.index.isin(last)].copy()  # select by rank
test2p.loc[ test2p['target']>=0.5, 'target' ] = 1
test2p.loc[ test2p['target']<0.5, 'target' ] = 0 
orig_len = new_train.shape[0]
new_train2 = pd.concat([new_train,test2p],axis=0)
new_train2.reset_index(drop=True,inplace=True)
print("before pseudo labeling: ", new_train.shape)
print("after pseudo labeling : ", new_train2.shape)
print("test data size        : ", new_test.shape)

categoricals = []
lgbm_params = {'objective': 'binary', 'metric': 'None', 'boosting_type': 'gbdt', 'tree_learner': 'serial', 'learning_rate': 0.03, "num_leaves": 10, 'random_seed':44,'max_depth': 5}

def modelling_pseudo(new_train, new_test, cons_len):
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
    real = np.array([])
    features_list = [i for i in X_train.columns if i != "chip_id"]
    feature_importance_df = pd.DataFrame(features_list, columns=["Feature"])
    mean_score = 0
    for i , (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        print("Fold "+str(i+1))
        X_train2 = X_train.iloc[train_index,:]
        y_train2 = y_train.iloc[train_index]

        X_test2 = X_train.iloc[test_index,:]
        y_test2 = y_train.iloc[test_index]
        
        X_test2 = X_test2[X_test2.index < cons_len]
        y_test2 = y_test2[y_test2.index < cons_len]
        
        X_train2.drop(["chip_id"], axis=1, inplace=True)
        X_test2.drop(["chip_id"], axis=1, inplace=True)
        lgb_train = lgb.Dataset(X_train2, y_train2)
        lgb_eval = lgb.Dataset(X_test2, y_test2, reference=lgb_train)
        
        clf = lgb.train(lgbm_params, lgb_train,valid_sets=[lgb_train, lgb_eval],
           num_boost_round=10000,early_stopping_rounds=100,verbose_eval = 1000, feval=pr_auc_metric, categorical_feature = categoricals) 
            
        valid_predict = clf.predict(X_test2, num_iteration = clf.best_iteration)
        mean_score += average_precision_score(y_test2,valid_predict) / n_folds
        valid = np.concatenate([valid, valid_predict])
        real = np.concatenate([real, y_test2])
        feature_importance_df["Fold_"+str(i+1)] = clf.feature_importance()
        models.append(clf)
        
    feature_importance_df["Average"] = np.mean(feature_importance_df.iloc[:,1:n_folds+1], axis=1)
    feature_importance_df["Std"] = np.std(feature_importance_df.iloc[:,1:n_folds+1], axis=1)
    feature_importance_df["Cv"] = feature_importance_df["Std"] / feature_importance_df["Average"]

    score = average_precision_score(real, valid)
    print("mean score = {}".format(mean_score))
    print("average precision score = {}".format(score))
    pred_value = np.zeros(X_test.shape[0])
    for model in models:
        pred_value += model.predict(X_test, num_iteration = model.best_iteration) / len(models)
    return score, pred_value, feature_importance_df

metric_lgb_pl, pred_value_lgb_pl, feature_importance_df_lgb_pl = modelling_pseudo(new_train2, new_test, orig_len)
