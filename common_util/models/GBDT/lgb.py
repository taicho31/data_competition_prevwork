def modelling_lgb(new_train, new_test):

    X_train = new_train.drop(['accuracy_group'],axis=1) 
    y_train = X_train.num_correct.copy()

    X_test = new_test.drop(["installation_id","accuracy_group", "game_session"], axis=1)
    X_test = X_test[sorted(X_test.columns.tolist())]

    n_folds=5
    skf=GroupKFold(n_splits = n_folds)
    pred_value = np.zeros([X_test.shape[0]])
    
    lgbm_params = {'objective': 'binary','eval_metric': 'auc','metric': 'auc', 'boosting_type': 'gbdt',
 'tree_learner': 'serial','bagging_fraction': 0.9605425291685099,'bagging_freq': 4,'colsample_bytree': 0.6784238046856443,
 'feature_fraction': 0.9792407844605087,'learning_rate': 0.017891320270412462,'max_depth': 7, 'random_seed':42,
 'min_data_in_leaf': 8,'min_sum_hessian_in_leaf': 17,'num_leaves': 17}

    valid_correct_num = pd.DataFrame(np.zeros([X_train.shape[0]]))
    for i , (train_index, test_index) in enumerate(skf.split(X_train, y_train, X_train["installation_id"])):
        print("Fold "+str(i+1))
        X_train2 = X_train.iloc[train_index,:]
        y_train2 = y_train.iloc[train_index]
        X_train2 = X_train2.drop(['installation_id'],axis=1)
    
        X_test2 = X_train.iloc[test_index,:]
        y_test2 = y_train.iloc[test_index]
        X_test2 = X_test2.drop(['installation_id'],axis=1)
            
        lgb_train = lgb.Dataset(X_train2, y_train2)
        lgb_eval = lgb.Dataset(X_test2, y_test2, reference=lgb_train)
        clf = lgb.train(lgbm_params, lgb_train,valid_sets=[lgb_train, lgb_eval],
            num_boost_round=10000,early_stopping_rounds=100,verbose_eval = 500,categorical_feature = categoricals)
        train_predict = clf.predict(X_train2, num_iteration = clf.best_iteration)
        test_predict = clf.predict(X_test2, num_iteration = clf.best_iteration)
        pred_value += clf.predict(X_test, num_iteration = clf.best_iteration) / len(n_folds)
            
        valid_correct_num.iloc[test_index] = test_predict.reshape(X_test2.shape[0], 1)
                
    print("logloss = \t {}".format(log_loss(y_train, valid_correct_num)))
    print("ROC = \t {}".format(roc_auc_score(y_train, valid_correct_num)))
    print('Accuracy score = \t {}'.format(accuracy_score(y_train, np.round(valid_correct_num))))
    print('Precision score = \t {}'.format(precision_score(y_train, np.round(valid_correct_num))))
    print('Recall score =   \t {}'.format(recall_score(y_train, np.round(valid_correct_num))))
    print('F1 score =      \t {}'.format(f1_score(y_train, np.round(valid_correct_num))))
    print(confusion_matrix(y_train, np.round(valid_correct_num)))
    
    return pred_value, valid_correct_num
pred_value, valid_correct_num = num_correct_calc(new_train, new_test)
