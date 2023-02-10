# preprocess need to be done in advance
from sklearn.model_selection import KFold
import datetime
import lightgbm as lgb

def adversrial_validation(tr, te):
    train_only_feat = list(set(train.columns) - set(test.columns))
    test_only_feat= list(set(test.columns) - set(train.columns))
    
    tr.drop(train_only_feat, axis=1, inplace=True)
    te.drop(test_only_feat, axis=1, inplace=True)
    
    tr['ad_target'] = 1
    te['ad_target'] = 0
    n_train = tr.shape[0]
    ad_df = pd.concat([tr, te], axis = 0)
    
    # Update column names
    predictors = [c for c in ad_df.columns if c != "ad_target"]
    print(predictors)
    
    NFOLD = 5
    ad_df = ad_df.iloc[np.random.permutation(len(ad_df))]
    ad_df.reset_index(drop = True, inplace = True)
    
    # lgb params
    lgb_params = {
         'objective':'binary',
         'max_depth': 5,
         'learning_rate': 0.2,
         "metric": 'auc',
        }
         #"min_child_samples": 20,
         #"boosting": "gbdt",
         #"feature_fraction": 0.9,
         #"bagging_freq": 1,
         #"bagging_fraction": 0.9 ,
         #"bagging_seed": 44,
         #'num_leaves': 50,
         #'min_data_in_leaf': 30, 
    
    # Get folds for k-fold CV
    folds = KFold(n_splits = NFOLD, shuffle = True, random_state = 0)
    fold = folds.split(ad_df)

    # Get target column name
    target = 'ad_target'
    
    eval_score = 0
    eval_preds = np.zeros(ad_df.shape[0])

    for i, (train_index, test_index) in enumerate(fold):
        print( "\n[{}] Fold {} of {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i+1, NFOLD))
        train_X, valid_X = ad_df[predictors].values[train_index], ad_df[predictors].values[test_index]
        train_y, valid_y = ad_df[target].values[train_index], ad_df[target].values[test_index]

        dtrain = lgb.Dataset(train_X, label = train_y,
                          feature_name = list(predictors)
                          )
        dvalid = lgb.Dataset(valid_X, label = valid_y,
                          feature_name = list(predictors)
                          )
        
        eval_results = {}
        
        bst = lgb.train(lgb_params, 
                         dtrain, 
                         valid_sets = [dtrain, dvalid], 
                         valid_names = ['train', 'valid'], 
                         evals_result = eval_results, 
                         num_boost_round = 5000,
                         early_stopping_rounds = 10,
                         verbose_eval = 10)
    
        print("\nRounds:", bst.best_iteration)
        print("AUC: ", eval_results['valid']['auc'][bst.best_iteration-1])
    
        eval_score += eval_results['valid']['auc'][bst.best_iteration-1]
   
        eval_preds[test_index] += bst.predict(valid_X, num_iteration = bst.best_iteration)
        if i == 0:
            feature_importance_df = pd.DataFrame(sorted(zip(predictors, bst.feature_importance())), columns=['Feature','Value1'])            
        else:
            feature_importance_df["Vaue"+str(i+1)] = bst.feature_importance()
    
    feature_importance_df["Average"] = np.mean(feature_importance_df.iloc[:,1:NFOLD+1], axis=1)
    feature_importance_df["Std"] = np.std(feature_importance_df.iloc[:,1:NFOLD+1], axis=1)
    feature_importance_df["Cv"] = feature_importance_df["Std"] / feature_importance_df["Average"]
    eval_score = round(eval_score/NFOLD,6)

    print("\nModel Report")
    print("AUC: ", eval_score) 
    
    lgb.plot_importance(bst, max_num_features = 20)
    
     #Get training rows that are most similar to test
    df_av = ad_df[[i_d, prev_target]].copy()
    df_av['preds'] = eval_preds
    df_av_train = df_av[df_av[prev_target] == 1]
    df_av_train = df_av_train.sort_values(by=['preds']).reset_index(drop=True)

    # Check distribution
    #df_av_train.preds.plot()

    # Store to feather
    #df_av_train[['ID', 'preds']].reset_index(drop=True).to_feather('adversarial_validation.ft')

    #because 0 is assigned to test data, pred close to 0 means that the data is similar to test data
    #print(df_av_train.head(20)) 
    
    return feature_importance_df

sam = adversrial_validation(train, test)
sam.sort_values("Value1", ascending=False).reset_index(drop=True)
