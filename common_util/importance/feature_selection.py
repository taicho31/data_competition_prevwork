class SelectionByModelImportance():
    def __init__(self, data, target_variable, valid_scheme, 
                 n_splits, random_seed,metric,
                 early_stopping_rounds, verbose_eval, cri, num, params):
        self.data = data
        self.target_variable = target_variable
        self.valid_scheme = valid_scheme
        self.n_splits = n_splits
        self.random_seed = random_seed 
        self.metric = metric
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.cri = cri 
        self.num = num 
        self.params = params
        
    def execute(self):
        import numpy as np
        import lightgbm as lgb
    
        lgbm_params = self.params
        cv = self.valid_scheme(n_splits = self.n_splits, shuffle=True, random_state=self.random_seed)
        y = self.data[self.target_variable]
        X = self.data.drop([self.target_variable],axis=1)
        
        features_list = [i for i in X_train.columns if i != "installation_id"]
        feature_importance_df = pd.DataFrame(features_list, columns=["Feature"])
        for i , (train_index, test_index) in enumerate(cv.split(X, y)):
            print("Fold "+str(i+1))
            X_train2 = X.iloc[train_index,:]
            y_train2 = y.iloc[train_index]
    
            X_test2 = X.iloc[test_index,:]
            y_test2 = y.iloc[test_index]
            
            lgb_train = lgb.Dataset(X_train2, y_train2)
            lgb_eval = lgb.Dataset(X_test2, y_test2, reference=lgb_train)
            clf = lgb.train(lgbm_params, lgb_train,valid_sets=[lgb_train, lgb_eval],
                            num_boost_round=10000,
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose_eval = self.verbose_eval)
            
            feature_importance_df["Fold_"+str(i+1)] = clf.feature_importance()
            
        feature_importance_df["Average"] = np.mean(feature_importance_df.iloc[:,1:self.n_splits+1], axis=1)
        feature_importance_df["Std"] = np.std(feature_importance_df.iloc[:,1:self.n_splits+1], axis=1)
        feature_importance_df["Cv"] = feature_importance_df["Std"] / feature_importance_df["Average"]
    
        if self.cri == "average":
            tmp = feature_importance_df.sort_values("Average", ascending = False).reset_index(drop=True).copy()
            feat = tmp[tmp.index <= self.num-1]["Feature"]
        elif self.cri == "cv":
            tmp = feature_importance_df.sort_values("Cv", ascending = True).reset_index(drop=True).copy()
            feat = tmp[tmp.index <= self.num-1]["Feature"]
        else:
            tmp = feature_importance_df.sort_values("Std", ascending = True).reset_index(drop=True).copy()
            feat = tmp[tmp.index <= self.num-1]["Feature"]
        return list(feat)

if __name__ == "__main__":
    
    params = {'objective': 'binary','eval_metric': 'auc','metric': 'auc', 'boosting_type': 'gbdt',
              'tree_learner': 'serial','bagging_fraction': 0.9605425291685099,'bagging_freq': 4,
              'colsample_bytree': 0.6784238046856443, 'feature_fraction': 0.9792407844605087,
              'learning_rate': 0.017891320270412462,'max_depth': 7,
              'min_data_in_leaf': 8,'min_sum_hessian_in_leaf': 17,'num_leaves': 17}

    feature_selector = SelectionByModelImportance(
                    data = train[feature+[target]], 
                    target_variable = target,
                    valid_scheme = StratifiedKFold, 
                    n_splits = 5, 
                    random_seed = 0, 
                    metric = roc_auc_score,
                    early_stopping_rounds = 20,
                    verbose_eval = 30,
                    cri = "average",
                    num = 3,
                    params = params
                   )

    features = feature_selector.execute()
    print(features)
