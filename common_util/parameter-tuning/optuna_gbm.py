# https://gist.github.com/smly/367c53e855cdaeea35736f32876b7416
# https://tech.preferred.jp/ja/blog/hyperparameter-tuning-with-optuna-integration-lightgbm-tuner/
# https://kiseno-log.com/2019/11/05/lightgbm%E3%81%A8optuna%E3%82%92%E5%B0%8E%E5%85%A5%E3%83%BB%E5%8B%95%E3%81%8B%E3%81%97%E3%81%A6%E3%81%BF%E3%82%8B/

class OptunaLGB():
    def __init__(self, data, target_variable, valid_scheme, 
                 n_splits, random_seed, 
                 metric, early_stopping_rounds, verbose_eval):
        self.data = data
        self.target_variable = target_variable
        self.valid_scheme = valid_scheme
        self.n_splits = n_splits
        self.random_seed = random_seed 
        self.metric = metric
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        
    def execute(self):
        import json
        import optuna.integration.lightgbm as lgb 
        y = self.data[self.target_variable]
        X = self.data.drop([self.target_variable],axis=1)

        cv = self.valid_scheme(n_splits = self.n_splits, shuffle=True, random_state=self.random_seed)

        valid = np.zeros([X.shape[0]])
        best_params_dict = {}
        for fold , (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
            print("Fold "+str(fold+1))
            X_train2 = X.iloc[train_index,:]
            y_train2 = y.iloc[train_index]

            X_test2 = X.iloc[test_index,:]
            y_test2 = y.iloc[test_index]
        
            lgb_train = lgb.Dataset(X_train2, y_train2)
            lgb_eval = lgb.Dataset(X_test2, y_test2, reference=lgb_train)
        
            lgbm_params = {'objective': 'binary', 'metric': 'auc'}
        
            clf = lgb.train(lgbm_params,
                            lgb_train,
                            valid_sets=[lgb_train, lgb_eval],
                            num_boost_round=10000, 
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose_eval = 0)  #feval=pr_auc_metric,
            
            best_params = clf.params
            best_params_dict[fold] = best_params

        score = self.metric(y, valid)
        print("Overall score = {}".format(score))
        return best_params_dict

if __name__ == "__main__":
    search = OptunaLGB(data = train[feature+[target]], 
                             target_variable = target,
                             valid_scheme = StratifiedKFold, 
                             n_splits = 5, 
                             random_seed = 0, 
                             metric = roc_auc_score,
                             early_stopping_rounds = 20,
                             verbose_eval = 30
                   )

    best_params_dict = search.execute()
    print(best_params_dict)
