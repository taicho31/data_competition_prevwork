# 学習データの特徴量X, 学習データのtarget Yを準備
# parametersで最適化したいparameterを設定
# CVもしくはtime series splitなどでvalidation schemeを設定
# 最適化したい関数を設定
# maximize or minimizeを設定

class ParamSearchByOptuna():
    def __init__(self, data, target_variable, valid_scheme, 
                 n_splits, random_seed, direction,trial_nums,metric,
                 early_stopping_rounds, verbose_eval):
        self.data = data
        self.target_variable = target_variable
        self.valid_scheme = valid_scheme
        self.n_splits = n_splits
        self.random_seed = random_seed 
        self.direction = direction 
        self.trial_nums = trial_nums
        self.metric = metric
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval

    def execute(self):
        import optuna
        from optuna import Trial
        
        print("seprarating features and target")
        y = self.data[self.target_variable]
        X = self.data.drop([self.target_variable],axis=1) 
        valid = np.zeros([X.shape[0]])
        cv = self.valid_scheme(n_splits = self.n_splits, shuffle=True, random_state = self.random_seed)
        
        def objective(trial: Trial, fast_check=True, target_meter=0, return_info=False):  
            model_parameters = {
                    'num_leaves': trial.suggest_int('num_leaves', 2, 128),
                    'objective': 'binary',
                    'max_depth': trial.suggest_int('max_depth', 1, 10),
                    'learning_rate': trial.suggest_uniform('learning_rate', 0.1, 1.0),
                    "boosting": "gbdt",
                    'random_seed': 0,
                    'tree_learner': 'serial',
                    "metric": "auc",
                    #'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                    #'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                    #"bagging_freq": 5,
                    #"bagging_fraction": trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
                    #"feature_fraction": trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                    #"verbosity": -1,
            }
            
            for i , (train_index, val_index) in enumerate(cv.split(X, y)):
                print("fold: ", i)
        
                X_train = X.iloc[train_index,:]
                y_train = y.iloc[train_index]
        
                X_valid = X.iloc[val_index,:]
                y_valid = y.iloc[val_index]
        
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        
                model = lgb.train(model_parameters,
                                  lgb_train,
                                  valid_sets=[lgb_train, lgb_eval],
                                  num_boost_round=10000,
                                  early_stopping_rounds=self.early_stopping_rounds,
                                  verbose_eval = self.verbose_eval,
                                  #feval=self.metric)
                                 )
        
                valid_predict = model.predict(X_valid, num_iteration = model.best_iteration)
                valid[val_index] = valid_predict
        
            valid_score = self.metric(y, valid)
    
            return valid_score

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.trial_nums)

        print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

        return study

if __name__ == "__main__":
    search = ParamSearchByOptuna(data = train[feature+[target]], 
                             target_variable = target,
                             valid_scheme = StratifiedKFold, 
                             n_splits = 5, 
                             random_seed = 0, 
                             direction = "maximize",
                             trial_nums = 1, 
                             metric = roc_auc_score,
                             early_stopping_rounds = 20,
                             verbose_eval = 30
                   )

    study = search.execute()
    print(study.best_trial.params)
