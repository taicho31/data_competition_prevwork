# https://blog.amedama.jp/entry/optuna-lightgbm-tunercv
from optuna.integration import lightgbm as lgb_opt

def lgbm_optuna(X, Y):
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
    }
    # 基本的には cv() 関数のオプションがそのまま渡せる
    tuner_cv = lgb_opt.LightGBMTunerCV(
        lgbm_params, lgb_train,
        num_boost_round=1000,
        early_stopping_rounds=100,
        verbose_eval=20,
        folds=folds,
    )
    
    # 最適なパラメータを探索する
    tuner_cv.run()

    # 最も良かったスコアとパラメータを書き出す
    print(f'Best score: {tuner_cv.best_score}')
    return  tuner_cv.best_params
    
#Y = new_train.target.copy()
#X = new_train.drop(["target"], axis=1).copy()
#random_state = 42
#lbl = preprocessing.LabelEncoder()
#lbl.fit(list(X["chip_id"]))
#X["chip_id"] = lbl.transform(list(X["chip_id"]))
#lgbm_optuna(X, Y)
