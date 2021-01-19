# null importance
# https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
def get_feature_importances(data, shuffle, target, features, seed=None):
    # Go over fold and keep track of CV score (train and valid) and feature importances
    
    # Shuffle target if required
    y = data[target].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data[target].copy().sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'n_jobs': 4
    }

    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[features]))
    
    return imp_df

def build_null_df(data, target, features):
    from tqdm import tqdm
    null_imp_df = pd.DataFrame()
    nb_runs = 80
    for i in tqdm(range(nb_runs)):
        # Get current run importances
        imp_df = get_feature_importances(data=data, target=target, features = features, shuffle=True)
        imp_df['run'] = i + 1 
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    return null_imp_df

def calculate_imp(actual_imp_df, null_imp_df):
    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
        gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
        feature_scores.append((_f, split_score, gain_score))

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
    return scores_df

def corr_calculate_imp(actual_imp_df, null_imp_df):
    correlation_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
        gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
        split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        correlation_scores.append((_f, split_score, gain_score))

    corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
    return corr_scores_df

def score_feature_selection(df=None, train_features=None, target=None):
    # Fit LightGBM 
    dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': .1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 13,
        'n_jobs': 4,
        'min_split_gain': .00001,
        'reg_alpha': .00001,
        'reg_lambda': .00001,
        'metric': 'auc'
    }
    
    # Fit the model
    hist = lgb.cv(
        params=lgb_params, 
        train_set=dtrain, 
        num_boost_round=2000,
        nfold=5,
        stratified=True,
        shuffle=True,
        early_stopping_rounds=50,
        verbose_eval=0,
        seed=17
    )
    # Return the last mean / std values 
    return hist['auc-mean'][-1], hist['auc-stdv'][-1]

# prepare for the data
np.random.seed(123)
# execute null importance
actual_imp_df = get_feature_importances(train, shuffle=False, target="Survived", features = feature, seed=123)
null_imp_df = build_null_df(data=train, target="Survived", features = feature)
scores_df = calculate_imp(actual_imp_df, null_imp_df)
corr_scores_df = corr_calculate_imp(actual_imp_df, null_imp_df)

for threshold in [0, 10, 20, 30 , 40, 50 ,60 , 70, 80 , 90, 95, 99]:
    split_feats = sorted(list(corr_scores_df[corr_scores_df.split_score >= threshold]["feature"]))
    #split_cat_feats = sorted(list(set(corr_scores_df[corr_scores_df.split_score >= threshold]["feature"]) & set(categorical_feats)))
    gain_feats = sorted(list(corr_scores_df[corr_scores_df.gain_score >= threshold]["feature"]) )
    #gain_cat_feats = sorted(list(set(corr_scores_df[corr_scores_df.gain_score >= threshold]["feature"]) & set(categorical_feats)))
                                                                                                 
    print('Results for threshold %3d' % threshold)
    split_results = score_feature_selection(df=train, train_features=split_feats, target=train[target])
    print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
    gain_results = score_feature_selection(df=train, train_features=gain_feats, target=train[target])
    print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))
