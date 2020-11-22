# https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
from collections import defaultdict
from collections import Counter

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

def modelling_sgk(new_train, new_test, lgbm_params): # stratify by target, group by session_id
    X_train = new_train.drop(['target'],axis=1).copy()
    y_train = new_train.target.copy()
    
    remove_features = []
    for i in X_train.columns:
        if (X_train[i].std() == 0) and i not in remove_features:
            remove_features.append(i)
    X_train = X_train.drop(remove_features, axis=1)
    X_test = new_test.copy()
    X_test = X_test.drop(remove_features, axis=1)
    groups = np.array(X_train.session_id.values)
    X_test = X_test.drop("session_id", axis=1)
    models = []
    
    n_folds = 5
    valid = np.array([])
    real = np.array([])
    evals_result = {}
    features_list = [i for i in X_train.columns if i != "session_id"]
    feature_importance_df = pd.DataFrame(features_list, columns=["Feature"])
    for i, (train_index, test_index) in enumerate(stratified_group_k_fold(X_train, y_train, groups, k=n_folds, seed=12)):
        print("Fold "+str(i+1))
        X_train2 = X_train.iloc[train_index,:]
        y_train2 = y_train.iloc[train_index]
        X_train2.drop("session_id", axis=1, inplace=True)

        X_test2 = X_train.iloc[test_index,:]
        y_test2 = y_train.iloc[test_index]
        X_test2.drop("session_id", axis=1, inplace=True)

        lgb_train = lgb.Dataset(X_train2, y_train2)
        lgb_eval = lgb.Dataset(X_test2, y_test2, reference=lgb_train)
        clf = lgb.train(lgbm_params, lgb_train,valid_sets=[lgb_train, lgb_eval],
            num_boost_round=10000,early_stopping_rounds=100,verbose_eval = 500, categorical_feature = categoricals)
        valid_predict = clf.predict(X_test2, num_iteration = clf.best_iteration)
        valid = np.concatenate([valid, valid_predict])
        real = np.concatenate([real, y_test2])
        feature_importance_df["Fold_"+str(i+1)] = clf.feature_importance()

        models.append(clf)
        
    feature_importance_df["Average"] = np.mean(feature_importance_df.iloc[:,1:n_folds+1], axis=1)
    feature_importance_df["Std"] = np.std(feature_importance_df.iloc[:,1:n_folds+1], axis=1)
    feature_importance_df["Cv"] = feature_importance_df["Std"] / feature_importance_df["Average"]

    roc = roc_auc_score(real, valid)
    print("ROC = {}".format(roc_auc_score(real, valid)))
    print(confusion_matrix(real, np.round(valid)))
    pred_value = np.zeros(X_test.shape[0])
    for model in models:
        pred_value += model.predict(X_test, num_iteration = model.best_iteration) / len(models)
    return roc, pred_value, feature_importance_df
    
roc_sgk1, pred_value, _ = modelling_sgk(new_train, new_test, lgbm_params)
