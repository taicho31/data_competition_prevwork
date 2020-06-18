from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

def permuted(df):
    for column_name in df.columns:
        permuted_df = df.copy()
        permuted_df[column_name] = np.random.permutation(permuted_df[column_name])
        yield column_name, permuted_df


def pimp(clf, X, y, cv=None, eval_func=roc_auc_score):
    base_scores = []
    permuted_scores = defaultdict(list)

    if cv is None:
        #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv = GroupKFold(n_splits=5)
        
    for train_index, test_index in cv.split(X, y, X["installation_id"]):
        # 学習用データと検証用データに分割する
        X_train2, y_train2 = X.iloc[train_index], y.iloc[train_index]
        X_test2, y_test2 = X.iloc[test_index], y.iloc[test_index]

        # 学習用データでモデルを学習する
        clf.fit(X_train2, y_train2)

        # まずは何もシャッフルしていないときのスコアを計算する
        y_pred_base = clf.predict(X_test2)
        base_score = eval_func(y_test2, y_pred_base)
        base_scores.append(base_score)

        # 特定のカラムをシャッフルした状態で推論したときのスコアを計算する
        permuted_X_test_gen = permuted(X_test2)
        for column_name, permuted_X_test in permuted_X_test_gen:
            y_pred_permuted = clf.predict(permuted_X_test)
            permuted_score = eval_func(y_test2, y_pred_permuted)
            permuted_scores[column_name].append(permuted_score)

    # 基本のスコアとシャッフルしたときのスコアを返す
    np_base_score = np.array(base_scores)
    dict_permuted_score = {name: np.array(scores) for name, scores in permuted_scores.items()}
    return np_base_score, dict_permuted_score

def score_difference_statistics(base, permuted):
    mean_base_score = base.mean()
    for column_name, scores in permuted.items():
        score_differences = scores - mean_base_score
        yield column_name, score_differences.mean(), score_differences.std()

# prepare for the data ---
X_train = new_train.drop(['accuracy_group'],axis=1) 
y_train = new_train.accuracy_group.copy()
y_train.loc[y_train <=1] = 0
y_train.loc[y_train >=2] = 1
lbl = preprocessing.LabelEncoder()
lbl.fit(list(X_train["installation_id"]))
X_train["installation_id"] = lbl.transform(list(X_train["installation_id"]))
remove_features = [i for i in X_train.columns if i in to_exclude]
for i in X_train.columns:
    if X_train[i].std() == 0 and i not in remove_features:
        remove_features.append(i)
X_train = X_train.drop(remove_features, axis=1)
X_train = X_train[sorted(X_train.columns.tolist())]  

# execute permutation importance 
clf = RandomForestClassifier(n_estimators=100)
base_score, permuted_scores = pimp(clf, X_train, y_train)
diff_stats = list(score_difference_statistics(base_score, permuted_scores))
pimp_df = pd.DataFrame(diff_stats, columns = ["feature", "score_mean", "score_std"])
pimp_df = pimp_df.sort_values("score_mean", ascending=True).reset_index(drop=True)
pimp_df.head(10)
