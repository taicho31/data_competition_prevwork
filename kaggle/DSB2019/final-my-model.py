#!/usr/bin/env python
# coding: utf-8

# - final modification

# In[1]:


import pandas as pd
import numpy as np
import warnings
import datetime
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
import lightgbm as lgb
from functools import partial
import json
import copy
import time
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from hyperopt import hp, tpe, Trials, fmin, space_eval
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows",1000)
np.set_printoptions(precision=8)
warnings.filterwarnings("ignore")
import random


# In[2]:


def qwk(a1, a2):
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)
    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))
    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)
    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)
    e = e / a1.shape[0]
    return np.round(1 - o / e, 8)


# In[3]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])
        return -qwk(y, X_p)
        #return -mod_qwk(y, X_p, weights=weights)
    
    def fit(self, X, y, random_flg = False):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        if random_flg:
            initial_coef = [np.random.uniform(0.4,0.5), np.random.uniform(0.5,0.6), np.random.uniform(0.6,0.7)]
        else:
            initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead') #Powell
        
    def predict(self, X, coef):
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

    def coefficients(self):
        return self.coef_['x']


# In[4]:


def eval_qwk_lgb_regr(y_pred, train_t):
    dist = Counter(train_t['accuracy_group'])
    for k in dist:
        dist[k] /= len(train_t)
    
    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_pred, acum * 100)

    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    y_pred = np.array(list(map(classify, y_pred)))
    
    return y_pred


# # install

# In[5]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/data-science-bowl-2019/train.csv')\ntrain_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')\ntest = pd.read_csv('../input/data-science-bowl-2019/test.csv')\n#specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')\nsample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')")


# # Preprocess and Feature engineering

# In[6]:


get_ipython().run_cell_magic('time', '', 'def encode_title(train, test):\n    train[\'title_event_code\'] = list(map(lambda x, y: str(x) + \'_\' + str(y), train[\'title\'], train[\'event_code\']))\n    test[\'title_event_code\'] = list(map(lambda x, y: str(x) + \'_\' + str(y), test[\'title\'], test[\'event_code\']))\n    list_of_title_eventcode = sorted(list(set(train[\'title_event_code\'].unique()).union(set(test[\'title_event_code\'].unique()))))\n    \n    train[\'type_world\'] = list(map(lambda x, y: str(x) + \'_\' + str(y), train[\'type\'], train[\'world\']))\n    test[\'type_world\'] = list(map(lambda x, y: str(x) + \'_\' + str(y), test[\'type\'], test[\'world\']))\n    list_of_type_world = sorted(list(set(train[\'type_world\'].unique()).union(set(test[\'type_world\'].unique()))))\n    \n    list_of_user_activities = sorted(list(set(train[\'title\'].unique()).union(set(test[\'title\'].unique()))))\n    list_of_event_code = sorted(list(set(train[\'event_code\'].unique()).union(set(test[\'event_code\'].unique()))))\n    list_of_worlds = sorted(list(set(train[\'world\'].unique()).union(set(test[\'world\'].unique()))))\n    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))\n    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))\n    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))\n    assess_titles = sorted(list(set(train[train[\'type\'] == \'Assessment\'][\'title\'].value_counts().index).union(set(test[test[\'type\'] == \'Assessment\'][\'title\'].value_counts().index))))\n\n    train[\'title\'] = train[\'title\'].map(activities_map)\n    test[\'title\'] = test[\'title\'].map(activities_map)\n    train[\'world\'] = train[\'world\'].map(activities_world)\n    test[\'world\'] = test[\'world\'].map(activities_world)\n\n    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype(\'int\')))\n    win_code[activities_map[\'Bird Measurer (Assessment)\']] = 4110\n    \n    train[\'timestamp\'] = pd.to_datetime(train[\'timestamp\'])\n    test[\'timestamp\'] = pd.to_datetime(test[\'timestamp\'])\n    \n    train["misses"] = train["event_data"].apply(lambda x: json.loads(x)["misses"] if "\\"misses\\"" in x else np.nan)\n    test["misses"] = test["event_data"].apply(lambda x: json.loads(x)["misses"] if "\\"misses\\"" in x else np.nan)\n        \n    train["true"] = train["event_data"].apply(lambda x: 1 if "true" in x and "correct" in x else 0)\n    test["true"] = test["event_data"].apply(lambda x: 1 if "true" in x and "correct" in x else 0)\n\n    train["false"] = train["event_data"].apply(lambda x: 1 if "false" in x and "correct" in x else 0)\n    test["false"] = test["event_data"].apply(lambda x: 1 if "false" in x and "correct" in x else 0)\n    \n    train["game_complete"] = train["event_data"].apply(lambda x: 1 if "game_completed" in x else 0)\n    test["game_complete"] = test["event_data"].apply(lambda x: 1 if "game_completed" in x else 0)\n    \n    #train["level"] = train["event_data"].apply(lambda x: json.loads(x)["level"] if "\\"level\\"" in x else np.nan)\n    #test["level"] = test["event_data"].apply(lambda x: json.loads(x)["level"] if "\\"level\\"" in x else np.nan)\n    \n    #train["round"] = train["event_data"].apply(lambda x: json.loads(x)["round"] if "\\"round\\"" in x else np.nan)\n    #test["round"] = test["event_data"].apply(lambda x: json.loads(x)["round"] if "\\"round\\"" in x else np.nan)\n               \n    return train, test, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, activities_world, list_of_title_eventcode, list_of_type_world\n\ntrain, test, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, activities_world, list_of_title_eventcode, list_of_type_world = encode_title(train, test)')


# In[7]:


def get_data(user_sample, test_set=False):
    last_activity = 0
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    title_eventcode_count = {str(ele): 0 for ele in list_of_title_eventcode}
    user_world_count = {"world_"+str(wor) : 0 for wor in activities_world.values()}
    event_code_count = {str(ev): 0 for ev in list_of_event_code}
    title_count = {actv: 0 for actv in list_of_user_activities}
    type_world_count = {str(ev): 0 for ev in list_of_type_world}
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    last_game_time_title = {'lgt_' + title: 0 for title in assess_titles}
    ac_game_time_title = {'agt_' + title: 0 for title in assess_titles}
    ac_true_attempts_title = {'ata_' + title: 0 for title in assess_titles}
    ac_false_attempts_title = {'afa_' + title: 0 for title in assess_titles}
    
    all_assessments = []
    accuracy_groups = {"0":0, "1":0, "2":0, "3":0}
    accumulated_accuracy_group = 0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0 
    accumulated_actions = 0
    counter = 0
    time_first_activity = user_sample.iloc[0]['timestamp']
    miss = 0
    crys_game_true = 0; crys_game_false = 0
    tree_game_true = 0; tree_game_false = 0
    magma_game_true = 0; magma_game_false = 0
    crys_game_acc = []; tree_game_acc = []; magma_game_acc = []
    durations = []
    prev_assess_title = -999
    assess_count = 1
    last_accuracy = -999
    prev_assess_start = -999; prev_assess_end = -999
    real_prev_assess_start = -999; real_prev_assess_end = -999
    real_assess_start = -999; real_assess_end = -999
    complete_games = 0
    no_result_count = 0
    crys_game_level = np.array([]); tree_game_level = np.array([]); magma_game_level = np.array([])
    crys_game_round = np.array([]); tree_game_round = np.array([]); magma_game_round = np.array([])
    
    for i, session in user_sample.groupby('game_session', sort=False):      
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
        session_world = session["world"].iloc[0]
        game_session = session["game_session"].iloc[0]
        
        if session_type != 'Assessment':
            if session_type == "Game":
                true = session['true'].sum()
                false = session['false'].sum() 
                if session_world == activities_world["CRYSTALCAVES"]:
                    crys_game_true += true
                    crys_game_false += false
                    crys_game_acc.append(true / (true + false) if (true + false) != 0 else 0)
                    #crys_game_level = np.concatenate([crys_game_level, session["level"]], axis=0)
                    #crys_game_round = np.concatenate([crys_game_round, session["round"]], axis=0)
                elif session_world == activities_world["TREETOPCITY"]:
                    tree_game_true += true
                    tree_game_false += false
                    tree_game_acc.append(true / (true + false) if (true + false) != 0 else 0)
                    #tree_game_level = np.concatenate([tree_game_level, session["level"]], axis=0)
                    #tree_game_round = np.concatenate([tree_game_round, session["round"]], axis=0)
                elif session_world == activities_world["MAGMAPEAK"]:
                    magma_game_true += true
                    magma_game_false += false
                    magma_game_acc.append(true / (true + false) if (true + false) != 0 else 0)
                    #magma_game_level = np.concatenate([magma_game_level, session["level"]], axis=0)
                    #magma_game_round = np.concatenate([magma_game_round, session["round"]], axis=0)
                else:
                    pass
                
        if (session_type == 'Assessment') & (test_set or len(session)>1): 
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            true_attempts = all_attempts['event_data'].str.contains('true').sum() # true in target assess
            false_attempts = all_attempts['event_data'].str.contains('false').sum() # false in target assessment
            assess_start = session.iloc[0,2]
            assess_end = session.iloc[-1,2]
            
            # from start of installation_id to the start of target assessment ------------------------
            features = user_activities_count.copy() # appearance of each type without duplicates
            features.update(title_eventcode_count.copy()) # apperance of combi of title and event_code
            features.update(user_world_count.copy()) # appearance of world with duplicates
            features.update(event_code_count.copy())
            features.update(title_count.copy())
            features.update(type_world_count.copy())
            features.update(last_accuracy_title.copy())
            features.update(last_game_time_title.copy())
            features.update(ac_game_time_title.copy())
            features.update(ac_true_attempts_title.copy())
            features.update(ac_false_attempts_title.copy())
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            ac_true_attempts_title['ata_' + session_title_text] += true_attempts
            ac_false_attempts_title['afa_' + session_title_text] += false_attempts
            last_game_time_title['lgt_' + session_title_text] = session['game_time'].iloc[-1]
            ac_game_time_title['agt_' + session_title_text] += session['game_time'].iloc[-1]
            features["misses"] = miss
            features['accumulated_actions'] = accumulated_actions
            features["no_complete_game"] = complete_games
            features["no_result_count"] = no_result_count 
            
            if true_attempts + false_attempts == 0:
                no_result_count += 1
            else:
                real_assess_start = session.iloc[0,2]
                real_assess_end = session.iloc[-1,2]

            if session_world == activities_world["CRYSTALCAVES"]:
                features["game_true"] = crys_game_true
                features["game_false"] = crys_game_false
                features['game_accuracy'] = crys_game_true / (crys_game_true + crys_game_false) if (crys_game_true + crys_game_false) != 0 else 0
                features["game_accuracy_std"] = np.std(crys_game_acc) if len(crys_game_acc) >=1 else 0
                features["last_game_acc"] = crys_game_acc[-1] if len(crys_game_acc) >=1 else 0
                #features["hightest_level"] = np.nanmax(crys_game_level) if len(crys_game_level[~np.isnan(crys_game_level)]) >=1 else -1
                #features["level_count"] = len(crys_game_level[~np.isnan(crys_game_level)])
                #features["round_count"] = len(crys_game_round[~np.isnan(crys_game_round)])
            elif session_world == activities_world["TREETOPCITY"]:
                features["game_true"] = tree_game_true
                features["game_false"] = tree_game_false
                features['game_accuracy'] = tree_game_true / (tree_game_true + tree_game_false) if (tree_game_true + tree_game_false) != 0 else 0
                features["game_accuracy_std"] = np.std(tree_game_acc) if len(tree_game_acc) >=1 else 0
                features["last_game_acc"] = tree_game_acc[-1] if len(tree_game_acc) >=1 else 0
                #features["hightest_level"] = np.nanmax(tree_game_level) if len(tree_game_level[~np.isnan(tree_game_level)]) >=1 else -1
                #features["level_count"] = len(tree_game_level[~np.isnan(tree_game_level)])
                #features["round_count"] = len(tree_game_round[~np.isnan(tree_game_round)])
            elif session_world == activities_world["MAGMAPEAK"]:
                features["game_true"] = magma_game_true
                features["game_false"] = magma_game_false
                features['game_accuracy'] = magma_game_true / (magma_game_true + magma_game_false) if (magma_game_true + magma_game_false) != 0 else 0
                features["game_accuracy_std"] = np.std(magma_game_acc) if len(magma_game_acc) >=1 else 0
                features["last_game_acc"] = magma_game_acc[-1] if len(magma_game_acc) >=1 else 0
                #features["hightest_level"] = np.nanmax(magma_game_level) if len(magma_game_level[~np.isnan(magma_game_level)]) >=1 else -1
                #features["level_count"] = len(magma_game_level[~np.isnan(magma_game_level)])
                #features["round_count"] = len(magma_game_round[~np.isnan(magma_game_round)])
            
            features['installation_id'] = session['installation_id'].iloc[-1]
            features['session_title'] = session_title
            features['game_session'] = game_session
            features["prev_assess_title"] = prev_assess_title
            prev_assess_title = session_title
            features["first_assessment"] = 1 if assess_count == 1 else 0
            assess_count += 1
            features["time_from_start"] = (assess_start - time_first_activity).seconds

            if prev_assess_end == -999:
                features["time_bet_assess"] = -999
            else:
                features["time_bet_assess"] = (assess_start - prev_assess_end).seconds
            prev_assess_start = assess_start
            prev_assess_end = assess_end
            if real_prev_assess_end == -999:
                features["time_bet_real_assess"] = -999
            else:
                features["time_bet_real_assess"] = (real_assess_start - real_prev_assess_end).seconds
            real_prev_assess_start = real_assess_start
            real_prev_assess_end = real_assess_end
            
            if durations == []: #span of timestamp in target assessment
                features['duration_mean'] = 0
                features['duration_std'] = 0
                features['duration_max'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
                features['duration_std'] = np.std(durations)
                features['duration_max'] = np.max(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)
            
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            features['last_assess_acc'] = last_accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
            last_accuracy = accuracy
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[str(features['accuracy_group'])] += 1
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)
                
            counter += 1
            
        complete_games += np.sum(session["game_complete"])
        miss += np.sum(session["misses"])
        user_world_count["world_"+str(session_world)] += session.shape[0]
        
        n_of_type_world = Counter(session['type_world']) 
        for key in n_of_type_world.keys():
            type_world_count[str(key)] += n_of_type_world[key]
            
        n_of_title = Counter(session['title']) 
        for key in n_of_title.keys():
            title_count[activities_labels[key]] += n_of_title[key]
            
        n_of_eventcode = Counter(session['event_code']) 
        for key in n_of_eventcode.keys():
            event_code_count[str(key)] += n_of_eventcode[key]
                        
        n_of_title_eventcode = Counter(session['title_event_code']) 
        for key in n_of_title_eventcode.keys():
            title_eventcode_count[str(key)] += n_of_title_eventcode[key]
        
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type
    if test_set:
        return all_assessments[-1], all_assessments[:-1]
    return all_assessments


# In[8]:


def get_train_and_test(train, test):
    compiled_train = []
    compiled_test = []

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=train.installation_id.nunique(), desc='Installation_id', position=0):
        compiled_train += get_data(user_sample)
    del train
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=test.installation_id.nunique(), desc='Installation_id', position=0):
        test_data, val_data = get_data(user_sample, test_set=True)
        compiled_test.append(test_data)
    del test
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)

    categoricals = ['session_title']
    return reduce_train, reduce_test, categoricals
new_train, new_test, categoricals = get_train_and_test(train, test)


# In[9]:


tmp = new_train[new_train.Game==0].copy()
tmp = tmp[tmp.Activity == 0].copy()
tmp = tmp[tmp.Clip == 0].copy()
tmp = tmp[tmp.Assessment ==0].copy()
remove_train_index = tmp.index
new_train = new_train[~new_train.index.isin(remove_train_index)].copy()


# In[10]:


#tmp2 = new_test[new_test.Game==0].copy()
#tmp2 = tmp2[tmp2.Activity == 0].copy()
#tmp2 = tmp2[tmp2.Clip == 0].copy()
#tmp2 = tmp2[tmp2.Assessment ==0].copy()
#random_test_index = tmp2.index


# # Feature selection

# In[11]:


def exclude(reduce_train, reduce_test, features):
    to_exclude = [] 
    ajusted_test = reduce_test.copy()
    for feature in features:
        if feature not in ['accuracy_group', 'installation_id', 'session_title', 'hightest_level']:
            data = reduce_train[feature]
            train_mean = data.mean()
            data = ajusted_test[feature] 
            test_mean = data.mean()
            try:
                ajust_factor = train_mean / test_mean
                if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:
                    to_exclude.append(feature)
                    print(feature)
                else:
                    ajusted_test[feature] *= ajust_factor
            except:
                to_exclude.append(feature)
                print(feature)
    return to_exclude, ajusted_test
features = [i for i in new_train.columns if i not in ["game_session"]]
to_exclude, ajusted_test = exclude(new_train, new_test, features)


# # modelling and prediction

# In[12]:


def num_correct_calc(new_train, new_test):
    X_train = new_train.drop(['accuracy_group'],axis=1) 
    X_train = pd.merge(X_train, train_labels[["game_session", "num_correct", "num_incorrect"]], on ="game_session")
    y_train = X_train.num_correct.copy()
    X_train = X_train.drop(['game_session', "num_correct", "num_incorrect"],axis=1) 
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(X_train["installation_id"]))
    X_train["installation_id"] = lbl.transform(list(X_train["installation_id"]))
    remove_features = [i for i in X_train.columns if "_4235" in i or i == "world_"+str(activities_world["NONE"])
                      or i in to_exclude]
    for i in X_train.columns:
        if X_train[i].std() == 0 and i not in remove_features:
            remove_features.append(i)
    X_train = X_train.drop(remove_features, axis=1)
    X_train = X_train[sorted(X_train.columns.tolist())]

    X_test = new_test.drop(["installation_id","accuracy_group", "game_session"], axis=1)
    X_test = X_test.drop(remove_features, axis=1)
    X_test = X_test[sorted(X_test.columns.tolist())]

    n_folds=5
    skf=GroupKFold(n_splits = n_folds)
    models = []
    lgbm_params = {'objective': 'binary','eval_metric': 'auc','metric': 'auc', 'boosting_type': 'gbdt',
 'tree_learner': 'serial','bagging_fraction': 0.9605425291685099,'bagging_freq': 4,'colsample_bytree': 0.6784238046856443,
 'feature_fraction': 0.9792407844605087,'learning_rate': 0.017891320270412462,'max_depth': 7, 'random_seed':42,
 'min_data_in_leaf': 8,'min_sum_hessian_in_leaf': 17,'num_leaves': 17}
    X_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) if x not in ["installation_id", "session_title", "accuracy_group"] else x for x in X_train.columns]
    X_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) if x not in ["installation_id", "session_title", "accuracy_group"] else x for x in X_test.columns]

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
            
        models.append(clf)
        valid_correct_num.iloc[test_index] = test_predict.reshape(X_test2.shape[0], 1)
                
    print("logloss = \t {}".format(log_loss(y_train, valid_correct_num)))
    print("ROC = \t {}".format(roc_auc_score(y_train, valid_correct_num)))
    print('Accuracy score = \t {}'.format(accuracy_score(y_train, np.round(valid_correct_num))))
    print('Precision score = \t {}'.format(precision_score(y_train, np.round(valid_correct_num))))
    print('Recall score =   \t {}'.format(recall_score(y_train, np.round(valid_correct_num))))
    print('F1 score =      \t {}'.format(f1_score(y_train, np.round(valid_correct_num))))
    print(confusion_matrix(y_train, np.round(valid_correct_num)))
    pred_value = np.zeros([X_test.shape[0]])
    for model in models:
        pred_value += model.predict(X_test, num_iteration = model.best_iteration) / len(models)
    return pred_value, valid_correct_num
pred_value, valid_correct_num = num_correct_calc(new_train, new_test)


# In[13]:


def num_incorrect_calc(new_train, new_test): 
    X_train = new_train.drop(['accuracy_group'],axis=1) 
    X_train = pd.merge(X_train, train_labels[["game_session", "num_correct", "num_incorrect"]], on ="game_session")
    y_train = X_train.num_incorrect.copy()
    y_train.loc[y_train >=2] = 2
    X_train = X_train.drop(['game_session', "num_correct", "num_incorrect"],axis=1) 
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(X_train["installation_id"]))
    X_train["installation_id"] = lbl.transform(list(X_train["installation_id"]))
    remove_features = [i for i in X_train.columns if "_4235" in i or i == "world_"+str(activities_world["NONE"]) 
                       or i in to_exclude]
    for i in X_train.columns:
        if X_train[i].std() == 0 and i not in remove_features:
            remove_features.append(i)
    X_train = X_train.drop(remove_features, axis=1)
    X_train = X_train[sorted(X_train.columns.tolist())]

    X_test = new_test.drop(["installation_id","accuracy_group", "game_session"], axis=1)
    X_test = X_test.drop(remove_features, axis=1)
    X_test = X_test[sorted(X_test.columns.tolist())]

    n_folds=5
    skf=GroupKFold(n_splits = n_folds)
    models = []
    lgbm_params = {'boosting_type': 'gbdt', 'metric': 'rmse', 'objective': 'regression', 'eval_metric': 'cappa',
                 'tree_learner': 'serial', 'bagging_fraction': 0.5644726008433546, 'bagging_freq': 5, 'seed':42,
                    'colsample_bytree': 0.5385657738791669,
                     'feature_fraction': 0.8415810198341436,
                     'learning_rate': 0.007905671344935965,
                     'max_depth': 25,
                     'min_data_in_leaf': 99,
                     'min_sum_hessian_in_leaf': 7,
                     'num_leaves': 28}
    
    X_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) if x not in ["installation_id", "session_title", "accuracy_group"] else x for x in X_train.columns]
    X_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) if x not in ["installation_id", "session_title", "accuracy_group"] else x for x in X_test.columns]
    valid_incorrect_num = pd.DataFrame(np.zeros([X_train.shape[0]]))
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
            num_boost_round=10000,early_stopping_rounds=100,verbose_eval = 500)
        train_predict = clf.predict(X_train2, num_iteration = clf.best_iteration)
        test_predict = clf.predict(X_test2, num_iteration = clf.best_iteration)
            
        models.append(clf)
        valid_incorrect_num.iloc[test_index] = test_predict.reshape(X_test2.shape[0], 1)
            
    print("RMSE = \t {}".format(np.sqrt(mean_squared_error(y_train, valid_incorrect_num))))    
    pred_value_incorrect = np.zeros([X_test.shape[0]])
    for model in models:
        pred_value_incorrect += model.predict(X_test, num_iteration = model.best_iteration) / len(models)
    return pred_value_incorrect, valid_incorrect_num
pred_value_incorrect, valid_incorrect_num  = num_incorrect_calc(new_train, new_test)


# In[14]:


train_exp_accuracy = valid_correct_num / (valid_correct_num + valid_incorrect_num)
test_exp_accuracy = pred_value / (pred_value + pred_value_incorrect)
best_score = 0
for i in range(20):
    optR = OptimizedRounder()
    optR.fit(np.array(train_exp_accuracy).reshape(-1,), new_train.accuracy_group, random_flg=True)
    coefficients = optR.coefficients()
    final_valid_pred = optR.predict(np.array(train_exp_accuracy).reshape(-1,), coefficients)
    score = qwk(new_train.accuracy_group, final_valid_pred)
    print(i, np.sort(coefficients), score)
    if score > best_score:
        best_score = score
        best_coefficients = coefficients
final_test_pred = pd.cut(np.array(test_exp_accuracy).reshape(-1,), [-np.inf] + list(np.sort(best_coefficients)) + [np.inf], labels = [0, 1, 2, 3])
#final_test_pred = pd.cut(np.array(test_exp_accuracy).reshape(-1,), [-np.inf] + list(np.sort([0.30278718, 0.42545025, 0.56396292])) + [np.inf], labels = [0, 1, 2, 3])
#final_test_pred = eval_qwk_lgb_regr(np.array(test_exp_accuracy).reshape(-1,), new_train)

sample_submission["accuracy_group"] = final_test_pred.astype(int)
sample_submission.to_csv('submission.csv', index=False)
sample_submission["accuracy_group"].value_counts(normalize = True)


# In[ ]:




