import pandas as pd
import feather
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from scipy.stats import norm, rankdata
warnings.filterwarnings('ignore')

#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)
                
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist

path_train_mod = '../input/train_mod.feather'
path_test_mod = '../input/test_mod.feather'

path_train_std = '../input/train_std.feather'
path_test_std = '../input/test_std.feather'

path_train_mod_std = '../input/train_mod_std.feather'
path_test_mod_std = '../input/test_mod_std.feather'

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

# -------------------------------------------------------------------------------
features = [c for c in train.columns if c not in ['ID_code', 'target']]

# standardized data ----------------------------------------------------------------- 
#print("Standardize original data")
#sc = StandardScaler()
#X_train_std = sc.fit_transform(X_train.values)
#X_test_std = sc.transform(X_test.values)

#X_train_std = train[features].copy()
#X_test_std = test[features].copy()

#for col in features:
#    X_train_std[col] = ((X_train_std[col] - X_train_std[col].mean()) 
#    / X_train_std[col].std()).astype('float32')

#    X_test_std[col] = ((X_test_std[col] - X_test_std[col].mean())
#    / X_test_std[col].std()).astype('float32')

#X_train_std = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
#X_test_std = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns)

#train_idcode_target = pd.DataFrame(train[["ID_code","target"]])
#train_std = pd.concat([train_idcode_target, X_train_std], axis=1)

#test_idcode = pd.DataFrame(test["ID_code"])
#test_std = pd.concat([test_idcode, X_test_std], axis=1)

#train_std.to_csv("../input/santander-customer-transaction-prediction/train_std.csv")
#test_std.to_csv("../input/santander-customer-transaction-prediction/test_std.csv")

#print("Writing standardized train to a feather files...")
#feather.write_dataframe(train_std, path_train_std)

#print("Writing standardized test to a feather files...")
#feather.write_dataframe(test_std, path_test_std)

print("--------------feature engineering-------------")
for df in [train, test]:#要約統計量
    df["max"] = df.iloc[:,2:].max(axis=1)
    df["min"] = df.iloc[:,2:].min(axis=1)
    df["std"] = df.iloc[:,2:].std(axis=1)
    df["skew"] = df.iloc[:,2:].skew(axis=1)
    df["kurtosis"] = df.iloc[:,2:].kurtosis(axis=1)
    df["median"] = df.iloc[:,2:].median(axis=1)
    df["mean"] = df.iloc[:,2:].mean(axis=1)

features = [c for c in train.columns if not c in ["target", "ID_code", "max", "min", "skew", "kurtosis", "median", "std", "mean"]]
for feature in features:
    train[feature+'^2'] = train[feature] * train[feature]
    train[feature+'^3'] = train[feature] * train[feature] * train[feature]
    train[feature+'^4'] = train[feature] * train[feature] * train[feature] * train[feature]
    test[feature+'^2'] = test[feature] * test[feature]
    test[feature+'^3'] = test[feature] * test[feature] * test[feature]
    test[feature+'^4'] = test[feature] * test[feature] * test[feature] * test[feature]
    train[feature+'_cp'] = rankdata(train[feature]).astype('float32')	
    test[feature+'_cp'] = rankdata(test[feature]).astype('float32')
    train['r2_'+feature] = np.round(train[feature], 2)
    test['r2_'+feature] = np.round(test[feature], 2)
    train['r1_'+feature] = np.round(train[feature], 1)
    test['r1_'+feature] = np.round(test[feature], 1) 
    #train["posneg_"+feature] = np.sign(train[feature])
    #test["posneg_"+feature] = np.sign(test[feature])
    if feature != "var_81":
        train["diff_var_81_"+feature] = train[feature] - train["var_81"]
        test["diff_var_81_"+feature] = test[feature] - test["var_81"]

#for i in range(200):
#    mean = np.mean(train["var_"+str(i)])
#    std = np.std(train["var_"+str(i)])
#    train['var_'+str(i)+"_outlier"]=train['var_'+str(i)].apply(lambda x: 0 if ( abs(x - mean) / std ) <=3 else (1))

#for i in range(200):
#    mean = np.mean(test["var_"+str(i)])
#    std = np.std(test["var_"+str(i)])
#    test['var_'+str(i)+"_outlier"]=test['var_'+str(i)].apply(lambda x: 0 if ( abs(x - mean) / std ) <=3 else (1))

print("train shape: {}". format(train.shape))
print("test shape: {}". format(test.shape))

#train, NAlist = reduce_mem_usage(train)
#print("_________________")
#print("")
#print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
#print("_________________")
#print("")
#print(NAlist)

#test, NAlist = reduce_mem_usage(test)
#print("_________________")
#print("")
#print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
#print("_________________")
#print("")
#print(NAlist) 

#train.to_csv("../input/santander-customer-transaction-prediction/train_mod.csv")  #2nd ver 2019.3.22
#test.to_csv("../input/santander-customer-transaction-prediction/test_mod.csv")    #2nd ver 2019.3.22

print("Writing modified train to a feather files...")
feather.write_dataframe(train, path_train_mod)

print("Writing modified test to a feather files...")
feather.write_dataframe(test, path_test_mod)

# modified standardized data ----------------------------------------------------------------- 
features = [c for c in train.columns if c not in ['ID_code', 'target']]

print("Standardize modified data")

#sc = StandardScaler()
#X_train_std = sc.fit_transform(X_train.values)
#X_test_std = sc.transform(X_test.values)

X_train_mod_std = train[features].copy()
X_test_mod_std = test[features].copy()

for col in features:
    X_train_mod_std[col] = ((X_train_mod_std[col] - X_train_mod_std[col].mean())
    / X_train_mod_std[col].std()).astype('float32')

    X_test_mod_std[col] = ((X_test_mod_std[col] - X_test_mod_std[col].mean())
    / X_test_mod_std[col].std()).astype('float32')

#X_train_std = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
#X_test_std = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns)

train_idcode_target = pd.DataFrame(train[["ID_code","target"]])
train_mod_std = pd.concat([train_idcode_target, X_train_mod_std], axis=1)

test_idcode = pd.DataFrame(test["ID_code"])
test_mod_std = pd.concat([test_idcode, X_test_mod_std], axis=1)

#train_std.to_csv("../input/santander-customer-transaction-prediction/train_mod_std.csv")
#test_std.to_csv("../input/santander-customer-transaction-prediction/test_mod_std.csv")

print("Writing modified standardized train to a feather files...")
feather.write_dataframe(train_mod_std, path_train_mod_std)

print("Writing modified standardized test to a feather files...")
feather.write_dataframe(test_mod_std, path_test_mod_std)

# undersampling and SMOTE ---------------------------------------------------
#positive_count_train = train["target"].sum()
#rus = RandomUnderSampler(ratio={0:positive_count_train*8, 1:positive_count_train}, random_state=0)

#features = [c for c in train.columns if c not in ['target']]
#target = train['target']
#X_train = train[features]
#X_train_undersampled, target_undersampled = rus.fit_sample(X_train, target)
#print('target_undersample:\n{}'.format(pd.Series(target_undersampled).value_counts()))

#smote = SMOTEENN(ratio={0:positive_count_train*99, 1:positive_count_train*10}, random_state=0)
#X_train_resampled, target_resampled = smote.fit_sample(X_train_undersampled, target_undersampled)
#print('target_resample:\n{}'.format(pd.Series(target_resampled).value_counts()))

