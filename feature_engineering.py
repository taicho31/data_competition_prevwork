import pandas as pd
import numpy as np
from tqdm import tqdm
from logging import getLogger
import feather
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.stats import norm, rankdata
import sys
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import category_encoders as ce
warnings.filterwarnings('ignore')

TRAIN = '../input/train.feather'
TEST = '../input/test.feather'

TRAIN_MOD = "../input/train_mod.feather"
TEST_MOD = "../input/test_mod.feather"

DIR = '../result/logfile'

logger = getLogger(__name__)

# reduce memory when creating modified data ----------------
#Based on this great kernel
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

log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)
    
handler = FileHandler(DIR + '_feature_engineering.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)
    
logger.info('start')
logger.info('Feature Engineering')

args = sys.argv
id_feature = args[1]
target_feature = args[2]
print("id_feature", id_feature)
print("target_feature", target_feature)

logger.info('install data')

train = feather.read_dataframe(TRAIN)
test = feather.read_dataframe(TEST)
logger.debug(train.head(5))
logger.debug(test.head(5))
logger.info('training data shape: {}'. format(train.shape))
logger.info('test data shape: {}'. format(test.shape))

# -------------------------------------------------------------------------------
features = [c for c in train.columns if c not in [id_feature, target_feature]]

logger.info('--------------feature engineering start-------------')
for df in [train, test]:#要約統計量
    df["max"] = df.iloc[:,2:].max(axis=1)
    df["min"] = df.iloc[:,2:].min(axis=1)
    df["std"] = df.iloc[:,2:].std(axis=1)
    df["skew"] = df.iloc[:,2:].skew(axis=1)
    df["kurtosis"] = df.iloc[:,2:].kurtosis(axis=1)
    df["median"] = df.iloc[:,2:].median(axis=1)
    df["mean"] = df.iloc[:,2:].mean(axis=1)

features = [c for c in train.columns if not c in ["target", "ID_code", "max", "min", "skew", "kurtosis", "median", "std", "mean"]  and c != id_feature and c != target_feature]

for df in [train, test]:
    for feature in features:
        df[feature+'^2'] = df[feature] * df[feature]
        #df[feature+'^3'] = df[feature] * df[feature] * df[feature]
        #df[feature+'^4'] = df[feature] * df[feature] * df[feature] * df[feature]
        #df[feature+'_cp'] = rankdata(df[feature]).astype('float32')
        #df['r2_'+feature] = np.round(df[feature], 2)
        #df['r1_'+feature] = np.round(df[feature], 1)

#for df in [train, test]:
#    if feature != "var_81":
#        df["diff_var_81_"+feature] = train[feature] - train["var_81"]

# category to variables
list_cols = list([c for c in train.columns if "object" in str(train[c].dtype)])
# 序数をカテゴリに付与して変換
ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')
train = ce_oe.fit_transform(train)
train[target_feature] = train[target_feature] - 1

logger.info('--------------feature engineering finish-------------')

logger.debug('training data shape after adding new features: {}'. format(train.shape))
logger.debug('test data shape after adding new features: {}'. format(test.shape))

train, NAlist = reduce_mem_usage(train)
logger.info('_________________')
logger.info('')
#logger.debug('Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ')
logger.info('_________________')
logger.info('')
logger.info(NAlist)

test, NAlist = reduce_mem_usage(test)
logger.info('_________________')
logger.info('')
#logger.debug('Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ')
logger.info('_________________')
logger.info('')
logger.info(NAlist)

logger.info('Writing modified training data to a feather files...')
feather.write_dataframe(train, TRAIN_MOD)

logger.info('Writing modified test data to a feather files...')
feather.write_dataframe(test, TEST_MOD)

logger.info('end')




