import pandas as pd
import numpy as np
from tqdm import tqdm
from logging import getLogger
import feather
import warnings
import sys
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
warnings.filterwarnings('ignore')

TRAIN_DATA = '../input/santander-customer-transaction-prediction/train.csv'
TEST_DATA = '../input/santander-customer-transaction-prediction/test.csv'

TRAIN_PATH = '../input/train.feather'
TEST_PATH = '../input/test.feather'

DIR = '../result/logfile'

logger = getLogger(__name__)

# reduce memory when creating modified data ----------------
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
    
handler = FileHandler(DIR + '_memory_reduce.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)
    
logger.info('start')
logger.info('Memory Reduce')

args = sys.argv
id_feature = args[1]
target_feature = args[2]
print("id_feature", id_feature)
print("target_feature", target_feature)
print(args)

logger.info('install data')

train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
logger.debug(train.head(5))
logger.debug(test.head(5))
logger.info('training data shape: {}'. format(train.shape))
logger.info('test data shape: {}'. format(test.shape))

# -------------------------------------------------------------------------------
features = [c for c in train.columns if c not in [id_feature, target_feature]]

logger.info('--------------memory reduce start-------------')
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

logger.info('Writing training data to a feather files...')
feather.write_dataframe(train, TRAIN_PATH)

logger.info('Writing test data to a feather files...')
feather.write_dataframe(test, TEST_PATH)

logger.info('end')

