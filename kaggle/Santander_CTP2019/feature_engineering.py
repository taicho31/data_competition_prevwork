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

logger.info('--------------feature engineering start-------------')

logger.info('-------------Feature removal----------------')
std_zero_feature = []
for i in [i for i in df.columns if i != "Date of Found"]:
    if df[i].std() == 0:
        std_zero_feature.append(i)

orig_input_features = [i for i in df.columns if i != target_feature]
input_features = [i for i in df.columns if i != target_feature and i not in std_zero_feature]

correlations = df[input_features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
corr_columns = ["level_0", "level_1", "value"]
correlations.columns = corr_columns
correlations = correlations.sort_values("value", ascending=False)

high_corr = correlations[correlations["value"]>=0.9]
remove_index = []
for i in range(high_corr.shape[0]-1):
    for j in range(i+1, high_corr.shape[0]):
        if high_corr.iloc[i]["level_0"] == high_corr.iloc[j]["level_1"] and high_corr.iloc[i]["level_1"] == high_corr.iloc[j]["level_0"]:
            remove_index.append(j)
            break
select_index = [i for i in range(high_corr.shape[0]) if i not in remove_index]
high_corr_no_duplicate = high_corr.iloc[select_index].reset_index()

high_corr_features = []
for i in range(high_corr_no_duplicate.shape[0]):
    if high_corr_no_duplicate.iloc[i]["level_0"] not in high_corr_features and high_corr_no_duplicate.iloc[i]["level_1"] not in high_corr_features:
        high_corr_features.append(high_corr_no_duplicate.iloc[i]["level_0"])
    elif high_corr_no_duplicate.iloc[i]["level_0"] in high_corr_features and high_corr_no_duplicate.iloc[i]["level_1"] not in high_corr_features:
        high_corr_features.append(high_corr_no_duplicate.iloc[i]["level_1"])
    elif high_corr_no_duplicate.iloc[i]["level_0"] not in high_corr_features and high_corr_no_duplicate.iloc[i]["level_1"] in high_corr_features:
        high_corr_features.append(high_corr_no_duplicate.iloc[i]["level_0"])

logger.info('--------------label encoding-------------')
un_test = ["Test Phase_製品審査", "Test Phase_Pana工場", "Test Phase_Honda工場", "Test Phase_From Other Models"]
# Test Phaseごとの集約値が効くのではないかと考え、カテゴリ値を集約しなおす
test_phase = [i for i in df.columns if "Test Phase_" in i]
for i in range(df.shape[0]):
    for j in range(len(test_phase)):
        if df.loc[i,test_phase[j]] ==1:
            df.loc[i,"Test_phase"] = test_phase[j]
            break

un_lead = ["Lead_Factory", "Lead_OtherSupp_NaviApp", "Lead_OtherDevice", "Lead_OtherSupp_ATT", "Lead_WG_HW-SW", "Lead_Knotty",
           "Lead_Peaks", "Lead_IOP"]
lead_component_variable = [i for i in df.columns if "Lead_" in i]
for i in range(df.shape[0]):
    for j in range(len(lead_component_variable)):
        if df.loc[i,lead_component_variable[j]] ==1:
            df.loc[i,"lead_component"] = lead_component_variable[j]
            break

un_hard = ["Hardware Version_1.5S", "Hardware Version_外部AMP", "Hardware Version_PP","Hardware Version_内蔵HD",
           "Hardware Version_内蔵AMP"]
hardware_variable = [i for i in df.columns if "Hardware" in i]
for i in range(df.shape[0]):
    for j in range(len(hardware_variable)):
        if df.loc[i,hardware_variable[j]] ==1:
            df.loc[i,"hardware_version"] = hardware_variable[j]
            break

defect_adding_variable = [i for i in df.columns if "Defect Adding" in i]
for i in range(df.shape[0]):
    for j in range(len(defect_adding_variable)):
        if df.loc[i,defect_adding_variable[j]] ==1:
            df.loc[i,"defect_adding"] = defect_adding_variable[j]
            break

repeatability_variable = [i for i in df.columns if "Repeatability" in i]
for i in range(df.shape[0]):
    for j in range(len(repeatability_variable)):
        if df.loc[i,repeatability_variable[j]] ==1:
            df.loc[i,"repeatability"] = repeatability_variable[j]
            break

# category to variables
list_cols = list([c for c in train.columns if "object" in str(train[c].dtype)])
# 序数をカテゴリに付与して変換
ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')
train = ce_oe.fit_transform(train)
train[target_feature] = train[target_feature] - 1

logger.info('-------------Feature addition----------------')
# 文章中のキーワードの出現回数
count_adb = []
count_ipod = []
count_ime = []
count_androidauto = []

for i in range(defect.shape[0]):
    mecab = MeCab.Tagger()
    parse = mecab.parse(defect.iloc[i]["Summary"])
    lines = parse.split('\n')
    items = (re.split('[\t,]', line) for line in lines)
    
    # 名詞をリストに格納
    words = [item[0] for item in items if (item[0] not in ('EOS', '', 't', 'ー') and item[1] == '名詞' and item[2] == '一般')]
    counter = Counter(words)
    
    count_adb.append(counter["ADB"]+counter["adb"])
    count_ipod.append(counter["ipod"]+counter["iPod"])
    count_ime.append(counter["ime"]+counter["IME"])
    count_androidauto.append(counter["AndroidAuto"])

new_df["summary_adb"] = count_adb
new_df["summart_ipod"] = count_ipod
new_df["summary_ime"] = count_ime
new_df["summary_androidauto"] = count_androidauto

day_feat = ["Date of Found", "Year", "Month", "Day", "Weekday"]
tmp_df = pd.concat([df, new_df], axis=1)
useless_features = [i for i in tmp_df.columns if i in high_corr_features or i in klog_features or i in file_features
                    or i in std_zero_feature or i in test_phase or i in lead_component_variable or i in hardware_variable or
                    i in defect_adding_variable or i in repeatability_variable]

# zero count
new_df_train = pd.DataFrame()
new_df_train["count_zero"] = len(input_features) - np.count_nonzero(train[input_features], axis=1)

# 基本統計量
new_df_train["max"] = train.iloc[:,2:].max(axis=1)
new_df_train["min"] = train.iloc[:,2:].min(axis=1)
new_df_train["std"] = train.iloc[:,2:].std(axis=1)
new_df_train["skew"] = train.iloc[:,2:].skew(axis=1)
new_df_train["kurtosis"] = train.iloc[:,2:].kurtosis(axis=1)
new_df_train["median"] = train.iloc[:,2:].median(axis=1)
new_df_train["mean"] = train.iloc[:,2:].mean(axis=1)

new_df_test = pd.DataFrame()
new_df_test["count_zero"] = len(input_features) - np.count_nonzero(test[input_features], axis=1)
new_df_test["max"] = test.iloc[:,2:].max(axis=1)
new_df_test["min"] = test.iloc[:,2:].min(axis=1)
new_df_test["std"] = test.iloc[:,2:].std(axis=1)
new_df_test["skew"] = test.iloc[:,2:].skew(axis=1)
new_df_test["kurtosis"] = test.iloc[:,2:].kurtosis(axis=1)
new_df_test["median"] = test.iloc[:,2:].median(axis=1)
new_df_test["mean"] = test.iloc[:,2:].mean(axis=1)

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

# 次元圧縮系----------------------------------------------------
#for i in range(1,20):
#    pca = PCA(n_components=i)
#    pca.fit(df[input_features])
#    if sum(pca.explained_variance_ratio_) >= 0.95:
#        break

#best = i
#print("best PCA components: ", best)
#pca_best = PCA(n_components=best)
#X_pca = pca_best.fit_transform(df[input_features])
#X_pca = pd.DataFrame(X_pca, columns=["pca_"+str(i) for i in range(1,best+1)])

#new_df = pd.concat([new_df, X_pca], axis=1)

#for i in range(1,20): # klogを集約する
#    pca = PCA(n_components=i)
#    pca.fit(df[klog_var])
#    if sum(pca.explained_variance_ratio_) >= 0.95:
#        break

#best = i
#print("best PCA components: ", best)
#pca_best_klog = PCA(n_components=best)
#X_pca_klog = pca_best_klog.fit_transform(df[klog_var])
#X_pca_klog = pd.DataFrame(X_pca_klog, columns=["pca_klog"+str(i) for i in range(1,best+1)])

# rtcを集約する
#for i in range(1,20):
#    pca = PCA(n_components=i)
#    pca.fit(df[rtc_features])
#    if sum(pca.explained_variance_ratio_) >= 0.95:
#        break

#best = i
#print("rtc best PCA components: ", best)
#pca_best_rtc = PCA(n_components=best)
#X_pca_rtc = pca_best_rtc.fit_transform(df[rtc_features])
#X_pca_rtc = pd.DataFrame(X_pca_rtc, columns=["pca_rtc"+str(i) for i in range(1,best+1)])

#new_df = pd.concat([new_df, X_pca_rtc], axis=1)

#new_df = pd.concat([new_df, X_pca_klog], axis=1)

# 集約------------------------------------------------------------
# new_df["null_no"] = df.isnull().sum(axis=1)
# new_df["non_zero"] = np.count_nonzero(df[input_features], axis=1)
# summary_df = df.groupby("lead_component").mean().reset_index()
# tp_summary_df = df.groupby("Test_phase").mean().reset_index()
# hard_df = df.groupby("hardware_version").mean().reset_index()
# defect_df = df.groupby("defect_adding").mean().reset_index()
# new_df["overall_len"] = df["Summary Len"] + df["Bug Description Len"] + df["Reappearance Procedure Len"] + df["Description Len"]
#for i in range(df.shape[0]): #　ここまでは組み込むことで性能が改善することを確認
#    new_df.loc[i,"Summary Len Ave"] = summary_df[summary_df["lead_component"] == df.loc[i,"lead_component"]]["Summary Len"].values[0]
#    new_df.loc[i,"Description Len Ave"] = summary_df[summary_df["lead_component"] == df.loc[i,"lead_component"]]["Description Len"].values[0]
#    new_df.loc[i,"Bug Description Len Ave"] = summary_df[summary_df["lead_component"] == df.loc[i,"lead_component"]]["Bug Description Len"].values[0]
#    new_df.loc[i,"Reappearance Procedure Len Ave"] = summary_df[summary_df["lead_component"] == df.loc[i,"lead_component"]]["Reappearance Procedure Len"].values[0]

#new_df["Summary Len diff"] = df["Summary Len"] - new_df["Summary Len Ave"]
#new_df["Description Len diff"] = df["Description Len"] - new_df["Description Len Ave"]
#new_df["Bug Description Len diff"] = df["Bug Description Len"] - new_df["Bug Description Len Ave"]
#new_df["Reappearance Procedure Len diff"] = df["Reappearance Procedure Len"] - new_df["Reappearance Procedure Len Ave"]

#for i in range(df.shape[0]): #　ここまでは組み込むことで性能が改善することを確認
#    new_df.loc[i,"Summary Len defect_Ave"] = defect_df[defect_df["defect_adding"] == df.loc[i,"defect_adding"]]["Summary Len"].values[0]
#    new_df.loc[i,"Description Len defect_Ave"] = defect_df[defect_df["defect_adding"] == df.loc[i,"defect_adding"]]["Description Len"].values[0]
#    new_df.loc[i,"Bug Description Len defect_Ave"] = defect_df[defect_df["defect_adding"] == df.loc[i,"defect_adding"]]["Bug Description Len"].values[0]
#    new_df.loc[i,"Reappearance Procedure Len defect_Ave"] = defect_df[defect_df["defect_adding"] == df.loc[i,"defect_adding"]]["Reappearance Procedure Len"].values[0]

#new_df["Summary Len defect_diff"] = df["Summary Len"] - new_df["Summary Len defect_Ave"]
#new_df["Description Len defect_diff"] = df["Description Len"] - new_df["Description Len defect_Ave"]
#new_df["Bug Description Len defect_diff"] = df["Bug Description Len"] - new_df["Bug Description Len defect_Ave"]
#new_df["Reappearance Procedure Len defect_diff"] = df["Reappearance Procedure Len"] - new_df["Reappearance Procedure Len defect_Ave"]

#new_df["summary_si"] = count_si
#new_df["summry_virtualkey"] = count_virtualkey
#new_df["summary_gps"] = count_gps

# cvは改善するが精度が下がる --------------
#count_0 = []
#count_03 = []
#count_07 = []
#for i in range(df.shape[0]):
#    day_diff = (df["Date of Found"][i] - df["Date of Found"])
#    day_diff = day_diff/np.timedelta64(1, "D")
#    day_diff = pd.DataFrame(day_diff)
#    count0 = day_diff[day_diff["Date of Found"]==0].shape[0]
#    count03 = day_diff[(day_diff["Date of Found"]>=0) & (day_diff["Date of Found"]<=3)].shape[0]
#    count07 = day_diff[(day_diff["Date of Found"]>=0) & (day_diff["Date of Found"]<=7)].shape[0]
#    count_0.append(count0)
#    count_03.append(count03)
#    count_07.append(count07)

#new_df["Found in 0days"] = count_0
#new_df["Found in 3days"] = count_03
#new_df["Found in 7days"] = count_07
# ---------------------------

# 乗算系-------------------------------------------------------------
# new_df["lda_319*123"] = df["lda_319"] * df["lda_123"]
# new_df["lda_260*147"] = df["lda_260"] * df["lda_147"]

# 比率系--------------------------------------------------
# new_df["ratio_340_desclen"] = df["lda_340"] * df["Description Len"]
# new_df["ratio_summary"] = df["Summary Len"] / new_df["overall_len"]
# new_df["bug_desc_summary"] = df["Bug Description Len"] / new_df["overall_len"]
# new_df["ratio_desc"] = df["Description Len"] / new_df["overall_len"]
# new_df["reappear_desc"] = df["Reappearance Procedure Len"] / new_df["overall_len"]

# ログ関連------------------------------------------------
#new_df["file_sum"] = df[file_var].sum(axis=1)
#new_df["klog_sum"] = df[klog_var].sum(axis=1)
#new_df["rtc_sum"] = df[rtc_var].sum(axis=1)
#new_df["ratio_file_klog"] = new_df["file_sum"] / (new_df["klog_sum"] + 0.1)
#new_df["ratio_klog_rtc"] = new_df["klog_sum"] / (new_df["rtc_sum"] + 0.1)
#new_df["ratio_rtc_file"] = new_df["rtc_sum"] / (new_df["file_sum"] + 0.1)

# lda系-----------------------------------------------------------
# ldaのlead_componentによる集計値　相関が高い変数を組み込む
#for j in [4, 3, 5, 17, 14, 13, 11, 29, 27, 25, 23]:
#    for i in range(df.shape[0]):
#        new_df.loc[i,"lda_"+ str(j) + " Ave"] = summary_df[summary_df["lead_component"] == df.loc[i,"lead_component"]]["lda_"+str(j)].values[0]


# その他系-------------------------------------------------------
# found_variables = [i for i in df.columns if "Found_" in i]
# new_df["max_found"] = np.argmax(df[found_variables].values, axis=1)
# lda_features = [i for i in input_features if "lda_" in i]
# new_df["max_lda"] = np.argmax(df[lda_features].values, axis=1)
# new_df["bin f_security"] = pd.cut(df["Found_Security"], 2, labels=["0","1"])
# new_df["bin f_security"] = pd.get_dummies(new_df["bin f_security"])
# new_df["std"] = new_df.std(axis=1)
# new_df["range"] = new_df.max(axis=1) - new_df.min(axis=1)

# new_df["max_found"] = np.argmax(df[found_variables].values, axis=1)
# new_df["min_found"] = np.argmin(df[found_variables].values, axis=1)
# new_df["sum_found"] = np.sum(df[found_variables].values, axis=1)
# new_df["sum_all"] = np.sum(df[input_features].values, axis=1)

#for i in lda_features:
#    cv = df[i].value_counts()
#    new_df[str(i)+"count"] = df[i].map(cv)

#logger.info('variables summary')
#mod_df = pd.concat([df, new_df], axis=1)
#final_features = [i for i in mod_df.columns if i in i not in useless_features and i not in day_feat and i != target_feature and i != "Type"]

#print("original input features    : ", len(orig_input_features))
#print("std zero features          : ", len(std_zero_feature))
#print("input_features             : ", len(input_features)) # original - std_zero
#print("----------------------------")
#print("useless_features         : ", len(useless_features))
#print("----------------------------")
#print("final feature              : ", len(final_features))


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




