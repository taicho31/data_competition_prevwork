import pandas as pd
import numpy as np
import glob

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")
ensemble=pd.DataFrame([])

prediction_list = glob.glob('../result/*.csv') # prediction list so far 
for ind in prediction_list:
    index = ind.find("submission") # submissionという名前が始まるインデックス番号
    df = pd.read_csv(ind)
    df = df.rename(columns={'target': 'target_'+str(index+11:-4)})
    df_result = df["target_"+str(index+11:-4)]
    ensemble = pd.concat([ensemble, df_result], axis=1)

print(ensemble.corr())

vote = np.mean(ensemble, axis=1)

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = vote
sub_df.to_csv("../result/submission_ensemble.csv", index=False)
