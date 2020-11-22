import pandas as pd
import numpy as np

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")
ensemble=pd.DataFrame([])

lr_result = pd.read_csv("../result/submission_lr.csv")
lr_result = lr_result.rename(columns={'target': 'target_lr'})
lr = lr_result["target_lr"]

#rfc_result = pd.read_csv("../result/submission_rfc.csv")
#rfc_result = rfc_result.rename(columns={'target': 'target_rfc'})
#rfc = rfc_result["target_rfc"]

lgb_result = pd.read_csv("../result/submission_lgb.csv")
lgb_result = lgb_result.rename(columns={'target': 'target_lgb'})
lgb = lgb_result["target_lgb"]

lgb_augment_result = pd.read_csv("../result/lgb_augment_submission.csv")
lgb_augment_result = lgb_augment_result.rename(columns={'target': 'target_lgb_augment'})
lgb_augment = lgb_augment_result["target_lgb_augment"]

gb_result = pd.read_csv("../result/submission_gb.csv")
gb_result = gb_result.rename(columns={'target': 'target_gb'})
gb = gb_result["target_gb"]

cat_result = pd.read_csv("../result/submission_cat.csv")
cat_result = cat_result.rename(columns={'target': 'target_cat'})
cat = cat_result["target_cat"]

#svm_result = pd.read_csv("../result/submission_svm.csv")
#svm_result = svm_result.rename(columns={'target': 'target_svm'})
#svm = svm_result["target_svm"]

#xgb_result = pd.read_csv("../result/submission_xgb.csv")
#xgb_result = xgb_result.rename(columns={'target': 'target_xgb'})
#xgb = xgb_result["target_xgb"]

etc_result = pd.read_csv("../result/submission_etc.csv")
etc_result = etc_result.rename(columns={'target': 'target_etc'})
etc = etc_result["target_etc"]

ensemble = pd.concat([ensemble, lr], axis=1)
#ensemble = pd.concat([ensemble, rfc], axis=1)
ensemble = pd.concat([ensemble, lgb], axis=1)
ensemble = pd.concat([ensemble, lgb_augment], axis=1)
ensemble = pd.concat([ensemble, gb], axis=1)
ensemble = pd.concat([ensemble, cat], axis=1)
#ensemble = pd.concat([ensemble, svm], axis=1)
#ensemble = pd.concat([ensemble, xgb], axis=1)
ensemble = pd.concat([ensemble, etc], axis=1)

print(ensemble.corr())

vote = np.mean(ensemble, axis=1)

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = vote
sub_df.to_csv("../result/submission_ensemble_ver2.csv", index=False)
