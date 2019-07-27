#! /bin/bash
ID=$(<feature_setting/id_feature.txt)
TARGET=$(<feature_setting/target_feature.txt)

echo "$ID"
echo "$TARGET"

#ls ../input/*.feather > /dev/null 2>&1
#if [ $? -ne 0 ]; then
#    echo "original feather file exists"
#else
#    python3 memory_reduce.py "$ID" "$TARGET"
#fi

python3 eda.py "$ID" "$TARGET"
#python3 feature_engineering.py "$ID" "$TARGET"
#python3 model/lgb_train.py $ID $TARGET
