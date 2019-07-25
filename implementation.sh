#! /bin/bash
ls ../input/*.feather > /dev/null 2>&1
if [$? -ne 0]; then
    echo "original feather file exists"
else
    python3 memory_reduce.py
fi

python3 feature_engineering.py
python3 lgb_train.py
