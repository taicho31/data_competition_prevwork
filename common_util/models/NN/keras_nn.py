import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

n_folds=5

skf=GroupKFold(n_splits = n_folds)
models = []

def get_lr(epoch): # change learning rate by epoch
    if epoch < 20:
        return 1e-3

    return 1e-4

def nn_modelling(train, test):
    # to keep reproducible results ----------------
    # https://keras.io/getting_started/faq/
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    random.seed(12345)
    tf.random.set_seed(1234)
    # ------------------------------------------
    
    features = [i for i in train.columns if i not in ["installation_id", "accuracy_group"]]
    categoricals = ['session_title']
    for cat in categoricals:
        enc = OneHotEncoder()
        train_cats = enc.fit_transform(train[[cat]])
        test_cats = enc.transform(test[[cat]])
        cat_cols = ['{}_{}'.format(cat, str(col)) for col in enc.active_features_]
        features += cat_cols
        train_cats = pd.DataFrame(train_cats.toarray(), columns=cat_cols)
        test_cats = pd.DataFrame(test_cats.toarray(), columns=cat_cols)
        train = pd.concat([train, train_cats], axis=1)
        test = pd.concat([test, test_cats], axis=1)

    # standardization -----
    scalar = MinMaxScaler()
    train[features] = scalar.fit_transform(train[features])
    test[features] = scalar.transform(test[features])
    # ---------------------
    
    X_train = train.drop(['accuracy_group'],axis=1) 
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(X_train["installation_id"]))
    X_train["installation_id"] = lbl.transform(list(X_train["installation_id"]))
    remove_features = []
    for i in categoricals:
        remove_features.append(i)
    for i in X_train.columns:
        if X_train[i].std() == 0 and i not in remove_features:
            remove_features.append(i)
    for i in high_corr_features:
        if i not in remove_features:
            remove_features.append(i)

    X_train = X_train.drop(remove_features, axis=1)
    X_train = X_train[sorted(X_train.columns.tolist())]
    y_train = new_train.accuracy_group
    
    X_test = test.drop(["installation_id","accuracy_group"], axis=1)
    X_test = X_test.drop(remove_features, axis=1)
    X_test = X_test[sorted(X_test.columns.tolist())]
    
    random_try = 30
    mean_qwk_score = 0
    for i , (train_index, test_index) in enumerate(skf.split(X_train, y_train, X_train["installation_id"])):    
        X_train2 = X_train.iloc[train_index,:]
        y_train2 = y_train.iloc[train_index]
        X_train2 = X_train2.drop(['installation_id'],axis=1)
    
        for try_time in range(random_try): 
            print("Fold "+str(i+1)+" random try " +str(try_time+1))
            X_test2 = X_train.iloc[test_index,:]
            y_test2 = y_train.iloc[test_index]
            
            X_test2, idx_val = get_random_assessment(X_test2)
            X_test2.drop(['installation_id'], inplace=True, axis=1) # 'past_target'
            y_test2 = y_test2.loc[idx_val]
        
            verbosity = 100
            model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(X_train2.shape[1],)),
                tf.keras.layers.Dense(200, activation='relu'), #, kernel_regularizer=tf.keras.regularizers.l2(0.001)
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(100, activation='tanh'),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.3),
                #tf.keras.layers.Dense(50, activation='relu'),
                #tf.keras.layers.LayerNormalization(),
                #tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(25, activation='relu'),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1, activation='relu')
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4), loss='mse') #
            #print(model.summary())
            save_best = tf.keras.callbacks.ModelCheckpoint('./nn_model.h5', save_best_only=True, verbose=1)
            early_stop = tf.keras.callbacks.EarlyStopping(patience=10)
        
            model.fit(X_train2, y_train2, 
                     validation_data=(X_test2, y_test2),
                    epochs=25,callbacks=[save_best]) #early_stop, LearningRateScheduler(get_lr)
            model.load_weights('./nn_model.h5')

            models.append(model)
            valid = np.array(model.predict(X_test2).reshape(X_test2.shape[0],))
            real = np.array(y_test2)
            # threshold optimization --------------
            best_score = 0
            for j in range(20):
                optR = OptimizedRounder()
                optR.fit(np.array(valid).reshape(-1,), real, random_flg=True)
                coefficients = optR.coefficients()
                final_valid_pred = optR.predict(np.array(valid).reshape(-1,), coefficients)
                score = qwk(real, final_valid_pred)
                print(j, np.sort(coefficients), score)
                if score > best_score:
                    best_score = score
                    best_coefficients = coefficients
            mean_qwk_score += best_score / (random_try * n_folds)
            if try_time == 0 and i == 0:
                final_coefficients = np.sort(best_coefficients) / (random_try * n_folds)
            else:
                final_coefficients += np.sort(best_coefficients) / (random_try * n_folds)
                           
    print("MEAN QWK = \t {}".format(mean_qwk_score))
    # test prediction  ------------------------
    pred_value = np.zeros([X_test.shape[0]])
    for model in models:
        pred_value += model.predict(X_test).reshape(X_test.shape[0],) / len(models)
    return pred_value, final_coefficients
pred_value, final_coefficients = nn_modelling(new_train, new_test)
