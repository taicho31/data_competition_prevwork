import numpy as np

from lightgbm import early_stopping, log_evaluation
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from xgboost import callback


class Common_LGB_Modelling:
    """
    Train and test data should contain the same selected features for ML models.
    Train, test data and target should be the same data type. (Pandas or Numpy)
    """

    def __init__(self, model_class, custom_callback=None):
        self.model_class = model_class
        self.custom_callback = custom_callback
        self.default_callback = [
            early_stopping(stopping_rounds=50),
            log_evaluation(100),
        ]

    def train(self, x_tr, y_tr, params):

        model = self.model_class(**params)
        model.fit(x_tr, y_tr)

        return model

    def train_and_valid(self, x_tr, y_tr, x_val, y_val, params):

        if self.custom_callback:
            callbacks = self.custom_callback
        else:
            callbacks = self.default_callback

        model = self.model_class(**params)
        model = model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], callbacks=callbacks)
        valid_pred = model.predict(x_val)

        return model, valid_pred

    def test(self, models, test):
        test_pred = [model.predict(test) for model in models]
        test_pred = np.mean(test_pred, axis=0)
        return test_pred

    def test_by_batch(self, models, test, batch_size):
        test_pred_all = []
        for idx in range(0, len(test), batch_size):
            test_pred_batch = [
                model.predict(test.iloc[idx : idx + batch_size]) for model in models
            ]
            test_pred_batch = np.mean(test_pred_batch, axis=0)
            test_pred_all.append(test_pred_batch)
        return np.concatenate(test_pred_all)

    def numpy_test_by_batch(self, models, test, batch_size):
        test_pred_all = []
        for idx in range(0, len(test), batch_size):
            test_pred_batch = [
                model.predict(test[idx : idx + batch_size]) for model in models
            ]
            test_pred_batch = np.mean(test_pred_batch, axis=0)
            test_pred_all.append(test_pred_batch)
        return np.concatenate(test_pred_all)


class Common_CB_Modelling:
    """
    Train and test data should contain the same selected features for ML models.
    Train, test data and target should be the same data type. (Pandas or Numpy)
    """

    def __init__(self, model_class):
        self.model_class = model_class

    def train(self, x_tr, y_tr, params, cat):

        train_pool = Pool(data=x_tr, label=y_tr, cat_features=cat)

        model = self.model_class(**params)
        model.fit(train_pool)

        return model

    def train_and_valid(self, x_tr, y_tr, x_val, y_val, params, cat):
        train_pool = Pool(data=x_tr, label=y_tr, cat_features=cat)
        valid_pool = Pool(data=x_val, label=y_val, cat_features=cat)

        model = self.model_class(**params)

        if isinstance(model, CatBoostClassifier):
            model.fit(
                train_pool,
                eval_set=[valid_pool],
                early_stopping_rounds=50,
                verbose_eval=100,
            )
            valid_pred = model.predict_proba(x_val)[:, 1]
        elif isinstance(model, CatBoostRegressor):
            model.fit(
                train_pool,
                eval_set=[valid_pool],
                early_stopping_rounds=50,
                verbose_eval=100,
            )
            valid_pred = model.predict(x_val)

        return model, valid_pred

    def test(self, models, test):
        if isinstance(models[0], CatBoostClassifier):
            test_pred = [model.predict_proba(test)[:, 1] for model in models]
        elif isinstance(models[0], CatBoostRegressor):
            test_pred = [model.predict(test) for model in models]
        test_pred = np.mean(test_pred, axis=0)
        return test_pred

    def test_by_batch(self, models, test, batch_size):
        test_pred_all = []
        for idx in range(0, len(test), batch_size):
            if isinstance(models[0], CatBoostClassifier):
                test_pred_batch = [
                    model.predict_proba(test.iloc[idx : idx + batch_size])[:, 1]
                    for model in models
                ]
            elif isinstance(models[0], CatBoostRegressor):
                test_pred_batch = [
                    model.predict(test.iloc[idx : idx + batch_size]) for model in models
                ]
            test_pred_batch = np.mean(test_pred_batch, axis=0)
            test_pred_all.append(test_pred_batch)
        return np.concatenate(test_pred_all)

    def numpy_test_by_batch(self, models, test, batch_size):
        test_pred_all = []
        for idx in range(0, len(test), batch_size):
            if isinstance(models[0], CatBoostClassifier):
                test_pred_batch = [
                    model.predict_proba(test[idx : idx + batch_size])[:, 1]
                    for model in models
                ]
            elif isinstance(models[0], CatBoostRegressor):
                test_pred_batch = [
                    model.predict(test[idx : idx + batch_size]) for model in models
                ]
            test_pred_batch = np.mean(test_pred_batch, axis=0)
            test_pred_all.append(test_pred_batch)
        return np.concatenate(test_pred_all)


class Common_XGB_Modelling:
    """
    Train and test data should contain the same selected features for ML models.
    Train, test data and target should be the same data type. (Pandas or Numpy)
    """

    def __init__(self, model_class):
        self.model_class = model_class

    def train_and_valid(self, x_tr, y_tr, x_val, y_val, params):

        model = self.model_class(
            **params, callbacks=[callback.EvaluationMonitor(rank=0, period=100)]
        )
        model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)])
        valid_pred = model.predict(x_val, iteration_range=(0, model.best_iteration))

        return model, valid_pred

    def test(self, models, test):
        test_pred = [
            model.predict(test, iteration_range=(0, model.best_iteration))
            for model in models
        ]
        test_pred = np.mean(test_pred, axis=0)
        return test_pred
