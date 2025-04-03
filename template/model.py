import numpy as np
import polars as pl

import matplotlib.pyplot as plt
import matplotlib_fontja
import seaborn as sns

from lightgbm import early_stopping, log_evaluation
from catboost import Pool


class BaseGBDTClass:
    def __init__(self, model_class, params):
        self.model_class = model_class
        self.params = params

    def train(self, x_tr, y_tr):
        raise NotImplementedError("train method must be implemented")

    def train_with_validation(self, x_tr, y_tr, x_val, y_val):
        raise NotImplementedError("train_with_validation method must be implemented")

    def predict(self, model, input_):
        raise NotImplementedError("predict method must be implemented")

    def extract_importance(self, model, features, imp_col_prefix=None):
        importance = model.feature_importances_
        if imp_col_prefix is not None:
            importance_col = imp_col_prefix + "_importance"
        else:
            importance_col = "importance"
        importance_df = pl.DataFrame({"feature": features, importance_col: importance})
        return importance_df

    def check_pred_distribution_diff(self, predictions, gt, title):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(predictions, ax=ax, label="prediction")
        sns.histplot(gt, ax=ax, label="ground truth")
        ax.legend()
        ax.grid()
        ax.set_title(title)

    def test(self, models, test):
        test_predictions = [self.predict(model, test) for model in models]
        test_predictions = np.mean(test_predictions, axis=0)
        return test_predictions


class LGBClass(BaseGBDTClass):
    def __init__(self, cat_features="auto", custom_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cat_features = cat_features
        if custom_callback:
            self.callbacks = custom_callback
        else:
            self.callbacks = [
                early_stopping(stopping_rounds=50),
                log_evaluation(100),
            ]

    def train(self, x_tr, y_tr):

        model = self.model_class(**self.params)
        model.fit(
            x_tr,
            y_tr,
            categorical_feature=self.cat_features,
        )

        return model

    def train_with_validation(self, x_tr, y_tr, x_val, y_val):

        model = self.model_class(**self.params)
        model = model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_val, y_val)],
            categorical_feature=self.cat_features,
            callbacks=self.callbacks,
        )
        return model

    def predict(self, model, input_):
        predictions = model.predict(input_)
        return predictions


class XGBClass(BaseGBDTClass):
    def __init__(self, verbose_eval_step: int, output_prob: bool, multi_label: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose_eval_step = verbose_eval_step
        self.output_prob = output_prob
        self.multi_label = multi_label

    def train(self, x_tr, y_tr):

        model = self.model_class(**self.params)
        model.fit(x_tr, y_tr)
        return model

    def train_with_validation(self, x_tr, y_tr, x_val, y_val):

        model = self.model_class(**self.params)
        model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=self.verbose_eval_step)
        return model

    def predict(self, model, input_):
        if self.output_prob:
            predictions = model.predict_proba(
                input_, iteration_range=(0, model.best_iteration)
            )
            if not self.multi_label:
                predictions = predictions[:, 1]
        else:
            predictions = model.predict(
                input_, iteration_range=(0, model.best_iteration)
            )
        return predictions


class CBClass(BaseGBDTClass):
    def __init__(self, cat, output_prob: bool, multi_label: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cat = cat
        self.multi_label = multi_label
        self.output_prob = output_prob

    def train(self, x_tr, y_tr):

        train_pool = Pool(data=x_tr, label=y_tr, cat_features=self.cat)

        model = self.model_class(**self.params)
        model.fit(train_pool)

        return model

    def train_with_validation(self, x_tr, y_tr, x_val, y_val):
        train_pool = Pool(data=x_tr, label=y_tr, cat_features=self.cat)
        valid_pool = Pool(data=x_val, label=y_val, cat_features=self.cat)

        model = self.model_class(**self.params)
        model.fit(
            train_pool,
            eval_set=[valid_pool],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        return model

    def predict(self, model, input_):
        if self.output_prob:
            predictions = model.predict_proba(input_)
            if not self.multi_label:
                predictions = predictions[:, 1]
        else:
            predictions = model.predict(input_)
        return predictions

    def extract_importance(self, model, features, imp_col_prefix=None):
        importance = model.get_feature_importance()
        if imp_col_prefix is not None:
            importance_col = imp_col_prefix + "_importance"
        else:
            importance_col = "importance"
        importance_df = pl.DataFrame({"feature": features, importance_col: importance})
        return importance_df


def param_tuning(objective, trial_num = 5, option = "minimize"):
                        
    study = optuna.create_study(direction=option) 
    study.optimize(objective, n_trials=trial_num)
    trial = study.best_trial
    best_parameters = trial.params
    best_value = trial.value
    print('Value: ', best_value)
    print('best_parameters: ', best_parameters)

    optuna.visualization.plot_param_importances(study).show()

    return best_parameters, best_value