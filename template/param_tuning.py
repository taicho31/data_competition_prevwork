import optuna
import numpy as np


class ObjectiveManager:
    """Class for HyperParameter tuning by Optuna."""

    def __init__(
        self,
        data,
        target,
        model_name,
        model,
        params,
        metric_func,
        train_valid_idx_pairs,
    ):
        self.data = data
        self.target = target
        self.model_name = model_name
        self.model = model
        self.params = params
        self.metric = metric_func
        self.train_valid_idx_pairs = train_valid_idx_pairs

    def __call__(self, trial):
        if self.model_name == "lgb":
            # https://lightgbm.readthedocs.io/en/latest/Parameters.html
            self.check_params = {
                "max_bin": trial.suggest_int("max_bin", 10, 500),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 50),
                "min_sum_hessian_in_leaf": trial.suggest_float(
                    "min_sum_hessian_in_leaf", 1e-8, 10.0, log=True
                ),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                # 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
                # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 10),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 10),
                "lambda_l2": trial.suggest_float("lambda_l2", 0, 0.1),
                "lambda_l1": trial.suggest_float("lambda_l1", 0, 0.1),
                "max_depth": trial.suggest_int("max_depth", 2, 100),
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "path_smooth": trial.suggest_int("path_smooth", 0, 10),
            }

        elif self.model_name == "xgb":
            # https://xgboost.readthedocs.io/en/stable/parameter.html
            self.check_params = {
                "verbosity": 0,
                "gamma": trial.suggest_float("gamma", 0, 1),
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
                "subsample": trial.suggest_float("subsample", 0, 1),
                "lambda": trial.suggest_float("lambda", 0, 1),
                "alpha": trial.suggest_float("alpha", 0, 1),
            }

        elif self.model_name == "cb":
            # https://catboost.ai/en/docs/references/training-parameters/
            self.check_params = {
                "verbose": False,
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                # 'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
                "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.5),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 10),
                "colsample_bylevel": trial.suggest_float("reg_lambda", 0, 1.0),
            }

        all_params = self.params.copy()
        all_params.update(self.check_params)
        score = self.calc_score(
            self.data, self.target, all_params, self.train_valid_idx_pairs
        )
        return score

    def calc_score(self, data, target, params, train_valid_idx_pairs):
        valid_output = np.zeros(len(target))
        for split, (tr_idx, val_idx) in enumerate(train_valid_idx_pairs):
            _, valid_pred = self.model.train_and_valid(
                data.loc[tr_idx],
                target.loc[tr_idx],
                data.loc[val_idx],
                target.loc[val_idx],
                params,
            )
            valid_output[val_idx] = valid_pred
        score = self.metric(target, valid_output)
        return score


def param_tuning(
    data,
    target,
    model_name,
    model,
    params,
    metric_func,
    train_valid_idx_pairs,
    trial_num=5,
    option="minimize",
):

    objective = ObjectiveManager(
        data, target, model_name, model, params, metric_func, train_valid_idx_pairs
    )

    study = optuna.create_study(direction=option)
    study.optimize(objective, n_trials=trial_num)
    trial = study.best_trial
    print("Value: ", trial.value)

    optuna.visualization.plot_param_importances(study).show()
    return trial.params
