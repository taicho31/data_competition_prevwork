def sample_scheduler_func(current_lr, eval_history, best_round, current_round, is_higher_better):
    """次のラウンドで用いる学習率を決定するための関数 (この中身を好きに改造する)

    :param current_lr: 現在の学習率 (指定されていない場合の初期値は None)
    :param eval_history: 検証用データに対する評価指標の履歴
    :param best_round: 現状で最も評価指標の良かったラウンド数
    :param is_higher_better: 高い方が性能指標として優れているか否か
    :return: 次のラウンドで用いる学習率

    NOTE: 学習を打ち切りたいときには callback.EarlyStopException を上げる
    """
    # default value when learning rate isn't set.
    current_lr = current_lr or 0.05

    #print(current_round, best_round+1)
    if current_round - (best_round+1) >= 7:
        current_lr /= 2
        print("Learning rate changed to {}".format(current_lr))

    # 小さすぎるとほとんど学習が進まないので下限も用意する
    min_threshold = 0.01
    current_lr = max(min_threshold, current_lr)

    return current_lr

class LrSchedulingCallback(object):
    """ラウンドごとの学習率を動的に制御するためのコールバック"""

    def __init__(self, strategy_func):
        # 学習率を決定するための関数
        self.scheduler_func = strategy_func
        # 検証用データに対する評価指標の履歴
        self.eval_metric_history = []

    def __call__(self, env):
        # get current learning rate
        current_lr = env.params.get('learning_rate')

        # extract result of validation data [0] = result of training data
        first_eval_result = env.evaluation_result_list[1]
        metric_score = first_eval_result[2]
        # 評価指標は大きい方が優れているか否か
        is_higher_better = first_eval_result[3]
        
        # 評価指標の履歴を更新する
        self.eval_metric_history.append(metric_score)
        # 現状で最も優れたラウンド数を計算する
        best_round_find_func = np.argmax if is_higher_better else np.argmin
        best_round = best_round_find_func(self.eval_metric_history)
        current_round = len(self.eval_metric_history)

        # 新しい学習率を計算する
        new_lr = self.scheduler_func(current_lr=current_lr,
                                     eval_history=self.eval_metric_history,
                                     best_round=best_round,
                                     current_round = current_round,
                                     is_higher_better=is_higher_better)
        # 次のラウンドで使う学習率を更新する
        update_params = {
            'learning_rate': new_lr,
        }
        env.model.reset_parameter(update_params)
        env.params.update(update_params)

## in the modelling ....

#lr_scheduler_cb = LrSchedulingCallback(strategy_func=sample_scheduler_func)
#callbacks = [lr_scheduler_cb,]

#clf = lgb.train(params, lgb_train,valid_sets=[lgb_train, lgb_eval], 
               num_boost_round=500,early_stopping_rounds=10,verbose_eval = 10, categorical_feature=categoricals, callbacks = callbacks,) 
