import numpy as np
import pandas as pd
import datasets

from reward_models.custom import custom_callbacks as cb


def preds_generator(model, dataset):
    pass


def test_retry_continue_callback_higher_better():
    callback = cb.RetryContinueCallback('dummy', preds_generator, higher_is_better=True)
    data = {'retried': [0, 0, 1, 0],
            'end_of_conversation': [0, 0, 1, 0],
            'text': ['a']*4}
    preds = np.array([-1, -4, 10, -5])
    metrics = callback.get_metrics(preds, pd.DataFrame.from_dict(data))
    assert metrics['retry_auc'] == 0.
    assert metrics['end_of_convo_auc'] == 0.
    assert metrics['retry_spearman'] < 0.
    assert metrics['end_of_convo_spearman'] < 0.


def test_retry_continue_callback_lower_better():
    callback = cb.RetryContinueCallback('dummy', preds_generator, higher_is_better=False)
    data = {'retried': [0, 0, 1, 0],
            'end_of_conversation': [0, 0, 1, 0],
            'text': ['a']*4}
    preds = np.array([-1, -4, 10, -5])
    metrics = callback.get_metrics(preds, pd.DataFrame.from_dict(data))
    assert metrics['retry_auc'] == 1.
    assert metrics['end_of_convo_auc'] == 1.
    assert metrics['retry_spearman'] > 0.
    assert metrics['end_of_convo_spearman'] > 0.


def test_human_ranking_callback_higher_better():
    callback = cb.HumanRankingCallback('dummy', preds_generator, higher_is_better=True)
    data = {'rank (lower is better)': [1, 2, 3, 4],
            'id': ['a']*4}
    preds = np.array([10, 9, 2, -100])
    metrics = callback.get_metrics(preds, pd.DataFrame.from_dict(data))
    assert metrics['spearman'] > 0.
    assert metrics['kendall_tau'] == 1.
    assert metrics['top_2_acc'] == 1.


def test_human_ranking_callback_lower_better():
    callback = cb.HumanRankingCallback('dummy', preds_generator, higher_is_better=False)
    data = {'rank (lower is better)': [1, 2, 3, 4],
            'id': ['a']*4}
    preds = np.array([10, 9, 2, -100])
    metrics = callback.get_metrics(preds, pd.DataFrame.from_dict(data))
    assert metrics['spearman'] < 0.
    assert metrics['kendall_tau'] == -1.
    assert metrics['top_2_acc'] == 0.


def test_reject_worst_message_callback_higher_better():
    callback = cb.RejectWorstMessageCallback('dummy', preds_generator, higher_is_better=True)
    data = {'label': [0, 1, 0, 0, 1, 0, 0, 0],
            'id': [1] * 4 + [2] * 4,
            'is_nsfw': [1] * 4 + [0] * 4,
            'new_user': [0] * 4 + [1] * 4}
    preds = np.array([9, -1, 4, 6, 10, 2, 3, 8])
    metrics = callback.get_metrics(preds, pd.DataFrame.from_dict(data))
    assert metrics['overall_acc'] == 0.5
    assert metrics['nsfw_acc'] == 1.
    assert metrics['sfw_acc'] == 0.
    assert metrics['new_user_acc'] == 0.
    assert metrics['old_user_acc'] == 1.


def test_reject_worst_message_callback_lower_better():
    callback = cb.RejectWorstMessageCallback('dummy', preds_generator, higher_is_better=False)
    data = {'label': [0, 1, 0, 0, 1, 0, 0, 0],
            'id': [1] * 4 + [2] * 4,
            'is_nsfw': [1] * 4 + [0] * 4,
            'new_user': [0] * 4 + [1] * 4}
    preds = np.array([9, -1, 4, 6, 10, 2, 3, 8])
    metrics = callback.get_metrics(preds, pd.DataFrame.from_dict(data))
    assert metrics['overall_acc'] == 0.5
    assert metrics['nsfw_acc'] == 0.
    assert metrics['sfw_acc'] == 1.
    assert metrics['new_user_acc'] == 1.
    assert metrics['old_user_acc'] == 0.


def test_retention_callback_higher_is_better():
    callback = cb.RetentionCallback('dummy', preds_generator, higher_is_better=True)
    data = {'d3': [0, 1, 0, 0, 1, 0, 0, 0],
            'd7': [0, 1, 0, 0, 1, 0, 0, 0],
            'd30': [1, 0, 1, 1, 0, 1, 1, 1],
            'is_new_user': [True] * 4 + [False] * 4,
            }
    preds = np.array([-5, 5, -9, 0, 12, -1, -2, 1])
    metrics = callback.get_metrics(preds, datasets.Dataset.from_dict(data))
    assert metrics['d3_auc'] == 1.
    assert metrics['d7_auc'] == 1.
    assert metrics['d30_auc'] == 0.


def test_retention_callback_lower_is_better():
    callback = cb.RetentionCallback('dummy', preds_generator, higher_is_better=False)
    data = {'d3': [0, 1, 0, 0, 1, 0, 0, 0],
            'd7': [0, 1, 0, 0, 1, 0, 0, 0],
            'd30': [1, 0, 1, 1, 0, 1, 1, 1],
            'is_new_user': [True] * 4 + [False] * 4,
            }
    preds = np.array([-5, 5, -9, 0, 12, -1, -2, 1])
    metrics = callback.get_metrics(preds, datasets.Dataset.from_dict(data))
    assert metrics['d3_auc'] == 0.
    assert metrics['d7_auc'] == 0.
    assert metrics['d30_auc'] == 1.
