from scipy.stats import kendalltau
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score as auc
from tqdm import tqdm
from transformers import TrainerCallback
import datasets
import numpy as np
import wandb

from reward_models.utils import group_apply
from reward_models.config import HF_TOKEN


class _HFCallbackMixin(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ds = datasets.load_dataset(self.hf_data_path, use_auth_token=HF_TOKEN)
            predictions = self.predict_batch(ds['train'])
            metrics = self.get_metrics(predictions, ds['train'])
            step = state.global_step if state.global_step > 0 else 1
            wandb.log(data={self.wandb_metrics_name: metrics}, step=step)

    def predict_batch(self, ds, batch_size=512):
        ixs = range(len(ds))
        indices = range(0, len(ds), batch_size)
        preds = [self.predict(ds.select(ixs[i:i+batch_size])) for i in tqdm(indices)]
        preds = np.concatenate(preds)
        return preds

    def predict(self, dataset):
        return self.preds_generator(self.model, dataset)

    def get_metrics(self, predictions, df):
        raise NotImplementedError

    def _invert_predictions_if_higher_is_better(self, predictions):
        return -predictions if self.higher_is_better else predictions


class RetryContinueCallback(_HFCallbackMixin):
    def __init__(self, model, preds_generator, higher_is_better=False):
        self.model = model
        self.preds_generator = preds_generator
        self.hf_data_path = 'ChaiML/14k_completions_evaluation'
        self.wandb_metrics_name = '14k_retry_continue'
        self.higher_is_better = higher_is_better

    def get_metrics(self, predictions, df):
        predictions = self._invert_predictions_if_higher_is_better(predictions)
        spearman_retry = spearmanr(predictions, df['retried'])
        spearman_eoc = spearmanr(predictions, df['end_of_conversation'])
        auc_retry = auc(df['retried'], predictions)
        auc_end_of_conversation = auc(df['end_of_conversation'], predictions)
        metrics = {
                'retry_spearman': spearman_retry.correlation,
                'retry_auc': auc_retry,
                'end_of_convo_spearman': spearman_eoc.correlation,
                'end_of_convo_auc': auc_end_of_conversation
                }
        return metrics


class RetentionCallback(_HFCallbackMixin):
    def __init__(self, model, preds_generator, higher_is_better=False):
        self.model = model
        self.preds_generator = preds_generator
        self.hf_data_path = 'ChaiML/14k_eval_retention_data'
        self.wandb_metrics_name = 'retention_eval'
        self.higher_is_better = higher_is_better

    def get_metrics(self, predictions, df):
        df = df.to_pandas()
        preds = -self._invert_predictions_if_higher_is_better(predictions)
        new_user_ix = np.array(df['is_new_user'])
        metrics = {}
        for day in ['d3', 'd7', 'd30']:
            flag = df[day]
            metrics[f'{day}_auc'] = auc(flag, preds)
            metrics[f'{day}_auc_new'] = auc(flag[new_user_ix], preds[new_user_ix])
            metrics[f'{day}_auc_old'] = auc(flag[~new_user_ix], preds[~new_user_ix])
        return metrics


class HumanRankingCallback(_HFCallbackMixin):
    def __init__(self, model, preds_generator, higher_is_better=False):
        self.model = model
        self.preds_generator = preds_generator
        self.hf_data_path = 'ChaiML/reward_model_manual_labeled_100_convos'
        self.wandb_metrics_name = '100_manual_labeled_convos'
        self.higher_is_better = higher_is_better

    def get_metrics(self, predictions, df):
        labels = np.array(df['rank (lower is better)'])
        ids = np.array(df['id'])
        predictions = self._invert_predictions_if_higher_is_better(predictions)
        ranked_prediction = group_apply(predictions, ids, self._rank)
        spearman = spearmanr(ranked_prediction, labels)
        values = np.vstack([ranked_prediction, labels]).T
        kendall = group_apply(values, ids, self._kendalltau, multiarg=True)
        top_2_acc = group_apply(values, ids, self._top_2_acc, multiarg=True)
        metrics = {
                'spearman': spearman.correlation,
                'kendall_tau': np.mean(kendall),
                'top_2_acc': np.mean(top_2_acc)
                }
        return metrics

    def _rank(self, predictions):
        return predictions.argsort().argsort()+1

    def _kendalltau(self, x, y):
        return kendalltau(x, y).correlation

    def _top_2_acc(self, x, y):
        s1 = set(np.argwhere(np.in1d(x, [1, 2])).flatten())
        s2 = set(np.argwhere(np.in1d(y, [1, 2])).flatten())
        return s1 == s2


class RejectWorstMessageCallback(_HFCallbackMixin):
    def __init__(self, model, preds_generator, higher_is_better=False):
        self.model = model
        self.preds_generator = preds_generator
        self.hf_data_path = 'ChaiML/rejection_sampling_4k_convo_data'
        self.wandb_metrics_name = 'reject_worst_messages'
        self.higher_is_better = higher_is_better

    def get_metrics(self, predictions, df):
        labels = np.array(df['label'])
        ids = np.array(df['id'])
        predictions = self._invert_predictions_if_higher_is_better(predictions)
        pred_labels = ss.np.group_apply(predictions, ids, self._label_worst_score)
        values = np.vstack([labels, pred_labels]).T
        accuracy = ss.np.group_apply(values, ids, lambda x, y: all(x == y), multiarg=True)
        nsfw_ix = np.array(df['is_nsfw']).astype(bool)
        new_user_ix = np.array(df['new_user']).astype(bool)
        metrics = {
                'overall_acc': accuracy.mean(),
                'nsfw_acc': accuracy[nsfw_ix].mean(),
                'sfw_acc': accuracy[~nsfw_ix].mean(),
                'new_user_acc': accuracy[new_user_ix].mean(),
                'old_user_acc': accuracy[~new_user_ix].mean()
                }
        return metrics

    def _label_worst_score(self, predictions):
        return predictions == np.max(predictions)
