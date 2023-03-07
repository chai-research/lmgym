from transformers import Trainer
from torch import nn


class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_function = nn.BCELoss()
        loss = loss_function(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class MultiLabelBCETrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self._num_lables = kwargs.pop('num_labels')
        self._custom_loss_func = nn.BCEWithLogitsLoss()
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self._custom_loss_func(
                logits.view(-1, self._num_lables),
                labels.view(-1, self._num_lables),
                )
        return (loss, outputs) if return_outputs else loss
