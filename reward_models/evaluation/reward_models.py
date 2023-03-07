import numpy as np

import torch
from transformers import AutoModelForSequenceClassification
from reward_models.evaluation.base_cls import BaseRewardModel, _GPT2MixIn

DEVICE = 0 if torch.cuda.is_available() else 'cpu'


class RandomRM(BaseRewardModel):
    def _predict(self, inputs, responses):
        out = [{'text': text, 'score': np.random.randn()} for text in responses]
        return out


class GPT2Model(BaseRewardModel, _GPT2MixIn):
    def __init__(self, hf_model_name):
        self.hf_model_name = hf_model_name
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

    def _load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.hf_model_name)
        model = model.to(DEVICE).eval()
        return model

    def _predict(self, inputs, responses):
        out = []
        for response in responses:
            model_inputs = inputs + response
            tokens = self._tokenize(model_inputs)
            logits = torch.softmax(self.model(**tokens).logits, dim=1)
            preds = float(logits[0][1])
            out.append({'text': response, 'score': preds})
        return out
