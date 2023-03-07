from transformers import AutoModelForSequenceClassification
import torch

from reward_models.evaluation import reward_models as rm
from reward_models.evaluation import base_models
from reward_models.evaluation.base_cls import BaseRewardModel, _GPT2MixIn


class GPT2Model(BaseRewardModel, _GPT2MixIn):
    def __init__(self, hf_model_name):
        self.hf_model_name = hf_model_name
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

    def _load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.hf_model_name)
        model = model.to(0).eval()
        return model

    def _predict(self, inputs, responses):
        out = []
        for response in responses:
            model_inputs = inputs + response
            tokens = self._tokenize(model_inputs)
            logits = torch.sigmoid(self.model(**tokens).logits)
            preds = float(logits[0][0])
            out.append({'text': response, 'score': preds})
        return out


def get_base_model():
    return base_models.GPTJ()


def get_reward_models():
    models = {
        'zl_cp1': GPT2Model('ChaiML/reward_models_v29_d1r_42_2500000_cp_38064'),
        'zl_cp2': GPT2Model('ChaiML/reward_models_v29_d1r_42_2500000_cp_39040'),
        'retry-xl': rm.GPT2Model('ChaiML/gpt2_xl_retry_and_continue_12m_v2'),
        }
    return models
