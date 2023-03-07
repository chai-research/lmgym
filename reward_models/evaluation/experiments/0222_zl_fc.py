import torch

from reward_models.evaluation import reward_models as rm
from reward_models.evaluation import base_models


def get_base_model():
    return base_models.GPTJ()


class ZLModel(rm.GPT2Model):
    def _predict(self, inputs, responses):
        out = []
        for response in responses:
            model_inputs = inputs + response
            tokens = self._tokenize(model_inputs)
            logits = self.model(**tokens).logits
            preds = torch.mean(torch.sigmoid(logits), axis=1)
            preds = float(preds[0])
            out.append({'text': response, 'score': preds})
        return out


def get_reward_models():
    out = {
        'ri_2mm': rm.GPT2Model('ChaiML/reward_models_rob_2mm_retry_accept_cp_62500'),
        'fc_8mm_1': rm.GPT2Model('ChaiML/reward_models_v2_respond180s_42_8000000_cp_14058'),
        'fc_8mm_2': rm.GPT2Model('ChaiML/reward_models_v2_respond180s_42_8000000_cp_31240'),
        'zl_2.5mm_1': ZLModel('ChaiML/reward_models_v01_d123r_42_2500000_cp_39060'),
        'zl_2.5mm_2': ZLModel('ChaiML/reward_models_v01_d123r_42_2500000_cp_76167'),
        }
    return out
