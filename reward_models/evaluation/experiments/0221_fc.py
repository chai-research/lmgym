from reward_models.evaluation import reward_models as rm
from reward_models.evaluation import base_models


def get_base_model():
    return base_models.GPTJ()


def get_reward_models():
    models = {
        'fc': rm.GPT2Model('ChaiML/reward_models_v2_respond180s_42_8000000_cp_14058'),
        'retry-xl': rm.GPT2Model('ChaiML/gpt2_xl_retry_and_continue_12m_v2'),
        }
    return models
