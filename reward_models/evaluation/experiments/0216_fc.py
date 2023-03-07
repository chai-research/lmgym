from reward_models.evaluation import reward_models as rm
from reward_models.evaluation import base_models


def get_base_model():
    return base_models.GPTJ()


def get_reward_models():
    out = {
        'fc': rm.GPT2Model('ChaiML/fc_new_data_reward_cp3900'),
        'retry-xl': rm.GPT2Model('ChaiML/gpt2_xl_retry_and_continue_12m_v2'),
        }
    return out
