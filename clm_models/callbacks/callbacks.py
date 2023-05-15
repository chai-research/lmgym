import os

from transformers import AutoModelForSequenceClassification
from clm_models.callbacks import prompts, cleanup


def get_callbacks(data_args, training_args, tokenizer):
    callbacks = []

    if "gpt" in tokenizer.name_or_path.lower():
        # GPT-J
        dep_callback_args = {
            'temperature': 0.72,
            'repetition_penalty': 1.13125,
            'max_new_tokens': 64,
            'top_p': 0.725,
            'top_k': 0,
            'do_sample': True,
            'eos_token_id': 198,
        }
    else:
        # LLaMa
        dep_callback_args = {
            'temperature': 0.7,
            'repetition_penalty': 1 / 0.85,
            'max_new_tokens': 128,
            'top_k': 40,
            'do_sample': True,
            'eos_token_id': 13,
        }

    if data_args.eval_prompt_path is not None:
        reward_models = None
        if data_args.add_reward_scores:
            rank = training_args.local_rank
            reward_models = {
                'continue_50m': _load_reward_model('ChaiML/reward_48m_gpt2_target_2', rank),
                'retry_12m': _load_reward_model('ChaiML/gpt2_retry_12m', rank),
                'stars_2m': _load_reward_model('ChaiML/3plus_stars_gpt2_reward', rank),
                'retry_and_continue_12m': _load_reward_model('ChaiML/gpt2_retry_and_continue_12m_v2', rank)
            }

        if data_args.eval_prompt_path is not None:
            callback = prompts.RecordExampleAnswersCallback(
                name='dep',
                path=data_args.eval_prompt_path,
                tokenizer=tokenizer,
                params=dep_callback_args,
                num_prompts=data_args.num_eval_prompts,
                reward_models=reward_models
            )
            callbacks.append(callback)

    if data_args.clean_enabled:
        callback = cleanup.CleanupCallback(
            pattern=os.path.join(training_args.output_dir, 'checkpoint-*', 'global_step*')
        )
        callbacks.append(callback)
    return callbacks


def _load_reward_model(name, local_rank):
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        use_auth_token=True
    )
    return model.to(local_rank)
