import os

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, set_seed
import numpy as np
import torch

from reward_models.experiments.v5a_convo_d123r_joint import data, config
from reward_models.custom import custom_trainer
from reward_models.custom import custom_callbacks as cb
from reward_models.custom import train_utils as utils


os.environ['TOKENIZERS_PARALLELISM'] = 'True'

TOKENIZER = data._get_tokenizer()


def compute_labels(ds):
    res = []
    f1 = np.array(ds['accepted']).astype(float)
    res.append(f1)
    f2 = np.array(ds['n_user_messages_remaining'])
    f2 = (f2 >= 4).astype(float)  # short msg we filter
    res.append(f2)
    f3 = np.array(ds['log_norm_time_delta_till_user_response'])
    f3[~np.isfinite(f3)] = -1
    f3 = ((f3 < 10) & (f3 > -1)).astype(float)  # initial binary value 10
    res.append(f3)
    for col in ['d1', 'd2', 'd3']:
        arr = np.isfinite(ds[col]).astype(float)
        res.append(arr)
    res = np.array(res).T
    return list(res)


def prepare_training_data(ds):
    ds = utils.add_label(ds, compute_labels)
    ds = utils.remove_unnecessary_columns(ds)
    ds = split_train_test_data(ds, config.DATA_SIZE_TRAIN, config.DATA_SIZE_TEST)
    return ds


def split_train_test_data(ds, train_size, test_size, split_ratio=0.01):
    if config.SHUFFLE_BEFORE_SPLIT:
        ds = ds.shuffle(seed=config.SEED_TRAIN)
    ds = ds.train_test_split(test_size=split_ratio, shuffle=False)
    train_size = min(train_size, len(ds['train']))
    train_ds = ds['train'].shuffle(seed=config.SEED_TRAIN).select(range(train_size))
    test_size = min(test_size, len(ds['test']))
    test_ds = ds['test'].shuffle(seed=config.SEED_TRAIN).select(range(test_size))
    return train_ds, test_ds


def load_hf_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=config.NUM_LABELS,
            problem_type='multi_label_classification',
            )
    if model_name == 'gpt2':
        model.config.pad_token_id = 50256
    return model


def get_trainer_object(model, save_path):
    training_args = TrainingArguments(
        output_dir=save_path,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.TRAIN_EPOCHS,
        evaluation_strategy='steps',
        eval_steps=config.EVAL_STEPS,
        save_strategy='steps',
        save_steps=config.EVAL_STEPS,
        logging_strategy='steps',
        logging_steps=config.EVAL_STEPS,
        fp16=True,
        optim='adamw_hf',
        logging_first_step=False,
        seed=config.SEED_TRAIN,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUM_STEPS,
        dataloader_num_workers=12,
        dataloader_pin_memory=True,
        report_to='wandb',
        )

    custom_cb = [
            cb.HumanRankingCallback(model, generate_preds, higher_is_better=True),
            cb.RejectWorstMessageCallback(model, generate_preds, higher_is_better=True),
            cb.RetryContinueCallback(model, generate_preds, higher_is_better=True),
            cb.RetentionCallback(model, generate_preds, higher_is_better=True),
            ]

    trainer = custom_trainer.MultiLabelBCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        callbacks=custom_cb,
        num_labels=config.NUM_LABELS,
        )
    return trainer


def generate_preds(model, dataset):
    tokenized = data.apply_tokenizer(
            dataset,
            TOKENIZER,
            return_tensors='pt',
            return_attention_mask=True,
            ).to(0)
    with torch.no_grad():
        output = model(**tokenized).logits
    preds = torch.mean(torch.sigmoid(output), axis=1).cpu().detach()
    return preds


print('loading training dataset...')
set_seed(config.SEED_TRAIN)
ds = load_dataset(config.DATASET, use_auth_token=config.HF_AUTH_TOKEN)
assert len(ds.keys()) == 1
ds = ds['train']
print(ds)

# print('sampling by id with max limit...')
# ixs1 = utils.get_sample_ixs_by_id_with_max_limit(ds, group='convo_id', limit=1, mode='reverse')
# # ixs2 = utils.get_sample_ixs_by_id_with_max_limit(ds['train'], group='convo_id', limit=1, mode='random', seed=config.SEED_TRAIN)
# # ixs3 = np.unique(np.concatenate([ixs1, ixs2]))
# ds = ds.select(ixs1)
# print(ds)

print('sampling training dataset...')
n_samples = config.DATA_SIZE_TRAIN + config.DATA_SIZE_TEST
ds = utils.get_ordered_sample_data(ds, n_samples)
print(ds)
# 1/0

train_set, val_set = prepare_training_data(ds)
# print_label_stats(train_set, fold='train')
# print_label_stats(val_set, fold='test')


print('initializing wandb')
if config.RANK == 0:
    utils.init_wandb(project='reward', train_config=config)

print('initializing model...')
load_path, save_path = utils.format_load_and_save_path_from_checkpoint(config)
model = load_hf_model(load_path)

print('configuring training args...')
trainer = get_trainer_object(model, save_path=save_path)
trainer.evaluate()
trainer.train()

if config.TORCH_SAVE_PATH is not None:
    print('saving torch model to {}'.format(config.TORCH_SAVE_PATH))
    model.eval()
    torch.save(model, config.TORCH_SAVE_PATH)
