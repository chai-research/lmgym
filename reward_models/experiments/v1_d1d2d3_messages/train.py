from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, set_seed
import numpy as np
import wandb
import torch
import config
from reward_models.custom import train_utils


def init_wandb():
    wandb.login(key=config.WANDB_AUTH_TOKEN)
    name = f'{config.MODEL}_data_seed_{config.SEED}_data_{config.DATA_SIZE_TRAIN}'
    wandb.init(project='reward', name=name)


def print_label_stats(ds, fold):
    print(f'### {fold} label stats ###')
    print('mean         ', np.mean(ds['labels']))
    print('std          ', np.std(ds['labels']))
    print('median       ', np.median(ds['labels']))


def compute_labels(ds):
    d1 = np.clip(np.nan_to_num(np.array(ds['d1'])), 0, 200) / 200
    d2 = np.clip(np.nan_to_num(np.array(ds['d2'])), 0, 200) / 200
    d3 = np.clip(np.nan_to_num(np.array(ds['d3'])), 0, 200) / 200
    return list(d1 + d2 + d3)


def prepare_training_data(ds):
    assert len(ds.keys()) == 1
    ds = ds['train']
    print("Total dataset size:", len(ds))
    ds = train_utils.sample_by_id_with_max_limit(ds, 'user_id', 30)
    print("Total dataset size after filter:", len(ds))
    print("Split train test...")
    ds_train, ds_val = train_utils.train_test_split_by_group_id(
        ds,
        config.DATA_SIZE_TRAIN,
        config.DATA_SIZE_TEST)
    print("Prepare columns...")
    ds_train = train_utils.prepare_ds_columns(ds_train, compute_labels)
    ds_val = train_utils.prepare_ds_columns(ds_val, compute_labels)
    print("Data preparation finished")
    return ds_train, ds_val


def load_hf_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type='regression',
        num_labels=1)
    if model_name == 'gpt2':
        model.config.pad_token_id = 50256
    return model


def get_trainer_object(model):
    training_args = TrainingArguments(
        output_dir=config.OUTPUT,
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
        logging_first_step=True,
        seed=config.SEED,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUM_STEPS,
        dataloader_num_workers=12,
        dataloader_pin_memory=True,
        report_to='wandb',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
    )
    return trainer


set_seed(config.SEED)

if config.RANK == 0:
    init_wandb()

print('loading training dataset...')
ds = load_dataset(config.DATASET, use_auth_token=config.HF_AUTH_TOKEN)
print(ds)

train_set, val_set = prepare_training_data(ds)
print_label_stats(train_set, fold='train')
print_label_stats(val_set, fold='test')

print('initializing model...')
model = load_hf_model(config.MODEL)

print('configuring training args...')
trainer = get_trainer_object(model)

print('evaluation...')
trainer.evaluate()

print('training...')
trainer.train()

if config.HF_UPLOAD_PATH is not None:
    print('uploading to huggingface {}'.format(config.HF_UPLOAD_PATH))
    model.push_to_hub(config.HF_UPLOAD_PATH, use_auth_token=config.HF_AUTH_TOKEN)

if config.TORCH_SAVE_PATH is not None:
    print('saving torch model to {}'.format(config.TORCH_SAVE_PATH))
    model.eval()
    torch.save(model, config.TORCH_SAVE_PATH)
