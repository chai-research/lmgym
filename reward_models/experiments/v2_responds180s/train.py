from datasets import load_metric, load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, set_seed
import numpy as np
import wandb
import torch
import config

from reward_models.custom import train_utils, custom_callbacks
from reward_models.experiments.v2b_responds180s.data import _get_tokenizer

metric = load_metric("accuracy")
tokenizer = _get_tokenizer()


def init_wandb():
    wandb.login(key=config.WANDB_AUTH_TOKEN)
    name = f'{config.EXP_TAG}_data_{config.DATA_SIZE_TRAIN}'
    wandb.init(project='reward', name=name)


def print_label_stats(ds, fold):
    print(f'### {fold} label stats ###')
    print('mean         ', np.mean(ds['labels']))
    print('std          ', np.std(ds['labels']))
    print('median       ', np.median(ds['labels']))


def compute_labels(ds):
    response_time = np.array(ds['time_delta_till_user_response'])
    label = [0 if (v == -1 or v > 180) else 1 for v in response_time]
    return label


def prepare_training_data(ds):
    assert len(ds.keys()) == 1
    ds = ds['train']
    print("Total dataset size:", len(ds))
    print("Prepare columns...")
    ds = train_utils.add_label(ds, compute_labels)
    # ds = train_utils.sample_by_id_with_max_limit(ds, 'user_id', 30)
    # print("Total dataset size after filter:", len(ds))
    print("Split train test...")
    ds_train, ds_val = train_utils.train_test_split_by_group_id(
        ds,
        config.DATA_SIZE_TRAIN,
        config.DATA_SIZE_TEST)
    ds_train = train_utils.remove_unnecessary_columns(ds_train)
    ds_val = train_utils.remove_unnecessary_columns(ds_val)
    print("Data preparation finished")
    return ds_train, ds_val


def load_hf_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2)
    if model_name == 'gpt2':
        model.config.pad_token_id = 50256
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def preds_generator(model, dataset):
    tokenized = tokenizer(
            dataset['text'],
            return_attention_mask=True,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=256
            ).to(0)
    preds = torch.softmax(model(**tokenized).logits, axis=1).cpu().detach()[:, 1]
    return preds


def get_trainer_object(model):
    callbacks = [
        custom_callbacks.RetryContinueCallback(
            model=model,
            preds_generator=preds_generator,
            higher_is_better=True),
        custom_callbacks.HumanRankingCallback(
            model=model,
            preds_generator=preds_generator,
            higher_is_better=True),
        custom_callbacks.RejectWorstMessageCallback(
            model=model,
            preds_generator=preds_generator,
            higher_is_better=True),
        ]

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
        compute_metrics=compute_metrics,
        callbacks=callbacks,
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
