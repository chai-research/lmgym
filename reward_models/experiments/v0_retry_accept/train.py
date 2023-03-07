import os
from datasets import load_metric, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, set_seed
import numpy as np
import wandb
import torch

from reward_models.custom import custom_callbacks as cb
from reward_models.config import HF_TOKEN


RANK = int(os.environ.get('LOCAL_RANK', 0))
print('running process rank {}'.format(RANK))


if RANK == 0:
    wandb.login(key='95713a8419108e9246736f20a564e81559a8e80f')

SEED = 100
print('using seed:', SEED)

set_seed(SEED)

DATA = int(os.environ['DATA'])
print('using data size:', DATA)

HF_UPLOAD_PATH = os.environ.get('HF_UPLOAD')
if HF_UPLOAD_PATH is not None:
    print('will upload to huggingface {}'.format(HF_UPLOAD_PATH))

TORCH_SAVE_PATH = os.environ.get('TORCH_SAVE_PATH')
if TORCH_SAVE_PATH is not None:
    print('will save torch model to path {}'.format(TORCH_SAVE_PATH))

MAX_LENGTH = 256
print('using max length', MAX_LENGTH)

if RANK == 0:
    wandb.init(
        project='reward',
        name='gpt2_data_seed_{}_data_{}'.format(SEED, DATA)
    )

MODEL = 'gpt2'
OUTPUT = '/tmp/reward_models_{}_{}'.format(SEED, DATA)
LOG_PATH = os.path.join(OUTPUT, 'callback_logs.json')

K = 2

TRAIN_EPOCHS = 2

# twice per epoch
BATCH_SIZE = 64
EVAL_STEPS = int(DATA / (BATCH_SIZE * 20))
print('evaluation steps:', EVAL_STEPS)


# MODEL #

print('loading model...')
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# DATASET #

def compute_label(dataset):
    # target is did the user not retry (high score = good) AND continued
    rows = zip(dataset['accepted'], dataset['remaining_user_messages'])
    return [int(accepted == 1) and int(remain >= K) for accepted, remain in rows]


tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    truncation_side='left',
    padding_side='right'  # for GPT2 padding side is always right
)


tokenizer.pad_token_id = ['<|endoftext|>']
assert tokenizer.pad_token_id == 50256

# training requires
model.config.pad_token_id = tokenizer.pad_token_id


def tokenize_function(examples):
    tokens = tokenizer(
        examples["input_ids"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
    return tokens


# SEPERATION DATASET #2: LARGE SAMPLE + DIFFERENT MODEL INPUTS + LABELLED

print('creating main training dataset...')
ds = load_dataset(
    'ChaiML/retry_target_50m',
    use_auth_token=HF_TOKEN
)


def clean_train_data(ds):
    for key in ds.keys():
        print('processing', key)
        ds[key] = ds[key].add_column('labels', compute_label(ds[key]))
        ds[key] = ds[key].remove_columns([
            'accepted', 'remaining_user_messages', 'conversation_length'
        ])
    return ds


tokenized = clean_train_data(ds)


print('mean label', np.mean(ds['train']['labels']))


# CREATE TRAINING DATASETS

train_set = tokenized['train'].shuffle(seed=SEED).select(range(DATA))
val_set = tokenized['test'].shuffle(seed=100).select(range(10000))

# TRAINING ARGUMENTS #
print('configuring arguments and metrics...')
training_args = TrainingArguments(
    output_dir=OUTPUT,
    evaluation_strategy='steps',
    eval_steps=EVAL_STEPS,
    learning_rate=1e-5,
    num_train_epochs=TRAIN_EPOCHS,
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=EVAL_STEPS,
    fp16=True,
    optim='adamw_hf',
    logging_first_step=True,
    seed=SEED,
    per_device_train_batch_size=16,
    dataloader_num_workers=12,
    dataloader_pin_memory=True,
    report_to='wandb'
)


# METRICS #

accuracy = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


def generate_softmax_preds(model, dataset):
    tokenized = tokenizer(
            dataset['text'],
            return_tensors='pt',
            return_attention_mask=True,
            padding='longest',
            truncation=True,
            max_length=256
            ).to(0)
    preds = torch.softmax(model(**tokenized).logits, axis=1).cpu().detach()[:, 1]
    return preds


CALLBACKS = [
        cb.HumanRankingCallback(model, generate_softmax_preds, higher_is_better=True),
        cb.RejectWorstMessageCallback(model, generate_softmax_preds, higher_is_better=True),
        cb.RetryContinueCallback(model, generate_softmax_preds, higher_is_better=True),
        ]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    compute_metrics=compute_metrics,
    callbacks=CALLBACKS,
)

print('evaluation...')
trainer.evaluate()

print('training...')
trainer.train()

if HF_UPLOAD_PATH is not None:
    print('uploading to huggingface {}'.format(HF_UPLOAD_PATH))
    model.push_to_hub(HF_UPLOAD_PATH, use_auth_token=HF_TOKEN)

if TORCH_SAVE_PATH is not None:
    print('saving torch model to {}'.format(TORCH_SAVE_PATH))
    model.eval()
    torch.save(model, TORCH_SAVE_PATH)
