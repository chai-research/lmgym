import os
import torch

EXP_TAG = 'v5a_convo_d123r_joint_6_labels'

# GPU
RANK = int(os.environ.get('LOCAL_RANK', 0))
GPU_COUNT = torch.cuda.device_count()

# DATA
DATASET = 'ChaiML/log_norm_time_delta_sample_rate_50'
DATA_SIZE_TRAIN = int(os.environ.get('DATA'))
DATA_SIZE_TEST = 10000
SEED_TRAIN = 42
SEED_TEST = 42
MAX_LENGTH = 256
SHUFFLE_BEFORE_SPLIT = True

# MODEL
MODEL = 'gpt2'
NUM_LABELS = 6
TRAIN_EPOCHS = 4
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
GRADIENT_ACCUM_STEPS = 1
EVAL_STEPS = int(DATA_SIZE_TRAIN / (BATCH_SIZE * 10))
PER_DEVICE_TRAIN_BATCH_SIZE = BATCH_SIZE // GRADIENT_ACCUM_STEPS // GPU_COUNT

# PATH
HF_UPLOAD_PATH = os.environ.get('HF_UPLOAD')
TORCH_SAVE_PATH = os.environ.get('TORCH_SAVE_PATH')
EXP_NAME = f'reward_models_{EXP_TAG}_{SEED_TRAIN}_{DATA_SIZE_TRAIN}'
OUTPUT = f'/models/checkpoints/{EXP_NAME}'
LOG_PATH = os.path.join(OUTPUT, 'callback_logs.json')

# CHECKPOINT
CHECKPOINT = None
