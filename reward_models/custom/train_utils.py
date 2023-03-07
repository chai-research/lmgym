from tqdm import tqdm
import numpy as np
import wandb

from reward_models.utils import get_timestamp
from reward_models.config import WANDB_AUTH_TOKEN


def init_wandb(project, train_config):
    name = f'{train_config.EXP_NAME}_{get_timestamp()}'
    wandb.login(key=WANDB_AUTH_TOKEN)
    wandb.init(project=project, name=name)


def add_label(ds, compute_label_function):
    return ds.add_column('labels', compute_label_function(ds))


def train_test_split_by_group_id(ds, train_size, val_size, seed=0, group_id='user_id'):
    assert len(ds) >= (train_size + val_size), \
        f'Data size ({train_size + val_size}) exceeds available data ({len(ds)})'
    sorting = np.argsort(ds[group_id])
    ds_train = ds.select(sorting[:train_size]).shuffle(seed=seed)
    ds_val = ds.select(sorting[train_size:train_size+val_size])
    return ds_train, ds_val


def remove_unnecessary_columns(ds):
    remaining_cols = [
        col for col in ds.column_names if col not in
        ['input_ids', 'attention_mask', 'labels']
    ]
    ds = ds.remove_columns(remaining_cols)
    return ds


def get_sorted_first_n_occurences_mask(vals, n):
    mask = np.full(len(vals), True)
    counter = 1
    for i in tqdm(range(1, len(vals))):
        counter = counter + 1 if vals[i] == vals[i-1] else 1
        mask[i] = counter <= n
    return mask


def sample_by_id_with_max_limit(ds, group, limit, mode='unchanged', seed=0):
    sample_ixs = get_sample_ixs_by_id_with_max_limit(ds, group, limit, mode, seed)
    return ds.select(sample_ixs)


def get_sample_ixs_by_id_with_max_limit(ds, group, limit, mode='unchanged', seed=0):
    assert mode in {'unchanged', 'random', 'reverse'}, 'unsupported mode'
    group_ids = np.array(ds[group])

    reordered_indices = get_reordered_indices(group_ids, mode, seed)
    reordered_group_ids = group_ids[reordered_indices]

    sorted_indices = np.argsort(reordered_group_ids, kind='stable')
    ordered_group_ids = reordered_group_ids[sorted_indices]

    mask = get_sorted_first_n_occurences_mask(ordered_group_ids, n=limit)
    mask = mask[np.argsort(sorted_indices)]
    mask = mask[np.argsort(reordered_indices)]
    sample_ixs = np.arange(len(mask))[mask]
    return sample_ixs


def get_reordered_indices(arr, mode, seed):
    assert mode in {'unchanged', 'random', 'reverse'}, 'unsupported mode'
    indices = np.arange(len(arr))
    if mode == 'random':
        ordered_indices = _random_shuffle_with_seed(indices, seed)
    elif mode == 'reverse':
        ordered_indices = indices[::-1]
    else:
        ordered_indices = indices
    return ordered_indices


def _random_shuffle_with_seed(arr, seed):
    with ss.np.set_temp_seed(seed):
        out = np.random.permutation(arr)
    return out


def get_ordered_sample_data(ds, n_samples, seed=0):
    ordered_sample_ixs = get_ordered_sample_ixs(ds, n_samples, seed)
    return ds.select(ordered_sample_ixs)


def get_ordered_sample_ixs(ds, n_samples, seed=0):
    n_rows = ds.num_rows
    n_samples = min(n_samples, n_rows)
    shuffle_ixs = _random_shuffle_with_seed(np.arange(n_rows), seed)
    ordered_sample_ixs = sorted(shuffle_ixs[:n_samples])
    return ordered_sample_ixs


def print_label_stats(ds):
    print(ds)
    labels = ds['labels']
    print('mean         ', np.mean(labels))
    print('median       ', np.median(labels))
    print('10 percentile', np.percentile(labels, 10))
    print('90 percentile', np.percentile(labels, 90))


def format_load_and_save_path_from_checkpoint(train_config):
    load_path = train_config.MODEL
    save_path = train_config.OUTPUT
    checkpoint = getattr(train_config, 'CHECKPOINT', None)
    if checkpoint is not None:
        load_path = f'{train_config.OUTPUT}/checkpoint-{checkpoint}'
        save_path = f'{train_config.OUTPUT}_start_cp_{checkpoint}'
    print(f'load model from {load_path}')
    print(f'save model to {load_path}')
    return load_path, save_path
