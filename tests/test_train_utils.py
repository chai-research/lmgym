import numpy as np
import pandas as pd
import datasets
import pytest
from reward_models.custom import train_utils as train_utils


def test_remove_unnecessary_columns_remove_all_but_three_colums(ds):
    ds = ds.add_column('labels', [1] * len(ds))
    ds = train_utils.remove_unnecessary_columns(ds)
    keep_columns = {'input_ids', 'attention_mask', 'labels'}
    assert len(set(ds.column_names) ^ keep_columns) == 0


def test_get_reordered_indices():
    with pytest.raises(AssertionError) as e:
        train_utils.get_reordered_indices(np.arange(10), mode='abc', seed=0)
    assert 'unsupported mode' in str(e)


def test_sample_by_id_with_max_limit_raises_on_incorrect_mode(ds):
    with pytest.raises(AssertionError) as e:
        train_utils.sample_by_id_with_max_limit(ds, 'user_id', 2, mode='abc')
    assert 'unsupported mode' in str(e)


def test_get_reordered_indices_random_seed():
    arr = np.random.randn(100)
    result1 = train_utils.get_reordered_indices(arr, mode='random', seed=0)
    result2 = train_utils.get_reordered_indices(arr, mode='random', seed=0)
    result3 = train_utils.get_reordered_indices(arr, mode='random', seed=1)
    assert all(result1 == result2)
    assert not all(result1 == result3)


def test_sample_by_id_with_max_limit_mode_unchanged(ds):
    output_ds = train_utils.sample_by_id_with_max_limit(ds, 'user_id', 2, mode='unchanged')
    expected_df = pd.DataFrame({
        'user_id': [0, 1, 1, 0, 2, 8, 2],
        'values':  [0, 1, 2, 3, 4, 5, 7]})
    expected_ds = datasets.Dataset.from_pandas(expected_df)
    assert np.all(output_ds['user_id'] == expected_ds['user_id'])
    assert np.all(output_ds['values'] == expected_ds['values'])


def test_sample_by_id_with_max_limit_mode_reverse(ds):
    output_ds = train_utils.sample_by_id_with_max_limit(ds, 'user_id', 2, mode='reverse')
    expected_df = pd.DataFrame({
        'user_id': [8, 0, 2, 2, 1, 0, 1],
        'values':  [5, 9, 1, 2, 3, 4, 5]})
    expected_ds = datasets.Dataset.from_pandas(expected_df)
    assert np.all(output_ds['user_id'] == expected_ds['user_id'])
    assert np.all(output_ds['values'] == expected_ds['values'])


def test_train_test_split_by_group_id(ds):
    ds_train, ds_val = train_utils.train_test_split_by_group_id(ds, train_size=8, val_size=2)
    assert len(set(ds_train['user_id']) & set(ds_val['user_id'])) <= 1
    assert len(ds_train) == 8
    assert len(ds_val) == 2


def test_get_ordered_sample_data(ds):
    ds = ds.add_column('range', np.arange(ds.num_rows))
    new_ds = train_utils.get_ordered_sample_data(ds, 5, seed=1)
    order = new_ds['range']
    assert (np.diff(order) > 0).all()

    new_ds_seed_1 = train_utils.get_ordered_sample_data(ds, 5, seed=1)
    order_seed_1 = new_ds_seed_1['range']
    assert np.equal(order, order_seed_1).all()

    new_ds_seed_2 = train_utils.get_ordered_sample_data(ds, 5, seed=2)
    order_seed_2 = new_ds_seed_2['range']
    assert not np.equal(order, order_seed_2).all()

    new_ds_2 = train_utils.get_ordered_sample_data(ds, 1000)
    order_2 = new_ds_2['range']
    assert np.equal(ds['range'], order_2).all()


@pytest.fixture
def ds():
    sample_df = pd.DataFrame({
        'user_id':          [0, 1, 1, 0, 2, 8, 1, 2, 1, 0, 1, 2, 2, 1, 0, 1],
        'values':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
        'input_ids':        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
        'attention_mask':   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    sample_ds = datasets.Dataset.from_pandas(sample_df)
    return sample_ds
