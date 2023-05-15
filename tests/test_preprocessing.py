from datasets import Dataset, DatasetDict

from clm_models.data_processing.filter_user_edits import (
    filter_users_edits,
)
from clm_models.data_processing.deduplication import Deduplicator
from clm_models.data_processing.user_sampling import (
    sample_dataset_by_users,
)
from clm_models.data_processing.prepare_edit_dataset import (
    prepare_edit_dataset,
)


def test_filter_users_edits():
    num_good_rows = 10
    num_bad_rows = 11
    edited_responses = ["a"] * num_good_rows
    edited_responses += ["a" * 1000] * num_bad_rows

    dataset_dict = {"edited_response": edited_responses}
    dataset = DatasetDict({"train": Dataset.from_dict(dataset_dict)})
    dataset = filter_users_edits(dataset)
    assert len(dataset["train"]) == num_good_rows


def test_sample_dataset_by_users():
    num_rows = 10
    dataset_dict = {
        "user_id": [0] * num_rows,
        "model_input": [""] * num_rows,
        "response": [""] * num_rows,
        "edited_response": [""] * num_rows,
    }

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(dataset_dict),
            "validation": Dataset.from_dict(dataset_dict),
        }
    )
    dataset = sample_dataset_by_users(dataset)
    assert (
        sum([len(dataset[dataset_split_name]) for dataset_split_name in dataset.keys()])
        == 2
    )


def test_prepare_edit_dataset():
    num_rows = 1
    dataset_dict = {
        "user_id": [0] * num_rows,
        "prompt": [""] * num_rows,
        "rejected": ["rejected"] * num_rows,
        "chosen": ["chosen"] * num_rows,
        "message_id": [0] * num_rows,
    }

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(dataset_dict),
        }
    )
    dataset = prepare_edit_dataset(dataset)
    assert set(dataset["train"].column_names) == {
        "input_text",
        "output_text",
        "user_id",
    }


def test_long_deduplicate_dataset():
    num_short_rows = 10
    num_long_rows = 10
    sample_short_text = "My name is Aleksey" * 10
    sample_long_text = "My name is Aleksey Korshuk" * 10

    edited_responses = [sample_short_text * 10] * num_short_rows
    edited_responses += [sample_long_text * 10] * num_long_rows

    dataset_dict = {"edited_response": edited_responses}
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(dataset_dict),
        }
    )
    deduplicator = Deduplicator()
    deduplicated_dataset = deduplicator.run(dataset)
    assert (
        len(deduplicated_dataset["train"]) == 2
    ), f'Recieved length {len(deduplicated_dataset["train"])}'


def test_short_deduplicate_dataset():
    num_short_rows = 10
    num_long_rows = 10
    sample_short_text = "My name is Aleksey"
    sample_long_text = "My name is Aleksey Korshuk"

    edited_responses = [sample_short_text * 10] * num_short_rows
    edited_responses += [sample_long_text * 10] * num_long_rows

    dataset_dict = {"edited_response": edited_responses}
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(dataset_dict),
        }
    )
    deduplicator = Deduplicator(min_num_tokens=50)
    deduplicated_dataset = deduplicator.run(dataset)
    assert (
        len(deduplicated_dataset["train"]) == num_short_rows + num_long_rows
    ), f'Recieved length {len(deduplicated_dataset["train"])}'


def test_jaccard_threshold_deduplicate_dataset():
    num_short_rows = 10
    num_long_rows = 10
    sample_short_text = "My name is Aleksey"
    sample_long_text = "My name is Aleksey Korshuk"

    edited_responses = [sample_short_text * 10] * num_short_rows
    edited_responses += [sample_long_text * 10] * num_long_rows

    dataset_dict = {"edited_response": edited_responses}
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(dataset_dict),
        }
    )
    deduplicator = Deduplicator(jaccard_threshold=0)
    deduplicated_dataset = deduplicator.run(dataset)
    assert (
        len(deduplicated_dataset["train"]) == 1
    ), f'Recieved length {len(deduplicated_dataset["train"])}'
