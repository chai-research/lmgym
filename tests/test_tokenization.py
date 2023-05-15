import numpy as np
import datasets
from transformers import AutoTokenizer

from clm_models.custom.tokenization import tokenize_function


def load_test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        'gpt2',
        padding_side='left',
        truncation_side='left'
    )
    try:
        tokenizer.pad_token_id = ['<|endoftext|>']
    except ValueError:
        # newer versions assign input_id
        tokenizer.pad_token_id = 50256
    assert tokenizer.pad_token_id == 50256
    return tokenizer


def test_sets_input_text_labels_to_disabled():
    tokenizer = load_test_tokenizer()
    examples = datasets.arrow_dataset.Batch({
        'input_text': [
            'the cat in',
            'my mouse in the house is',
            'what'
        ],
        'output_text': [
            ' the hat',
            ' hungry.',
            ' is the problem over here'
        ]
    })

    tokenized = tokenize_function(
        examples,
        tokenizer,
        block_size=7,  # chosen to trigger both padding AND truncation
        train_to_probs=False
    )

    assert tokenized['input_ids'] == [
        [50256, 50256, 1169, 3797, 287, 262, 6877],
        [10211, 287, 262, 2156, 318, 14720, 13],
        [50256, 10919, 318, 262, 1917, 625, 994]
    ]

    assert tokenized['attention_mask'] == [
        [0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1],
    ]

    assert tokenized['labels'] == [
        [-100, -100, -100, -100, -100, 262, 6877],
        [-100, -100, -100, -100, -100, 14720, 13],
        [-100, -100, 318, 262, 1917, 625, 994]
    ]

    assert set(tokenized.keys()) == {'input_ids', 'attention_mask', 'labels'}


def test_assigns_probabilities_to_input_data():
    tokenizer = load_test_tokenizer()

    examples = datasets.arrow_dataset.Batch({
        'input_text': [
            'the cat in',
            'my mouse in the house is',
            'what'
        ],
        'output_text': [
            ' the hat',
            ' hungry.',
            ' is the problem over here'
        ],
        'tokens': [
            [262, 6877],
            [14720, 13],
            [318, 262, 1917, 625, 994]
        ],
        'logprobs': [
            [-1.2, -2.3],
            [-0.123, -1.4],
            [-4.1, -0.5, -0.3, -0.111, -1.57]
        ]
    })

    tokenized = tokenize_function(
        examples,
        tokenizer,
        block_size=7,  # chosen to trigger both padding AND truncation
        train_to_probs=True
    )

    expected = [
        [-100, -100, -100, -100, -100, 0.3011942, 0.100258],
        [-100, -100, -100, -100, -100, 0.88426, 0.2465969],
        [-100, -100, 0.01657267, 0.60653, 0.740818, 0.894938, 0.208045]
    ]
    assert np.isclose(tokenized['probs'], expected).all()

    assert tokenized['input_ids'] == [
        [50256, 50256, 1169, 3797, 287, 262, 6877],
        [10211, 287, 262, 2156, 318, 14720, 13],
        [50256, 10919, 318, 262, 1917, 625, 994]
    ]

    assert tokenized['attention_mask'] == [
        [0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1],
    ]

    assert tokenized['labels'] == [
        [-100, -100, -100, -100, -100, 262, 6877],
        [-100, -100, -100, -100, -100, 14720, 13],
        [-100, -100, 318, 262, 1917, 625, 994]
    ]

    assert set(tokenized.keys()) == {'input_ids', 'attention_mask', 'labels', 'probs'}
