import torch
import numpy as np

from copy import deepcopy


def tokenize_function(examples, tokenizer, block_size, train_to_probs):
    input_texts = examples['input_text']
    output_texts = examples['output_text']
    data = [input_ + output_ for input_, output_ in zip(input_texts, output_texts)]

    inputs = tokenizer(
        data,
        padding="max_length",
        max_length=block_size,
        truncation=True,
        return_token_type_ids=False
    )

    inputs["labels"] = deepcopy(inputs.input_ids)
    batch_size = len(inputs["labels"])

    inputs = disable_input_text_tokens(tokenizer, output_texts, inputs, batch_size)

    if train_to_probs:
        inputs = add_token_probabilities(inputs, examples, batch_size)

    return inputs


def add_token_probabilities(inputs, examples, batch_size):
    """When we want to train to soft targets. """
    disabled_grid = torch.tensor(-100.).expand_as(torch.tensor(inputs['labels']))
    inputs["probs"] = disabled_grid.tolist()

    for batch in range(batch_size):
        tokens, logprobs = examples['tokens'][batch], examples['logprobs'][batch]
        assert len(tokens) == len(logprobs)

        for token in range(0, len(tokens)):
            # rare: sometimes the first input token can get joined with the input text
            # e.g something\nMe:\"reply" the : and the " get combined into a single token
            if not tokens[-token - 1] == inputs['labels'][batch][-token - 1]:
                print('probability tokens do not match output text')

            inputs["probs"][batch][-token - 1] = np.exp(logprobs[-token - 1])

    return inputs


def disable_input_text_tokens(tokenizer, output_texts, inputs, batch_size):
    """We only want to train on the output_text tokens so set others to -100. """
    output_lengths = [len(tokenizer(output_string).input_ids) for output_string in output_texts]

    for batch in range(batch_size):
        num_input_tokens = len(inputs['labels'][batch]) - output_lengths[batch]
        for token in range(0, num_input_tokens):
            inputs["labels"][batch][token] = -100

    return inputs
