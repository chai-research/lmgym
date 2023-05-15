import torch

from clm_models.custom.losses import (
    _select_probability_by_index,
    _filter_disabled_tokens,
    soft_binary_entropy_loss,
    _shift_and_flatten_inputs
)


def test_select_probability_by_index():
    probs = torch.tensor([
        [0.1, 0.5, 0.2, 0.2],
        [0.1, 0.1, 0.1, 0.7],
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.2, 0.3, 0.3],
    ])
    assert probs.shape == torch.Size([4, 4])

    target_tokens = torch.tensor([1, 3, 2, 3])

    train_probs = _select_probability_by_index(probs, target_tokens)

    expected = torch.tensor([0.5, 0.7, 0.3, 0.3])
    assert torch.all(train_probs == expected)


def test_select_probability_by_index_handles_one_value():
    probs = torch.tensor([[0.9707, 0.0293]])
    assert probs.shape == torch.Size([1, 2])

    target_tokens = torch.tensor([1])

    train_probs = _select_probability_by_index(probs, target_tokens)

    # previous was returning single float value
    assert train_probs.shape == torch.Size([1])

    expected = torch.tensor([0.0293])
    assert torch.all(train_probs == expected)


def test_filter_disabled_tokens():
    probs = torch.tensor([
        [0.1, 0.5, 0.2, 0.2],
        [0.1, 0.1, 0.1, 0.7],
        [0.3, 0.2, 0.3, 0.2],
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.2, 0.3, 0.3],
        [0.1, 0.4, 0.4, 0.1],
    ])
    assert probs.shape == torch.Size([6, 4])

    target = torch.tensor([[-100, 3, -100, -100, 1, 2]])
    assert target.shape == torch.Size([1, 6])

    target_probs = torch.tensor([[-100, 0.2, -100, -100, 0.1, 0.1]])
    assert target_probs.shape == torch.Size([1, 6])

    res_probs, res_target, res_target_probs = _filter_disabled_tokens(
        probs, target, target_probs)

    expected = torch.tensor([3, 1, 2])
    assert torch.all(res_target == expected)

    expected = torch.tensor([0.2, 0.1, 0.1])
    assert torch.all(res_target_probs == expected)

    expected = torch.tensor([
        [0.1, 0.1, 0.1, 0.7],
        [0.2, 0.2, 0.3, 0.3],
        [0.1, 0.4, 0.4, 0.1],
    ])
    assert torch.all(res_probs == expected)


def test_one_dimensional_one_probability_regression_test():
    logits = torch.tensor([[[-10., -13.], [-6.3, -9.8], [-13, 1.2]]])
    targets = torch.tensor([[-100, -100, 1]])
    probs = torch.tensor([[-100, -100, 0.4]])

    loss = soft_binary_entropy_loss(logits, targets, probs)
    assert torch.isclose(loss, torch.tensor(1.42975)).item()


def test_one_dimensional_multiple_probability_regression_test():
    logits = torch.tensor([[[-10., -13.], [-6.3, -9.8], [-13, 1.2]]])
    targets = torch.tensor([[1, 0, 1]])
    probs = torch.tensor([[0.3, 0.7, 0.4]])

    loss = soft_binary_entropy_loss(logits, targets, probs)
    assert torch.isclose(loss, torch.tensor(1.189168)).item()


def test_batch_one_probability_regression_test():
    logits = torch.tensor([
        [[-10., -13.], [-6.3, -9.8], [-13, 1.2]],
        [[-3., 2.1], [1.1, 0.3], [-2.31, 0.324]],
    ])

    targets = torch.tensor([[-100, -100, 0], [-100, -100, 1]])
    probs = torch.tensor([[-100, -100, 0.2], [-100, -100, 0.7]])

    assert logits.shape == torch.Size([2, 3, 2])
    assert targets.shape == torch.Size([2, 3])
    assert probs.shape == torch.Size([2, 3])

    loss = soft_binary_entropy_loss(logits, targets, probs)
    assert torch.isclose(loss, torch.tensor(1.8804252)).item()


def test_handles_extreme_probabilities():
    logits = torch.tensor([
        [[-10., -13.], [-6.3, -9.8], [-13, 1.2]],
        [[-3., 2.1], [1.1, 0.3], [-2.31, 0.324]],
    ])

    targets = torch.tensor([[-100, -100, 0], [-100, -100, 1]])
    probs = torch.tensor([[-100, -100, 0.0], [-100, -100, 1.0]])

    assert logits.shape == torch.Size([2, 3, 2])
    assert targets.shape == torch.Size([2, 3])
    assert probs.shape == torch.Size([2, 3])

    loss = soft_binary_entropy_loss(logits, targets, probs)
    assert torch.isclose(loss, torch.tensor(2.3504252)).item()


def test_handles_extreme_logits():
    # previously would do torch.log(0) => nan
    logits = torch.tensor([
        [[-10., -13.], [-100., 40.3], [-13, -13.5]],
        [[-3., 2.1], [1.1, 0.3], [-2.31, 0.324]],
    ])

    targets = torch.tensor([[-100, -100, 0], [-100, -100, 1]])
    probs = torch.tensor([[-100, -100, 0.3], [-100, -100, 0.7]])

    assert logits.shape == torch.Size([2, 3, 2])
    assert targets.shape == torch.Size([2, 3])
    assert probs.shape == torch.Size([2, 3])

    loss = soft_binary_entropy_loss(logits, targets, probs)
    assert torch.isclose(loss, torch.tensor(15.46555137)).item()


def test_batch_two_probabilities_regression_test():
    logits = torch.tensor([
        [[-10., -13.], [-6.3, -9.8], [-13, 1.2]],
        [[-3., 2.1], [1.1, 0.3], [-2.31, 0.324]],
    ])

    targets = torch.tensor([[-100, 1, 0], [0, 1, 1]])
    probs = torch.tensor([[-100, 0.3, 0.2], [0.12, 0.83, 0.7]])

    loss = soft_binary_entropy_loss(logits, targets, probs)
    assert torch.isclose(loss, torch.tensor(1.39562)).item()


def test_shift_and_flatten_inputs():
    logits = torch.tensor([
        [[-10., -13.], [-6.3, -9.8], [-13, 1.2]],
        [[-3., 2.1], [1.1, 0.3], [-2.31, 0.324]],
    ])
    assert logits.shape == torch.Size([2, 3, 2])

    targets = torch.tensor([[-100, 1, 0], [0, 1, 1]])
    assert targets.shape == torch.Size([2, 3])

    probs = torch.tensor([[-100, 0.3, 0.2], [0.12, 0.83, 0.7]])
    assert targets.shape == torch.Size([2, 3])

    logits, target, probs = _shift_and_flatten_inputs(logits, targets, probs)

    expected = torch.tensor([
        [-10., -13.], [-6.3, -9.8], [-3., 2.1], [1.1, 0.3],
    ])
    assert torch.all(expected == logits)

    expected = torch.tensor([[1, 0, 1, 1]])
    assert torch.all(expected == target)

    expected = torch.tensor([[0.3, 0.2, 0.83, 0.7]])
    assert torch.all(expected == probs)
