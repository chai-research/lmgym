import torch

from transformers import Trainer


class SoftTargetTrainer(Trainer):
    """Train against probabilities. """

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        probs = inputs.pop('probs')

        outputs = model(**inputs)
        logits = outputs[0]

        loss = soft_binary_entropy_loss(logits, labels, probs)
        return (loss, outputs) if return_outputs else loss


def soft_binary_entropy_loss(logits, target_tokens, target_probs):
    """
    logits - (batch_size, num_tokens, vocab_size)
    target - (batch_size, num_tokens)
    probability - (batch_size, num_tokens)

    e.g we are saying the final token should be 7365 with probability 0.35
    so no shifting has occured by this point

    logits - tokenize(the cat in the hat)
    target - [-100, -100, -100, -100, 7365]
    probability - [-100, -100, -100, -100, 0.35]
    """
    batch_size, num_tokens, vocab_size = logits.shape
    assert target_tokens.shape == torch.Size([batch_size, num_tokens])
    assert target_probs.shape == torch.Size([batch_size, num_tokens])

    logits, target_tokens, target_probs = _shift_and_flatten_inputs(
        logits,
        target_tokens,
        target_probs
    )

    assert logits.shape == torch.Size([batch_size * (num_tokens - 1), vocab_size])
    assert target_tokens.shape == torch.Size([1, batch_size * (num_tokens - 1)])
    assert target_probs.shape == torch.Size([1, batch_size * (num_tokens - 1)])

    train_probs = torch.softmax(logits, dim=-1)

    train_probs, target_tokens, target_probs = _filter_disabled_tokens(
        train_probs,
        target_tokens,
        target_probs
    )

    assert train_probs.shape[-1] == vocab_size
    num_training_examples, _ = train_probs.shape
    assert target_tokens.shape == torch.Size([num_training_examples])
    assert target_probs.shape == torch.Size([num_training_examples])

    train_token_probs = _select_probability_by_index(train_probs, target_tokens)
    assert train_token_probs.shape == torch.Size([num_training_examples])

    # identical results to BCELoss but works with half precision
    # loss = torch.nn.BCELoss()(train_token_probs, target_probs)
    # clip to -100 like BCELoss to prevent -inf losses
    loss = - target_probs * torch.clip(torch.log(train_token_probs), min=-100)
    loss = loss - (1 - target_probs) * torch.clip(torch.log(1 - train_token_probs), min=-100)

    return loss.mean()


def _select_probability_by_index(output_probs, target_tokens):
    target_tokens = target_tokens.unsqueeze(dim=-1)
    num_output_probs = output_probs.shape[0]
    assert target_tokens.shape == torch.Size([num_output_probs, 1])
    probs = torch.gather(output_probs, -1, target_tokens).squeeze(dim=1)
    return probs


def _filter_disabled_tokens(output_probs, target, probs):
    # -100 is used by pytorch to indicate to not train on this token
    ignore_ix = target != -100
    length = target.shape[-1]
    target = target.masked_select(ignore_ix)
    probs = probs.masked_select(ignore_ix)
    assert ignore_ix.shape == torch.Size([1, length])
    output_probs = output_probs[ignore_ix[0]]
    return output_probs, target, probs


def _shift_and_flatten_inputs(logits, target, probability):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()
    shift_probs = probability[..., 1:].contiguous()

    # Flatten the tokens
    flattened_logits = shift_logits.view(-1, shift_logits.size(-1))
    flattened_labels = shift_labels.view(1, -1)
    flattened_probs = shift_probs.view(1, -1)
    return flattened_logits, flattened_labels, flattened_probs
