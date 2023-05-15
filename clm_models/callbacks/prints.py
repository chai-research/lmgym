"""
Evaluate a model against a fixed set of prompts to get a sense of its performance
during training
"""

from transformers import TrainerCallback
import torch
import logging

logger = logging.getLogger(__name__)


from contextlib import contextmanager


import time


@contextmanager
def timer(key):
    start = time.time()
    yield
    duration = time.time() - start
    print('{} took {}s'.format(key, duration))


class PrintDebugCallback(TrainerCallback):
    """Log responses to fixed questions to wandb. """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        print('starting debug evaluation callback')
        model = kwargs.get('model')

        print('expected', 0.35)
        print_prob('the cat in the', ' bat', self.tokenizer, model)

        print('expected', 0.4, 0.5)
        print_prob('the mouse in the', ' house', self.tokenizer, model)
        print_prob('the mouse in the house', ' test', self.tokenizer, model)

        print('expected', 0.17)
        print_prob('the cat in the', ' test', self.tokenizer, model)

        print('expected', 0.24)
        print_prob('the cat', ' test', self.tokenizer, model)


@torch.no_grad()
def print_prob(text, target, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt')['input_ids'].to(model.device)
    next_logits = model(tokens).logits
    probs = torch.softmax(next_logits, dim=-1)[0][-1]
    token = tokenizer(target)['input_ids'][0]
    print(text)
    print(target, 100 * probs[token].item())
    print()
