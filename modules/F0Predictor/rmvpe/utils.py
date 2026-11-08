import sys
from functools import reduce

import librosa
import numpy as np
import torch
from torch.nn.modules.module import _addindent

from .constants import *  # noqa: F403


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count

    
def to_local_average_cents(salience, center=None, thred=0.05):
    """Average the nine bins around each peak, with device-local tensors."""
    if salience.ndim not in (1, 2):
        raise ValueError("salience must be 1D or 2D")
    single = salience.ndim == 1
    values = salience[None] if single else salience
    values = values.float()
    bins = torch.arange(values.shape[-1], device=values.device)
    if center is None:
        center = values.argmax(dim=-1)
    else:
        center = torch.as_tensor(center, device=values.device).reshape(-1)
    weights = values * ((bins[None] - center[:, None]).abs() <= 4)
    mapping = 20 * bins + CONST  # noqa: F405
    cents = (weights * mapping).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-9)
    cents = torch.where(weights.max(dim=-1).values > thred, cents, 0)
    return cents[0] if single else cents


def to_viterbi_cents(salience, thred=0.05):
    bins = np.arange(N_CLASS)  # noqa: F405
    transition = np.maximum(30 - abs(bins[:, None] - bins[None, :]), 0).astype(float)
    transition /= transition.sum(axis=1, keepdims=True)
    prob = salience.float().T
    prob = prob / prob.sum(dim=0, keepdim=True).clamp_min(1e-9)
    path = librosa.sequence.viterbi(prob.detach().cpu().numpy(), transition).astype(np.int64)
    return to_local_average_cents(salience, center=path, thred=thred)
