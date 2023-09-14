"""
Based on https://github.com/bfs18/tacotron2
"""
import itertools

import torch


UPDATE_GAF_EVERY_N_STEP = 10


def grads_for_params(loss, parameters, optimizer):
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    grads = []
    for p in parameters:
        grads.append(p.grad.detach().clone())
    optimizer.zero_grad()
    return grads


def calc_grad_norm(grads, method='max'):
    if method == 'max':
        return torch.stack([torch.max(torch.abs(g)) for g in grads])
    elif method == 'l1':
        return torch.stack([torch.sum(torch.abs(g)) for g in grads])
    else:
        raise ValueError('Unsupported method [{}]'.format(method))


def calc_grad_adapt_factor(loss1, loss2, parameters, optimizer):
    # return a factor for loss2 to make the greatest gradients
    # for loss1 and loss2 in similar scale.
    parameters, parameters_backup = itertools.tee(parameters)
    grads1 = grads_for_params(loss1, parameters, optimizer)
    grads2 = grads_for_params(loss2, parameters_backup, optimizer)

    norms1 = calc_grad_norm(grads1)
    norms2 = calc_grad_norm(grads2)

    indices = (norms1 != 0) & (norms2 != 0)
    norms1 = norms1[indices]
    norms2 = norms2[indices]

    return torch.min(norms1 / norms2)
    # return torch.mean(norms1 / norms2)