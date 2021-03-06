import math
import torch
from torch.optim.optimizer import Optimizer


class op_Adam_lop_Adam(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hypergrad_lr (float, optional): hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        `Online Learning Rate Adaptation with Hypergradient Descent`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Online Learning Rate Adaptation with Hypergradient Descent:
        https://openreview.net/forum?id=BkrsAzWAb
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), lr_betas=(0.9, 0.999), lr_eps=1e-8, eps=1e-8,
                 weight_decay=0, hypergrad_lr=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps, lr_betas=lr_betas, lr_eps=lr_eps,
                        weight_decay=weight_decay, hypergrad_lr=hypergrad_lr)
        super(op_Adam_lop_Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'op_Adam_lop_Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Exponential moving average of hypergradient values
                    state['exp_avg_h'] = grad.new_tensor(0)
                    # Exponential moving average of squared hypergradient values
                    state['exp_avg_h_sq'] = grad.new_tensor(0)

                # References and beta1, beta2 coefficients for Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # References and beta1_h, beta2_h coefficients, in Hypergradient Adam (HD Adam) for the learning rate
                exp_avg_h, exp_avg_h_sq = state['exp_avg_h'], state['exp_avg_h_sq']
                beta1_h, beta2_h = group['lr_betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                if state['step'] > 1:
                    prev_bias_correction1 = 1 - beta1 ** (state['step'] - 1)
                    prev_bias_correction2 = 1 - beta2 ** (state['step'] - 1)

                    # Hypergradient for Adam optimizer:
                    h = torch.dot(grad.view(-1),
                                  torch.div(exp_avg, exp_avg_sq.sqrt().add_(group['eps'])).view(-1)) * math.sqrt(
                        prev_bias_correction2) / prev_bias_correction1
                    h = -h

                    # Hypergradient Adam (HD Adam) for the learning rate:
                    exp_avg_h.mul_(beta1_h).add_(1 - beta1_h, h)
                    exp_avg_h_sq.mul_(beta2_h).addcmul_(1 - beta2_h, h, h)
                    denom_ = exp_avg_h_sq.sqrt().add_(group['lr_eps'])
                    # denom_ = torch.sum(exp_avg_sq).add_(group['lr_eps'])

                    bias_correction1_ = 1 - beta1_h ** state['step']
                    bias_correction2_ = 1 - beta2_h ** state['step']
                    step_size_ = group['hypergrad_lr'] * math.sqrt(bias_correction2_) / bias_correction1_

                    group['lr'] = group['lr'] - step_size_ * exp_avg_h / denom_

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss