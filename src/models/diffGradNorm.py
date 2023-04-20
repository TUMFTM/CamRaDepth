import math
import torch
from torch.optim.optimizer import Optimizer
from torch import linalg as LA

version_higher = ( torch.__version__ >= "1.5.0" )

class diffGradNorm(Optimizer):
    r"""Implements diffGradNorm algorithm. Modified from diffGrad in PyTorch
    Arguments:
        args (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    reference: AdaNorm: Adaptive Gradient Norm Correction based Optimizer for CNNs
               WACV 2023
    """

    def __init__(self, args, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(diffGradNorm, self).__init__(args, defaults)

    def __setstate__(self, state):
        super(diffGradNorm, self).__setstate__(state)

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
                    raise RuntimeError('diffGradNorm does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Previous gradient
                    state['previous_grad'] = torch.zeros_like(p.data)
                    state['exp_grad_norm'] = 0

                exp_avg, exp_avg_sq, previous_grad, exp_grad_norm = state['exp_avg'], state['exp_avg_sq'], state['previous_grad'], state['exp_grad_norm']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Gradient Norm Correction
                grad_norm = LA.norm(grad)
                exp_grad_norm = 0.95*exp_grad_norm + 0.05*grad_norm
                if exp_grad_norm > grad_norm:
                    grad1 = grad * exp_grad_norm / (grad_norm + 1e-8)
                else:
                    grad1 = grad
                state['exp_grad_norm'] = exp_grad_norm.clone()
                
                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad1, alpha=1 - beta1)
                # exp_avg.mul_(beta1).add_(1 - beta1, grad1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # computer diffgrad coefficient
                diff = abs(previous_grad - grad)
                #print(diff)
                dfc = 1. / (1. + torch.exp(-diff))
                state['previous_grad'] = grad.clone()

                exp_avg1 = exp_avg * dfc

                step_size = group['lr'] * math.sqrt(bias_correction2) / (bias_correction1 + 1e-8)

                p.data.addcdiv_(exp_avg1, denom, value=-step_size)


        return loss

