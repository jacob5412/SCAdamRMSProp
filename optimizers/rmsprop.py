import torch
import torch.optim as optim


class RMSProp(optim.Optimizer):
    def __init__(self, params, alpha=0.001, beta=0.0, epsilon=1e-8):
        defaults = dict(alpha=alpha, beta=beta, epsilon=epsilon)
        super(RMSProp, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("RMSProp does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(p.data)
                    state["momentum"] = torch.zeros_like(p.data)

                square_avg = state["square_avg"]
                momentum = state["momentum"]
                state["step"] += 1
                step = state["step"]

                alpha = group["alpha"]
                beta = group["beta"]
                epsilon = group["epsilon"]

                alpha_t = alpha / (step**0.5)
                theta_t = 1 - (1 / step)

                square_avg.mul_(theta_t).addcmul_(1 - theta_t, grad, grad)

                momentum.mul_(beta).add_(1 - beta, grad)

                p.data.addcdiv_(-alpha_t, momentum, square_avg.sqrt().add(epsilon))

        return loss
