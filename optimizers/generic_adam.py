import torch
import torch.optim as optim


class GenericAdam(optim.Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9, r=0.5, epsilon=1e-8):
        defaults = dict(lr=lr, beta=beta, r=r, epsilon=epsilon)
        super(GenericAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                beta, r, epsilon = group["beta"], group["r"], group["epsilon"]

                state["step"] += 1
                t = state["step"]
                alpha_t = group["lr"] / (t**0.5)
                theta_t = 1 - ((0.001 + 0.999 * r) / (t**r))

                m.mul_(beta).add_(1 - beta, grad)
                v.mul_(theta_t).addcmul_(1 - theta_t, grad, grad)

                p.data.addcdiv_(-alpha_t, m, (v.sqrt() + epsilon))

        return loss
