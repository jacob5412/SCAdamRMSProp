import torch
import torch.optim as optim


class AMSGrad(optim.Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9, theta=0.999, epsilon=1e-8):
        defaults = dict(lr=lr, beta=beta, theta=theta, epsilon=epsilon)
        super(AMSGrad, self).__init__(params, defaults)

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
                    state["v_hat"] = torch.zeros_like(p.data)

                m, v, v_hat = state["m"], state["v"], state["v_hat"]
                beta, theta, epsilon = group["beta"], group["theta"], group["epsilon"]

                state["step"] += 1
                t = state["step"]
                alpha_t = group["lr"] / (t**0.5)

                m.mul_(beta).add_(1 - beta, grad)
                v.mul_(theta).addcmul_(1 - theta, grad, grad)
                torch.maximum(v_hat, v, out=v_hat)

                p.data.addcdiv_(-alpha_t, m, (v_hat.sqrt() + epsilon))

        return loss
