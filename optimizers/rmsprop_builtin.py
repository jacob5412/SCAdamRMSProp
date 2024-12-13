from torch.optim import RMSprop


class RMSPropBuiltin:

    def __init__(self, params, alpha=0.001, beta=0.0, epsilon=1e-8):
        self.base_alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.step_count = 0
        self.optimizer = RMSprop(params, lr=alpha, momentum=beta, eps=epsilon)

    def step(self, closure=None):
        self.step_count += 1
        step = self.step_count

        alpha_t = self.base_alpha / (step**0.5)
        theta_t = 1 - (1 / step)

        for group in self.optimizer.param_groups:
            group["lr"] = alpha_t
            group["theta_t"] = theta_t

        return self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()
