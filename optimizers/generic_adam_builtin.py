from torch.optim import Adam


class GenericAdamBuiltin:

    def __init__(self, params, lr=0.001, beta=0.9, r=0.5, epsilon=1e-8):
        self.base_lr = lr
        self.beta = beta
        self.r = r
        self.epsilon = epsilon
        self.t = 0
        self.optimizer = Adam(params, lr=lr, betas=(beta, 0.999), eps=epsilon)

    def step(self, closure=None):
        self.t += 1

        alpha_t = self.base_lr / (self.t**0.5)
        theta_t = 1 - ((0.001 + 0.999 * self.r) / (self.t**self.r))

        for group in self.optimizer.param_groups:
            group["lr"] = alpha_t
            group["theta_t"] = theta_t

        return self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()
