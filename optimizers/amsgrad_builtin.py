from torch.optim import Adam


class AMSGradBuiltin:
    """
    Wrapper around PyTorch's built-in Adam optimizer with a custom learning rate decay.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, amsgrad=False):
        self.base_lr = lr
        self.t = 0
        self.optimizer = Adam(params, lr=lr, betas=betas, eps=eps, amsgrad=amsgrad)

    def step(self, closure=None):
        # Increment the step count
        self.t += 1

        # Dynamically adjust the learning rate
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr / (self.t**0.5)

        # Perform a step
        return self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()
