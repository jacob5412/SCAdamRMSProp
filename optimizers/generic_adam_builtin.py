from torch.optim import Adam


class GenericAdamBuiltin:
    """
    Wrapper around PyTorch's built-in Adam optimizer with custom learning rate
    and theta decay.
    """

    def __init__(self, params, lr=0.001, beta=0.9, r=0.5, epsilon=1e-8):
        """
        Args:
            params: Iterable of parameters to optimize.
            lr: Base learning rate.
            beta: Coefficient for the first moment estimate.
            r: Decay factor for theta_t.
            epsilon: Term added to the denominator for numerical stability.
        """
        self.base_lr = lr
        self.beta = beta
        self.r = r
        self.epsilon = epsilon
        self.t = 0
        self.optimizer = Adam(params, lr=lr, betas=(beta, 0.999), eps=epsilon)

    def step(self, closure=None):
        # Increment the step count
        self.t += 1

        # Dynamically adjust the learning rate and theta
        alpha_t = self.base_lr / (self.t**0.5)
        theta_t = 1 - ((0.001 + 0.999 * self.r) / (self.t**self.r))

        # Update learning rates for each parameter group
        for group in self.optimizer.param_groups:
            group["lr"] = alpha_t
            group["theta_t"] = theta_t  # Save theta_t for debugging purposes

        # Perform the optimization step
        return self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()
