from torch.optim import RMSprop


class RMSPropBuiltin:
    """
    Wrapper around PyTorch's built-in RMSprop optimizer to include dynamic learning rate
    and squared gradient decay adjustments.
    """

    def __init__(self, params, alpha=0.001, beta=0.0, epsilon=1e-8):
        """
        Args:
            params: Iterable of parameters to optimize.
            alpha: Base learning rate.
            beta: Momentum factor.
            epsilon: Term added to the denominator for numerical stability.
        """
        self.base_alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.step_count = 0
        self.optimizer = RMSprop(params, lr=alpha, momentum=beta, eps=epsilon)

    def step(self, closure=None):
        # Increment the step count
        self.step_count += 1
        step = self.step_count

        # Compute dynamic learning rate and decay factor
        alpha_t = self.base_alpha / (step**0.5)  # Dynamic learning rate
        theta_t = 1 - (1 / step)  # Decay factor for squared gradients

        # Update learning rates for each parameter group
        for group in self.optimizer.param_groups:
            group["lr"] = alpha_t
            group["theta_t"] = theta_t  # Store theta_t for debugging or tracking purposes

        # Perform the optimization step
        return self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()
