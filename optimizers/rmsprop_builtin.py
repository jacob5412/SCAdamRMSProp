from torch.optim import RMSprop


class RMSPropBuiltin:
    """
    A custom wrapper around PyTorch's RMSProp optimizer.
    Includes dynamic learning rate adjustment and a scaling factor for experimentation.
    """

    def __init__(self, params, alpha=0.001, beta=0.0, epsilon=1e-8):
        """
        Initializes the RMSPropBuiltin optimizer.

        Args:
        - params: Parameters to be optimized.
        - alpha: Base learning rate (default: 0.001).
        - beta: Momentum factor (default: 0.0).
        - epsilon: Term added to the denominator for numerical stability (default: 1e-8).
        """
        self.base_alpha = alpha  # Store the base learning rate
        self.beta = beta  # Momentum factor
        self.epsilon = epsilon  # Small value to avoid division by zero
        self.step_count = 0  # Counter to track optimization steps

        # Initialize the RMSprop optimizer
        self.optimizer = RMSprop(params, lr=alpha, momentum=beta, eps=epsilon)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
        - closure: A closure that reevaluates the model and returns the loss (optional).

        Returns:
        - The result of the RMSprop optimizer's step method.
        """
        # Increment the step count
        self.step_count += 1
        step = self.step_count

        # Compute time-adjusted learning rate (alpha_t)
        alpha_t = self.base_alpha / (step**0.5)

        # Compute a custom scaling factor (theta_t)
        theta_t = 1 - (1 / step)

        # Update the learning rate and scaling factor for each parameter group
        for group in self.optimizer.param_groups:
            group["lr"] = alpha_t  # Adjust the learning rate
            group["theta_t"] = theta_t  # Custom scaling factor (used for experimentation)

        # Perform the optimization step
        return self.optimizer.step(closure)

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()
