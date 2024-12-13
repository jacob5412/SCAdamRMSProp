from torch.optim import Adam


class GenericAdamBuiltin:
    """
    A generic wrapper around the PyTorch Adam optimizer that allows for custom modifications
    to the learning rate and parameter updates with additional control variables.
    """

    def __init__(self, params, lr=0.001, beta=0.9, r=0.5, epsilon=1e-8):
        """
        Initializes the GenericAdamBuiltin optimizer.

        Args:
        - params: Parameters of the model to be optimized.
        - lr: Base learning rate (default: 0.001).
        - beta: Coefficient for the first moment estimate (default: 0.9).
        - r: A custom parameter to control the decay rate (default: 0.5).
        - epsilon: Small value to prevent division by zero (default: 1e-8).
        """
        self.base_lr = lr  # Store the base learning rate
        self.beta = beta  # First moment decay coefficient
        self.r = r  # Decay control parameter
        self.epsilon = epsilon  # Numerical stability constant
        self.t = 0  # Time step counter

        # Initialize the Adam optimizer with default settings for second moment coefficient and epsilon
        self.optimizer = Adam(params, lr=lr, betas=(beta, 0.999), eps=epsilon)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
        - closure: A closure that reevaluates the model and returns the loss (optional).

        Returns:
        - The result of the Adam optimizer's step method.
        """
        self.t += 1  # Increment the time step

        # Compute the learning rate adjustment factor
        alpha_t = self.base_lr / (self.t**0.5)
        # Compute a custom scaling factor based on parameter r
        theta_t = 1 - ((0.001 + 0.999 * self.r) / (self.t**self.r))

        # Update the learning rate and custom factor for each parameter group
        for group in self.optimizer.param_groups:
            group["lr"] = alpha_t  # Adjust learning rate
            group["theta_t"] = theta_t  # Custom scaling factor

        # Perform the optimization step
        return self.optimizer.step(closure)

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()
