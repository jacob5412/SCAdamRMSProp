import torch


class GenericAdamCER:
    """
    Implementation of a customized Adam optimizer variant (CER - Custom Exponential Regularization).
    Includes dynamic learning rate and parameter constraints.
    """

    def __init__(self, x_init, F=(-1, 1), beta1=0.9, beta2=0.999, epsilon=1e-8, r=0.5, device="cpu"):
        """
        Initializes the optimizer with the given parameters.

        Args:
        - x_init: Initial value of the parameter to optimize.
        - F: Tuple (min, max) representing the bounds for clamping the parameter values.
        - beta1: Coefficient for the first moment estimate (default: 0.9).
        - beta2: Coefficient for the second moment estimate (default: 0.999).
        - epsilon: Small value to prevent division by zero (default: 1e-8).
        - r: Custom decay parameter controlling the dynamic adjustment of learning rate (default: 0.5).
        - device: The device to store tensors (default: "cpu").
        """
        self.x = torch.tensor(x_init, dtype=torch.float32, device=device, requires_grad=False)  # Optimized parameter
        self.F = F  # Bounds for the parameter values
        self.beta1 = beta1  # First moment decay coefficient
        self.beta2 = beta2  # Second moment decay coefficient
        self.epsilon = epsilon  # Numerical stability constant
        self.r = r  # Custom decay control parameter

        self.m = torch.zeros_like(self.x)  # First moment (momentum)
        self.v = torch.zeros_like(self.x)  # Second moment (variance)
        self.t = 0  # Time step counter

    def step(self, grad):
        """
        Performs a single optimization step.

        Args:
        - grad: Gradient of the loss with respect to the parameter.

        Returns:
        - The updated parameter value.
        """
        self.t += 1  # Increment the time step

        # Compute the base learning rate adjustment
        alpha_t = 0.5 / torch.sqrt(torch.tensor(self.t, dtype=torch.float32, device=self.x.device))

        # Ensure the gradient is a tensor
        if not isinstance(grad, torch.Tensor):
            grad = torch.tensor(grad, dtype=torch.float32, device=self.x.device)

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        # Correct bias for first and second moments
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Compute \(\theta_t = 1 - \frac{0.01 + 0.99r}{t^r}\)
        theta_t = 1 - ((0.01 + 0.99 * self.r) / (self.t**self.r))

        # Compute the step size for the update
        step_size = alpha_t * theta_t / (torch.sqrt(v_hat) + self.epsilon)

        # Update the parameter
        self.x = self.x - step_size * m_hat

        # Clamp the parameter to the specified bounds
        self.x = torch.clamp(self.x, self.F[0], self.F[1])

        return self.x.item()
