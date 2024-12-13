from torch.optim import Adam


class AMSGradBuiltin:
    """
    A custom optimizer wrapper around PyTorch's Adam optimizer with AMSGrad option.
    Modifies the learning rate schedule based on the current step.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, amsgrad=False):
        """
        Initializes the optimizer with given parameters.

        Args:
        - params: Model parameters to optimize.
        - lr: Base learning rate (default: 0.001).
        - betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999)).
        - eps: Term added to denominator for numerical stability (default: 1e-8).
        - amsgrad: Boolean flag to enable AMSGrad variant of Adam (default: False).
        """
        self.base_lr = lr  # Store the base learning rate
        self.t = 0  # Time step counter
        # Initialize the Adam optimizer with AMSGrad option
        self.optimizer = Adam(params, lr=lr, betas=betas, eps=eps, amsgrad=amsgrad)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
        - closure: A closure that reevaluates the model and returns the loss (optional).

        Returns:
        - The result of the optimizer's step method.
        """
        self.t += 1  # Increment the step counter

        # Adjust the learning rate for each parameter group based on the step
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr / (self.t**0.5)

        # Perform the optimization step
        return self.optimizer.step(closure)

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()
