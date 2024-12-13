import torch
import torch.optim as optim


class RMSProp(optim.Optimizer):
    """
    A custom implementation of the RMSProp optimizer.
    Includes dynamic learning rate adjustments and momentum updates for optimization.
    """

    def __init__(self, params, alpha=0.001, beta=0.0, epsilon=1e-8):
        """
        Initializes the RMSProp optimizer.

        Args:
        - params: Model parameters to optimize.
        - alpha: Base learning rate (default: 0.001).
        - beta: Momentum factor (default: 0.0).
        - epsilon: Term added to the denominator for numerical stability (default: 1e-8).
        """
        defaults = dict(alpha=alpha, beta=beta, epsilon=epsilon)
        super(RMSProp, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
        - closure: A closure that reevaluates the model and returns the loss (optional).

        Returns:
        - The loss value if a closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Loop over all parameter groups
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue  # Skip if there is no gradient

                grad = p.grad.data  # Gradient of the parameter
                if grad.is_sparse:
                    raise RuntimeError("RMSProp does not support sparse gradients")

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0  # Step counter
                    state["square_avg"] = torch.zeros_like(p.data)  # Running average of squared gradients
                    state["momentum"] = torch.zeros_like(p.data)  # Momentum buffer

                square_avg = state["square_avg"]  # Retrieve running average of squared gradients
                momentum = state["momentum"]  # Retrieve momentum buffer
                state["step"] += 1  # Increment the step counter
                step = state["step"]

                # Retrieve hyperparameters
                alpha = group["alpha"]
                beta = group["beta"]
                epsilon = group["epsilon"]

                # Compute dynamic learning rate and scaling factor
                alpha_t = alpha / (step**0.5)  # Time-decayed learning rate
                theta_t = 1 - (1 / step)  # Dynamic scaling factor

                # Update the running average of squared gradients
                square_avg.mul_(theta_t).addcmul_(1 - theta_t, grad, grad)

                # Update the momentum term
                momentum.mul_(beta).add_(1 - beta, grad)

                # Update the parameter using RMSProp formula with adjustments
                p.data.addcdiv_(-alpha_t, momentum, square_avg.sqrt().add(epsilon))

        return loss
