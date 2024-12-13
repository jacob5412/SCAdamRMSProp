import torch
import torch.optim as optim


class GenericAdam(optim.Optimizer):
    """
    A custom implementation of the Adam optimizer with additional control over the second moment
    decay rate using a dynamic scaling factor theta_t.
    """

    def __init__(self, params, lr=0.001, beta=0.9, r=0.5, epsilon=1e-8):
        """
        Initializes the GenericAdam optimizer.

        Args:
        - params: Model parameters to optimize.
        - lr: Learning rate (default: 0.001).
        - beta: Coefficient for the first moment estimate (default: 0.9).
        - r: Exponent controlling the decay of theta_t (default: 0.5).
        - epsilon: Small value to prevent division by zero (default: 1e-8).
        """
        defaults = dict(lr=lr, beta=beta, r=r, epsilon=epsilon)
        super(GenericAdam, self).__init__(params, defaults)

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
                    continue  # Skip if there is no gradient for this parameter

                grad = p.grad.data  # Gradient of the parameter
                state = self.state[p]  # State dictionary for the parameter

                # Initialize state variables if not already present
                if len(state) == 0:
                    state["step"] = 0  # Step counter
                    state["m"] = torch.zeros_like(p.data)  # First moment (momentum)
                    state["v"] = torch.zeros_like(p.data)  # Second moment (variance)

                # Retrieve state variables and hyperparameters
                m, v = state["m"], state["v"]
                beta, r, epsilon = group["beta"], group["r"], group["epsilon"]

                # Increment the step counter
                state["step"] += 1
                t = state["step"]

                # Compute the learning rate scaling factor
                alpha_t = group["lr"] / (t**0.5)
                # Compute the dynamic scaling factor theta_t
                theta_t = 1 - ((0.001 + 0.999 * r) / (t**r))

                # Update biased first moment estimate
                m.mul_(beta).add_(1 - beta, grad)
                # Update biased second moment estimate with theta_t
                v.mul_(theta_t).addcmul_(1 - theta_t, grad, grad)

                # Update the parameter using Adam update rule with theta_t adjustment
                p.data.addcdiv_(-alpha_t, m, (v.sqrt() + epsilon))

        return loss
