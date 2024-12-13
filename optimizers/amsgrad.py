import torch
import torch.optim as optim


class AMSGrad(optim.Optimizer):
    """
    Implementation of the AMSGrad variant of the Adam optimizer.
    AMSGrad improves the convergence properties of Adam by using a maximum of past squared gradients
    for normalization instead of an exponential average.
    """

    def __init__(self, params, lr=0.001, beta=0.9, theta=0.999, epsilon=1e-8):
        """
        Initializes the AMSGrad optimizer with the given parameters.

        Args:
        - params: Model parameters to optimize.
        - lr: Learning rate (default: 0.001).
        - beta: Coefficient for the first moment estimate (default: 0.9).
        - theta: Coefficient for the second moment estimate (default: 0.999).
        - epsilon: Small value to prevent division by zero (default: 1e-8).
        """
        defaults = dict(lr=lr, beta=beta, theta=theta, epsilon=epsilon)
        super(AMSGrad, self).__init__(params, defaults)

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

        # Iterate over all parameter groups
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue  # Skip if no gradient is available

                grad = p.grad.data  # Gradient of the parameter
                state = self.state[p]  # State dictionary for the parameter

                # Initialize state variables if not already present
                if len(state) == 0:
                    state["step"] = 0  # Step counter
                    state["m"] = torch.zeros_like(p.data)  # First moment estimate
                    state["v"] = torch.zeros_like(p.data)  # Second moment estimate
                    state["v_hat"] = torch.zeros_like(p.data)  # Maximum second moment

                # Retrieve state variables
                m, v, v_hat = state["m"], state["v"], state["v_hat"]
                beta, theta, epsilon = group["beta"], group["theta"], group["epsilon"]

                # Increment the step counter
                state["step"] += 1
                t = state["step"]
                # Compute the time-adjusted learning rate
                alpha_t = group["lr"] / (t**0.5)

                # Update biased first moment estimate
                m.mul_(beta).add_(1 - beta, grad)
                # Update biased second moment estimate
                v.mul_(theta).addcmul_(1 - theta, grad, grad)
                # Update the maximum of second moment estimate
                torch.maximum(v_hat, v, out=v_hat)

                # Update the parameter using AMSGrad formula
                p.data.addcdiv_(-alpha_t, m, (v_hat.sqrt() + epsilon))

        return loss
