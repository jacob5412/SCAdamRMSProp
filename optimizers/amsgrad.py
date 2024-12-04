import torch
import torch.optim as optim


class AMSGrad(optim.Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9, theta=0.999, epsilon=1e-8):
        """
        Implements AMSGrad optimizer using beta and theta.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate (default: 0.001).
            beta (float): Coefficient for the first moment estimate (default: 0.9).
            theta (float): Coefficient for the second moment estimate (default: 0.999).
            epsilon (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        """
        defaults = dict(lr=lr, beta=beta, theta=theta, epsilon=epsilon)
        super(AMSGrad, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss (float): The loss value, if closure is provided; otherwise, None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)  # First moment
                    state["v"] = torch.zeros_like(p.data)  # Second moment
                    state["v_hat"] = torch.zeros_like(p.data)  # Max of second moment

                m, v, v_hat = state["m"], state["v"], state["v_hat"]
                beta, theta, epsilon = group["beta"], group["theta"], group["epsilon"]

                # Update step
                state["step"] += 1
                t = state["step"]
                alpha_t = group["lr"] / (t**0.5)  # Learning rate decay dynamically

                # Update moments
                m.mul_(beta).add_(1 - beta, grad)  # First moment
                v.mul_(theta).addcmul_(1 - theta, grad, grad)  # Second moment
                torch.maximum(v_hat, v, out=v_hat)  # Keep the maximum value of v

                # Parameter update
                p.data.addcdiv_(-alpha_t, m, (v_hat.sqrt() + epsilon))

        return loss
