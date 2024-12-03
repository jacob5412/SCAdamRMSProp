import torch
import torch.optim as optim


class RMSProp(optim.Optimizer):
    def __init__(self, params, alpha=0.001, epsilon=1e-8):
        """
        RMSProp optimizer with step-size decay.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            alpha (float): Base learning rate (default: 0.001).
            epsilon (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        """
        defaults = dict(alpha=alpha, epsilon=epsilon)
        super(RMSProp, self).__init__(params, defaults)

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
                if grad.is_sparse:
                    raise RuntimeError("RMSProp does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(p.data)

                square_avg = state["square_avg"]
                state["step"] += 1
                step = state["step"]
                alpha = group["alpha"]
                epsilon = group["epsilon"]

                # Dynamic learning rate and theta_t
                alpha_t = alpha / (step**0.5)  # Dynamic LR
                theta_t = 1 - (1 / step)  # Decay factor for squared gradients

                # Update the moving average of squared gradients
                square_avg.mul_(theta_t).addcmul_(1 - theta_t, grad, grad)

                # Parameter update
                p.data.addcdiv_(-alpha_t, grad, square_avg.sqrt().add(epsilon))

        return loss