import torch


class GenericAdamCER:
    def __init__(self, x_init, F=(-1, 1), beta1=0.9, beta2=0.999, epsilon=1e-8, r=0.5, device="cpu"):
        self.x = torch.tensor(x_init, dtype=torch.float32, device=device, requires_grad=False)
        self.F = F
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.r = r

        self.m = torch.zeros_like(self.x)
        self.v = torch.zeros_like(self.x)
        self.t = 0

    def step(self, grad):
        self.t += 1

        alpha_t = 0.5 / torch.sqrt(torch.tensor(self.t, dtype=torch.float32, device=self.x.device))

        if not isinstance(grad, torch.Tensor):
            grad = torch.tensor(grad, dtype=torch.float32, device=self.x.device)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Compute \(\theta_t = 1 - \frac{0.01 + 0.99r}{t^r}\)
        theta_t = 1 - ((0.01 + 0.99 * self.r) / (self.t**self.r))

        # Compute step size
        step_size = alpha_t * theta_t / (torch.sqrt(v_hat) + self.epsilon)

        self.x = self.x - step_size * m_hat
        self.x = torch.clamp(self.x, self.F[0], self.F[1])

        return self.x.item()
