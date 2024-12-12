import torch

class GenericAdamCER:
    def __init__(self, x_init, F=(-1, 1), beta1=0.9, beta2=0.999, epsilon=1e-8, r=0.5, device="cpu"):
        self.x = torch.tensor(x_init, dtype=torch.float32, device=device, requires_grad=False)  # Initial value
        self.F = F  # Constraint set
        self.beta1 = beta1  # First moment decay rate
        self.beta2 = beta2  # Second moment decay rate
        self.epsilon = epsilon  # Small number to avoid division by zero
        self.r = r  # Decay exponent for theta_t

        # Adam state variables
        self.m = torch.zeros_like(self.x)  # First moment estimate
        self.v = torch.zeros_like(self.x)  # Second moment estimate
        self.t = 0  # Time step

    def step(self, grad):
        self.t += 1

        # Compute learning rate \(\alpha_t = 0.5 / \sqrt{t}\)
        alpha_t = 0.5 / torch.sqrt(torch.tensor(self.t, dtype=torch.float32, device=self.x.device))

        # Convert grad to tensor if not already
        if not isinstance(grad, torch.Tensor):
            grad = torch.tensor(grad, dtype=torch.float32, device=self.x.device)

        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        # Correct bias in moment estimates
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Compute \(\theta_t = 1 - \frac{0.01 + 0.99r}{t^r}\)
        theta_t = 1 - ((0.01 + 0.99 * self.r) / (self.t ** self.r))

        # Compute step size
        step_size = alpha_t * theta_t / (torch.sqrt(v_hat) + self.epsilon)

        # Update x
        self.x = self.x - step_size * m_hat

        # Apply constraints (projection onto F)
        self.x = torch.clamp(self.x, self.F[0], self.F[1])

        return self.x.item()
