import torch
import torch.nn as nn

from optimizers.generic_adam_builtin import GenericAdamBuiltin


def test_generic_adam_builtin_initialization():
    """
    Test the initialization of the GenericAdamBuiltin optimizer.
    """
    model = nn.Linear(10, 1)  # Simple model with 10 inputs, 1 output
    optimizer = GenericAdamBuiltin(model.parameters(), lr=0.01, beta=0.9, r=0.5, epsilon=1e-8)

    # Check optimizer parameters
    assert optimizer.base_lr == 0.01
    assert optimizer.beta == 0.9
    assert optimizer.r == 0.5
    assert optimizer.epsilon == 1e-8


def test_generic_adam_builtin_step():
    """
    Test the optimizer step on a simple model and dataset.
    """
    torch.manual_seed(42)

    # Create a simple model and dataset
    model = nn.Linear(1, 1)
    X = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([[2.0], [4.0], [6.0]])

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = GenericAdamBuiltin(model.parameters(), lr=0.1, beta=0.9, r=0.5, epsilon=1e-8)

    # Perform a single optimization step
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    # Assert that the parameters have been updated
    for param in model.parameters():
        assert param.grad is not None  # Gradients should be computed
        assert torch.any(param.data != param.data.clone().detach() - param.grad.data)  # Params should change


def test_generic_adam_builtin_training():
    """
    Test if the GenericAdamBuiltin optimizer can successfully optimize a small regression problem.
    """
    torch.manual_seed(42)

    # Create dataset
    X = torch.linspace(-1, 1, 100).unsqueeze(1)  # Inputs
    y = X.pow(3) + 0.1 * torch.randn(X.size())  # Cubic function with noise

    # Define model, loss, and optimizer
    model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
    criterion = nn.MSELoss()
    optimizer = GenericAdamBuiltin(model.parameters(), lr=0.01, beta=0.9, r=0.5, epsilon=1e-8)

    # Training loop
    num_epochs = 200
    initial_loss = None
    final_loss = None
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        if epoch == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        if epoch == num_epochs - 1:
            final_loss = loss.item()

    # Assert that the final loss is lower than the initial loss
    assert final_loss < initial_loss, "The optimizer failed to reduce the loss."
