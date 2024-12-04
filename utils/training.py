import time

import torch
from sklearn.metrics import accuracy_score


def train_model(
    model, train_loader, test_loader, device, criterion, optimizer_class, optimizer_kwargs, num_epochs, print_epochs=1
):
    """
    Train the model using a specified optimizer with configurable parameters.

    Args:
        model (nn.Module): The neural network model (e.g., LeNet).
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to train on (CPU/GPU).
        criterion (Loss): Loss function (e.g., CrossEntropyLoss).
        optimizer_class (class): Optimizer class (e.g., GenericAdam, SGD).
        optimizer_kwargs (dict): Keyword arguments for the optimizer.
        num_epochs (int): Number of epochs.
        print_epochs (int): Frequency of printing results.

    Returns:
        dict: Training losses, test losses, training accuracies, test accuracies.
    """
    # Initialize optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    # Metrics to store
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Measure training accuracy and loss
        train_loss = epoch_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

        # Measure test accuracy and loss
        test_loss /= len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Measure epoch duration
        epoch_duration = time.time() - start_time

        # Print metrics at the specified interval
        if (epoch + 1) % print_epochs == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, "
                f"Time: {epoch_duration:.2f}s"
            )

    # Return all metrics
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
    }
