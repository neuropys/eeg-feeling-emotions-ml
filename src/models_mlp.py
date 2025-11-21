import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class MLPNet(nn.Module):
    """
    Simple fully-connected neural network (Multi-Layer Perceptron)
    for EEG feature classification.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Final classification layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, input_dim)
        return self.net(x)


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size: int = 64, device: str = "cpu"):
    """
    Convert numpy arrays to PyTorch tensors and create DataLoaders.
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu"
):
    """
    Train the MLP model using cross-entropy loss and Adam optimizer.

    Returns
    -------
    model : nn.Module
        Trained model.
    history : dict
        Training (and optionally validation) loss per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        # ---- Training ----
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(X_batch)          # (batch_size, num_classes)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        # ---- Validation ----
        epoch_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    val_loss += loss.item() * X_val.size(0)

            epoch_val_loss = val_loss / len(val_loader.dataset)
            history["val_loss"].append(epoch_val_loss)

        if epoch_val_loss is not None:
            print(f"Epoch {epoch:02d} | train_loss={epoch_train_loss:.4f} | val_loss={epoch_val_loss:.4f}")
        else:
            print(f"Epoch {epoch:02d} | train_loss={epoch_train_loss:.4f}")

    return model, history


def predict_mlp(model: nn.Module, data_loader: DataLoader, device: str = "cpu"):
    """
    Run the model on all batches in data_loader and return predicted class indices.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            outputs = model(X_batch.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())

    return torch.cat(all_preds, dim=0).numpy()
