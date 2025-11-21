import os
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset_emotions import load_emotions_dataset, train_test_split_scaled
from models_mlp import MLPNet, create_dataloaders, train_mlp, predict_mlp


def main():
    # ---- Paths ----
    project_root = r"C:\Users\zekib\PycharmProjects\eeg-feeling-emotions"
    csv_path = os.path.join(project_root, "data", "raw", "emotions.csv")

    # ---- Device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- 1. Load dataset ----
    X, y, le, feature_names = load_emotions_dataset(csv_path)
    print("X shape:", X.shape)   # e.g., (2132, 2548)
    print("y shape:", y.shape)
    print("Classes:", le.classes_)

    # ---- 2. Train/test split + scaling ----
    X_train, X_test, y_train, y_test, scaler = train_test_split_scaled(
        X, y, test_size=0.2, random_state=42
    )

    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)

    print("Input dim:", input_dim)
    print("Number of classes:", num_classes)

    # ---- 3. Create DataLoaders ----
    train_loader, test_loader = create_dataloaders(
        X_train, y_train,
        X_test, y_test,
        batch_size=64,
        device=device
    )

    # ---- 4. Define MLP model ----
    hidden_dims = [256, 128]  # you can tune this
    model = MLPNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
        dropout=0.3
    )

    # ---- 5. Train model ----
    print("\nTraining MLP...")
    model, history = train_mlp(
        model,
        train_loader=train_loader,
        val_loader=test_loader,  # using test as "validation" here for simplicity
        num_epochs=25,
        lr=1e-3,
        device=device
    )

    # ---- 6. Evaluate on test set ----
    print("\nEvaluating on test set...")
    y_pred = predict_mlp(model, test_loader, device=device)

    print("\nClassification report (MLP):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nConfusion matrix (MLP):")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
