import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from dataset_emotions import load_emotions_dataset, train_test_split_scaled


def main():
    project_root = r"C:\Users\zekib\PycharmProjects\PythonProject\eeg-feeling-emotions"
    csv_path = os.path.join(project_root, "data", "raw", "emotions.csv")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load data
    X, y, le, feature_names = load_emotions_dataset(csv_path)
    X_train, X_test, y_train, y_test, scaler = train_test_split_scaled(X, y, test_size=0.2)

    # 2. Train RF (similar to train_rf.py)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 3. Check performance again (just to be sure)
    y_pred = rf.predict(X_test)
    print("RandomForest test performance:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 4. Feature importances
    importances = rf.feature_importances_  # shape: (n_features,)
    importances = np.array(importances)

    # 5. Create a DataFrame for easier analysis
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    # Sort by importance
    fi_df = fi_df.sort_values("importance", ascending=False)

    # Save all importances
    fi_path = os.path.join(processed_dir, "rf_feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"\nSaved full feature importances to: {fi_path}")

    # Show top 20 in console
    print("\nTop 20 most important features:")
    print(fi_df.head(20))

    # 6. Plot top 20
    top_k = 20
    top_df = fi_df.head(top_k)
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Top 20 RandomForest feature importances")
    plt.tight_layout()

    plot_path = os.path.join(processed_dir, "rf_feature_importances_top20.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved top-20 feature importance plot to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
