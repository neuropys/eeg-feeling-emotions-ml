import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from dataset_emotions import load_emotions_dataset, train_test_split_scaled


def main():
    # ---- Paths ----
    project_root = r"C:\Users\zekib\PycharmProjects\PythonProject\eeg-feeling-emotions"
    csv_path = os.path.join(project_root, "data", "raw", "emotions.csv")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # ---- 1. Load + split data ----
    X, y, le, feature_names = load_emotions_dataset(csv_path)
    X_train, X_test, y_train, y_test, scaler = train_test_split_scaled(X, y, test_size=0.2)

    # ---- 2. Train RF again (same as train_rf.py) ----
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Check performance to be sure
    y_pred = rf.predict(X_test)
    print("RandomForest test performance:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ---- 3. Raw feature importances ----
    importances = np.array(rf.feature_importances_)  # shape: (n_features,)
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    # Sort descending
    fi_df = fi_df.sort_values("importance", ascending=False)

    # Save full list
    full_path = os.path.join(processed_dir, "rf_feature_importances_all.csv")
    fi_df.to_csv(full_path, index=False)
    print(f"\nSaved full feature importance table to: {full_path}")

    print("\nTop 20 most important features:")
    print(fi_df.head(20))

    # ---- 4. Plot top 20 ----
    top_k = 20
    top_df = fi_df.head(top_k)
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Top 20 RF feature importances")
    plt.tight_layout()

    top_plot = os.path.join(processed_dir, "rf_feature_importances_top20.png")
    plt.savefig(top_plot, dpi=200)
    print(f"\nSaved top-20 importance plot to: {top_plot}")
    plt.show()

    # ---- 5. Aggregate by channel and feature type ----
    # We assume names like 'mean_0_a', 'fft_741_b', potentially with a leading '# '
    channel_importance = {}
    type_importance = {}

    for feat, imp in zip(feature_names, importances):
        # Remove any leading '# ' characters
        clean = feat.lstrip("# ").strip()

        parts = clean.split("_")

        # feature type is the first part: 'mean', 'std', 'fft', ...
        feat_type = parts[0] if len(parts) > 1 else "unknown"

        # channel code is assumed to be the last part: 'a', 'b', 'c', 'd'
        chan_code = parts[-1]
        if chan_code not in ["a", "b", "c", "d"]:
            chan_code = "unknown"

        # accumulate importance per channel
        channel_importance[chan_code] = channel_importance.get(chan_code, 0.0) + imp

        # accumulate importance per feature type
        type_importance[feat_type] = type_importance.get(feat_type, 0.0) + imp

    # Convert to DataFrames & normalize to percentage
    chan_df = pd.DataFrame(
        [{"channel_code": k, "importance": v} for k, v in channel_importance.items()]
    )
    chan_df["importance_pct"] = chan_df["importance"] / chan_df["importance"].sum() * 100.0
    chan_df = chan_df.sort_values("importance_pct", ascending=False)

    type_df = pd.DataFrame(
        [{"feature_type": k, "importance": v} for k, v in type_importance.items()]
    )
    type_df["importance_pct"] = type_df["importance"] / type_df["importance"].sum() * 100.0
    type_df = type_df.sort_values("importance_pct", ascending=False)

    # Save them
    chan_path = os.path.join(processed_dir, "rf_channel_importance.csv")
    type_path = os.path.join(processed_dir, "rf_featuretype_importance.csv")

    chan_df.to_csv(chan_path, index=False)
    type_df.to_csv(type_path, index=False)

    print("\nChannel-level importance:")
    print(chan_df)

    print("\nFeature-type-level importance:")
    print(type_df)

    # ---- 6. Optional: bar plots for channel + type ----
    plt.figure(figsize=(6, 4))
    plt.bar(chan_df["channel_code"], chan_df["importance_pct"])
    plt.ylabel("Importance (%)")
    plt.xlabel("Channel code (a/b/c/d)")
    plt.title("RF importance per channel group")
    plt.tight_layout()
    chan_plot = os.path.join(processed_dir, "rf_channel_importance.png")
    plt.savefig(chan_plot, dpi=200)
    print(f"\nSaved channel importance plot to: {chan_plot}")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar(type_df["feature_type"], type_df["importance_pct"])
    plt.ylabel("Importance (%)")
    plt.xlabel("Feature type")
    plt.title("RF importance per feature type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    type_plot = os.path.join(processed_dir, "rf_featuretype_importance.png")
    plt.savefig(type_plot, dpi=200)
    print(f"Saved feature-type importance plot to: {type_plot}")
    plt.show()


if __name__ == "__main__":
    main()
