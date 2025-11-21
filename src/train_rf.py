import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from dataset_emotions import load_emotions_dataset, train_test_split_scaled


def main():
    project_root = r"C:\Users\zekib\PycharmProjects\PythonProject\eeg-feeling-emotions"
    csv_path = os.path.join(project_root, "data", "raw", "emotions.csv")

    # 1. Load dataset
    X, y, le, feature_names = load_emotions_dataset(csv_path)
    print("X shape:", X.shape)   # e.g., (2132, 2548)
    print("y shape:", y.shape)
    print("Classes:", le.classes_)  # ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

    # 2. Split & scale
    X_train, X_test, y_train, y_test, scaler = train_test_split_scaled(X, y, test_size=0.2)

    # 3. Train RandomForest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = rf.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
