import pandas as pd
import os

project_root = r"C:\Users\zekib\PycharmProjects\PythonProject\eeg-feeling-emotions"
csv_path = os.path.join(project_root, "data", "raw", "emotions.csv")

df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("First 10 columns:", df.columns[:10].tolist())
print("Last 5 columns:", df.columns[-5:].tolist())

if "label" in df.columns:
    print("\nLabel value counts:")
    print(df["label"].value_counts())
else:
    print("\nNo 'label' column found. Last column is:", df.columns[-1])
